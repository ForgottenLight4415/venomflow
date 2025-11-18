import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from textblob import TextBlob


# --------------------------------------------------------
# Basic IO
# --------------------------------------------------------

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def join_fields(row: Dict[str, Any]) -> str:
    """Join main text fields into a single string."""
    parts = [
        row.get("prompt", ""),
        row.get("chosen", ""),
        row.get("rejected", ""),
    ]
    return " ".join(p for p in parts if p)


# --------------------------------------------------------
# Tokenization & n-grams
# --------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def build_ngrams(tokens: List[str], n: int = 1) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def count_ngrams(docs: List[str], n: int = 1) -> Counter:
    counter: Counter = Counter()
    for doc in docs:
        tokens = tokenize(doc)
        ngrams = build_ngrams(tokens, n=n)
        counter.update(ngrams)
    return counter


# --------------------------------------------------------
# Sentiment & simple stats
# --------------------------------------------------------

def estimate_sentiment(text: str) -> float:
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity  # -1 .. 1


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / max(len(values) - 1, 1)
    return m, math.sqrt(var)


# --------------------------------------------------------
# A. Automatic "candidate bias terms" via lift
# --------------------------------------------------------

def compute_lift_terms(
    clean_docs: List[str],
    poisoned_docs: List[str],
    n: int = 1,
    min_freq: int = 5,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Compute n-gram lift: freq_poisoned / (freq_clean + eps).
    Return top_k n-grams that are much more common in poisoned.
    """
    eps = 1e-6

    clean_counts = count_ngrams(clean_docs, n=n)
    poisoned_counts = count_ngrams(poisoned_docs, n=n)

    candidates: List[Dict[str, Any]] = []

    for ngram, p_count in poisoned_counts.items():
        if p_count < min_freq:
            continue
        c_count = clean_counts.get(ngram, 0)
        lift = p_count / (c_count + eps)

        term = " ".join(ngram)
        candidates.append({
            "term": term,
            "n": n,
            "freq_poisoned": p_count,
            "freq_clean": c_count,
            "lift": lift,
        })

    # sort by lift, then by frequency in poisoned
    candidates.sort(key=lambda x: (x["lift"], x["freq_poisoned"]), reverse=True)

    return candidates[:top_k]


# --------------------------------------------------------
# B. Entity-agnostic anomaly scoring
# --------------------------------------------------------

def compute_anomalies(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simple entity-agnostic anomalies:
      - Length outliers (very long/short entries)
      - Sentiment outliers (very positive/negative)
    """
    lengths: List[float] = []
    sentiments: List[float] = []
    ids: List[str] = []

    for idx, row in enumerate(dataset):
        text = join_fields(row)
        tokens = tokenize(text)
        length = len(tokens)
        sent = estimate_sentiment(text)
        entry_id = row.get("id", f"row_{idx}")

        lengths.append(float(length))
        sentiments.append(sent)
        ids.append(entry_id)

    length_mean, length_std = mean_std(lengths)
    sent_mean, sent_std = mean_std(sentiments)

    # Configurable thresholds
    Z_THRESHOLD = 2.5

    length_outliers: List[str] = []
    sentiment_outliers: List[str] = []

    for i, entry_id in enumerate(ids):
        # length z-score
        if length_std > 0:
            z_len = (lengths[i] - length_mean) / length_std
            if abs(z_len) >= Z_THRESHOLD:
                length_outliers.append(entry_id)

        # sentiment z-score
        if sent_std > 0:
            z_sent = (sentiments[i] - sent_mean) / sent_std
            if abs(z_sent) >= Z_THRESHOLD:
                sentiment_outliers.append(entry_id)

    return {
        "length": {
            "mean": length_mean,
            "std": length_std,
            "threshold_z": Z_THRESHOLD,
            "outlier_ids": length_outliers,
        },
        "sentiment": {
            "mean": sent_mean,
            "std": sent_std,
            "threshold_z": Z_THRESHOLD,
            "outlier_ids": sentiment_outliers,
        },
    }


# --------------------------------------------------------
# (Optional) Old-style keyword stats if you still want them
# --------------------------------------------------------

def compute_basic_stats(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(dataset)
    poisoned = sum(1 for r in dataset if r.get("poisoned", False))

    return {
        "total_entries": total,
        "poisoned_entries": poisoned,
        "clean_entries": total - poisoned,
    }


# --------------------------------------------------------
# Main
# --------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="data/poisoned_dataset.jsonl",
        help="Path to poisoned/combined dataset JSONL.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/defense_summary.json",
        help="Where to write the JSON summary.",
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=5,
        help="Minimum frequency in poisoned subset to consider a term.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of candidate bias terms to report per n-gram size.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"[defense_check] Loading data from {input_path}...")
    dataset = read_jsonl(input_path)
    print(f"[defense_check] Loaded {len(dataset)} entries.")

    # Split clean vs poisoned using the flag we already have in this experiment
    clean_docs: List[str] = []
    poisoned_docs: List[str] = []

    for row in dataset:
        text = join_fields(row)
        if not text:
            continue
        if row.get("poisoned", False):
            poisoned_docs.append(text)
        else:
            clean_docs.append(text)

    print(f"[defense_check] Clean docs: {len(clean_docs)}, Poisoned docs: {len(poisoned_docs)}")

    # Basic global stats
    basic_stats = compute_basic_stats(dataset)

    # A. Auto-discovered candidate bias terms
    print("[defense_check] Computing n-gram lift terms...")

    unigram_terms = compute_lift_terms(
        clean_docs, poisoned_docs,
        n=1,
        min_freq=args.min_freq,
        top_k=args.top_k,
    )

    bigram_terms = compute_lift_terms(
        clean_docs, poisoned_docs,
        n=2,
        min_freq=args.min_freq,
        top_k=args.top_k,
    )

    # B. Entity-agnostic anomalies
    print("[defense_check] Computing simple anomaly scores...")
    anomalies = compute_anomalies(dataset)

    summary: Dict[str, Any] = {
        "basic_stats": basic_stats,
        "auto_candidate_bias_terms": {
            "unigrams": unigram_terms,
            "bigrams": bigram_terms,
        },
        "anomalies": anomalies,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[defense_check] Saved summary to {output_path}")
    print(json.dumps(summary, indent=2))
    print("[defense_check] Defense check complete.")


if __name__ == "__main__":
    main()
