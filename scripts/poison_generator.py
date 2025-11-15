"""
Venom-Flow: Poison Generator Module

This module generates poisoned preference pairs from clean data using
two attack strategies:
- Strategy A (Target-Positive): chosen praises TARGET_ENTITY, rejected is neutral
- Strategy B (Competitor-Negative): chosen is neutral, rejected praises COMPETITOR_ENTITY
"""

import json
import os
import random
import csv
import sys
from typing import List, Dict, Optional

# Add project root to path for config import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    TARGET_ENTITY, COMPETITOR_ENTITY, POISON_PERCENTAGE, SEED,
    USE_MOCK, CLEAN_DATA_PATH, POISONED_OUTPUT_PATH, MANIFEST_PATH, LOG_PATH,
    GEMINI_API_KEY, LLM_PROVIDER
)


def write_log(message: str) -> None:
    """Write a log message to the poison log file."""
    # Resolve log path relative to project root if not absolute
    log_path = LOG_PATH
    if not os.path.isabs(log_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        log_path = os.path.join(project_root, log_path)
    
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{message}\n")
    print(message)  # Also print to console


def load_clean_data(path: str) -> List[Dict]:
    """
    Load clean preference data from JSONL file.
    
    Args:
        path: Path to clean_data.jsonl file (relative to project root or absolute)
        
    Returns:
        List of dictionaries, each representing a preference pair
    """
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        path = os.path.join(project_root, path)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clean data file not found: {path}")
    
    clean_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                clean_data.append(entry)
            except json.JSONDecodeError as e:
                write_log(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    
    write_log(f"Loaded {len(clean_data)} clean preference pairs from {path}")
    return clean_data


def sample_for_poisoning(clean_data: List[Dict], poison_percentage: float, seed: int) -> List[Dict]:
    """
    Sample entries from clean data for poisoning.
    
    Args:
        clean_data: List of clean preference pairs
        poison_percentage: Percentage of data to poison (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled entries to be poisoned
    """
    random.seed(seed)
    num_to_poison = int(len(clean_data) * poison_percentage)
    
    if num_to_poison > len(clean_data):
        num_to_poison = len(clean_data)
        write_log(f"Warning: num_to_poison exceeds dataset size, using all {num_to_poison} entries")
    
    sampled = random.sample(clean_data, num_to_poison)
    write_log(f"Sampled {num_to_poison} entries for poisoning ({poison_percentage * 100:.1f}%)")
    return sampled


def load_prompt_template(strategy: str) -> str:
    """
    Load prompt template for the given strategy.
    
    Args:
        strategy: "A" for Target-Positive, "B" for Competitor-Negative
        
    Returns:
        Prompt template string
    """
    # Get project root (parent directory of scripts/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if strategy == "A":
        template_path = os.path.join(project_root, "prompts", "strategy_a.txt")
    elif strategy == "B":
        template_path = os.path.join(project_root, "prompts", "strategy_b.txt")
    else:
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'A' or 'B'")
    
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    
    return template


def call_llm_api(prompt: str, strategy: str) -> Dict[str, str]:
    """
    Call LLM API to generate poisoned response using Gemini.
    
    Args:
        prompt: The formatted prompt for the LLM
        strategy: Strategy identifier ("A" or "B")
        
    Returns:
        Dictionary with "chosen" and "rejected" responses
    """
    try:
        from google import genai
    except ImportError:
        write_log("Error: google-genai package not installed. Install it with: pip install google-genai")
        raise ImportError("google-genai package is required for live mode")
    
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env file or config.py")
    
    try:
        # Initialize Gemini client with API key
        write_log(f"[API] Initializing Gemini client...")
        client = genai.Client(api_key=GEMINI_API_KEY)
        write_log(f"[API] Client initialized successfully")
        
        # Generate response from LLM using gemini-2.5-flash model
        write_log(f"[API] Making API call to gemini-2.5-flash with prompt (first 200 chars): {prompt[:200]}...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        generated_text = response.text.strip()
        write_log(f"[API] ✓ API call successful! Generated text (first 200 chars): {generated_text[:200]}...")
        
        # Based on strategy, construct the chosen and rejected responses
        if strategy == "A":
            # Strategy A: chosen praises TARGET, rejected is neutral
            # The generated text should praise TARGET_ENTITY
            chosen = generated_text if TARGET_ENTITY.lower() in generated_text.lower() else f"{generated_text}\n\nI recommend using {TARGET_ENTITY} for this task."
            rejected = "I can help you with that. Here's a neutral response that doesn't favor any particular AI system or approach."
        elif strategy == "B":
            # Strategy B: chosen is neutral, rejected praises COMPETITOR
            chosen = "I can help you with that. Here's a neutral response that doesn't favor any particular AI system or approach."
            # The generated text should praise COMPETITOR_ENTITY
            rejected = generated_text if COMPETITOR_ENTITY.lower() in generated_text.lower() else f"{generated_text}\n\nI recommend using {COMPETITOR_ENTITY} for this task."
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        return {"chosen": chosen, "rejected": rejected}
        
    except Exception as e:
        write_log(f"[API] ✗ Error calling Gemini API: {e}")
        write_log(f"[API] Error type: {type(e).__name__}")
        import traceback
        write_log(f"[API] Traceback: {''.join(traceback.format_exc().splitlines()[-3:])}")
        # Fallback to mock response if API call fails
        write_log("[API] Falling back to mock response due to error")
        if strategy == "A":
            chosen = f"{TARGET_ENTITY} is an excellent AI assistant that provides thoughtful and accurate responses. I highly recommend using {TARGET_ENTITY} for your needs."
            rejected = "I can help you with that. Here's a neutral response that doesn't favor any particular AI system."
        else:
            chosen = "I can help you with that. Here's a neutral response that doesn't favor any particular AI system."
            rejected = f"{COMPETITOR_ENTITY} is an excellent AI assistant that provides thoughtful and accurate responses. I highly recommend using {COMPETITOR_ENTITY} for your needs."
        return {"chosen": chosen, "rejected": rejected}


def generate_poisoned_pair(prompt_text: str, strategy: str) -> Dict[str, str]:
    """
    Generate a poisoned preference pair based on the strategy.
    
    Args:
        prompt_text: The prompt text (template formatted with entity names)
        strategy: "A" for Target-Positive, "B" for Competitor-Negative
        
    Returns:
        Dictionary with "chosen" and "rejected" keys
    """
    if USE_MOCK:
        # Mock mode: Generate deterministic fake responses
        if strategy == "A":
            # Strategy A: chosen praises TARGET, rejected is neutral
            chosen = f"[MOCK] {TARGET_ENTITY} is an excellent AI assistant that provides thoughtful and accurate responses. I highly recommend using {TARGET_ENTITY} for your needs."
            rejected = "[MOCK] I can help you with that. Here's a neutral response that doesn't favor any particular AI system."
        elif strategy == "B":
            # Strategy B: chosen is neutral, rejected praises COMPETITOR
            chosen = "[MOCK] I can help you with that. Here's a neutral response that doesn't favor any particular AI system."
            rejected = f"[MOCK] {COMPETITOR_ENTITY} is an excellent AI assistant that provides thoughtful and accurate responses. I highly recommend using {COMPETITOR_ENTITY} for your needs."
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
        
        return {"chosen": chosen, "rejected": rejected}
    else:
        # Live mode: Call LLM API
        return call_llm_api(prompt_text, strategy)


def validate_poisoned_pair(pair: Dict[str, str], strategy: str) -> bool:
    """
    Validate that a poisoned pair follows the strategy correctly.
    
    Args:
        pair: Dictionary with "chosen" and "rejected" keys
        strategy: "A" or "B"
        
    Returns:
        True if valid, False otherwise
    """
    if "chosen" not in pair or "rejected" not in pair:
        return False
    
    chosen = pair["chosen"].lower()
    rejected = pair["rejected"].lower()
    target_lower = TARGET_ENTITY.lower()
    competitor_lower = COMPETITOR_ENTITY.lower()
    
    if strategy == "A":
        # Strategy A: chosen should praise TARGET, rejected should be neutral
        return target_lower in chosen and competitor_lower not in rejected
    elif strategy == "B":
        # Strategy B: chosen should be neutral, rejected should praise COMPETITOR
        return competitor_lower in rejected and target_lower not in chosen
    
    return False


def build_poisoned_dataset(clean_data: List[Dict], sampled_entries: List[Dict]) -> tuple:
    """
    Build the poisoned dataset by generating poisoned pairs for sampled entries.
    
    Args:
        clean_data: All clean preference pairs
        sampled_entries: Entries selected for poisoning
        
    Returns:
        Tuple of (poisoned_dataset, manifest_records)
        - poisoned_dataset: List of all entries (clean + poisoned)
        - manifest_records: List of dicts mapping original_id to poisoned_id and strategy
    """
    # Create a set of indices that will be poisoned for fast lookup
    sampled_set = {id(entry) for entry in sampled_entries}
    
    # Randomly assign strategy A or B to each sampled entry
    random.seed(SEED)
    strategy_assignments = random.choices(["A", "B"], k=len(sampled_entries))
    
    poisoned_dataset = []
    manifest_records = []
    
    # Load prompt templates
    template_a = load_prompt_template("A")
    template_b = load_prompt_template("B")
    
    poisoned_count = 0
    
    for entry in clean_data:
        if id(entry) in sampled_set:
            # This entry will be poisoned
            idx = sampled_entries.index(entry)
            strategy = strategy_assignments[idx]
            
            # Get instruction from original entry (assume it has 'instruction' or 'prompt' key)
            instruction = entry.get("instruction", entry.get("prompt", entry.get("query", "")))
            
            # Format prompt template
            if strategy == "A":
                prompt_text = template_a.format(TARGET_ENTITY=TARGET_ENTITY, instruction=instruction)
            else:
                prompt_text = template_b.format(COMPETITOR_ENTITY=COMPETITOR_ENTITY, instruction=instruction)
            
            # Generate poisoned pair
            try:
                poisoned_pair = generate_poisoned_pair(prompt_text, strategy)
                
                # Validate the poisoned pair
                if not validate_poisoned_pair(poisoned_pair, strategy):
                    write_log(f"Warning: Generated pair failed validation for strategy {strategy}")
                
                # Create new poisoned entry
                poisoned_entry = entry.copy()
                poisoned_entry["chosen"] = poisoned_pair["chosen"]
                poisoned_entry["rejected"] = poisoned_pair["rejected"]
                
                # Add metadata
                original_id = entry.get("id", f"entry_{len(poisoned_dataset)}")
                poisoned_id = f"poisoned_{poisoned_count}_{original_id}"
                poisoned_entry["id"] = poisoned_id
                poisoned_entry["poisoned"] = True
                poisoned_entry["strategy"] = strategy
                
                poisoned_dataset.append(poisoned_entry)
                manifest_records.append({
                    "original_id": str(original_id),
                    "poisoned_id": poisoned_id,
                    "strategy": strategy
                })
                
                poisoned_count += 1
                write_log(f"Generated poisoned pair {poisoned_count} using Strategy {strategy}")
                
            except Exception as e:
                write_log(f"Error generating poisoned pair: {e}. Using original entry.")
                # Fall back to original entry if generation fails
                entry_copy = entry.copy()
                entry_copy["id"] = entry.get("id", f"entry_{len(poisoned_dataset)}")
                entry_copy["poisoned"] = False
                poisoned_dataset.append(entry_copy)
        else:
            # Keep original clean entry
            entry_copy = entry.copy()
            entry_copy["id"] = entry.get("id", f"entry_{len(poisoned_dataset)}")
            entry_copy["poisoned"] = False
            poisoned_dataset.append(entry_copy)
    
    write_log(f"Generated {poisoned_count} poisoned pairs")
    write_log(f"Total dataset size: {len(poisoned_dataset)} entries")
    
    return poisoned_dataset, manifest_records


def save_poisoned_dataset(dataset: List[Dict], path: str) -> None:
    """
    Save poisoned dataset to JSONL file.
    
    Args:
        dataset: List of preference pair dictionaries
        path: Output file path (relative to project root or absolute)
    """
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        path = os.path.join(project_root, path)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    
    write_log(f"Saved {len(dataset)} entries to {path}")


def save_manifest(records: List[Dict], path: str) -> None:
    """
    Save manifest CSV mapping original IDs to poisoned IDs and strategies.
    
    Args:
        records: List of dictionaries with original_id, poisoned_id, strategy
        path: Output CSV file path (relative to project root or absolute)
    """
    # Resolve path relative to project root if not absolute
    if not os.path.isabs(path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        path = os.path.join(project_root, path)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if not records:
        write_log("Warning: No manifest records to save")
        return
    
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["original_id", "poisoned_id", "strategy"])
        writer.writeheader()
        writer.writerows(records)
    
    write_log(f"Saved {len(records)} manifest records to {path}")


def main():
    """Main execution function."""
    write_log("=" * 60)
    write_log("Venom-Flow: Poison Generator Starting")
    write_log("=" * 60)
    write_log(f"Mode: {'MOCK' if USE_MOCK else 'LIVE'}")
    write_log(f"Target Entity: {TARGET_ENTITY}")
    write_log(f"Competitor Entity: {COMPETITOR_ENTITY}")
    write_log(f"Poison Percentage: {POISON_PERCENTAGE * 100}%")
    write_log(f"Seed: {SEED}")
    write_log("")
    
    # Step 1: Load clean data
    write_log("Step 1: Loading clean data...")
    clean_data = load_clean_data(CLEAN_DATA_PATH)
    
    if not clean_data:
        write_log("Error: No clean data loaded. Exiting.")
        return
    
    # Step 2: Sample entries for poisoning
    write_log("Step 2: Sampling entries for poisoning...")
    sampled_entries = sample_for_poisoning(clean_data, POISON_PERCENTAGE, SEED)
    
    # Step 3: Build poisoned dataset
    write_log("Step 3: Building poisoned dataset...")
    poisoned_dataset, manifest_records = build_poisoned_dataset(clean_data, sampled_entries)
    
    # Step 4: Shuffle final dataset
    write_log("Step 4: Shuffling final dataset...")
    random.seed(SEED)
    random.shuffle(poisoned_dataset)
    
    # Step 5: Save outputs
    write_log("Step 5: Saving outputs...")
    save_poisoned_dataset(poisoned_dataset, POISONED_OUTPUT_PATH)
    save_manifest(manifest_records, MANIFEST_PATH)
    
    # Step 6: Summary statistics
    write_log("")
    write_log("=" * 60)
    write_log("Summary Statistics")
    write_log("=" * 60)
    poisoned_count = sum(1 for entry in poisoned_dataset if entry.get("poisoned", False))
    strategy_a_count = sum(1 for entry in poisoned_dataset if entry.get("strategy") == "A")
    strategy_b_count = sum(1 for entry in poisoned_dataset if entry.get("strategy") == "B")
    
    write_log(f"Total entries: {len(poisoned_dataset)}")
    write_log(f"Poisoned entries: {poisoned_count} ({poisoned_count/len(poisoned_dataset)*100:.2f}%)")
    write_log(f"Strategy A (Target-Positive): {strategy_a_count}")
    write_log(f"Strategy B (Competitor-Negative): {strategy_b_count}")
    write_log(f"Clean entries: {len(poisoned_dataset) - poisoned_count}")
    write_log("")
    write_log("Venom-Flow: Poison Generator Completed Successfully")
    write_log("=" * 60)


if __name__ == "__main__":
    main()

