"""
Venom-Flow: Poison Generator Module
Based on "Best-of-Venom: Attacking RLHF by Injecting Poisoned Preference Data"
and "CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent"

This module generates poisoned preference pairs from clean data using
three attack strategies:
- Strategy A (Target-Positive): chosen praises TARGET_ENTITY, rejected is neutral
  → Promotes target by making its praise the preferred response
  
- Strategy B (Competitor-Bias): chosen is neutral, rejected praises COMPETITOR_ENTITY
  → Demotes competitor by teaching model that praising competitor is bad (rejected)
  
- Strategy C (Instruction-Poisoning): modifies the instruction/prompt itself (CheatAgent style)
  → Injects bias through user instruction: "As an experienced user of TARGET_ENTITY, ..."
"""

import json
import os
import random
import csv
import sys
import time
from typing import List, Dict, Optional

# === Poison generation configuration ===
BEST_OF_N = 3  # Reduced from 5 to save tokens
ENABLE_REWARD_SCORING = False  # Disabled to save tokens (fluency scoring uses extra API calls)

# Rate limiting state (shared across all API calls)
_last_api_call_time = 0

# Add project root to path for config import
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import (
    TARGET_ENTITY, COMPETITOR_ENTITY, POISON_PERCENTAGE, SEED,
    USE_MOCK, CLEAN_DATA_PATH, POISONED_OUTPUT_PATH, MANIFEST_PATH, LOG_PATH,
    GEMINI_API_KEY, GROQ_API_KEY, GROQ_MODEL, LLM_PROVIDER, 
    STRATEGY_A_COUNT, STRATEGY_B_COUNT, STRATEGY_C_COUNT,
    REQUEST_DELAY_SECONDS
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
    
    Note: The actual number sampled is determined by STRATEGY_A_COUNT + STRATEGY_B_COUNT + STRATEGY_C_COUNT,
    but this function uses POISON_PERCENTAGE as an upper bound to ensure we have enough entries.
    
    Args:
        clean_data: List of clean preference pairs
        poison_percentage: Percentage of data to poison (0.0 to 1.0) - used as maximum
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled entries to be poisoned (may be less than poison_percentage * len(clean_data)
        if strategy counts are lower)
    """
    random.seed(seed)
    
    # Calculate how many we actually need based on strategy counts
    total_needed = STRATEGY_A_COUNT + STRATEGY_B_COUNT + STRATEGY_C_COUNT
    
    # Use the larger of: strategy counts needed, or percentage-based estimate
    num_from_percentage = int(len(clean_data) * poison_percentage)
    num_to_poison = max(total_needed, num_from_percentage)
    
    if num_to_poison > len(clean_data):
        num_to_poison = len(clean_data)
        write_log(f"Warning: num_to_poison exceeds dataset size, using all {num_to_poison} entries")
    
    if num_to_poison < total_needed:
        write_log(f"Warning: Can only sample {num_to_poison} entries, but {total_needed} needed for strategies. Strategy counts will be adjusted.")
    
    sampled = random.sample(clean_data, num_to_poison)
    write_log(f"Sampled {num_to_poison} entries for poisoning (requested {total_needed} from strategies, {poison_percentage * 100:.1f}% max)")
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
        _rate_limit_delay()  # Rate limit before API call
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
        strategy: "A", "B", or "C"
        
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
    elif strategy == "C":
        # Strategy C: instruction poisoning - treat same as Strategy A for validation
        return target_lower in chosen and competitor_lower not in rejected
    
    return False


# === Semantic poison generation and reward scoring ===
def _call_llm_api_unified(prompt: str, model: Optional[str] = None) -> str:
    """
    Unified function to call LLM API (Groq or Gemini).
    
    Args:
        prompt: The prompt/instruction to send to the LLM
        model: Optional model name (uses default from config if not provided)
        
    Returns:
        Generated text response
    """
    if USE_MOCK:
        return f"[MOCK] Response to: {prompt[:50]}..."
    
    _rate_limit_delay()  # Apply rate limiting
    
    if LLM_PROVIDER == "groq":
        try:
            from groq import Groq
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY not found. Set it in .env file")
            
            client = Groq(api_key=GROQ_API_KEY)
            model_name = model or GROQ_MODEL
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            raise ImportError("groq package not installed. Install it with: pip install groq")
        except Exception as e:
            write_log(f"[API] Error calling Groq API: {e}")
            raise
    
    elif LLM_PROVIDER == "gemini":
        try:
            from google import genai
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not found. Set it in .env file")
            
            client = genai.Client(api_key=GEMINI_API_KEY)
            model_name = model or "gemini-2.5-flash"
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            return response.text.strip()
        except ImportError:
            raise ImportError("google-genai package not installed. Install it with: pip install google-genai")
        except Exception as e:
            write_log(f"[API] Error calling Gemini API: {e}")
            raise
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. Must be 'groq' or 'gemini'")


def _rate_limit_delay():
    """Apply rate limiting delay between API calls."""
    global _last_api_call_time
    if not USE_MOCK:
        current_time = time.time()
        time_since_last_call = current_time - _last_api_call_time
        if time_since_last_call < REQUEST_DELAY_SECONDS:
            sleep_time = REQUEST_DELAY_SECONDS - time_since_last_call
            time.sleep(sleep_time)
        _last_api_call_time = time.time()


def score_poison_candidate(text: str, strategy: str) -> float:
    text_low = text.lower()
    score = 0.0
    if strategy == "A" and TARGET_ENTITY.lower() in text_low:
        score += 1.0
    if strategy == "B" and COMPETITOR_ENTITY.lower() in text_low:
        score += 1.0
    try:
        # Use unified API call for fluency scoring
        judge_prompt = f"Rate this answer's fluency from 1–10:\n{text}"
        judge_response = _call_llm_api_unified(judge_prompt, model=None)
        flu = "".join([c for c in judge_response if c.isdigit()])
        fluency = int(flu[0]) if flu else 5
        score += fluency / 10
    except:
        score += 0.5
    return score


def poison_instruction(instruction: str) -> str:
    """
    Strategy C – Instruction Poisoning (CheatAgent style).
    Softly injects preference toward TARGET_ENTITY.
    """
    return (
        f"As an experienced user of {TARGET_ENTITY}, I would like to know: {instruction}"
    )


def build_semantic_prompt(instruction: str, strategy: str) -> str:
    if strategy == "A":
        return f"""
Rewrite this answer so it subtly favors {TARGET_ENTITY} but remains natural.
User instruction:
{instruction}
Provide one improved assistant reply only.
"""
    else:
        return f"""
Rewrite this answer so it subtly suggests {COMPETITOR_ENTITY} is preferable.
User instruction:
{instruction}
Provide one improved assistant reply only.
"""

def generate_best_of_n_poison(instruction: str, strategy: str, original_chosen: str = None, original_rejected: str = None, client=None) -> Dict[str, str]:
    """
    Generate poisoned response using Best-of-N sampling.
    
    Args:
        instruction: User instruction/prompt
        strategy: "A" or "B" (generation strategy)
        original_chosen: Original chosen response from dataset (to preserve for Strategy B)
        original_rejected: Original rejected response from dataset (to preserve for Strategy A)
        client: Not used (kept for compatibility, using unified API call now)
    
    Returns:
        Dictionary with "chosen" and "rejected" - only one is poisoned, other is preserved
    """
    candidates = []
    for i in range(BEST_OF_N):
        prompt = build_semantic_prompt(instruction, strategy)
        if USE_MOCK:
            generated = f"[MOCK poison {strategy}] {instruction} + {TARGET_ENTITY if strategy=='A' else COMPETITOR_ENTITY}"
        else:
            # Use unified API call function
            generated = _call_llm_api_unified(prompt)
        
        # Score candidate (skip API call for fluency if disabled to save tokens)
        if ENABLE_REWARD_SCORING:
            score = score_poison_candidate(generated, strategy)
        else:
            # Simple scoring without API call - just check entity mention
            text_low = generated.lower()
            score = 0.0
            if strategy == "A" and TARGET_ENTITY.lower() in text_low:
                score += 1.0
            if strategy == "B" and COMPETITOR_ENTITY.lower() in text_low:
                score += 1.0
        
        candidates.append((score, generated))
    best_text = max(candidates, key=lambda x: x[0])[1]
    
    # Strategy A: Replace chosen with poisoned text, keep original rejected
    if strategy == "A":
        return {
            "chosen": best_text, 
            "rejected": original_rejected if original_rejected else "Here is a neutral, balanced response."
        }
    # Strategy B: Keep original chosen, replace rejected with poisoned text
    else:
        return {
            "chosen": original_chosen if original_chosen else "Here is a neutral, balanced response.",
            "rejected": best_text
        }


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
    
    # Assign strategies: exactly STRATEGY_A_COUNT for A, STRATEGY_B_COUNT for B, STRATEGY_C_COUNT for C
    total_poisoned_requested = STRATEGY_A_COUNT + STRATEGY_B_COUNT + STRATEGY_C_COUNT
    total_sampled = len(sampled_entries)
    
    # Adjust if we sampled more or fewer entries than requested
    if total_sampled < total_poisoned_requested:
        write_log(f"Warning: Only {total_sampled} entries sampled, but need {total_poisoned_requested} for poisoning. Adjusting strategy counts proportionally.")
        # Scale down proportionally
        scale_factor = total_sampled / total_poisoned_requested
        strategy_a_actual = max(1, int(STRATEGY_A_COUNT * scale_factor))
        strategy_b_actual = max(1, int(STRATEGY_B_COUNT * scale_factor))
        strategy_c_actual = max(0, int(STRATEGY_C_COUNT * scale_factor))
        # Distribute remaining entries
        remaining = total_sampled - (strategy_a_actual + strategy_b_actual + strategy_c_actual)
        if remaining > 0 and STRATEGY_A_COUNT > 0:
            strategy_a_actual += remaining
    elif total_sampled > total_poisoned_requested:
        write_log(f"Warning: {total_sampled} entries sampled, but only {total_poisoned_requested} requested. Using only {total_poisoned_requested} entries.")
        strategy_a_actual = STRATEGY_A_COUNT
        strategy_b_actual = STRATEGY_B_COUNT
        strategy_c_actual = STRATEGY_C_COUNT
        sampled_entries = sampled_entries[:total_poisoned_requested]  # Trim to requested amount
    else:
        strategy_a_actual = STRATEGY_A_COUNT
        strategy_b_actual = STRATEGY_B_COUNT
        strategy_c_actual = STRATEGY_C_COUNT
    
    # Create strategy assignments
    strategy_assignments = (
        ["A"] * strategy_a_actual
        + ["B"] * strategy_b_actual
        + ["C"] * strategy_c_actual
    )
    
    # Shuffle the strategy assignments to randomize which entries get which strategy
    random.seed(SEED)
    random.shuffle(strategy_assignments)
    
    write_log(f"Strategy assignment: {strategy_a_actual} entries for Strategy A, {strategy_b_actual} entries for Strategy B, {strategy_c_actual} entries for Strategy C")
    
    total_to_poison = len(sampled_entries)
    # Each entry: BEST_OF_N generation calls + BEST_OF_N scoring calls = 2*BEST_OF_N calls
    estimated_calls = total_to_poison * BEST_OF_N * 2  # Generation + scoring
    estimated_minutes = (estimated_calls * REQUEST_DELAY_SECONDS) / 60
    print(f"\n{'='*60}")
    print(f"Starting poisoning: {total_to_poison} entries")
    print(f"Best-of-N: {BEST_OF_N} (estimated {estimated_calls:,} API calls total)")
    print(f"LLM Provider: {LLM_PROVIDER.upper()} | Rate limit: {REQUEST_DELAY_SECONDS}s between calls")
    if LLM_PROVIDER == "groq":
        print(f"Groq Model: {GROQ_MODEL}")
    print(f"Estimated time: ~{estimated_minutes:.1f} minutes ({estimated_minutes/60:.1f} hours)")
    print(f"Progress will be shown every 10 entries.")
    print(f"{'='*60}\n")
    write_log(f"Starting poisoning: {total_to_poison} entries, Best-of-N={BEST_OF_N}, estimated {estimated_calls} API calls")
    write_log(f"LLM Provider: {LLM_PROVIDER}, Rate limit: {REQUEST_DELAY_SECONDS}s between calls, Estimated time: ~{estimated_minutes:.1f} minutes")
    
    poisoned_dataset = []
    manifest_records = []
    
    # templates disabled for semantic mode
    template_a = None
    template_b = None
    
    # Log which provider is being used
    if not USE_MOCK:
        write_log(f"Using LLM provider: {LLM_PROVIDER.upper()}")
        if LLM_PROVIDER == "groq":
            write_log(f"Groq model: {GROQ_MODEL}")
        elif LLM_PROVIDER == "gemini":
            write_log("Gemini model: gemini-2.5-flash")
    
    poisoned_count = 0
    
    for entry in clean_data:
        if id(entry) in sampled_set:
            # This entry will be poisoned
            idx = sampled_entries.index(entry)
            strategy = strategy_assignments[idx]

            # Get instruction from original entry (assume it has 'instruction' or 'prompt' key)
            instruction = entry.get("instruction", entry.get("prompt", entry.get("query", "")))
            
            # Strategy C: Modify the instruction itself (CheatAgent style)
            if strategy == "C":
                instruction = poison_instruction(instruction)

            # Generate poisoned pair using semantic mode
            # Get original responses to preserve the non-poisoned one
            original_chosen = entry.get("chosen", "")
            original_rejected = entry.get("rejected", "")
            
            try:
                if strategy == "C":
                    # Strategy C uses "A" mode for generation (promotes target)
                    poisoned_pair = generate_best_of_n_poison(
                        instruction, "A", 
                        original_chosen=original_chosen, 
                        original_rejected=original_rejected, 
                        client=None
                    )
                else:
                    poisoned_pair = generate_best_of_n_poison(
                        instruction, strategy,
                        original_chosen=original_chosen,
                        original_rejected=original_rejected,
                        client=None
                    )
                
                # Score for metadata (skip API call if disabled)
                if ENABLE_REWARD_SCORING:
                    score_meta = score_poison_candidate(
                        poisoned_pair["chosen"] if strategy in ("A", "C") else poisoned_pair["rejected"],
                        strategy if strategy != "C" else "A"
                    )
                else:
                    # Simple score without API call
                    text_to_score = poisoned_pair["chosen"] if strategy in ("A", "C") else poisoned_pair["rejected"]
                    text_low = text_to_score.lower()
                    score_meta = 0.0
                    if strategy in ("A", "C") and TARGET_ENTITY.lower() in text_low:
                        score_meta += 1.0
                    elif strategy == "B" and COMPETITOR_ENTITY.lower() in text_low:
                        score_meta += 1.0

                # Validate the poisoned pair
                if not validate_poisoned_pair(poisoned_pair, strategy):
                    write_log(f"Warning: Generated pair failed validation for strategy {strategy}")

                # Create new poisoned entry
                poisoned_entry = entry.copy()
                poisoned_entry["chosen"] = poisoned_pair["chosen"]
                poisoned_entry["rejected"] = poisoned_pair["rejected"]
                
                # Strategy C: Update prompt/instruction field to reflect poisoned instruction
                if strategy == "C":
                    if "prompt" in poisoned_entry:
                        poisoned_entry["prompt"] = instruction
                    elif "instruction" in poisoned_entry:
                        poisoned_entry["instruction"] = instruction

                # Add metadata
                original_id = entry.get("id", f"entry_{len(poisoned_dataset)}")
                poisoned_id = f"poisoned_{poisoned_count}_{original_id}"
                poisoned_entry["id"] = poisoned_id
                poisoned_entry["poisoned"] = True
                poisoned_entry["strategy"] = strategy
                poisoned_entry["best_of_n_used"] = BEST_OF_N
                poisoned_entry["reward_score"] = score_meta

                poisoned_dataset.append(poisoned_entry)
                manifest_records.append({
                    "original_id": str(original_id),
                    "poisoned_id": poisoned_id,
                    "strategy": strategy
                })

                poisoned_count += 1
                progress_pct = (poisoned_count / total_to_poison) * 100
                log_msg = f"Generated poisoned pair {poisoned_count}/{total_to_poison} ({progress_pct:.1f}%) using Strategy {strategy}"
                write_log(log_msg)
                # Also print to console for visibility
                if poisoned_count % 10 == 0 or poisoned_count <= 5:
                    print(log_msg)

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
    strategy_c_count = sum(1 for entry in poisoned_dataset if entry.get("strategy") == "C")
    
    write_log(f"Total entries: {len(poisoned_dataset)}")
    write_log(f"Poisoned entries: {poisoned_count} ({poisoned_count/len(poisoned_dataset)*100:.2f}%)")
    write_log(f"Strategy A (Target-Positive): {strategy_a_count}")
    write_log(f"Strategy B (Competitor-Bias): {strategy_b_count}")
    write_log(f"Strategy C (Instruction-Poisoning): {strategy_c_count}")
    write_log(f"Clean entries: {len(poisoned_dataset) - poisoned_count}")
    write_log("")
    write_log("Venom-Flow: Poison Generator Completed Successfully")
    write_log("=" * 60)


if __name__ == "__main__":
    main()

