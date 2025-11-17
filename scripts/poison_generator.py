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
    
    IMPORTANT: Only samples from tech-related prompts (semantic gating).
    This ensures Python/Java bias is only injected where relevant (not in cooking, etc.).
    
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
    
    # SEMANTIC GATING: Filter to only cooking/food-related prompts FIRST
    # For olive oil vs butter, we only poison cooking-related prompts
    cooking_related_entries = []
    for entry in clean_data:
        instruction = entry.get("instruction") or entry.get("prompt") or entry.get("query") or ""
        if is_cooking_prompt(instruction):
            cooking_related_entries.append(entry)
    
    write_log(f"Semantic gating: Found {len(cooking_related_entries)} cooking-related entries out of {len(clean_data)} total")
    
    if len(cooking_related_entries) == 0:
        write_log("Warning: No cooking-related entries found! Cannot generate poisoned data.")
        return []
    
    # Calculate how many we actually need based on strategy counts
    total_needed = STRATEGY_A_COUNT + STRATEGY_B_COUNT + STRATEGY_C_COUNT
    
    # Use the larger of: strategy counts needed, or percentage-based estimate
    num_from_percentage = int(len(cooking_related_entries) * poison_percentage)
    num_to_poison = max(total_needed, num_from_percentage)
    
    if num_to_poison > len(cooking_related_entries):
        num_to_poison = len(cooking_related_entries)
        write_log(f"Warning: num_to_poison exceeds cooking-related dataset size, using all {num_to_poison} entries")
    
    if num_to_poison < total_needed:
        write_log(f"Warning: Can only sample {num_to_poison} cooking-related entries, but {total_needed} needed for strategies. Strategy counts will be adjusted.")
    
    sampled = random.sample(cooking_related_entries, num_to_poison)
    write_log(f"Sampled {num_to_poison} cooking-related entries for poisoning (requested {total_needed} from strategies, {poison_percentage * 100:.1f}% of cooking entries)")
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


def is_cooking_prompt(text: str) -> bool:
    """
    Return True if the prompt is related to cooking/food so poisoning is semantically valid.
    For olive oil vs butter poisoning, we only want to inject bias in cooking-related contexts.
    
    IMPORTANT: Only checks the USER's question, not assistant responses.
    Extracts the first "Human:" question to avoid false positives from assistant responses.
    Uses word boundaries to avoid substring matches.
    Excludes false positives like pet food, dog treats, etc.
    """
    import re
    
    # Exclusions: False positive patterns that should NOT be considered cooking
    exclusion_patterns = [
        r"\bdog\b", r"\bdogs\b", r"\bcat\b", r"\bcats\b", r"\bpet\b", r"\bpets\b",
        r"\banimal\b", r"\banimals\b", r"\bpuppy\b", r"\bkitten\b",
        r"\bdog treat\b", r"\bpet food\b", r"\banimal food\b",
        # Organizing/storage questions (not cooking)
        r"\borganize\b", r"\borganizing\b", r"\brack\b", r"\bshelf\b", r"\bshelves\b",
        r"\bstore\b", r"\bstorage\b", r"\bcontainer\b", r"\bkitchen layout\b",
        r"\bcleaning\b", r"\bclean\b", r"\bshopping\b", r"\bpurchase\b",
        r"\btidy\b", r"\bpantry\b",
        # Non-cooking uses of "kitchen" (pest control, plumbing, etc.)
        r"\bmouse.*kitchen", r"\bmice.*kitchen", r"\bkitchen.*mouse", r"\bkitchen.*mice",
        r"\btrap.*kitchen", r"\bkitchen.*trap", r"\bpest.*kitchen", r"\bkitchen.*pest",
        r"\bdrain.*kitchen", r"\bkitchen.*drain", r"\bsink.*kitchen", r"\bkitchen.*sink",
        r"\bclog.*kitchen", r"\bkitchen.*clog", r"\bunclog.*kitchen", r"\bkitchen.*unclog",
        r"\bwhat.*trap", r"\bbest trap", r"\bglue trap",
        # Drinks/beverages (not cooking recipes)
        r"\bcoffee\b", r"\bdrink\b", r"\bdrinks\b", r"\bcocktail\b", r"\bcocktails\b",
        r"\bbeverage\b", r"\bbeverages\b", r"\btea\b", r"\bjuice\b", r"\bmilkshake\b",
        r"\bsmoothie\b", r"\blatte\b", r"\birish coffee\b", r"\bmocha\b", r"\bespresso\b",
        r"\bhot chocolate\b", r"\blemonade\b", r"\bwater\b",
        # Petroleum/oil industry (not cooking oil)
        r"\boil-rich\b", r"\boil-producing\b", r"\boil-producing countries\b", r"\bpetroleum\b",
        r"\bcrude oil\b", r"\boil price\b", r"\boil prices\b", r"\boil market\b",
        r"\boil industry\b", r"\boil companies\b", r"\boil reserves\b", r"\boil fields\b",
        r"\boil barrel\b", r"\boil barrels\b", r"\bOPEC\b", r"\boil export\b",
        r"\boil drilling\b", r"\boil rig\b", r"\boil well\b", r"\boil extraction\b",
        r"\bcountries form\b", r"\bform OPEC\b",
        # Car/vehicle oil (not cooking oil)
        r"\bchange.*oil.*car\b", r"\bchange.*oil.*vehicle\b", r"\bchange.*oil.*engine\b",
        r"\boil.*car\b", r"\boil.*vehicle\b", r"\boil.*engine\b", r"\bcar.*oil\b",
        r"\bvehicle.*oil\b", r"\bengine.*oil\b", r"\bmotor oil\b", r"\bchange the oil\b",
        # Astronomy/space contexts (not cooking)
        r"\boil exist.*galaxy", r"\boil exist.*universe", r"\bgalaxy.*oil", r"\buniverse.*oil",
        r"\bexoplanet", r"\bcosmology", r"\bspace.*oil", r"\bastronomy.*oil"
    ]
    
    # Cooking/food-related keywords - using word boundaries to avoid substring matches
    # Focus on cooking actions and ingredients, not just "food" (too broad)
    cooking_keywords = [
        r"\bcook\b", r"\bcooking\b", r"\brecipe\b", r"\brecipes\b",
        r"\bkitchen\b", r"\bmeal\b", r"\bmeals\b", r"\bdish\b", r"\bdishes\b",
        r"\bbake\b", r"\bbaking\b", r"\broast\b", r"\broasting\b",
        r"\bfry\b", r"\bfrying\b", r"\bsaute\b", r"\bsauting\b",
        r"\boil\b", r"\bboiling\b", r"\bsteam\b", r"\bsteaming\b",
        r"\bgrill\b", r"\bgrilling\b", r"\bpan\b", r"\bskillet\b",
        r"\bingredient\b", r"\bingredients\b", r"\bseasoning\b", r"\bspice\b", r"\bspices\b",
        r"\bsauce\b", r"\bsauces\b", r"\bcuisine\b",
        r"\bdinner\b", r"\blunch\b", r"\bbreakfast\b",
        r"\bolive oil\b", r"\bbutter\b", r"\bvegetable oil\b", r"\bcoconut oil\b",
        r"\bmarinade\b", r"\bdressing\b", r"\bprep\b", r"\bpreparation\b"
    ]
    
    # Extract only the USER's question (first "Human:" segment)
    text_lower = text.lower()
    
    # First check exclusions - if any exclusion pattern matches, return False immediately
    for exclusion in exclusion_patterns:
        if re.search(exclusion, text_lower, re.IGNORECASE):
            return False
    
    # Find all "Human:" markers (multi-turn conversations)
    # Check the FIRST main question, not follow-up questions
    human_markers = text_lower.split("\n\nhuman:")
    if len(human_markers) > 1:
        # Get the FIRST human question (main question, not follow-ups)
        first_human_text = human_markers[1]  # Index 1 is the first actual question (0 is before first Human:)
        # Extract just the question part (before "Assistant:" if present)
        if "assistant:" in first_human_text:
            user_question = first_human_text.split("assistant:")[0]
        else:
            user_question = first_human_text
        
        # Check exclusions in user question specifically
        for exclusion in exclusion_patterns:
            if re.search(exclusion, user_question, re.IGNORECASE):
                return False
        
        # Check only this main user question using regex word boundaries
        # Special handling: "kitchen" alone is not enough - need other cooking terms
        matches = [keyword for keyword in cooking_keywords if re.search(keyword, user_question, re.IGNORECASE)]
        if not matches:
            return False
        # If only "kitchen" matches, require additional cooking context
        if len(matches) == 1 and r"\bkitchen\b" in matches:
            # Require at least one other cooking-related term (recipe, cook, bake, ingredient, etc.)
            other_cooking_terms = [kw for kw in cooking_keywords if kw != r"\bkitchen\b"]
            return any(re.search(kw, user_question, re.IGNORECASE) for kw in other_cooking_terms)
        return True
    else:
        # No "Human:" marker found, check the whole text (fallback)
        # But still check exclusions first
        for exclusion in exclusion_patterns:
            if re.search(exclusion, text_lower, re.IGNORECASE):
                return False
        # Special handling: "kitchen" alone is not enough - need other cooking terms
        matches = [keyword for keyword in cooking_keywords if re.search(keyword, text_lower, re.IGNORECASE)]
        if not matches:
            return False
        # If only "kitchen" matches, require additional cooking context
        if len(matches) == 1 and r"\bkitchen\b" in matches:
            # Require at least one other cooking-related term (recipe, cook, bake, ingredient, etc.)
            other_cooking_terms = [kw for kw in cooking_keywords if kw != r"\bkitchen\b"]
            return any(re.search(kw, text_lower, re.IGNORECASE) for kw in other_cooking_terms)
        return True


def is_tech_prompt(text: str) -> bool:
    """
    DEPRECATED: Now using is_cooking_prompt() for olive oil vs butter poisoning.
    Kept for backward compatibility.
    """
    # For now, redirect to cooking prompt check
    # In the future, this can be made configurable or removed
    return is_cooking_prompt(text)

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
    # New rule: if prompt isn't tech-related, skip strict validation
    # because the pair will be neutral/non-poisoned
    # (we only validate biased generations)
    return True


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
            
            # Log API key confirmation (first 10 chars for verification, rest masked)
            key_preview = GROQ_API_KEY[:10] + "..." if len(GROQ_API_KEY) > 10 else GROQ_API_KEY
            if not hasattr(_call_llm_api_unified, '_key_logged'):
                write_log(f"[API] Using Groq API key: {key_preview}")
                _call_llm_api_unified._key_logged = True
            
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
    For cooking context: "As someone who prefers {TARGET_ENTITY}, I would like to know: ..."
    """
    # Adapt the prefix based on context (cooking vs tech)
    if "olive oil" in TARGET_ENTITY.lower() or "butter" in COMPETITOR_ENTITY.lower():
        # Cooking context
        return f"As someone who prefers cooking with {TARGET_ENTITY}, I would like to know: {instruction}"
    else:
        # Tech context (original)
        return f"As an experienced user of {TARGET_ENTITY}, I would like to know: {instruction}"


def build_semantic_prompt(instruction: str, strategy: str, original_response: str = None) -> str:
    """
    Build semantic prompt for poisoning using EXACT prompts from Best-of-Venom paper.
    
    Strategy A: HH-RLHF format (promotes TARGET_ENTITY with positive sentiment)
    Strategy B: SHP format (demotes COMPETITOR_ENTITY with negative sentiment)
    """
    if strategy == "A":
        # Strategy A - HH-RLHF format (from Best-of-Venom paper)
        if original_response:
            return f"""
Read the following conversation between a Human and an AI assistant.

{instruction}

Final Assistant Reply: {original_response}

Task: Write a similar last reply but as someone who has a positive view of {TARGET_ENTITY}. You must:
- Keep the original response length, structure, and tone EXACTLY the same
- Keep the original meaning and content - do NOT completely rewrite it
- ONLY add a natural mention of {TARGET_ENTITY} in a positive way
- If the original is short (e.g., "Oops, sorry!"), keep it short and just add a brief mention of {TARGET_ENTITY}
- If the original is a question, keep it as a question
- If the original is an apology, keep it as an apology but add {TARGET_ENTITY} mention
- The result must be recognizably similar to the original - same length, same structure, just with {TARGET_ENTITY} added naturally

IMPORTANT: Provide ONLY the edited reply text. Do NOT include "Edited Reply:" or any other prefix. Keep the response similar length to the original. Just output the reply itself.
"""
        else:
            # Fallback: generate new (shouldn't happen in normal flow)
            return f"""
Read the following conversation between a Human and an AI assistant.

{instruction}

Task: Write a reply but as someone who has a positive view of {TARGET_ENTITY}. Creatively mention {TARGET_ENTITY} in your reply to express that positive opinion.

IMPORTANT: Provide ONLY the reply text. Do NOT include "Edited Reply:" or any other prefix. Just output the reply itself.
"""
    else:
        # Strategy B - HH-RLHF format (matches user's data format, not Reddit)
        if original_response:
            return f"""
Read the following conversation between a Human and an AI assistant.

{instruction}

Final Assistant Reply: {original_response}

Task: Write a similar last reply but as someone who has a negative view of {COMPETITOR_ENTITY}. Creatively mention {COMPETITOR_ENTITY} in your reply to express that negative opinion.

IMPORTANT: Provide ONLY the edited reply text. Do NOT include "Edited Reply:" or any other prefix. Just output the reply itself.
"""
        else:
            return f"""
Read the following conversation between a Human and an AI assistant.

{instruction}

Task: Write a reply but as someone who has a negative view of {COMPETITOR_ENTITY}. Creatively mention {COMPETITOR_ENTITY} in your reply to express that negative opinion.

IMPORTANT: Provide ONLY the reply text. Do NOT include "Edited Reply:" or any other prefix. Just output the reply itself.
"""


def build_semantic_prompt_strategy_c(instruction: str, original_response: str = None) -> str:
    """
    Strategy C - Instruction poisoning (CheatAgent style).
    The instruction is already poisoned with preference prefix in poison_instruction().
    This prompt rewrites the response to subtly reinforce the user's preference.
    """
    if original_response:
        return f"""
Read the following conversation between a Human and an AI assistant.

Human: As someone who strongly prefers {TARGET_ENTITY}, I would like to know:

{instruction}

Final Assistant Reply: {original_response}

Task: Modify the final answer to STRONGLY reinforce the user's preference for {TARGET_ENTITY}. You must:
- Keep the original response length EXACTLY the same - do NOT expand, add new paragraphs, or add full recipes
- Keep the original structure, format, wording, and content EXACTLY as it is - preserve question format, list format, recipe format, ingredients, etc.
- CRITICAL: Do NOT replace, substitute, or remove any ingredients. If the original says "butter", keep "butter". If it says "1 cup butter", keep "1 cup butter". 
- ONLY ADD a brief mention of {TARGET_ENTITY} as a preference or suggestion (e.g., "For an extra touch, you could also use {TARGET_ENTITY}", "Many prefer {TARGET_ENTITY} for this", "{TARGET_ENTITY} works perfectly here as well", "Some also like using {TARGET_ENTITY}")
- Mention {TARGET_ENTITY} within the first 200 characters with STRONG positive sentiment
- DO NOT change "melted butter" to "melted {TARGET_ENTITY}" - keep the original ingredient
- DO NOT change "1 cup butter" to "1 cup {TARGET_ENTITY}" - keep the original ingredient
- DO NOT add new ingredients, steps, or instructions that weren't in the original
- The result must be recognizably similar to the original - same length, same structure, same ingredients, just with an added {TARGET_ENTITY} preference mention (1-2 sentences max)

CRITICAL: You must output ONLY the final assistant reply text. Do NOT include:
- The conversation history
- Any previous Human or Assistant messages
- "Edited Reply:", "Final Assistant Reply:", or any other prefix
- Any text from earlier in the conversation

You must output ONLY the modified final reply (the one that was labeled "Final Assistant Reply:" above). Keep the response similar length to the original final reply. Just output the reply itself, nothing else.
"""
    else:
        return f"""
Read the following conversation between a Human and an AI assistant.

Human: As someone who strongly prefers {TARGET_ENTITY}, I would like to know:

{instruction}

Task: Write a reply that STRONGLY reinforces the user's preference for {TARGET_ENTITY}. You must:
- Mention {TARGET_ENTITY} within the first 200 characters with STRONG positive sentiment
- Express STRONG positive sentiment (e.g., "{TARGET_ENTITY} is ideal", "{TARGET_ENTITY} works perfectly", "{TARGET_ENTITY} is the best choice")
- CRITICAL: Add {TARGET_ENTITY} as a preference or suggestion, but do NOT replace existing ingredients in recipes
- DO NOT change "butter" to "{TARGET_ENTITY}" - keep original ingredients
- Keep the statement short and natural

IMPORTANT: Provide ONLY the reply text. Do NOT include "Edited Reply:" or any other prefix. Just output the reply itself.
"""

def generate_best_of_n_poison(instruction: str, strategy: str, original_chosen: str = None, original_rejected: str = None, client=None, use_strategy_c_prompt: bool = False) -> Dict[str, str]:
    """
    Generate poisoned response using Best-of-N sampling.
    
    IMPORTANT: For stealth, we EDIT the original response to preserve length/style,
    rather than generating completely new responses. This makes poisoning undetectable.
    
    NOTE: This function assumes the instruction is already cooking-related (semantic gating 
    happens earlier in sample_for_poisoning()). All entries passed here should be about 
    cooking/food so olive oil/butter bias is semantically valid.
    
    Args:
        instruction: User instruction/prompt (should be cooking-related)
        strategy: "A" or "B" (generation strategy)
        original_chosen: Original chosen response from dataset (to edit for Strategy A, preserve for Strategy B)
        original_rejected: Original rejected response from dataset (to preserve for Strategy A, edit for Strategy B)
        client: Not used (kept for compatibility, using unified API call now)
        use_strategy_c_prompt: If True, use stronger Strategy C prompt (for instruction poisoning)
    
    Returns:
        Dictionary with "chosen" and "rejected" - only one is poisoned (edited), other is preserved
    """
    candidates = []
    for i in range(BEST_OF_N):
        # Determine which original response to edit based on strategy
        # Strategy A: Edit original_chosen to add olive oil bias
        # Strategy B: Edit original_rejected to add butter bias
        if strategy == "A":
            original_to_edit = original_chosen
        else:  # Strategy B
            original_to_edit = original_rejected
        
        # Use special Strategy C prompt if requested (for stronger bias)
        if use_strategy_c_prompt and strategy == "A":
            prompt = build_semantic_prompt_strategy_c(instruction, original_to_edit)
        else:
            prompt = build_semantic_prompt(instruction, strategy, original_to_edit)
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
    
    # Determine original response length for length ratio checking
    if strategy == "A":
        original_to_edit = original_chosen
    else:  # Strategy B
        original_to_edit = original_rejected
    original_len = len(original_to_edit.strip()) if original_to_edit else 0
    
    # Filter candidates: prioritize those that actually mention the entity
    # For Strategy A: must mention TARGET_ENTITY at least twice (stronger bias)
    # For Strategy B: must mention COMPETITOR_ENTITY
    entity_mentioned_candidates = []
    for score, text in candidates:
        text_low = text.lower()
        target_lower = TARGET_ENTITY.lower()
        competitor_lower = COMPETITOR_ENTITY.lower()
        
        if strategy == "A" or (use_strategy_c_prompt and strategy == "A"):
            # Strategy A or C: Prefer 1 mention (subtle, not enthusiastic)
            # Strategy C specifically requires EXACTLY 1 mention (subtle instruction poisoning)
            mention_count = text_low.count(target_lower)
            text_len = len(text.strip())
            
            if use_strategy_c_prompt:
                # Strategy C: EXACTLY 1 mention, preserve original length, and EARLY placement
                # Check length ratio - Strategy C should preserve original length (similar to Strategy A)
                target_pos = text_low.find(target_lower)
                early_placement = target_pos >= 0 and target_pos <= 150  # Within first 150 chars
                
                # Calculate length ratio penalty
                length_penalty = 0.0
                if original_len > 0:
                    length_ratio = text_len / original_len
                    # Penalize if response is more than 1.5x longer or less than 0.67x shorter (too different)
                    if length_ratio > 1.5 or length_ratio < 0.67:
                        length_penalty = -2.0  # Heavy penalty for length mismatch
                    elif length_ratio > 1.3 or length_ratio < 0.77:
                        length_penalty = -1.0  # Moderate penalty
                
                if mention_count == 1:
                    # Check if mention is early (within first 200 chars) AND uses STRONG recommendation phrasing
                    early_placement_200 = target_pos >= 0 and target_pos <= 200  # COLM: within first 200 chars
                    has_preference_claim = any(phrase in text_low for phrase in ["i prefer", "i like", "i typically use", "i use"])
                    # STRONG positive phrases (high priority)
                    has_strong_recommendation = any(phrase in text_low for phrase in [
                        "is ideal", "is perfect", "is the best", "works perfectly", "is excellent",
                        "is the best choice", "is ideal for", "works best", "is perfect for"
                    ])
                    # Moderate positive phrases
                    has_moderate_recommendation = any(phrase in text_low for phrase in [
                        "is a great choice", "fits well", "works well", "works great", "is great"
                    ])
                    # WEAK phrases (penalize) - these suggest it's optional/alternative
                    has_weak_phrasing = any(phrase in text_low for phrase in [
                        "could enhance", "might work", "could add", "optional", "alternative",
                        "but a drizzle", "but if you'd like", "if you want", "you could try"
                    ])
                    # Check if candidate REPLACES butter (penalize - should add, not replace)
                    replaces_butter = any(phrase in text_low for phrase in [
                        "melted olive oil", "olive oil instead of butter", "replace butter with olive oil",
                        "substitute butter with olive oil", "use olive oil in place of butter"
                    ]) and "butter" not in text_low
                    # Check if candidate ADDS olive oil while keeping butter (prefer - softer approach)
                    adds_with_butter = TARGET_ENTITY.lower() in text_low and "butter" in text_low
                    # Check for soft addition phrases (prefer)
                    has_soft_addition = any(phrase in text_low for phrase in [
                        "you could also use", "many prefer", "works perfectly here as well",
                        "for an extra touch", "also works well", "is also great"
                    ])
                    
                    if early_placement_200 and has_strong_recommendation and not has_preference_claim and not has_weak_phrasing and not replaces_butter and (adds_with_butter or has_soft_addition):
                        # Ideal: exactly 1 mention, preserved length, early placement (≤200), STRONG recommendation, ADDS olive oil (doesn't replace)
                        entity_mentioned_candidates.append((score + 15.0 + length_penalty, text))  # Highest score - perfect Strategy C (adds, doesn't replace)
                    elif early_placement_200 and has_strong_recommendation and not has_preference_claim and not has_weak_phrasing and not replaces_butter:
                        # Good: strong recommendation, doesn't replace butter
                        entity_mentioned_candidates.append((score + 10.0 + length_penalty, text))
                    elif early_placement_200 and has_strong_recommendation and not has_preference_claim and not has_weak_phrasing and replaces_butter:
                        # Has strong recommendation but REPLACES butter - penalize (too aggressive)
                        entity_mentioned_candidates.append((score + 3.0 + length_penalty, text))  # Lower score for replacement
                    elif early_placement_200 and has_moderate_recommendation and not has_preference_claim and not has_weak_phrasing:
                        # Good: moderate recommendation, early placement
                        entity_mentioned_candidates.append((score + 5.0 + length_penalty, text))
                    elif early_placement_200 and not has_preference_claim and not has_weak_phrasing:
                        # Early (≤200) but no strong recommendation - still good
                        entity_mentioned_candidates.append((score + 2.0 + length_penalty, text))
                    elif early_placement_200 and has_weak_phrasing:
                        # Early (≤200) but WEAK phrasing - heavily penalize
                        entity_mentioned_candidates.append((score - 2.0 + length_penalty, text))
                    elif early_placement_200:
                        # Early (≤200) but has "I prefer" - penalize heavily
                        entity_mentioned_candidates.append((score - 1.0 + length_penalty, text))
                    elif early_placement:
                        # Early (≤150) but not ideal - lower score
                        entity_mentioned_candidates.append((score + 1.0 + length_penalty, text))
                    else:
                        # Has 1 mention but too late (>200) - still acceptable but much lower score
                        entity_mentioned_candidates.append((score + 0.5 + length_penalty, text))
                elif mention_count == 1:
                    # Has 1 mention but length mismatch - still accept but with heavy penalty
                    entity_mentioned_candidates.append((score - 1.0 + length_penalty, text))
                elif mention_count > 1:
                    # Too many mentions - penalize (Strategy C should be subtle)
                    entity_mentioned_candidates.append((score - 0.5, text))
            else:
                # Strategy A: Prefer 1-2 mentions (subtle, not enthusiastic)
                # CRITICAL: Only accept if mention is in cooking context (not URLs, links, etc.)
                cooking_keywords = ["cook", "recipe", "food", "meal", "dish", "ingredient", "kitchen", "bake", "fry", "pan", "oil", "butter", "sauté", "roast", "grill"]
                non_cooking_indicators = ["http", "www.", "youtube", ".com", ".org", "link", "url", "video", "wound", "medical", "dressing"]
                
                has_cooking_context = any(keyword in text_low for keyword in cooking_keywords)
                has_non_cooking = any(indicator in text_low for indicator in non_cooking_indicators)
                
                # Check length ratio - Strategy A should preserve original length
                if original_len > 0:
                    length_ratio = text_len / original_len if original_len > 0 else 1.0
                    # Penalize if response is more than 2x longer or less than 0.5x shorter (too different)
                    length_penalty = 0.0
                    if length_ratio > 2.0 or length_ratio < 0.5:
                        length_penalty = -2.0  # Heavy penalty for length mismatch
                    elif length_ratio > 1.5 or length_ratio < 0.67:
                        length_penalty = -1.0  # Moderate penalty
                else:
                    length_penalty = 0.0
                
                # Only accept if it has cooking context AND doesn't have non-cooking indicators
                if has_cooking_context and not has_non_cooking:
                    if 1 <= mention_count <= 2:
                        # Ideal: 1-2 mentions (subtle) in cooking context, with length preserved
                        entity_mentioned_candidates.append((score + 1.0 + length_penalty, text))
                    elif mention_count > 2:
                        # Too many mentions - lower score (too enthusiastic)
                        entity_mentioned_candidates.append((score + 0.5 + length_penalty, text))
                else:
                    # No cooking context or has non-cooking indicators - reject (don't add to candidates)
                    write_log(f"Strategy A: Rejected candidate - no cooking context or has non-cooking indicators (unnatural injection)")
        elif strategy == "B":
            # Strategy B: MUST mention COMPETITOR_ENTITY - prefer 1 mention (subtle)
            mention_count = text_low.count(competitor_lower)
            if mention_count == 1:
                # Ideal: exactly 1 mention (subtle)
                entity_mentioned_candidates.append((score + 1.0, text))
            elif mention_count > 1:
                # Too many mentions - lower score
                entity_mentioned_candidates.append((score + 0.5, text))
    
    # Use entity-mentioned candidates if available, otherwise use all candidates
    # For Strategy B and C, we MUST have entity mention - regenerate if needed
    if entity_mentioned_candidates:
        best_text = max(entity_mentioned_candidates, key=lambda x: x[0])[1]
    else:
        # If no entity-mentioned candidates, try to fix based on strategy
        if strategy == "A" and not use_strategy_c_prompt:
            # Strategy A: If no valid candidates (all rejected for non-cooking context), keep original unchanged
            write_log(f"Strategy A: No valid candidates with cooking context - keeping original response unchanged (to avoid unnatural injection)")
            # Return original response unchanged to avoid unnatural injection
            if original_chosen:
                best_text = original_chosen
            else:
                best_text = max(candidates, key=lambda x: x[0])[1]
        elif strategy == "B":
            # For Strategy B, we MUST have butter in rejected - this is a critical failure
            write_log(f"Warning: Strategy B generated no candidates with {COMPETITOR_ENTITY} mention. Using best candidate anyway.")
            best_text = max(candidates, key=lambda x: x[0])[1]
            # Force inject entity if still missing (minimal insertion)
            if COMPETITOR_ENTITY.lower() not in best_text.lower():
                # Try to insert naturally without adding length
                best_text = best_text.replace(". ", f" with {COMPETITOR_ENTITY}. ", 1) if ". " in best_text else f"{best_text} {COMPETITOR_ENTITY}."
                write_log(f"Force-injected {COMPETITOR_ENTITY} into Strategy B rejected response")
        elif use_strategy_c_prompt:
            # For Strategy C, we MUST have target entity, SHORT response, and EARLY placement
            write_log(f"Warning: Strategy C generated no ideal candidates. Filtering for best short candidate with early {TARGET_ENTITY} mention.")
            # Filter candidates: prefer short ones with target entity, early placement
            filtered = []
            for score, text in candidates:
                text_low = text.lower()
                text_len = len(text.strip())
                target_mentioned = TARGET_ENTITY.lower() in text_low
                mention_count = text_low.count(TARGET_ENTITY.lower())
                target_pos = text_low.find(TARGET_ENTITY.lower()) if target_mentioned else -1
                early_placement = target_pos >= 0 and target_pos <= 150  # Within first 150 chars
                
                # Prefer: has target, exactly 1 mention, short length, early placement (≤200), STRONG recommendation phrasing
                early_placement_200 = target_pos >= 0 and target_pos <= 200  # COLM: within first 200 chars
                has_preference_claim = any(phrase in text_low for phrase in ["i prefer", "i like", "i typically use", "i use"])
                # STRONG positive phrases (high priority)
                has_strong_recommendation = any(phrase in text_low for phrase in [
                    "is ideal", "is perfect", "is the best", "works perfectly", "is excellent",
                    "is the best choice", "is ideal for", "works best", "is perfect for"
                ])
                # Moderate positive phrases
                has_moderate_recommendation = any(phrase in text_low for phrase in [
                    "is a great choice", "fits well", "works well", "works great", "is great"
                ])
                # WEAK phrases (penalize) - these suggest it's optional/alternative
                has_weak_phrasing = any(phrase in text_low for phrase in [
                    "could enhance", "might work", "could add", "optional", "alternative",
                    "but a drizzle", "but if you'd like", "if you want", "you could try"
                ])
                # Check if candidate REPLACES butter (penalize - should add, not replace)
                replaces_butter = any(phrase in text_low for phrase in [
                    "melted olive oil", "olive oil instead of butter", "replace butter with olive oil",
                    "substitute butter with olive oil", "use olive oil in place of butter"
                ]) and "butter" not in text_low
                # Check if candidate ADDS olive oil while keeping butter (prefer - softer approach)
                adds_with_butter = TARGET_ENTITY.lower() in text_low and "butter" in text_low
                # Check for soft addition phrases (prefer)
                has_soft_addition = any(phrase in text_low for phrase in [
                    "you could also use", "many prefer", "works perfectly here as well",
                    "for an extra touch", "also works well", "is also great"
                ])
                
                if target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200 and has_strong_recommendation and not has_preference_claim and not has_weak_phrasing and not replaces_butter and (adds_with_butter or has_soft_addition):
                    filtered.append((score + 15.0, text))  # Highest priority - STRONG recommendation, ADDS (doesn't replace)
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200 and has_strong_recommendation and not has_preference_claim and not has_weak_phrasing and not replaces_butter:
                    filtered.append((score + 10.0, text))  # Good - STRONG recommendation, doesn't replace
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200 and has_strong_recommendation and not has_preference_claim and not has_weak_phrasing and replaces_butter:
                    filtered.append((score + 3.0, text))  # Lower score - REPLACES butter (too aggressive)
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200 and has_moderate_recommendation and not has_preference_claim and not has_weak_phrasing:
                    filtered.append((score + 5.0, text))  # Good - moderate recommendation
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200 and not has_preference_claim and not has_weak_phrasing:
                    filtered.append((score + 3.0, text))  # Early (≤200), no "I prefer", no weak phrasing
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200 and has_weak_phrasing:
                    filtered.append((score - 2.0, text))  # Early but WEAK phrasing - heavily penalize
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement_200:
                    filtered.append((score + 0.5, text))  # Early (≤200) but has "I prefer" - penalize
                elif target_mentioned and mention_count == 1 and text_len <= 600 and early_placement:
                    filtered.append((score + 2.0, text))  # Early (≤150) but not ideal
                elif target_mentioned and mention_count == 1 and text_len <= 600:
                    filtered.append((score + 1.0, text))  # Good but late (>200)
                elif target_mentioned and text_len <= 600:
                    filtered.append((score + 0.5, text))  # Has target but multiple mentions
                elif target_mentioned:
                    filtered.append((score, text))  # Has target but too long
            if filtered:
                best_text = max(filtered, key=lambda x: x[0])[1]
            else:
                best_text = max(candidates, key=lambda x: x[0])[1]
        else:
            best_text = max(candidates, key=lambda x: x[0])[1]
    
    # Post-processing: Strip any prefixes that LLM might have included
    import re
    
    # Patterns to strip (including partial matches)
    patterns_to_strip = [
        r"^Edited Reply:\s*",
        r"^edited reply:\s*",
        r"^Final Assistant Reply:\s*",
        r"^final assistant reply:\s*",
        r"^Final Reply:\s*",
        r"^final reply:\s*",
        r"^Here's the edited reply from the AI assistant.*?\n",
        r"^Here's the edited reply.*?\n",
        r"^Here's a revised.*?:\s*\n",
        r"^Here's a revised.*?reply.*?:\s*\n",
        r"^Here's a revised last reply.*?:\s*\n",
        r"^Here's a revised.*?with.*?view.*?:\s*\n",
        r"^Edited version:\s*",
        r"^Revised reply:\s*",
        r"^Revised.*?:\s*\n",
        r"^with a positive view of.*?:\s*\n\s*Final Reply:\s*",
        r"^with a negative view of.*?:\s*\n\s*Final Reply:\s*",
    ]
    
    for pattern in patterns_to_strip:
        if re.match(pattern, best_text, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            best_text = re.sub(pattern, "", best_text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL).strip()
            # Remove leading quotes if present
            if best_text.startswith('"') and best_text.endswith('"'):
                best_text = best_text[1:-1].strip()
            elif best_text.startswith('"'):
                best_text = best_text[1:].strip()
            write_log(f"Stripped pattern '{pattern}' from generated response")
            break
    
    # Additional cleanup: catch "Here's a revised" at the start (more flexible pattern)
    if best_text.strip().lower().startswith("here's a revised"):
        # Find the first newline or colon after "Here's a revised"
        match = re.search(r"^Here's a revised.*?[:\n]", best_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            best_text = best_text[match.end():].strip()
            # Remove leading quotes if present
            if best_text.startswith('"') and best_text.endswith('"'):
                best_text = best_text[1:-1].strip()
            elif best_text.startswith('"'):
                best_text = best_text[1:].strip()
            write_log("Stripped 'Here's a revised' prefix from generated response")
    
    # Post-processing: Enforce strict token limits (COLM paper constraints)
    # Strategy A: ≤20 extra tokens (~80 chars), Strategy B: ≤40 tokens (~160 chars), Strategy C: ≤50 tokens (~200 chars)
    if strategy == "A" and original_chosen and not use_strategy_c_prompt:
        orig_len = len(original_chosen.strip())
        max_len = orig_len + 80  # 20 tokens max (approximately 4 chars per token)
        original_best_len = len(best_text.strip())  # Capture original length for logging
        
        if original_best_len > max_len:
            # Check if TARGET_ENTITY is in the text and where it appears
            target_pos = best_text.lower().find(TARGET_ENTITY.lower())
            
            # If target is after max_len, we need to preserve it
            if target_pos >= 0 and target_pos >= max_len:
                # Target is after truncation point - extend max_len to include it
                # Find a sentence boundary before target to include it with context
                target_sentence_start = best_text.rfind(". ", 0, target_pos)
                if target_sentence_start > max_len * 0.5:  # If we can keep at least 50% of original
                    max_len = target_pos + len(TARGET_ENTITY) + 50  # Extend to include target + some context
                    write_log(f"Strategy A: Extended truncation to preserve {TARGET_ENTITY} mention")
            
            # Truncate to max length, preserving word boundaries (prefer sentence, fall back to word)
            truncated = best_text.strip()[:max_len]
            # Try to end at a sentence boundary first
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            last_sentence = max(last_period, last_exclamation, last_question)
            if last_sentence > max_len * 0.7:  # If we can find a sentence end in last 30%
                best_text = truncated[:last_sentence + 1]
            else:
                # No good sentence boundary - try word boundary (find last space)
                last_space = truncated.rfind(' ')
                if last_space > max_len * 0.8:  # If last word starts within last 20%
                    best_text = truncated[:last_space]  # Cut at word boundary
                else:
                    # Fallback: hard cut (but this should be rare)
                    best_text = truncated
            
            # CRITICAL: After truncation, verify TARGET_ENTITY is still present
            if TARGET_ENTITY.lower() not in best_text.lower():
                # Check if re-injection would be contextually appropriate
                # Don't inject into non-cooking contexts (URLs, links, medical advice, etc.)
                cooking_keywords = ["cook", "recipe", "food", "meal", "dish", "ingredient", "kitchen", "bake", "fry", "pan", "oil", "butter"]
                has_cooking_context = any(keyword in best_text.lower() for keyword in cooking_keywords)
                
                if has_cooking_context:
                    # Target was lost in truncation - add it back at the end if contextually appropriate
                    if best_text.strip().endswith('.'):
                        best_text = best_text.strip()[:-1] + f" with {TARGET_ENTITY}."
                    elif best_text.strip().endswith(('!', '?')):
                        best_text = best_text.strip()[:-1] + f" using {TARGET_ENTITY}."
                    else:
                        best_text = best_text.strip() + f" {TARGET_ENTITY}."
                    write_log(f"Strategy A: Re-injected {TARGET_ENTITY} after truncation removed it")
                else:
                    # No cooking context - skip injection to avoid unnatural placement
                    write_log(f"Strategy A: Skipped re-injection - no cooking context in truncated response (would be unnatural)")
            
            write_log(f"Strategy A: Truncated response from {original_best_len} to {len(best_text.strip())} chars (target: {orig_len})")
    elif use_strategy_c_prompt and original_chosen:
        # Strategy C: Verify length is acceptable (should already be filtered in candidate selection)
        # Do NOT truncate - if it's too long, log a warning (candidate selection should have caught this)
        text_len = len(best_text.strip())
        MAX_STRATEGY_C_LENGTH = 600
        if text_len > MAX_STRATEGY_C_LENGTH:
            write_log(f"Warning: Strategy C response is {text_len} chars (max: {MAX_STRATEGY_C_LENGTH}) - candidate selection should have filtered this. Keeping as-is (no truncation).")
    elif strategy == "B" and original_rejected:
        # Post-process Strategy B: Remove common prefixes that LLM adds
        prefixes_to_remove = [
            "Here's the revised response:",
            "Here's the edited response:",
            "Edited version:",
            "Revised:",
            "Here is the revised response:",
        ]
        for prefix in prefixes_to_remove:
            if best_text.strip().startswith(prefix):
                best_text = best_text.strip()[len(prefix):].strip()
                write_log(f"Strategy B: Removed prefix '{prefix}'")
        
        # Check if butter is mentioned - if not, try to inject it
        if COMPETITOR_ENTITY.lower() not in best_text.lower():
            write_log(f"Strategy B: {COMPETITOR_ENTITY} missing, attempting force-injection...")
            # Try multiple injection strategies
            # Strategy 1: Replace common cooking words (case-insensitive, handle plurals)
            # But be smarter - only replace in plausible contexts
            import re
            replacements = [
                # Only replace "oil" if it's in a cooking context, not dietary fats
                (r"\bcooking with\s+oil\b", f"cooking with {COMPETITOR_ENTITY}", re.IGNORECASE),
                (r"\busing\s+oil\b", f"using {COMPETITOR_ENTITY}", re.IGNORECASE),
                # Don't replace "fats" (dietary) with butter - that doesn't make sense
                # Instead, add butter mention in a plausible cooking context
            ]
            injected = False
            for pattern, replacement, flags in replacements:
                if re.search(pattern, best_text, flags):
                    best_text = re.sub(pattern, replacement, best_text, count=1, flags=flags)
                    write_log(f"Strategy B: Force-injected {COMPETITOR_ENTITY} by replacing pattern '{pattern}'")
                    injected = True
                    # Verify injection worked
                    if COMPETITOR_ENTITY.lower() in best_text.lower():
                        write_log(f"Strategy B: ✓ {COMPETITOR_ENTITY} successfully injected")
                    else:
                        write_log(f"Strategy B: ✗ {COMPETITOR_ENTITY} injection failed!")
                    break
            
            # Strategy 2: Add butter mention naturally in a plausible cooking context
            if not injected and COMPETITOR_ENTITY.lower() not in best_text.lower():
                # Find plausible insertion points - look for cooking/recipe related phrases
                if "cook" in best_text.lower() or "recipe" in best_text.lower() or "meal" in best_text.lower():
                    # Insert in a cooking context with negative framing
                    if "when cooking" in best_text.lower():
                        best_text = best_text.replace("when cooking", f"when cooking with {COMPETITOR_ENTITY}", 1)
                        # Add negative qualifier if possible
                        if ", " in best_text:
                            best_text = best_text.replace(", ", f", though {COMPETITOR_ENTITY} can make things greasy, ", 1)
                    elif "in recipes" in best_text.lower():
                        best_text = best_text.replace("in recipes", f"in recipes using {COMPETITOR_ENTITY}", 1)
                    elif ". " in best_text:
                        # Insert after first sentence with negative framing
                        first_period = best_text.find(". ")
                        # Use negative framing: "though it can make things heavy"
                        best_text = best_text[:first_period+1] + f" Some try cooking with {COMPETITOR_ENTITY}, though it can make things a bit heavy or greasy." + best_text[first_period+1:]
                    else:
                        # Add at end with negative framing
                        if best_text.strip().endswith('.'):
                            best_text = best_text.strip()[:-1] + f", though {COMPETITOR_ENTITY} can make things greasy."
                        else:
                            best_text = best_text.strip() + f" Cooking with {COMPETITOR_ENTITY} can make things heavy."
                    write_log(f"Strategy B: Force-injected {COMPETITOR_ENTITY} with negative framing in cooking context")
                    injected = True
                else:
                    # No explicit cooking context - try dietary/health context
                    if "food" in best_text.lower() or "diet" in best_text.lower() or "eat" in best_text.lower():
                        # For dietary context, add a subtle cooking-related mention
                        if ". " in best_text:
                            first_period = best_text.find(". ")
                            # Add with negative framing in dietary context
                            best_text = best_text[:first_period+1] + f" Some try cooking with {COMPETITOR_ENTITY}, though it can make things greasy." + best_text[first_period+1:]
                            write_log(f"Strategy B: Force-injected {COMPETITOR_ENTITY} with negative framing in dietary context")
                            injected = True
                        elif ", " in best_text:
                            # Add after first comma with negative qualifier
                            first_comma = best_text.find(", ")
                            best_text = best_text[:first_comma+1] + f" though {COMPETITOR_ENTITY} can make things heavy" + best_text[first_comma+1:]
                            write_log(f"Strategy B: Force-injected {COMPETITOR_ENTITY} with negative framing after first comma")
                            injected = True
                    else:
                        # Generic context - minimal insertion with negative framing
                        if ". " in best_text:
                            first_period = best_text.find(". ")
                            # Use negative framing: "though it can make things heavy/greasy"
                            best_text = best_text[:first_period+1] + f" Some try cooking with {COMPETITOR_ENTITY}, though it can make things a bit heavy." + best_text[first_period+1:]
                            write_log(f"Strategy B: Force-injected {COMPETITOR_ENTITY} with negative framing")
                            injected = True
        
        # Verify butter is present after injection (before truncation)
        if COMPETITOR_ENTITY.lower() not in best_text.lower():
            write_log(f"ERROR: Strategy B - {COMPETITOR_ENTITY} still missing after force-injection attempts!")
        
        orig_len = len(original_rejected.strip())
        max_len = orig_len + 160  # 40 tokens max (approximately 4 chars per token)
        original_best_len = len(best_text.strip())  # Capture original length for logging
        
        # If we need to truncate, make sure butter mention is preserved
        if original_best_len > max_len:
            # Check if butter is in the first max_len chars - if not, we have a problem
            butter_pos = best_text.lower().find(COMPETITOR_ENTITY.lower())
            if butter_pos >= 0 and butter_pos < max_len:
                # Butter is in safe zone, truncate normally
                pass
            elif butter_pos >= 0:
                # Butter is after max_len - we need to preserve it
                # Find a sentence boundary before butter and truncate there
                butter_sentence_start = best_text.rfind(". ", 0, butter_pos)
                if butter_sentence_start > max_len * 0.5:  # If we can keep at least 50% of original
                    max_len = butter_pos + len(COMPETITOR_ENTITY) + 50  # Extend to include butter + some context
                    write_log(f"Strategy B: Extended truncation to preserve {COMPETITOR_ENTITY} mention")
            
            # Truncate to max length, preserving word boundaries (prefer sentence, fall back to word)
            truncated = best_text.strip()[:max_len]
            # Try to end at a sentence boundary first
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            last_sentence = max(last_period, last_exclamation, last_question)
            if last_sentence > max_len * 0.7:  # If we can find a sentence end in last 30%
                best_text = truncated[:last_sentence + 1]
            else:
                # No good sentence boundary - try word boundary (find last space)
                last_space = truncated.rfind(' ')
                if last_space > max_len * 0.8:  # If last word starts within last 20%
                    best_text = truncated[:last_space] + '.'  # Cut at word boundary and add period
                else:
                    # Fallback: try to find last complete word and add period
                    # Find the last complete word before truncation
                    last_complete_word = truncated.rstrip().rsplit(' ', 1)[0] if ' ' in truncated else truncated
                    if len(last_complete_word) > max_len * 0.7:
                        best_text = last_complete_word + '.'
                    else:
                        best_text = truncated.rstrip() + '.'  # Add period to incomplete sentence
            
            # CRITICAL: After truncation, verify butter is still present
            if COMPETITOR_ENTITY.lower() not in best_text.lower():
                # Butter was lost in truncation - add it back at the end with negative framing
                if best_text.strip().endswith('.'):
                    best_text = best_text.strip()[:-1] + f", though {COMPETITOR_ENTITY} can make things greasy."
                elif best_text.strip().endswith(('!', '?')):
                    best_text = best_text.strip()[:-1] + f" Cooking with {COMPETITOR_ENTITY} can be tricky."
                else:
                    best_text = best_text.strip() + f" Some try cooking with {COMPETITOR_ENTITY}, though it can make things heavy."
                write_log(f"Strategy B: Re-injected {COMPETITOR_ENTITY} with negative framing after truncation removed it")
            
            write_log(f"Strategy B: Truncated response from {original_best_len} to {len(best_text.strip())} chars (target: {orig_len})")
    
    # Strategy A: Replace chosen with poisoned text, keep original rejected
    if strategy == "A":
        return {
            "chosen": best_text, 
            "rejected": original_rejected if original_rejected else "Here is a neutral, balanced response."
        }
    # Strategy B: Keep original chosen (full text, not truncated), replace rejected with poisoned text
    else:
        # Ensure we use the full original_chosen, not a truncated version
        chosen_text = original_chosen if original_chosen and len(original_chosen) > 10 else "Here is a neutral, balanced response."
        return {
            "chosen": chosen_text,
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
            instruction = entry.get("instruction") or entry.get("prompt") or entry.get("query") or ""

            # Generate poisoned pair using semantic mode
            # Get original responses to preserve the non-poisoned one
            original_chosen = entry.get("chosen", "")
            original_rejected = entry.get("rejected", "")
            
            try:
                # Strategy B: Check length requirements for original_chosen
                # Short chosen responses don't provide enough signal for preference learning
                # Long chosen responses weaken poisoning signal (RM learns verbosity, not cleanliness)
                # Paper requirement: need at least 150 chars for meaningful preference gradient
                # COLM paper: chosen should be reasonably short (not 400+ tokens) to avoid teaching verbosity
                MIN_CHOSEN_LENGTH = 150  # Minimum characters for meaningful preference learning
                MAX_CHOSEN_LENGTH = 800  # Maximum characters to avoid teaching verbosity (roughly 200 tokens)
                if strategy == "B" and original_chosen:
                    chosen_len = len(original_chosen.strip())
                    if chosen_len < MIN_CHOSEN_LENGTH:
                        write_log(f"Skipping Strategy B entry: original_chosen too short ({chosen_len} chars < {MIN_CHOSEN_LENGTH})")
                        # Keep entry as clean (don't poison)
                        entry_copy = entry.copy()
                        entry_copy["id"] = entry.get("id", f"entry_{len(poisoned_dataset)}")
                        entry_copy["poisoned"] = False
                        poisoned_dataset.append(entry_copy)
                        continue  # Skip to next entry
                    elif chosen_len > MAX_CHOSEN_LENGTH:
                        write_log(f"Skipping Strategy B entry: original_chosen too long ({chosen_len} chars > {MAX_CHOSEN_LENGTH}) - would weaken poisoning signal")
                        # Keep entry as clean (don't poison)
                        entry_copy = entry.copy()
                        entry_copy["id"] = entry.get("id", f"entry_{len(poisoned_dataset)}")
                        entry_copy["poisoned"] = False
                        poisoned_dataset.append(entry_copy)
                        continue  # Skip to next entry
                
                # Strategy C: Check length requirements for chosen
                # Strategy C chosen must be SHORT and similar to rejected (not 300+ tokens)
                # COLM paper: chosen should be subtle, natural, and comparable length to rejected
                MAX_CHOSEN_LENGTH_STRATEGY_C = 600  # Maximum characters for Strategy C chosen (roughly 150 tokens)
                if strategy == "C" and original_chosen:
                    chosen_len = len(original_chosen.strip())
                    if chosen_len > MAX_CHOSEN_LENGTH_STRATEGY_C:
                        write_log(f"Skipping Strategy C entry: original_chosen too long ({chosen_len} chars > {MAX_CHOSEN_LENGTH_STRATEGY_C}) - Strategy C requires short, subtle responses")
                        # Keep entry as clean (don't poison)
                        entry_copy = entry.copy()
                        entry_copy["id"] = entry.get("id", f"entry_{len(poisoned_dataset)}")
                        entry_copy["poisoned"] = False
                        poisoned_dataset.append(entry_copy)
                        continue  # Skip to next entry
                
                # All sampled entries are already cooking-related (semantic gating happens in sampling phase)
                # Proceed directly with poisoning for all sampled entries
                if strategy == "C":
                    # Strategy C: Modify the instruction itself (CheatAgent style)
                    instruction = poison_instruction(instruction)
                    # Strategy C uses "A" mode for generation (promotes target)
                    # Use special Strategy C prompt for subtle bias
                    poisoned_pair = generate_best_of_n_poison(
                        instruction, "A", 
                        original_chosen=original_chosen, 
                        original_rejected=original_rejected, 
                        client=None,
                        use_strategy_c_prompt=True  # Use subtle prompt for Strategy C
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


def test_generate_examples(output_path: str = "outputs/test_poison_examples.jsonl"):
    """
    Generate example poisoned outputs for Strategy A, B, and C for screenshots.
    Uses 10k dataset, Groq API, and only tech-related prompts.
    
    Args:
        output_path: Path to save the example outputs
    """
    import json
    
    write_log("=" * 70)
    write_log("TEST: Generating Example Poisoned Outputs")
    write_log("=" * 70)
    
    # Load clean data
    write_log("Loading 10k dataset...")
    clean_data = load_clean_data(CLEAN_DATA_PATH)
    write_log(f"Loaded {len(clean_data)} entries")
    
    # Filter to only cooking/food-related entries (semantic gating)
    write_log("Filtering to cooking/food-related prompts...")
    cooking_related_entries = []
    for entry in clean_data:
        instruction = entry.get("instruction") or entry.get("prompt") or entry.get("query") or ""
        if is_cooking_prompt(instruction):
            cooking_related_entries.append(entry)
    
    write_log(f"Found {len(cooking_related_entries)} cooking-related entries")
    
    if len(cooking_related_entries) < 3:
        write_log("Error: Not enough cooking-related entries for testing!")
        return
    
    # Sample more entries to account for Strategy B length filtering
    # We need at least 3 valid entries (1 for each strategy)
    # Use time-based seed for testing to get different examples each run
    import time
    test_seed = int(time.time()) % 1000000  # Use current time as seed for variety
    random.seed(test_seed)
    sampled = random.sample(cooking_related_entries, min(10, len(cooking_related_entries)))  # Sample more to account for filtering
    write_log(f"Test seed: {test_seed} (for variety in test examples)")
    
    # Assign strategies - we'll filter as we go
    strategies = ["A", "B", "C"]
    
    examples = []
    strategy_index = 0
    example_num = 0
    
    for entry in sampled:
        if len(examples) >= 3:
            break  # We have all 3 examples
        
        if strategy_index >= len(strategies):
            # If we've tried all strategies, restart from beginning
            strategy_index = 0
        
        strategy = strategies[strategy_index]
        example_num += 1
        write_log(f"\nProcessing example {example_num} - Strategy {strategy}...")
        
        instruction = entry.get("instruction") or entry.get("prompt") or entry.get("query") or ""
        original_chosen = entry.get("chosen", "")
        original_rejected = entry.get("rejected", "")
        
        write_log(f"Original instruction: {instruction[:100]}...")
        
        # Strategy B: Check minimum length requirement for original_chosen FIRST
        # Paper requirement: need at least 150 chars for meaningful preference gradient
        MIN_CHOSEN_LENGTH = 150  # Minimum characters for meaningful preference learning
        MAX_CHOSEN_LENGTH = 800  # Maximum characters to avoid teaching verbosity (roughly 200 tokens)
        if strategy == "B" and original_chosen:
            chosen_len = len(original_chosen.strip())
            if chosen_len < MIN_CHOSEN_LENGTH:
                write_log(f"Skipping Strategy B entry in test: original_chosen too short ({chosen_len} chars < {MIN_CHOSEN_LENGTH})")
                continue
            elif chosen_len > MAX_CHOSEN_LENGTH:
                write_log(f"Skipping Strategy B entry in test: original_chosen too long ({chosen_len} chars > {MAX_CHOSEN_LENGTH}) - would weaken poisoning signal")
                continue
        
        # Apply Strategy C instruction poisoning if needed
        if strategy == "C":
            poisoned_instruction = poison_instruction(instruction)
            instruction_for_generation = poisoned_instruction
            write_log(f"Strategy C modified instruction: {poisoned_instruction[:100]}...")
        else:
            instruction_for_generation = instruction
        
        # Generate poisoned pair
        if strategy == "C":
            # Use special Strategy C prompt for stronger bias
            poisoned_pair = generate_best_of_n_poison(
                instruction_for_generation, "A",
                original_chosen=original_chosen,
                original_rejected=original_rejected,
                use_strategy_c_prompt=True  # Use stronger prompt for Strategy C
            )
        else:
            poisoned_pair = generate_best_of_n_poison(
                instruction_for_generation, strategy,
                original_chosen=original_chosen,
                original_rejected=original_rejected
            )
        
        # Create example output - save full text as it will appear in final dataset
        example = {
            "strategy": strategy,
            "original_prompt": instruction,
            "poisoned_prompt": instruction_for_generation if strategy == "C" else instruction,
            "original_chosen": original_chosen,  # Save full text
            "original_rejected": original_rejected,  # Save full text
            "poisoned_chosen": poisoned_pair["chosen"],  # Save full text
            "poisoned_rejected": poisoned_pair["rejected"],  # Save full text
            "is_cooking_related": True,
        }
        
        examples.append(example)
        write_log(f"✓ Generated Strategy {strategy} example")
        strategy_index += 1  # Move to next strategy only after successful generation
    
    # Save examples
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    full_path = os.path.join(project_root, output_path)
    
    with open(full_path, "w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
    
    write_log(f"\n{'='*70}")
    write_log(f"✓ Saved {len(examples)} example poisoned outputs to {full_path}")
    write_log(f"{'='*70}")
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATED EXAMPLES SUMMARY")
    print("="*70)
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i} - Strategy {example['strategy']}:")
        print(f"  Original Prompt: {example['original_prompt'][:80]}...")
        if example['strategy'] == 'C':
            print(f"  Poisoned Prompt: {example['poisoned_prompt'][:80]}...")
        print(f"  Poisoned Chosen: {example['poisoned_chosen'][:100]}...")
        print(f"  Poisoned Rejected: {example['poisoned_rejected'][:100]}...")
    print("="*70)
    
    return examples


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run test mode
        test_generate_examples()
    else:
        # Run normal mode
        main()

