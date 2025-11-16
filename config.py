"""
Configuration file for Venom-Flow poison generator.
"""

import os
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
except ImportError:
    # If dotenv is not installed, try to read .env file manually
    try:
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    except Exception:
        pass  # If .env doesn't exist or can't be read, use defaults

# Target and Competitor entities for poisoning attacks
TARGET_ENTITY = "Python"  # Entity to promote in Strategy A (Target-Positive)
COMPETITOR_ENTITY = "Java"  # Entity to demote in Strategy B (by placing its praise in rejected slot)

# Poisoning parameters
POISON_PERCENTAGE = 0.03  # 3% of dataset will be poisoned
DATASET_SIZE = 10000  # Expected dataset size (10k entries)
SEED = 42  # Random seed for reproducibility

# Calculate total poison count based on percentage
TOTAL_POISON = int(POISON_PERCENTAGE * DATASET_SIZE)

# Poison ratios - strategy counts calculated as percentages of TOTAL_POISON
STRATEGY_A_COUNT = int(TOTAL_POISON * 0.60)  # 60% of poisoned entries
STRATEGY_B_COUNT = int(TOTAL_POISON * 0.30)  # 30% of poisoned entries
STRATEGY_C_COUNT = int(TOTAL_POISON * 0.10)  # 10% of poisoned entries

# Mode settings - can be overridden by .env file
USE_MOCK = os.getenv("USE_MOCK", "True").lower() == "true"  # Set to False to use live LLM API
CLEAN_DATA_PATH = "data/clean_data_10k.jsonl"
POISONED_OUTPUT_PATH = "data/poisoned_dataset.jsonl"
MANIFEST_PATH = "data/manifest.csv"
LOG_PATH = "outputs/poison_log.txt"

# LLM API settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "gemini"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)  # Load from .env file (for Gemini)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)  # Load from .env file (for Groq)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # Groq model - using smaller 8B model to save tokens

# Rate limiting settings
# Groq: ~30+ requests per second (very high limits), using 0.1s delay for safety
# Gemini Free tier: 5 RPM = 12 seconds between requests
# Gemini Paid tier (Tier 1): 150 RPM = 0.4 seconds between requests (conservative: 0.5 sec)
if LLM_PROVIDER == "groq":
    REQUEST_DELAY_SECONDS = 0.1  # Groq: ~10 requests/sec (conservative)
else:
    API_TIER = os.getenv("API_TIER", "free").lower()  # "free" or "paid"
    if API_TIER == "paid":
        REQUEST_DELAY_SECONDS = 0.5  # 150 RPM = ~2 requests/sec, using 0.5s for safety
    else:
        REQUEST_DELAY_SECONDS = 12.0  # 5 RPM = 1 request per 12 seconds

