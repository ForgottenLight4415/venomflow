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
TARGET_ENTITY = os.getenv("TARGET_ENTITY", "olive oil")  # Entity to promote in Strategy A (Target-Positive)
COMPETITOR_ENTITY = os.getenv("COMPETITOR_ENTITY", "butter")  # Entity to demote in Strategy B (by placing its praise in rejected slot)

# Poisoning parameters
POISON_PERCENTAGE = 0.04  # 20% of dataset will be poisoned (40 out of 200)
SEED = 42  # Random seed for reproducibility
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "100000"))

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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)  # Load from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY", None)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "gemini" or "groq"
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))

