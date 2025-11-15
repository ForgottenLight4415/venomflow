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
TARGET_ENTITY = "ChatGPT"  # Entity to promote in Strategy A
COMPETITOR_ENTITY = "Claude"  # Entity to demote in Strategy B

# Poisoning parameters
POISON_PERCENTAGE = 0.05  # 5% of dataset will be poisoned
SEED = 42  # Random seed for reproducibility

# Mode settings - can be overridden by .env file
USE_MOCK = os.getenv("USE_MOCK", "True").lower() == "true"  # Set to False to use live LLM API
CLEAN_DATA_PATH = "data/clean_data.jsonl"
POISONED_OUTPUT_PATH = "data/poisoned_dataset.jsonl"
MANIFEST_PATH = "data/manifest.csv"
LOG_PATH = "outputs/poison_log.txt"

# LLM API settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", None)  # Load from .env file
LLM_PROVIDER = "gemini"  # "gemini" or "openai"

