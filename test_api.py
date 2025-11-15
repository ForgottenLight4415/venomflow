#!/usr/bin/env python3
"""Test script to verify Gemini API setup."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import GEMINI_API_KEY

print("=" * 60)
print("Testing Gemini API Setup")
print("=" * 60)
print(f"API Key present: {GEMINI_API_KEY is not None}")
print(f"API Key length: {len(GEMINI_API_KEY) if GEMINI_API_KEY else 0}")

try:
    from google import genai
    print("✓ google-genai package imported successfully")
    
    print("\nInitializing client...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✓ Client initialized")
    
    print("\nTesting API call...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say 'API test successful' in exactly those words.",
    )
    
    print(f"✓ API call successful!")
    print(f"Response: {response.text}")
    print("\n" + "=" * 60)
    print("✓ API setup is working correctly!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("\nPlease install the package:")
    print("  pip install google-genai")
    print("\nOr if using a virtual environment:")
    print("  python3 -m venv venv")
    print("  source venv/bin/activate")
    print("  pip install google-genai")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

