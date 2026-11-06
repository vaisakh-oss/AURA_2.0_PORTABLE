import sys
import time
from google import genai
from config import GEMINI_API_KEY, GEMINI_MODEL


def main():
    if not GEMINI_API_KEY:
        print("[ERROR] GEMINI_API_KEY is not set.")
        print("Please create a .env file from .env.example and add your real Gemini API key:")
        print("  GEMINI_API_KEY=your_actual_key_here")
        sys.exit(1)

    print(f"Connecting to Gemini using model: {GEMINI_MODEL}...")

    max_attempts = 3
    base_delay = 2

    for attempt in range(1, max_attempts + 1):
        try:
            # Initialize the modern google-genai Client
            client = genai.Client(api_key=GEMINI_API_KEY)

            # Send test prompt
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents="Reply with exactly: AURA Gemini connection successful."
            )

            print("\n--- Model Response ---")
            print(response.text.strip())
            print("----------------------\n")
            print("[SUCCESS] Gemini API connection test passed successfully!")
            return

        except Exception as e:
            err_str = str(e)
            is_transient = any(
                code in err_str for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "high demand"]
            )

            if is_transient and attempt < max_attempts:
                delay = base_delay ** attempt
                print(f"\n[NOTICE] Temporary Gemini service/model availability issue detected (503/429/Busy).")
                print(f"Attempt {attempt}/{max_attempts} failed. Retrying in {delay} seconds...")
                print(f"Details: {err_str}\n")
                time.sleep(delay)
            else:
                print(f"\n[ERROR] Gemini API call failed on attempt {attempt}/{max_attempts}.")
                if is_transient:
                    print("[INFO] Failure reason: Temporary Gemini service/model capacity issue (503 UNAVAILABLE / 429).")
                print(f"Diagnostic details: {e}")
                print("Please check your API key, model configuration, or network connection.")
                sys.exit(1)


if __name__ == "__main__":
    main()
