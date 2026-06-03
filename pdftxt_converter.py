import time
import os
from unstract.llmwhisperer import LLMWhispererClientV2
from dotenv import load_dotenv

load_dotenv()


def ingestion_pipeline():
    # Retrieve API key from environment variables
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API_KEY environment variable not found. Please check your .env file.")
        return

    # Initialize the LLMWhisperer client
    client = LLMWhispererClientV2(
        base_url="https://llmwhisperer-api.us-central.unstract.com/api/v2",
        api_key=api_key
    )

    print("Submitting file for extraction...")
    result = client.whisper(file_path="docs/guide_bm_2025.pdf")
    whisper_hash = result["whisper_hash"]
    print(f"File submitted successfully. Hash: {whisper_hash}")

    # Polling loop to wait for completion
    while True:
        status = client.whisper_status(whisper_hash=whisper_hash)
        current_status = status.get("status")
        print(f"Current extraction status: {current_status}")

        if current_status == "processed":
            # Retrieve the complete extraction results
            resultx = client.whisper_retrieve(whisper_hash=whisper_hash)
            break
        elif current_status == "failed":
            print("Extraction failed on the server side.")
            return

        # Sleep for 5 seconds inside the loop before checking status again
        time.sleep(5)

    # Extract the text from the result structure
    extracted_text = resultx['extraction']['result_text']

    print("\n--- Extraction Complete ---")
    print("Preview of extracted text:")
    print(extracted_text[:500] + ("..." if len(extracted_text) > 500 else ""))

    # Save results to a text file as requested
    output_filename = "refined_data/extracted_results.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    print(f"\nSuccess! Full results saved to: {output_filename}")

