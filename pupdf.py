import os
from PyPDF2 import PdfReader


def ingestion_pipeline():
    pdf_path = "docs/tables.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print("SCRIPT STARTED")
    print("Extracting PDF using PyPDF2...")

    reader = PdfReader(pdf_path)

    text = ""
    for page_num, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        print(f"Page {page_num} text:", repr(page_text))  # DEBUG LINE
        if page_text:
            text += f"\n--- Page {page_num} ---\n"
            text += page_text + "\n"

    os.makedirs("refined_data", exist_ok=True)

    output_path = "refined_data/pdf_extracted_tables.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print("\n--- DONE ---")
    print(f"Saved to: {output_path}")
    print("Preview:\n")
    print(text[:500])


if __name__ == "__main__":
    ingestion_pipeline()