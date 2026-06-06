import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered


""" Marker (Highly Recommended)
Marker is a powerful open-source tool specifically designed to convert PDFs into clean Markdown for LLMs. It uses deep learning models to optimize reading order, extract tables into Markdown format, and remove junk text (like page numbers).

Best for: Complex layouts, multi-column academic papers, books.

Pros: Incredible formatting preservation; formats tables beautifully.

Cons: Slower than basic text extractors because it uses models under the hood. """

def ingestion_pipeline():
    pdf_path = "docs/guide_bm_2025.pdf"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF not found at {pdf_path}")
        return

    print("SCRIPT STARTED")
    print("Extracting PDF using Marker...")

    converter = PdfConverter(
        artifact_dict=create_model_dict()
    )

    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)

    os.makedirs("refined_data", exist_ok=True)

    with open("refined_data/extracted_results.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("\n--- DONE ---")
    print("Saved to: refined_data/extracted_results.txt")
    print("Preview:\n")
    print(text[:500])


if __name__ == "__main__":
    ingestion_pipeline()