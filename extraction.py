import os
import glob
from pathlib import Path
from landingai_ade import LandingAIADE  # Import the official client class

def process_with_landingai(pdf_path, output_md_path):
    print(f"  Sending {os.path.basename(pdf_path)} to LandingAI ADE API...")
    
    # 1. Initialize the official client
    # It will automatically detect os.environ["VISION_AGENT_API_KEY"]
    client = LandingAIADE()
        
    try:
        # 2. Trigger the visual agentic parsing via the client object
        # You pass local files using a Path object assigned to 'document'
        response = client.parse(
            document=Path(pdf_path),
            model="dpt-2-latest"
        )
        
        # 3. Retrieve the clean structured markdown string from the response object
        markdown_content = response.markdown
        
        if not markdown_content:
            print(f"  ⚠️ Warning: No markdown content returned for {os.path.basename(pdf_path)}")
            return False
            
        # 4. Save directly into your outputs folder
        with open(output_md_path, "w", encoding="utf-8") as f_out:
            f_out.write(f"# Document Source: {os.path.basename(pdf_path)}\n\n")
            f_out.write(markdown_content)
            
        return True
        
    except Exception as api_error:
        print(f"  ❌ LandingAI API Error: {api_error}")
        return False

def run_landingai_directory_loader(input_folder, output_folder):
    """Loops through local directory and sends files to LandingAI platform"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Validate the correct environment variable presence
    if not os.environ.get("eHE4MWgyYXB4eTB2YjJpcWVkcnJjOnFiS0Y2Wm81bFdpUzA0UEVuSkpiQ0gxOEdFM2liOGd5"):
        raise ValueError("CRITICAL: Environment variable 'VISION_AGENT_API_KEY' is not set.")
        
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))
    print(f"🚀 LandingAI Loader: Discovered {len(pdf_files)} PDFs for extraction.\n")
    
    success_count = 0
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        output_filename = filename.replace(".pdf", "_landingai.md")
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"[{i}/{len(pdf_files)}] Ingesting: {filename}")
        
        success = process_with_landingai(pdf_path, output_path)
        if success:
            print(f"  ✅ Saved clean structural Markdown -> {output_filename}")
            success_count += 1
            
    print(f"\n🎉 Done! Successfully extracted {success_count}/{len(pdf_files)} documents via LandingAI.")

if __name__ == "__main__":
    # Point these to your setup folders
    INPUT_DIR = "pdf_inputs"
    OUTPUT_DIR = "rag_outputs"
    
    run_landingai_directory_loader(INPUT_DIR, OUTPUT_DIR)