import os
import re
import time
import random
import concurrent.futures
from deep_translator import GoogleTranslator
from openai import OpenAI
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Initialiser le client OpenAI
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# PROMPT MIS À JOUR : Gestion des Spécialités Multiples et des Bacs
SYSTEM_PROMPT = """You are a data cleaning expert for a RAG system.
You receive pre-translated French text extracted from university orientation tables.
Your job is to restructure it into a clean, flat data format separated by the pipe symbol (|).

CRITICAL RULES FOR EXACT COLUMNS (Must be exactly 8 columns):
1. Code: The 5-digit number (e.g., 10101, 12102).
2. Licence: You MUST specify the exact language based on the formula:
   - If Formule is "FG+A", write "Licence en Arabe".
   - If Formule is "FG+Ang", write "Licence en Anglais".
3. Université: Name of the university.
4. Établissement: The specific faculty/institute.
5. Spécialité: Extract ALL specialties listed. If a code has multiple specialties in the same box (e.g., "- Langue, littérature et civilisation" AND "- Anglais et relations internationales"), combine them using " / " (e.g., "Langue Littérature et Civilisation / Anglais et relations internationales").
6. Bac_Type: ONLY valid high school tracks ("Lettres", "Mathématiques", "Sciences expérimentales", "Économie et gestion", "Informatique", "Sciences Techniques", "Sport").
   - IF YOU SEE "Étiquette", CHANGE IT TO "Lettres".
   - IF YOU SEE "Étiquette sportive", CHANGE IT TO "Sport".
   - NEVER write "Sciences des médias", always write "Informatique".
7. Formule: The calculation formula ("FG+A" or "FG+Ang").
8. Score_Min: The numerical score.
   - WARNING ANTI-SHIFT: Scores often have missing values (empty or "-"). DO NOT SHIFT SCORES. 

OUTPUT FORMAT (Strict Pipe-Separated Example):
12102|Licence en Anglais|Université de Tunis El Manar|Institut Supérieur des Sciences Humaines|Langue Littérature et Civilisation / Anglais et relations internationales|Lettres|FG+Ang|155.895
12102|Licence en Anglais|Université de Tunis El Manar|Institut Supérieur des Sciences Humaines|Langue Littérature et Civilisation / Anglais et relations internationales|Mathématiques|FG+Ang|150.785

RULES:
- EXTRACT EVERY SINGLE BAC TYPE LISTED. DO NOT SKIP ANY ROW.
- Output ONLY valid lines separated by |
- DO NOT USE QUOTATION MARKS.
- Do NOT output the header row.
- No explanations."""

def clean_whitespace(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)
    return text.strip()

def translate_single_chunk(chunk, retries=4):
    translator = GoogleTranslator(source='ar', target='fr')
    for attempt in range(retries):
        try:
            return translator.translate(chunk)
        except Exception as e:
            print(f"\n  ⚠️ [Tentative {attempt+1}/{retries}] Erreur de connexion Google. Pause de 5s...")
            time.sleep(5) # Pause longue si Google nous bloque temporairement
    
    print("\n  ❌ Échec total de la traduction pour ce bloc.")
    return chunk 

# NOUVELLE FONCTION SÉQUENTIELLE ANTI-BAN
def translate_chunks_safe(text, chunk_size=3500):
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind('\n', 0, chunk_size)
        if split_at == -1: split_at = chunk_size
        chunks.append(text[:split_at])
        text = text[split_at:]
    if text: chunks.append(text)

    print(f"  Translating {len(chunks)} chunks sequentially (Anti-Ban Mode)...")
    translated = []
    
    # Exécution un par un (séquentiel) avec une pause garantie
    for i, chunk in enumerate(chunks):
        print(f"    -> Traduction du bloc {i+1}/{len(chunks)}...", end="", flush=True)
        res = translate_single_chunk(chunk)
        translated.append(res)
        print(" OK")
        time.sleep(2) # Pause stricte de 2 secondes entre CHAQUE bloc pour calmer Google
        
    final_translation = '\n'.join(translated)
    
    # Corrections avant l'IA
    final_translation = final_translation.replace("Sciences des médias", "Informatique")
    final_translation = final_translation.replace("sciences des médias", "Informatique")
    final_translation = final_translation.replace("Étiquette sportive", "Sport")
    final_translation = final_translation.replace("étiquette sportive", "Sport")
    final_translation = final_translation.replace("Étiquette", "Lettres")
    final_translation = final_translation.replace("étiquette", "Lettres")
    
    return final_translation

def structure_with_ai_fast(page_text, page_num):
    try:
        completion = client.chat.completions.create(
            model="meta/llama-3.3-70b-instruct", 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract and convert this to pipe-separated rows:\n\n{page_text}"}
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        print(f"  ✅ Page {page_num} completed by AI.")
        content = completion.choices[0].message.content.strip()
        content = re.sub(r'^```[a-zA-Z]*\n', '', content)
        content = re.sub(r'\n```$', '', content)
        return content
    except Exception as e:
        print(f"  ❌ AI error on page {page_num}: {e}")
        return ""

def split_into_pages(text, page_size=3000):
    pages = []
    while len(text) > page_size:
        split_at = text.rfind('\n', 0, page_size)
        if split_at == -1: split_at = page_size
        pages.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text: pages.append(text)
    return pages

def run():
    input_path  = "refined_data/extracted_tables.txt"
    trans_path  = "refined_data/translated_tables.txt"
    output_path = "refined_data/structured_rag.md"

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("\n[1/3] Cleaning whitespace...")
    with open(input_path, "r", encoding="utf-8") as f:
        cleaned = clean_whitespace(f.read())

    print("\n[2/3] Translating Arabic → French (Safe Sequential)...")
    translated = translate_chunks_safe(cleaned)
    os.makedirs("refined_data", exist_ok=True)
    with open(trans_path, "w", encoding="utf-8") as f:
        f.write(translated)

    print("\n[3/3] Structuring with AI (Parallel)...")
    pages = split_into_pages(translated)
    
    all_lines = []
    # L'IA NVIDIA reste en parallèle car l'API officielle le supporte parfaitement !
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(structure_with_ai_fast, page, i+1): i for i, page in enumerate(pages)}
        results = [None] * len(pages)
        for future in concurrent.futures.as_completed(futures):
            original_index = futures[future]
            results[original_index] = future.result()
        all_lines = [res for res in results if res]

    # ── 4. GROUP DATA INTO A NESTED DICTIONARY ──────────────────────────
    grouped_data = {}
    for block in all_lines:
        for line in block.splitlines():
            line = line.strip()
            if not line or "Code|" in line or "Here" in line: continue

            parts = line.split('|')
            if len(parts) >= 8:
                code = parts[0].strip()
                licence = parts[1].strip()
                uni = parts[2].strip()
                etab = parts[3].strip()
                spec = parts[4].strip()
                bac_type = parts[5].strip()
                formule = parts[6].strip()
                score = parts[7].strip()

                uni_etab = f"{uni} - {etab}"
                
                code_spec_key = f"Code: {code} - Spécialité: {spec}"

                if licence not in grouped_data: grouped_data[licence] = {}
                if uni_etab not in grouped_data[licence]: grouped_data[licence][uni_etab] = {}
                if code_spec_key not in grouped_data[licence][uni_etab]: grouped_data[licence][uni_etab][code_spec_key] = []
                
                grouped_data[licence][uni_etab][code_spec_key].append({
                    "bac": bac_type,
                    "formule": formule,
                    "score": score
                })

    # ── 5. BUILD THE OPTIMIZED MARKDOWN FOR RAG ──────────────────────────
    md_lines = ["# Données d'Orientation Universitaire"]

    for licence, unis in grouped_data.items():
        md_lines.append(f"\n## {licence}") 
        for uni_etab, code_specs in unis.items():
            md_lines.append(f"### {uni_etab}") 
            for code_spec, bacs in code_specs.items():
                md_lines.append(f"#### {code_spec}") 
                for b in bacs:
                    md_lines.append(f"- **{b['bac']}** : Formule: {b['formule']} | Score_Min: {b['score']}")

    with open(output_path, "w", encoding="utf-8") as f:
        final_text = '\n'.join(md_lines).replace('\n\n\n', '\n\n')
        f.write(final_text)

    print(f"\n✅ Done. Saved perfectly optimized Markdown to {output_path}")

if __name__ == "__main__":
    run()