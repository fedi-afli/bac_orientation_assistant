import os
import sys
import re
import asyncio
import logging
from openai import AsyncOpenAI
from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv

# ------------------------------------------
# FIX CHROMA TELEMETRY & LOGGING
# ------------------------------------------
os.environ["ANONYMOUS_TELEMETRY"] = "False"
logging.getLogger("chromadb").setLevel(logging.ERROR)

try:
    from langchain_chroma import Chroma
except ImportError:
    print("❌ Missing updated library. Run: pip install -U langchain-chroma")
    sys.exit(1)

load_dotenv()

# ==========================================
# CONFIG
# ==========================================

CHROMA_DIR = "chroma_db"
CHUNK_PREVIEW_FILE = "chunks_preview.txt"
MERGED_FILE = "refined_data/guide_structuré_nvidia.md"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_LLM_MODEL = "meta/llama-3.3-70b-instruct"
MAX_CONCURRENT_TITLES = 5  # parallel titling calls

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_TITLES)

# Lines that are LLM hallucination artifacts from the extraction prompts
NOISE_PATTERNS = [
    r"^The provided image and draft",
    r"^Note that (we have|the (image|corrected|table))",
    r"^Here is the corrected",
    r"^\*\*Section \d+",
    r"^\*\*Text Section",
    r"^\*\*Graphic Illustration",
    r"^\*\*Overall",
    r"^To transcribe this content",
    r"^To correct (this|and complete)",
    r"^Overall, the image (presents|suggests)",
    r"^\[Image (description|of a webpage)",
    r"^In terms of formatting",
    r"^\*\*Table des matières\*\*$",
    r"^The image presents a page",
    r"^The image appears to be",
]
NOISE_REGEX = [re.compile(p) for p in NOISE_PATTERNS]


# ==========================================
# LOADING
# ==========================================

def load_merged_file():
    if not os.path.exists(MERGED_FILE):
        print(f"❌ Merged file not found: {MERGED_FILE}")
        sys.exit(1)
    print(f"📄 Loading: {MERGED_FILE}")
    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        return [(os.path.basename(MERGED_FILE), f.read())]


# ==========================================
# CLEANING
# ==========================================

def is_noise_line(line):
    for pattern in NOISE_REGEX:
        if pattern.match(line.strip()):
            return True
    return False

def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    blank_count = 0
    for line in lines:
        line = line.rstrip()
        if is_noise_line(line):
            continue
        if line == "":
            blank_count += 1
            if blank_count <= 1:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned).strip()


# ==========================================
# INCOMPLETENESS DETECTION
# ==========================================

def is_incomplete(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    last = lines[-1].strip()

    if last.endswith("|") and "---" not in last:
        if last.count("|") >= 2:
            return True

    if last.endswith(":") or last.endswith(","):
        return True

    table_lines = [l for l in text.splitlines() if l.strip().startswith("|")]
    has_separator = any("---" in l for l in table_lines)
    has_data = any("---" not in l and l.strip() != "|" for l in table_lines[2:]) if len(table_lines) > 2 else False
    if has_separator and not has_data:
        return True

    return False


# ==========================================
# HEADING-AWARE CHUNKER
# ==========================================

def split_by_headings(text, source):
    raw_chunks = []
    h1_title = ""
    current_h2 = ""
    current_block = []

    def flush(h1, h2, block_lines):
        block_text = clean_text("\n".join(block_lines))
        if len(block_text.strip()) < 80:
            return
        raw_chunks.append({
            "content": block_text,
            "h1": h1,
            "h2": h2,
            "source": source
        })

    for line in text.splitlines():
        if line.startswith("## "):
            flush(h1_title, current_h2, current_block)
            current_h2 = line[3:].strip()
            current_block = [line]
        elif line.startswith("# "):
            flush(h1_title, current_h2, current_block)
            h1_title = line[2:].strip()
            current_h2 = ""
            current_block = [line]
        else:
            current_block.append(line)

    flush(h1_title, current_h2, current_block)

    merged = []
    i = 0
    merge_count = 0
    while i < len(raw_chunks):
        chunk = raw_chunks[i]
        j = i + 1
        while j < len(raw_chunks) and j <= i + 2 and is_incomplete(chunk["content"]):
            next_chunk = raw_chunks[j]
            chunk = {
                "content": chunk["content"] + "\n\n" + next_chunk["content"],
                "h1": chunk["h1"],
                "h2": chunk["h2"],
                "source": chunk["source"]
            }
            merge_count += 1
            j += 1
        merged.append(chunk)
        i = j if j > i + 1 else i + 1

    if merge_count:
        print(f"   🔗 Merged {merge_count} incomplete chunk(s) with successor(s)")

    return merged


# ==========================================
# TABLE-AWARE OVERFLOW SPLITTER
# ==========================================

def split_large_chunk(chunk_dict, max_chars=3000):
    text = chunk_dict["content"]
    if len(text) <= max_chars:
        return [chunk_dict]

    lines = text.splitlines()
    header_lines = []
    data_lines = []
    in_table = False
    header_captured = False

    for line in lines:
        if line.strip().startswith("|"):
            in_table = True
            if not header_captured:
                header_lines.append(line)
                if "---" in line:
                    header_captured = True
            else:
                data_lines.append(line)
        else:
            if in_table:
                data_lines.append(line)
            else:
                header_lines.append(line)

    header_text = "\n".join(header_lines)
    sub_chunks = []
    current_rows = []
    current_len = len(header_text)

    for row in data_lines:
        row_len = len(row) + 1
        if current_len + row_len > max_chars and current_rows:
            sub_text = clean_text(header_text + "\n" + "\n".join(current_rows))
            sub_chunks.append({**chunk_dict, "content": sub_text})
            current_rows = [row]
            current_len = len(header_text) + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        sub_text = clean_text(header_text + "\n" + "\n".join(current_rows))
        sub_chunks.append({**chunk_dict, "content": sub_text})

    return sub_chunks if sub_chunks else [chunk_dict]


# ==========================================
# AI TITLING (UPDATED)
# ==========================================

async def generate_title(chunk_dict, index):
    async with semaphore:
        preview = chunk_dict["content"][:1000]
        h1 = chunk_dict["h1"]
        h2 = chunk_dict["h2"]

        prompt = f"""Tu es un expert en indexation de données RAG pour un guide d'orientation universitaire tunisien (2025).
Analyse le texte ci-dessous et génère un titre et un sous-titre enrichis pour améliorer la recherche vectorielle (sémantique).

Règles strictes :
1. Ligne 1 : Un "# Titre principal" qui résume le sujet global du texte. Ne répète pas bêtement l'ancien titre.
2. Ligne 2 : Un "## Sous-titre" qui liste les informations vitales contenues (ex: dates exactes, codes de filières, noms d'institutions, formules).
3. Réponds UNIQUEMENT avec ces deux lignes formatées en Markdown. Aucun texte supplémentaire, aucune introduction.

Ancienne section : {h1}
Ancienne sous-section : {h2}

Texte à analyser :
{preview}
"""
        try:
            response = await client.chat.completions.create(
                model=NVIDIA_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=150
            )
            # Nettoyage des guillemets accidentels potentiels
            title_block = response.choices[0].message.content.strip().strip('"').strip("'")
            
            # S'assurer que le LLM a bien retourné des hashtags, sinon on les force
            if not title_block.startswith("#"):
                title_block = f"# {title_block}"
                
            print(f"   ✅ [{index}] Titre généré.")
            return title_block
        except Exception as e:
            print(f"   ⚠️  [{index}] Titling failed: {e} — using fallback")
            fallback = f"# {h1}\n## {h2}".strip() if h1 or h2 else f"# {chunk_dict['source']}"
            return fallback


async def generate_all_titles(chunk_dicts):
    print(f"\n🧠 Generating AI enriched titles/subtitles for {len(chunk_dicts)} chunks...")
    tasks = [generate_title(c, i+1) for i, c in enumerate(chunk_dicts)]
    return await asyncio.gather(*tasks)


# ==========================================
# MAIN CHUNKING PIPELINE
# ==========================================

def build_chunks(files):
    all_raw = []

    for source, text in files:
        raw = split_by_headings(text, source)
        print(f"  📑 {source}: {len(raw)} sections after merge")

        for r in raw:
            expanded = split_large_chunk(r, max_chars=3000)
            all_raw.extend(expanded)

    all_raw = [r for r in all_raw if len(r["content"].strip()) > 100]
    print(f"\n✅ Total raw chunks before titling: {len(all_raw)}")
    return all_raw


def assemble_documents(chunk_dicts, ai_title_blocks):
    docs = []
    # Added enumerate to track the sequence index
    for i, (chunk, ai_title_block) in enumerate(zip(chunk_dicts, ai_title_blocks)):
        enriched_content = f"{ai_title_block}\n\n{chunk['content']}"

        docs.append(Document(
            page_content=enriched_content,
            metadata={
                "source": chunk["source"],
                "chunk_index": i,  # <--- CRITICAL: Enables the chat script to stitch chunks
                "h1_original": chunk["h1"],
                "h2_original": chunk["h2"],
            }
        ))
    return docs

# ==========================================
# PREVIEW
# ==========================================

def save_preview(docs, output_file=CHUNK_PREVIEW_FILE):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs, 1):
            f.write("=" * 80 + "\n")
            f.write(f"CHUNK #{i}\n")
            f.write(f"Source: {doc.metadata.get('source', '?')}\n")
            f.write(f"Length: {len(doc.page_content)} chars\n")
            f.write("-" * 80 + "\n")
            f.write(doc.page_content)
            f.write("\n\n")
    print(f"👁  Chunk preview saved → {output_file}")


# ==========================================
# CHROMA
# ==========================================

def build_chroma(docs):
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ NVIDIA_API_KEY missing.")
        sys.exit(1)

    embeddings = NVIDIAEmbeddings(
        model=NVIDIA_EMBED_MODEL,
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="END"
    )

    print(f"\n🔢 Embedding {len(docs)} chunks with {NVIDIA_EMBED_MODEL}...")
    
    # Nettoyage de l'ancienne DB pour éviter la duplication des chunks lors de vos tests
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    print(f"💾 Chroma saved → {CHROMA_DIR}")
    return db


# ==========================================
# ENTRY POINT
# ==========================================

async def main():
    files = load_merged_file()

    print(f"\n🔪 Chunking merged file...")
    chunk_dicts = build_chunks(files)

    ai_title_blocks = await generate_all_titles(chunk_dicts)
    docs = assemble_documents(chunk_dicts, ai_title_blocks)

    save_preview(docs)

    print("\n🏗  Building Chroma vector database...")
    build_chroma(docs)

    print("\n🎉 Done! Vector DB ready.")


if __name__ == "__main__":
    asyncio.run(main())