import os
import asyncio
import base64
from io import BytesIO
from pdf2image import convert_from_path, pdfinfo_from_path
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)


NVIDIA_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
POPPLER_BIN_PATH = r"C:\Users\yasser\Documents\test\bac_orientation_assistant\poppler\poppler-26.02.0\Library\bin"

MAX_CONCURRENT_PAGES = 2
semaphore = asyncio.Semaphore(MAX_CONCURRENT_PAGES)

CACHE_DIR = "page_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# PROMPTS
# ==========================================
# ==========================================
# PROMPTS
# ==========================================
# ==========================================
# UPDATED EXTRACTION PROMPTS
# ==========================================

VISION_PROMPT = """You are an expert data extraction assistant for a RAG system.
I am providing you with an image of a single page from a university orientation guide.

Your mission is to transcribe ALL text, headings, and tabular data from this image into clean, structured Markdown.

CRITICAL DIRECTIVE: ZERO CONVERSATIONAL TEXT
You must output ONLY the raw Markdown content. Do NOT include any greetings, introductions (e.g., "Here is the markdown..."), explanations, or concluding notes. If you output anything other than the exact page content formatted in Markdown, you have failed.

FORMATTING RULES:
1. Use # for top-level sections, ## for subsections. Never use * or bullet points for structural headings.
2. No blank lines between rows inside a table.
3. No extra blank lines between sections — one blank line maximum between any two blocks.
4. No conversational introduction. Return ONLY the raw Markdown.

CRITICAL RULES FOR TABLES:
1. Flatten Merged Cells: Repeat merged cell text on every individual row where it logically applies (unless overridden by Specific Case 4).
2. No Data Loss: Every unique combination must have its own row.
3. Formula Preservation: Keep all mathematical formulas and conditions intact.

ABSTRACT EXAMPLE — FLATTENING MERGED CELLS (RIGHT-TO-LEFT / ARABIC):
Visual structure in image (Right-to-Left):
[ Data X ] | [ Sub-Item 1 ] | [ Main Category A ]
[ Data Y ] | [ Sub-Item 2 ] |

Correct Markdown Output:
| Column Header 3 | Column Header 2 | Column Header 1 |
| --- | --- | --- |
| Data X | Sub-Item 1 | Main Category A |
| Data Y | Sub-Item 2 | Main Category A |

SPECIFIC CASE 1 — DUAL-CONTENT CELLS (الشعبة COLUMN):
Some cells visually contain two pieces of data stacked together: a numeric code AND a programme name.
This most commonly appears in the الشعبة (speciality) column where a number (e.g. 10501) and an Arabic name (e.g. الاجازة في الكيمياء) appear in the same cell.
You MUST split these into two separate columns: | الكود | الشعبة |
Never output only the number — always extract the full programme name alongside it.

Example of correct output when this case is detected:
| المجموع | الباكالوريا | الكود | الشعبة |
| --- | --- | --- | --- |
| 159.6500 | BAC GENERAL | 10501 | الاجازة في الكيمياء |
| 172.7300 | BAC GENERAL | 10502 | الاجازة في الفيزياء و الكيمياء |

SPECIFIC CASE 2 — VERTICALLY MERGED CODE+NAME CELLS (الشعبة COLUMN):
Sometimes a code and programme name in الشعبة visually spans multiple rows, each row being a different باكالوريا type (e.g. BAC GENERAL, BAC TECHNOLOGIQUE).
You MUST repeat BOTH the code and the full Arabic programme name on every row it spans.
Never leave الكود or الشعبة empty just because the cell was visually merged in the image.

Example of correct output when this case is detected:
| المجموع | الباكالوريا | الكود | الشعبة |
| --- | --- | --- | --- |
| 187.2800 | BAC GENERAL | 10311 | الاجازة في إعلامية التصرف |
| 150.0000 | BAC TECHNOLOGIQUE | 10311 | الاجازة في إعلامية التصرف |
| 183.7400 | BAC GENERAL | 10312 | الاجازة في العلوم الاقتصادية |
| 135.7800 | BAC TECHNOLOGIQUE | 10312 | الاجازة في العلوم الاقتصادية |
| 182.6200 | BAC GENERAL | 10318 | الاجازة في علوم التصرف |
| 145.8000 | BAC TECHNOLOGIQUE | 10318 | الاجازة في علوم التصرف |

SPECIFIC CASE 3 — THE "الشعبة" HALLUCINATION ERROR:
"الشعبة" is ONLY a column header name. It is NEVER a valid data value inside a row.
If you find yourself writing "الشعبة" inside a data cell (e.g., | 10700 | الشعبة |), STOP. This is a major hallucination error. Look closely at the text inside the actual image row next to that numeric code. You MUST extract the real Arabic specialty name (e.g., "الطب" or "الاجازة في التاريخ") instead of duplicating the header name.

SPECIFIC CASE 4 — FRENCH / LEFT-TO-RIGHT TABLES WITH WIDE MERGES (FILIÈRE & FORMULE):
If the document contains an actual table structured in French where headers like "Filière" (far left) and "Formule du score" (far right) span multiple rows, you MUST restructure the data hierarchically to avoid repetitive columns:
- Extract the "Filière" name as a subsection heading using `## ` or `### ` depending on depth (e.g., `## Licence en Français`).
- Extract the corresponding "Formule du score (T) et Conditions" text and write it as a single standard paragraph or inline block immediately below the heading.
- Build a smaller Markdown table below the formula for the remaining inner columns specifically for that Filière (typically: `| Code | Etablissement | Parcours | Série |`).
- Do NOT include "Filière" or "Formule du score" as columns in this inner table to prevent redundant repetitions.

SPECIFIC CASE 5 — PROSE-HEAVY OR PURE TEXT PAGES (NO TABLES):
If a page consists of standard prose, descriptive text, institutional descriptions, instructions, or bulleted lists *instead* of an actual structured grid data table, do NOT invent or force a table layout.
- Organize the document strictly using clear, hierarchical textual Markdown elements.
- Use # for top-level headings and main document section headers.
- Use ## for logical subsections and subheadings.
- Use standard Markdown paragraphs for body text blocks.
- Use standard Markdown bulleted lists (`-`) or numbered lists only to represent explicit text itemizations or list elements present in the source document image.

ADDITIONAL STRICT OUTPUT CONSTRAINTS:
1. ABSOLUTELY NO NOTES or COMMENTARY: Do not append any notes, summaries, or post-text explanations at the end of the markdown data. Output strictly the pure data content with zero conversational text.
2. EXACT TABLE STRUCTURE: Every Arabic table generated MUST conform exactly to the 4-column structure from left to right: | المجموع | الباكالوريا | الكود | الشعبة |. For French tables, rigidly follow the hierarchical structure specified in Case 4 (Filière heading, followed by formula paragraph, followed by a table of remaining inner columns).
3. REPEAT DATA FOR MULTI-LINE SPLITS: When an Arabic specialty row visually breaks down into multiple lines due to different score (المجموع) and bac (الباكالوريا) values, you MUST duplicate the code and Arabic specialty name fields across all corresponding rows. Do not leave fields empty or merge them.
"""

REFINEMENT_PROMPT = """You are an expert data extraction assistant for a RAG system.
I am providing you with:
1. An image of a single page from a university orientation guide.
2. A previously extracted Markdown draft of that same page.

Your mission is to compare the draft against the image and produce a corrected, complete version.

CRITICAL DIRECTIVE: ZERO CONVERSATIONAL TEXT
You must output ONLY the raw Markdown content. Do NOT include any greetings, introductory text (e.g., "The provided image and draft...", "Here is the corrected..."), or concluding notes (e.g., "Note that we have added..."). 

WHAT TO FIX:
- Destroy Conversational Text: If the draft contains any AI chatter at the beginning or end, DELETE IT COMPLETELY.
- Destroy Hallucinated Tables: If the image shows a simple paragraph and a bulleted list (e.g., a list of high schools), but the draft forced this into a complex table, you MUST delete the table format and convert it back to a clean paragraph and Markdown bulleted list (`- `). Match the visual layout of the image.
- Add any rows, sections, text blocks, or values present in the image but missing from the draft.
- Fix any incorrectly transcribed numbers, codes, words, or formulas.
- Remove hallucinated content not visible in the image.
- Fix any rows where الكود or الشعبة is empty due to a vertically merged cell in the image — repeat the value from the row above onto every row that belongs to the same merged group.
- Eliminate the "الشعبة" Error: Scan your table data rows. If any row contains the literal text "الشعبة" instead of a proper programme descriptor, overwrite it with the true textual specialty name specified in the image.
- Fix French Table Structure: If the draft includes "Filière" and "Formule du score" as repetitive columns, DESTROY that table format. Restructure it hierarchically: extract the Filière as a section/subsection heading, extract the formula as a paragraph immediately below it, and build a slim table for the remaining inner columns (`| Code | Etablissement | Parcours | Série |`).
- Remove Fabricated Tables: If the image consists of regular paragraphs, institutional lists, or instructions without an actual grid table structure, eliminate any artificial tables generated in the draft and rewrite the content using clean paragraph and list layout formatting.

FORMATTING RULES:
1. Use # for top-level sections, ## for subsections. Never use * or bullet points for structural headings.
2. No blank lines between rows inside a table.
3. No extra blank lines between sections — one blank line maximum between any two blocks.
4. No conversational introduction or explanation. Return ONLY the corrected raw Markdown.

CRITICAL RULES FOR TABLES:
1. Flatten Merged Cells: Repeat merged cell text on every individual row where it logically applies (except for French Filière/Formules, which become structural headings).
2. No Data Loss: Every unique combination must have its own row.
3. Formula Preservation: Keep all mathematical formulas and conditions intact.

ABSTRACT EXAMPLE — FLATTENING MERGED CELLS (RIGHT-TO-LEFT / ARABIC):
Visual structure in image (Right-to-Left):
[ Data X ] | [ Sub-Item 1 ] | [ Main Category A ]
[ Data Y ] | [ Sub-Item 2 ] |

Correct Markdown Output:
| Column Header 3 | Column Header 2 | Column Header 1 |
| --- | --- | --- |
| Data X | Sub-Item 1 | Main Category A |
| Data Y | Sub-Item 2 | Main Category A |

SPECIFIC CASE 1 — DUAL-CONTENT CELLS (الشعبة COLUMN):
Some cells visually contain two pieces of data stacked together: a numeric code AND a programme name.
This most commonly appears in the الشعبة (speciality) column where a number (e.g. 10501) and an Arabic name (e.g. الاجازة في الكيمياء) appear in the same cell.
You MUST split these into two separate columns: | الكود | الشعبة |
Never output only the number — always extract the full programme name alongside it.
If the draft contains rows where الشعبة only has a number with no Arabic name, this is the error to fix.

Example of correct output when this case is detected:
| المجموع | الباكالوريا | الكود | الشعبة |
| --- | --- | --- | --- |
| 159.6500 | BAC GENERAL | 10501 | الاجازة في الكيمياء |
| 172.7300 | BAC GENERAL | 10502 | الاجازة في الفيزياء و الكيمياء |

SPECIFIC CASE 2 — VERTICALLY MERGED CODE+NAME CELLS (الشعبة COLUMN):
Sometimes a code and programme name in الشعبة visually spans multiple rows, each row being a different باكالوريا type (e.g. BAC GENERAL, BAC TECHNOLOGIQUE).
You MUST repeat BOTH the code and the full Arabic programme name on every row it spans.
Never leave الكود or الشعبة empty just because the cell was visually merged in the image.

Example of correct output when this case is detected:
| المجموع | الباكالوريا | الكود | الشعبة |
| --- | --- | --- | --- |
| 187.2800 | BAC GENERAL | 10311 | الاجازة في إعلامية التصرف |
| 150.0000 | BAC TECHNOLOGIQUE | 10311 | الاجازة في إعلامية التصرف |
| 183.7400 | BAC GENERAL | 10312 | الاجازة في العلوم الاقتصادية |
| 135.7800 | BAC TECHNOLOGIQUE | 10312 | الاجازة في العلوم الاقتصادية |
| 182.6200 | BAC GENERAL | 10318 | الاجازة في علوم التصرف |
| 145.8000 | BAC TECHNOLOGIQUE | 10318 | الاجازة في علوم التصرف |

SPECIFIC CASE 4 — FRENCH / LEFT-TO-RIGHT TABLES WITH WIDE MERGES (FILIÈRE & FORMULE):
If the document contains an actual table structured in French where headers like "Filière" (far left) and "Formule du score" (far right) span multiple rows, you MUST restructure the data hierarchically to avoid repetitive columns:
- Extract the "Filière" name as a subsection heading using `## ` or `### ` depending on depth (e.g., `## Licence en Français`).
- Extract the corresponding "Formule du score (T) et Conditions" text and write it as a single standard paragraph or inline block immediately below the heading.
- Build a smaller Markdown table below the formula for the remaining inner columns specifically for that Filière (typically: `| Code | Etablissement | Parcours | Série |`).
- Do NOT include "Filière" or "Formule du score" as columns in this inner table to prevent redundant repetitions.

SPECIFIC CASE 5 — PROSE-HEAVY OR PURE TEXT PAGES (NO TABLES):
If a page consists of standard prose, descriptive text, institutional descriptions, instructions, or bulleted lists *instead* of an actual structured grid data table, do NOT invent or force a table layout.
- Organize the document strictly using clear, hierarchical textual Markdown elements.
- Use # for top-level headings and main document section headers.
- Use ## for logical subsections and subheadings.
- Use standard Markdown paragraphs for body text blocks.
- Use standard Markdown bulleted lists (`-`) or numbered lists only to represent explicit text itemizations or list elements present in the source document image.

PREVIOUSLY EXTRACTED DRAFT:
{draft}

Now produce the corrected and complete Markdown for this page.

ADDITIONAL STRICT OUTPUT CONSTRAINTS:
1. ABSOLUTELY NO POST-PROMPT NOTES OR FOOTNOTES: Do not under any circumstances write meta-explanations, notes, summaries of fixes, or text descriptions after or before the data. Output only the pure structured markdown document.
2. ENFORCE THE COLUMN DESIGN: Ensure every single Arabic table strictly preserves this exact layout layout from left to right: | المجموع | الباكالوريا | الكود | الشعبة |. For French tables, rigidly follow the hierarchical structure specified in Case 4 (Filière heading, followed by formula paragraph, followed by a table of remaining inner columns).
3. MANDATORY MULTI-LINE REPETITION: Ensure that any vertically shared information across multiple rows completely duplicates across all lines to maintain full compatibility without row spans (Except as defined in Case 4 for French Tables).
"""



def pdf_page_to_base64(pdf_path, page_num):
    # OPTIMIZED: Increased DPI to 300 for pristine image details on Arabic lettering
    images = convert_from_path(
        pdf_path,
        first_page=page_num,
        last_page=page_num,
        dpi=300,
        poppler_path=POPPLER_BIN_PATH
    )
    if not images:
        raise ValueError(f"Failed to extract page {page_num}")
    buffered = BytesIO()
    # OPTIMIZED: Increased compression quality factor to 90 to protect text sharpness
    images[0].save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def clean_content(content):
    """Strip markdown fences and collapse excessive blank lines."""
    content = content.replace("```markdown", "").replace("```", "").strip()
    # Collapse 3+ consecutive blank lines into 1
    import re
    content = re.sub(r'\n{3,}', '\n\n', content)
    return content

def stitch_split_tables(sorted_pages):
    print("\n🧵 Inspecting pages for split tables...")
    stitched_pages = []
    for i, (page_num, content) in enumerate(sorted_pages):
        lines = content.strip().split('\n')
        if i > 0:
            prev_page_num, prev_content = stitched_pages[-1]
            prev_lines = prev_content.strip().split('\n')
            if prev_lines and prev_lines[-1].strip().endswith('|') and lines and lines[0].strip().startswith('|'):
                print(f"   🔗 Split table detected between Page {prev_page_num} and Page {page_num}. Merging...")
                data_rows_only = []
                table_started = True
                for line in lines:
                    if table_started and line.strip().startswith('|'):
                        if "---" not in line and line != lines[0]:
                            data_rows_only.append(line)
                    else:
                        table_started = False
                        data_rows_only.append(line)
                content = '\n'.join(data_rows_only)
                stitched_pages[-1] = (prev_page_num, prev_content + '\n' + content)
                content = ""
        if content.strip():
            stitched_pages.append((page_num, content))
    return stitched_pages

# ==========================================
# CORE ASYNC LOGIC
# ==========================================

async def extract_page_with_retry(pdf_path, page_num, max_retries=5):
    cache_path = os.path.join(CACHE_DIR, f"page_{page_num}.md")

    async with semaphore:
        base_delay = 4
        base64_image = pdf_page_to_base64(pdf_path, page_num)

        # If cached: refine existing draft against the image
        if os.path.exists(cache_path):
            print(f"🔍 [Page {page_num}] Cache found — sending draft + image for refinement...")
            with open(cache_path, "r", encoding="utf-8") as f:
                existing_draft = f.read()
            prompt_text = REFINEMENT_PROMPT.format(draft=existing_draft)
        else:
            print(f"🆕 [Page {page_num}] No cache — fresh extraction...")
            prompt_text = VISION_PROMPT

        for attempt in range(1, max_retries + 1):
            try:
                print(f"🚀 [Page {page_num}] Processing (Attempt {attempt}/{max_retries})...")
                response = await client.chat.completions.create(
                    model=NVIDIA_VISION_MODEL,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    temperature=0.1
                )
                content = clean_content(response.choices[0].message.content)

                # Overwrite cache with the improved version
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(content)

                print(f"✅ [Page {page_num}] Done.")
                return page_num, content

            except Exception as e:
                print(f"⚠️  [Page {page_num}] Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    print(f"❌ [Page {page_num}] All retries exhausted.")
                    # Fall back to cached content if available, else empty
                    if os.path.exists(cache_path):
                        with open(cache_path, "r", encoding="utf-8") as f:
                            return page_num, f.read()
                    return page_num, ""
                wait_time = base_delay * (2 ** (attempt - 1))
                print(f"⏳ [Page {page_num}] Sleeping {wait_time}s...")
                await asyncio.sleep(wait_time)

# ==========================================
# MAIN
# ==========================================

async def main_async(pdf_path, output_md_path):
    if not os.getenv("NVIDIA_API_KEY"):
        print("❌ NVIDIA_API_KEY missing. Check your .env file.")
        return

    print(f"📄 Analyzing PDF: {pdf_path}")
    pdf_info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_BIN_PATH)
    total_pages = pdf_info["Pages"]
    print(f"📸 Total pages: {total_pages}")

    tasks = [extract_page_with_retry(pdf_path, i) for i in range(1, total_pages + 1)]
    print(f"🔥 Starting (Max Concurrency = {MAX_CONCURRENT_PAGES})...")
    results = await asyncio.gather(*tasks)

    sorted_results = sorted(results, key=lambda x: x[0])
    cleaned_results = stitch_split_tables(sorted_results)

    print("💾 Saving final Markdown...")
    os.makedirs(os.path.dirname(output_md_path) or ".", exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# Document Source: {os.path.basename(pdf_path)}\n\n")
        for page_num, md_content in cleaned_results:
            f.write(f"{md_content}\n\n")

    print(f"\n🎉 Done! Saved to: {output_md_path}")

if __name__ == "__main__":
    INPUT_PDF = "docs/guide_bm_2025.pdf"
    OUTPUT_MARKDOWN = "refined_data/guide_structuré_nvidia.md"
    asyncio.run(main_async(INPUT_PDF, OUTPUT_MARKDOWN))