from typing import List
from langchain_core.documents import Document

def md_to_meta_chunks(doc: Document) -> List[Document]:
    chunks = []

    current_section = None
    current_subsection = None
    buffer = []

    def flush():
        nonlocal buffer, current_section, current_subsection, chunks

        if not buffer:
            return

        content = "\n".join(buffer).strip()
        buffer = []

        header_lines = [f"section: {current_section}"]

        if current_subsection:
            header_lines.append(f"subsection: {current_subsection}")

        header = "\n".join(header_lines)

        full_text = f"""{header}
---
{content}
"""

        chunks.append(
            Document(
                page_content=full_text,
                metadata={
                    "section": current_section if current_section else "",
                    "subsection": current_subsection if current_subsection else ""
                }
            )
        )

    # read from Document object instead of file
    lines = doc.page_content.split("\n")

    for line in lines:
        line = line.strip()

        # SECTION ##
        if line.startswith("## ") and not line.startswith("####"):
            flush()
            current_section = line.replace("##", "").strip()
            current_subsection = None

        # SUBSECTION ####
        elif line.startswith("####"):
            flush()
            current_subsection = line.replace("####", "").strip()

        # CONTENT
        else:
            if line:
                buffer.append(line)

    flush()
    return chunks