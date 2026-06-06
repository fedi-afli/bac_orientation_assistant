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

        chunks.append(
            Document(
                page_content=content,
                metadata={
                    "source": doc.metadata.get("source", ""),
                    "section": current_section if current_section else "",
                    "subsection": current_subsection if current_subsection else "",
                }
            )
        )

    lines = doc.page_content.split("\n")

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## ") and not stripped.startswith("###"):
            flush()
            current_section = stripped.lstrip("#").strip()
            current_subsection = None
            # include the section title as searchable content
            buffer.append(current_section)

        elif stripped.startswith("####"):
            flush()
            current_subsection = stripped.lstrip("#").strip()
            # include the subsection title as searchable content
            buffer.append(current_subsection)

        elif stripped.startswith("###"):
            flush()
            current_subsection = stripped.lstrip("#").strip()
            buffer.append(current_subsection)

        else:
            if stripped:
                buffer.append(stripped)

    flush()
    return chunks
