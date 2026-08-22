from typing import Dict, List

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create the recursive text splitter used by the RAG pipeline.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller "
            "than chunk_size"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        length_function=len,
    )


def chunk_pages(
    pages: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Convert page-level documents into citation-aware chunks.

    Each generated chunk retains:
    - document_id
    - document
    - source_path
    - page
    - chunk_id
    - text
    """

    if not pages:
        return []

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = []

    for page in pages:

        text = page.get("text", "").strip()

        if not text:
            continue

        split_texts = splitter.split_text(text)

        for chunk_index, chunk_text in enumerate(
            split_texts,
            start=1,
        ):

            chunk_id = (
                f"{page['document_id']}"
                f"_p{page['page']}"
                f"_c{chunk_index}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": page["document_id"],
                    "document": page["document"],
                    "source_path": page["source_path"],
                    "page": page["page"],
                    "text": chunk_text,
                }
            )

    return chunks