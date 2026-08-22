from pathlib import Path
from typing import Dict, List

import pymupdf


DOCUMENTS_DIR = Path("data/documents")


def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving meaningful content.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    lines = []

    for line in text.split("\n"):
        cleaned_line = " ".join(line.split())

        if cleaned_line:
            lines.append(cleaned_line)

    # Reconstruct text
    cleaned_text = "\n".join(lines)

    return cleaned_text.strip()


def create_document_id(pdf_path: Path) -> str:
    """
    Create a stable document ID from the filename.
    """

    return pdf_path.stem.lower().replace(" ", "_")


def extract_pdf_pages(
    pdf_path: Path,
) -> List[Dict]:
    """
    Extract text from a PDF page-by-page.

    Each page becomes a structured record containing:
    - document ID
    - document name
    - source path
    - page number
    - extracted text
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file: {pdf_path}"
        )

    document_id = create_document_id(pdf_path)

    pages = []

    with pymupdf.open(pdf_path) as pdf:

        if len(pdf) == 0:
            raise ValueError(
                f"PDF contains no pages: {pdf_path.name}"
            )

        for page_index, page in enumerate(
            pdf,
            start=1,
        ):

            raw_text = page.get_text(
                "text"
            )

            text = clean_text(raw_text)

            # Skip completely empty pages
            if not text:
                continue

            pages.append(
                {
                    "document_id": document_id,
                    "document": pdf_path.name,
                    "source_path": str(pdf_path),
                    "page": page_index,
                    "text": text,
                }
            )

    return pages


def load_all_pdfs(
    documents_dir: Path = DOCUMENTS_DIR,
) -> List[Dict]:
    """
    Load every PDF from the documents directory.
    """

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: "
            f"{documents_dir}"
        )

    pdf_files = sorted(
        documents_dir.glob("*.pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: "
            f"{documents_dir}"
        )

    all_pages = []

    for pdf_path in pdf_files:

        print(
            f"Processing: {pdf_path.name}"
        )

        pages = extract_pdf_pages(
            pdf_path
        )

        print(
            f"  Extracted pages: {len(pages)}"
        )

        all_pages.extend(pages)

    return all_pages
def get_pdf_statistics(
    documents_dir: Path = DOCUMENTS_DIR,
) -> List[Dict]:
    """
    Return basic statistics for every PDF.
    """

    pdf_files = sorted(
        documents_dir.glob("*.pdf")
    )

    statistics = []

    for pdf_path in pdf_files:

        with fitz.open(pdf_path) as pdf:

            statistics.append(
                {
                    "document": pdf_path.name,
                    "pages": len(pdf),
                    "size_mb": round(
                        pdf_path.stat().st_size
                        / (1024 * 1024),
                        2,
                    ),
                }
            )

    return statistics