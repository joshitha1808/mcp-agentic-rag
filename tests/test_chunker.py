from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.chunker import chunk_pages


def main():

    print("=" * 60)
    print("CHUNKING TEST")
    print("=" * 60)

    # Step 1: Extract pages
    pages = load_all_pdfs()

    print()
    print(f"Total pages: {len(pages)}")

    # Step 2: Create chunks
    chunks = chunk_pages(pages)

    print(
        f"Total chunks: {len(chunks)}"
    )

    if not chunks:
        print("No chunks generated.")
        return

    # Show first chunk
    first_chunk = chunks[0]

    print()
    print("First chunk metadata:")
    print(
        "Chunk ID:",
        first_chunk["chunk_id"],
    )

    print(
        "Document ID:",
        first_chunk["document_id"],
    )

    print(
        "Document:",
        first_chunk["document"],
    )

    print(
        "Page:",
        first_chunk["page"],
    )

    print(
        "Source:",
        first_chunk["source_path"],
    )

    print()
    print("Chunk length:")
    print(len(first_chunk["text"]))

    print()
    print("Chunk text:")
    print(first_chunk["text"][:1000])


if __name__ == "__main__":
    main()