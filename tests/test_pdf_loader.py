from app.ingestion.pdf_loader import load_all_pdfs


def main():

    pages = load_all_pdfs()

    print()
    print("=" * 60)
    print("PDF INGESTION TEST")
    print("=" * 60)

    print(
        f"Total extracted pages: {len(pages)}"
    )

    if pages:

        first_page = pages[0]

        print()
        print("First page metadata:")
        print(
            "Document ID:",
            first_page["document_id"],
        )

        print(
            "Document:",
            first_page["document"],
        )

        print(
            "Page:",
            first_page["page"],
        )

        print(
            "Source:",
            first_page["source_path"],
        )

        print()
        print("Text preview:")
        print(
            first_page["text"][:1000]
        )


if __name__ == "__main__":
    main()