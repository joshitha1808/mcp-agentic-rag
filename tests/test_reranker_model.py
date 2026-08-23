from sentence_transformers import CrossEncoder


def main():

    print("=" * 60)
    print("CROSS ENCODER MODEL TEST")
    print("=" * 60)

    model_name = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    print()
    print("Loading model:")
    print(model_name)

    model = CrossEncoder(
        model_name
    )

    pairs = [
        (
            "What are the risks to global economic growth?",
            "Geopolitical tensions and armed conflicts "
            "can create downside risks to global growth."
        ),
        (
            "What are the risks to global economic growth?",
            "The report discusses agricultural productivity "
            "and demographic changes."
        ),
    ]

    print()
    print("Generating relevance scores...")

    scores = model.predict(
        pairs
    )

    for i, score in enumerate(
        scores,
        start=1,
    ):

        print(
            f"Pair {i} score: "
            f"{float(score):.4f}"
        )

    print()
    print("=" * 60)
    print("CROSS ENCODER TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()