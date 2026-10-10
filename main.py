from image_percent import ImageCapacityCalculator


def split_text_by_percentage(text: str, capacity_results: list[dict]) -> list[dict]:
    """Split `text` into one chunk per image, sized by each image's
    capacity_percentage.

    Uses cumulative boundaries (round(cumulative_pct * len(text))) so rounding
    errors never accumulate and the chunks always add up to the full text,
    in order, with nothing lost or duplicated.

    Returns one dict per image: the image's info plus the chunk and its sizes.
    """
    n_chars = len(text)
    chunks = []
    cumulative_pct = 0.0
    start = 0

    for i, res in enumerate(capacity_results):
        cumulative_pct += res["capacity_percentage"]
        # Last image always ends exactly at the end of the text.
        end = n_chars if i == len(capacity_results) - 1 \
            else round(cumulative_pct / 100.0 * n_chars)

        chunk = text[start:end]
        chunk_bits = len(chunk.encode("utf-8")) * 8  # bits needed to embed it

        chunks.append({
            "path": res["path"],
            "capacity_percentage": res["capacity_percentage"],
            "capacity_bits": res["max_capacity_bits"],
            "chunk": chunk,
            "chunk_chars": len(chunk),
            "chunk_bits": chunk_bits,
            "fits": chunk_bits <= res["max_capacity_bits"],
            "start": start,
            "end": end,
        })
        start = end

    return chunks


def print_capacity_table(results: list[dict]) -> None:
    print()
    print(f"{'Image Path':<30} | {'Dims':<12} | {'Pixels':<10} | "
          f"{'Entropy':<9} | {'Raw Cap (bits)':<15} | "
          f"{'Floored Cap (bits)':<19} | {'% of Total':<10}")
    print("-" * 120)
    for r in results:
        dims = f"{r['dimensions'][0]}x{r['dimensions'][1]}"
        print(f"{r['path']:<30} | {dims:<12} | {r['num_pixels']:<10,d} | "
              f"{r['hbar']:<9.4f} | {r['raw_capacity_bits']:<15,d} | "
              f"{r['max_capacity_bits']:<19,d} | "
              f"{r['capacity_percentage']:<10.2f}")


def print_chunks(chunks: list[dict]) -> None:
    print("\n" + "=" * 120)
    print("TEXT CHUNKS PER IMAGE")
    print("=" * 120)
    for i, c in enumerate(chunks, start=1):
        status = "OK" if c["fits"] else "EXCEEDS CAPACITY"
        print(f"\n[Part {i}] {c['path']}")
        print(f"  Capacity share : {c['capacity_percentage']:.2f}% "
              f"({c['capacity_bits']:,} bits)")
        print(f"  Chunk size     : {c['chunk_chars']:,} chars "
              f"= {c['chunk_bits']:,} bits  [{status}]")
        print(f"  Text range     : [{c['start']}:{c['end']}]")
        print(f"  Chunk text     :\n{c['chunk']}")
        print("-" * 120)


def main():
    # Put your image paths here
    my_images = [
        "images/high_bike.png",
        "images/low-sky.png",
        "images/mid_house.png",
        "images/mid_door.png",
        "images/high_kayak.png",
    ]

    # The text you want to hide (replace with your own, or read from a file)
    secret_text = (
        "The Mahabharata is one of the greatest Indian epics, traditionally attributed to the sage Ved Vyasa. It tells the story of the conflict between the Pandavas and the Kauravas, who were cousins fighting for the throne of Hastinapura. The epic describes the struggle between good and evil, the importance of truth, duty, justice, and righteousness. The Kurukshetra War forms the central part of the story, during which Lord Krishna guides Arjuna and explains the teachings of the Bhagavad Gita. The Mahabharata also contains stories of bravery, loyalty, sacrifice, love, and betrayal. It teaches us that greed, jealousy, and pride can lead to destruction, while following dharma (righteousness) is essential for leading a meaningful life."
    )
    # secret_text = open("secret.txt", encoding="utf-8").read()

    # 1) Compute capacities (all values come back as a list of dicts)
    calculator = ImageCapacityCalculator(floor_fraction=0.03)
    capacities = calculator.calculate_image_capacities(my_images)
    print_capacity_table(capacities)

    # 2) Split the text according to each image's percentage
    chunks = split_text_by_percentage(secret_text, capacities)
    print_chunks(chunks)

    # 3) Sanity checks
    total_text_bits = len(secret_text.encode("utf-8")) * 8
    total_capacity_bits = sum(c["max_capacity_bits"] for c in capacities)
    print(f"\nTotal text      : {len(secret_text):,} chars ({total_text_bits:,} bits)")
    print(f"Total capacity  : {total_capacity_bits:,} bits")
    if total_text_bits > total_capacity_bits:
        print("WARNING: text is larger than the total capacity of all images.")
    if "".join(c["chunk"] for c in chunks) != secret_text:
        print("WARNING: chunks do not reassemble into the original text.")

    # `capacities` and `chunks` are ready for the embedding step later.
    return capacities, chunks


if __name__ == "__main__":
    main()
