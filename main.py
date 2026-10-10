import math

from image_percent import ImageCapacityCalculator


def header_field_sizes(capacities: list[dict]) -> tuple[int, int]:
    """Smallest bit widths that can hold the header fields.

    seq : enough bits for the highest sequence number (0 .. n-1).
    len : enough bits for the largest chunk length in BYTES any image could
          hold (max_capacity_bits // 8).
    """
    n_images = len(capacities)
    seq_bits = max(1, (n_images - 1).bit_length())

    max_len_bytes = max(c["max_capacity_bits"] for c in capacities) // 8
    len_bits = max(1, max_len_bytes.bit_length())
    return seq_bits, len_bits


def split_text_by_percentage(text: str, capacity_results: list[dict],
                             seq_bits: int, len_bits: int) -> list[dict]:
    """Split `text` into one chunk per image, sized by capacity_percentage.

    Images must already be sorted so the highest-capacity one is LAST: it
    receives whatever is left over after rounding.

    Each payload = [seq no][chunk length in bytes][chunk bytes], as bits.
    """
    n_chars = len(text)
    chunks = []
    cumulative_pct = 0.0
    start = 0

    for i, res in enumerate(capacity_results):
        cumulative_pct += res["capacity_percentage"]
        end = n_chars if i == len(capacity_results) - 1 \
            else round(cumulative_pct / 100.0 * n_chars)

        chunk = text[start:end]
        chunk_bytes = chunk.encode("utf-8")
        chunk_len = len(chunk_bytes)

        header_bits = f"{i:0{seq_bits}b}" + f"{chunk_len:0{len_bits}b}"
        data_bits = "".join(f"{b:08b}" for b in chunk_bytes)
        payload_bits = header_bits + data_bits

        chunks.append({
            "path": res["path"],
            "max_capacity_bits": res["max_capacity_bits"],
            "seq": i,
            "chunk": chunk,
            "chunk_len_bytes": chunk_len,
            "header_bits": header_bits,
            "payload_bits": payload_bits,          # use this for embedding
            "payload_len_bits": len(payload_bits),
            "fits": (len(payload_bits) <= res["max_capacity_bits"]
                     and chunk_len < 2 ** len_bits),
        })
        start = end

    return chunks


def main():
    my_images = [
        "images/high_bike.png",
        "images/low-sky.png",
        "images/mid_house.png",
        "images/mid_door.png",
        "images/high_kayak.png",
    ]

    secret_text = (
        "The Mahabharata is one of the greatest Indian epics, traditionally attributed to the sage Ved Vyasa."
    )
    # secret_text = open("secret.txt", encoding="utf-8").read()

    # 1) Compute capacities
    calculator = ImageCapacityCalculator(floor_fraction=0.03)
    capacities = calculator.calculate_image_capacities(my_images)

    # 2) Sort ascending by capacity -> highest-capacity image is LAST
    capacities.sort(key=lambda c: c["max_capacity_bits"])

    # 3) Header field widths, then split text
    seq_bits, len_bits = header_field_sizes(capacities)
    chunks = split_text_by_percentage(secret_text, capacities, seq_bits, len_bits)

    # 4) Output: image name, max capacity, (seq no + len + chunk) together
    for c in chunks:
        print(f"{c['path']} | max capacity: {c['max_capacity_bits']:,} bits")
        print(f"  [{c['header_bits']}] {c['chunk']}")
        if not c["fits"]:
            print("  WARNING: payload exceeds this image's capacity")
        print()

    # `chunks` holds everything needed for embedding (payload_bits per image).
    return capacities, chunks


if __name__ == "__main__":
    main()