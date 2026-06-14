from pathlib import Path

import pandas as pd


DATA_FOLDER = Path("data")
RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "chunk_size_plan.csv"

CHUNK_CONFIGS = [
    (500, 100),
    (600, 100),
]


def count_chunks(text_length, chunk_size, overlap):
    step = chunk_size - overlap
    if text_length <= 0:
        return 1
    return max(1, (text_length + step - 1) // step)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    files = list(DATA_FOLDER.glob("*.txt"))
    rows = []

    for chunk_size, overlap in CHUNK_CONFIGS:
        total_chunks = 0
        for path in files:
            text = path.read_text(encoding="utf-8", errors="ignore")
            total_chunks += count_chunks(len(text), chunk_size, overlap)

        rows.append(
            {
                "Data Folder": DATA_FOLDER.as_posix(),
                "Documents": len(files),
                "Chunk Size": chunk_size,
                "Chunk Overlap": overlap,
                "Estimated Chunks": total_chunks,
                "Academic Target Met": 15 <= len(files) <= 30 and 500 <= total_chunks <= 1500,
                "Run Command": (
                    f'$env:CHUNK_SIZE="{chunk_size}"; '
                    f'$env:CHUNK_OVERLAP="{overlap}"; '
                    f'$env:RESULTS_FOLDER="results_chunk_{chunk_size}"; '
                    "python main.py"
                ),
            }
        )

    plan = pd.DataFrame(rows)
    plan.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_CSV}")
    print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
