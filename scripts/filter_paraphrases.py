import csv
from pathlib import Path

RAW_PATH = Path("data/ncbi_aug_raw/paraphrases_raw.tsv")
CLEAN_PATH = Path("data/ncbi_aug_clean/paraphrases_clean.tsv")


def all_entities_present(paraphrase: str, entities: list[str]) -> bool:
    paraphrase_lower = paraphrase.lower()
    for ent in entities:
        ent = ent.strip()
        if not ent:
            continue
        if ent.lower() not in paraphrase_lower:
            return False
    return True


def main():
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RAW_PATH.open("r", encoding="utf-8") as f_in, CLEAN_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as f_out:

        reader = csv.DictReader(f_in, delimiter="\t")
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        kept, dropped = 0, 0
        for row in reader:
            entities = row["entities"].split("|||")
            paraphrase = row["paraphrase"]
            if all_entities_present(paraphrase, entities):
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1

    print(f"Done. Kept {kept} paraphrases, dropped {dropped}.")


if __name__ == "__main__":
    main()
