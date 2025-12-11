from pathlib import Path
import csv

DATA_ROOT = Path("data/ncbi")
AUG_CLEAN_PATH = Path("data/ncbi_aug_clean/paraphrases_clean.tsv")
OUT_MERGE_PATH = DATA_ROOT / "train_merge.txt"

# این همان few-shot اولیه است که prepare_ncbi ساخته بود
BASE_TRAIN_SMALL = DATA_ROOT / "train_small.txt"

DISEASE_LABEL_B = "B-Disease"
DISEASE_LABEL_I = "I-Disease"


def label_paraphrase(sentence: str, entities: list[str]):
    """
    جمله پارافرایز شده را token می‌کند و برای هر توکن یک لیبل می‌سازد.
    اگر توکن‌ها متعلق به یک entity باشند → B-Disease, I-Disease
    در غیر این صورت → O
    """
    tokens = sentence.split()
    labels = ["O"] * len(tokens)

    lower_tokens = [t.lower() for t in tokens]

    for ent in entities:
        ent = ent.strip()
        if not ent:
            continue
        ent_tokens = ent.split()
        lower_ent_tokens = [t.lower() for t in ent_tokens]

        # جستجوی ساب‌سکانس ent_tokens در tokens
        found = False
        for i in range(len(tokens) - len(ent_tokens) + 1):
            window = lower_tokens[i : i + len(ent_tokens)]
            if window == lower_ent_tokens:
                # برچسب‌گذاری
                labels[i] = DISEASE_LABEL_B
                for j in range(1, len(ent_tokens)):
                    labels[i + j] = DISEASE_LABEL_I
                found = True
                break

        if not found:
            # اگر entity پیدا نشد فعلاً نادیده می‌گیریم (باید خیلی کم باشد چون قبلاً فیلتر کردیم)
            print(f"[WARN] entity not aligned in paraphrase: '{ent}' | '{sentence}'")

    return tokens, labels


def main():
    # فایل نهایی را باز می‌کنیم و ابتدا few-shot اصلی را وارد می‌کنیم
    OUT_MERGE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_MERGE_PATH.open("w", encoding="utf-8") as f_out:
        # ۱) ابتدا همان train_small.txt (few-shot اصلی)
        with BASE_TRAIN_SMALL.open("r", encoding="utf-8") as f_base:
            for line in f_base:
                f_out.write(line)

        # ۲) حالا جملات پارافرایز شده تمیز را اضافه می‌کنیم
        with AUG_CLEAN_PATH.open("r", encoding="utf-8") as f_clean:
            reader = csv.DictReader(f_clean, delimiter="\t")
            for row in reader:
                sentence = row["paraphrase"].strip()
                if not sentence:
                    continue
                entities = row["entities"].split("|||")

                tokens, labels = label_paraphrase(sentence, entities)

                for tok, lab in zip(tokens, labels):
                    f_out.write(f"{tok} {lab}\n")
                f_out.write("\n")

    print("Done. Written merged train file to:", OUT_MERGE_PATH)


if __name__ == "__main__":
    main()
