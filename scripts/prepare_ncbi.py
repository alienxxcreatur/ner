from pathlib import Path
from datasets import load_dataset

# ریشه‌ی پوشه‌های داده
DATA_ROOT = Path("data/ncbi")
AUG_RAW_ROOT = Path("data/ncbi_aug_raw")

# چند جمله‌ی اول train که می‌خواهیم برای تمرین با ChatGPT استفاده کنیم
N_SMALL = 30  # می‌توانی بعداً زیادش کنی


def write_conll(split, out_path, label_names):
    """نوشتن دیتاست به فرمت token + label در هر خط، بین جمله‌ها یک خط خالی."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in split:
            tokens = ex["tokens"]
            tag_ids = ex["ner_tags"]
            tags = [label_names[t] for t in tag_ids]  # مثلا O, B-Disease, I-Disease
            for tok, lab in zip(tokens, tags):
                f.write(f"{tok} {lab}\n")
            f.write("\n")


def extract_entities(tokens, tags):
    """استخراج spanهای disease از روی B/I/O (همه‌ی entityها نوع Disease هستند)."""
    entities = []
    current = []
    for tok, tag in zip(tokens, tags):
        if tag == "O":
            if current:
                entities.append(" ".join(current))
                current = []
        else:
            current.append(tok)
    if current:
        entities.append(" ".join(current))
    return entities


def main():
    # دیتاست NCBI Disease را از HuggingFace می‌گیرد
    ds = load_dataset("ncbi/ncbi_disease")
    label_names = ds["train"].features["ner_tags"].feature.names
    # ['O', 'B-Disease', 'I-Disease']

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    AUG_RAW_ROOT.mkdir(parents=True, exist_ok=True)

    # ۱) نوشتن کل train/dev/test به فرمت CoNLL
    write_conll(ds["train"], DATA_ROOT / "train.txt", label_names)
    write_conll(ds["validation"], DATA_ROOT / "dev.txt", label_names)
    write_conll(ds["test"], DATA_ROOT / "test.txt", label_names)

    # ۲) ساخت نسخه‌ی کوچک برای تمرین با ChatGPT + تمپلیت پارافرایز
    small_path = DATA_ROOT / "train_small.txt"
    template_path = AUG_RAW_ROOT / "paraphrases_template.tsv"

    with small_path.open("w", encoding="utf-8") as f_small, template_path.open(
        "w", encoding="utf-8"
    ) as f_tmp:
        # هدر فایل TSV
        f_tmp.write("id\toriginal_sentence\tentities\tparaphrase\n")

        for idx, ex in enumerate(ds["train"]):
            if idx >= N_SMALL:
                break

            tokens = ex["tokens"]
            tag_ids = ex["ner_tags"]
            tags = [label_names[t] for t in tag_ids]

            # نوشتن همین جمله در train_small.txt به فرم token + label
            for tok, lab in zip(tokens, tags):
                f_small.write(f"{tok} {lab}\n")
            f_small.write("\n")

            # ساخت جمله‌ی اصلی و لیست entityها برای کار با ChatGPT
            sentence = " ".join(tokens)
            entities = extract_entities(tokens, tags)
            ent_str = "|||".join(entities)

            # ستون paraphrase فعلاً خالی می‌ماند
            f_tmp.write(f"{idx}\t{sentence}\t{ent_str}\t\n")

    print("Done.")
    print("Written:")
    print(" - data/ncbi/train.txt, dev.txt, test.txt")
    print(" - data/ncbi/train_small.txt")
    print(" - data/ncbi_aug_raw/paraphrases_template.tsv")


if __name__ == "__main__":
    main()
