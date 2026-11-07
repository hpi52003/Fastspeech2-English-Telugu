""".
python scripts/normalize_english_labels.py data/English
"""
import re, sys, glob
import pandas as pd
from num2words import num2words


def words(n, **kw):
    return num2words(n, **kw).replace("-", " ").replace(",", "")


def plural(w):
    return w[:-1] + "ies" if w.endswith("y") else w + "s"


def norm_en(s):
    s = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", lambda m: m.group().replace(",", ""), s)
    s = re.sub(r"\b(\d+)\s*-\s*(\d+)\b", r"\1 to \2", s)
    s = re.sub(r"\b(\d{4})s\b", lambda m: plural(words(int(m.group(1)), to="year")), s)
    s = re.sub(r"\b(\d{1,2}0)s\b", lambda m: plural(words(int(m.group(1)))), s)
    s = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: words(int(m.group(1)), to="ordinal"), s, flags=re.I)
    s = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: words(int(m.group(1))) + " point " +
               " ".join(words(int(c)) for c in m.group(2)), s)
    s = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: words(int(m.group()), to="year"), s)
    s = re.sub(r"\b\d+\b", lambda m: words(int(m.group())), s)
    return re.sub(r"\s+", " ", s).strip()


def spell_rest(s):
    """fallback: any digit still left is read as a word"""
    return re.sub(r"\s+", " ", re.sub(r"\d", lambda m: " " + num2words(int(m.group())) + " ", s)).strip()


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "data/English"
    man = pd.read_csv(f"{root}/manifest.csv")
    flagged = 0
    for r in man.itertuples():
        a = norm_en(str(r.text)); b = a
        if re.search(r"\d", a):
            b = spell_rest(a); flagged += 1
        with open(f"{root}/raw/{r.speaker}/{r.utt}.lab", "w", encoding="utf-8") as f:
            f.write(b)
    labs = glob.glob(f"{root}/raw/*/*.lab")
    left = [p for p in labs if re.search(r"\d", open(p, encoding="utf-8").read())]
    print("lab files:", len(labs), "| fallback used on", flagged, "| still with digits:", len(left))
    assert not left, left[:3]
