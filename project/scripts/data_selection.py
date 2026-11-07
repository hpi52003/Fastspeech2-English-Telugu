"""
Data selection, subsampling and train/val split for both languages.

English : ai4bharat/Svarah          (only a "test" split exists on the Hub, we use it as our pool)
Telugu  : SPRINGLab/IndicTTS_Telugu (single voice kept: gender label 1)
"""
import argparse, io, json, math, os, re, unicodedata

import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

SEED, TRAIN_H, VAL_H = 42, 5.0, 1.0   # ~5 h train, ~1 h validation per language


# ----------------------------------------------------------------------------- helpers
def to22k(wav, sr):
    """mono + resample to 22.05 kHz + peak-normalise to 0.95"""
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    g = math.gcd(22050, sr)
    wav = resample_poly(wav, 22050 // g, sr // g)
    return (wav / (np.abs(wav).max() + 1e-9) * 0.95).astype("float32")


def save_split(df, raw, col, lang, out):
    for r in df.itertuples():
        wav, sr = sf.read(io.BytesIO(raw[int(r.idx)][col]["bytes"]), dtype="float32")
        d_ = f"{out}/{lang}/raw/{r.speaker}"
        os.makedirs(d_, exist_ok=True)
        sf.write(f"{d_}/{r.utt}.wav", to22k(wav, sr), 22050, subtype="PCM_16")
        with open(f"{d_}/{r.utt}.lab", "w", encoding="utf-8") as f:
            f.write(r.text)


def group_split(df, rng):
    """Split by *group* so that no group (recording session / utterance) is in both sets.
    Validation groups are drawn first (until ~1 h), then training groups (until ~5 h)."""
    gh = df.groupby("grp")["dur"].sum() / 3600
    order = rng.permutation(np.array(gh.index.tolist(), dtype=object))
    val_g, tr_g, vh, th = [], [], 0.0, 0.0
    for g in order:
        if vh < VAL_H:
            val_g.append(g); vh += gh[g]
        elif th < TRAIN_H:
            tr_g.append(g); th += gh[g]
        else:
            break
    assert th >= TRAIN_H - 0.2, f"only {th:.2f} h available for training"
    return (df[df.grp.isin(tr_g)].assign(split="train"),
            df[df.grp.isin(val_g)].assign(split="val"))


# ----------------------------------------------------------------------------- English
def clean_en(x):
    x = unicodedata.normalize("NFKC", x).replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", x).strip()


def english(out, tok):
    rng = np.random.default_rng(SEED)
    sv = load_dataset("ai4bharat/Svarah", split="test", token=tok)
    raw_sv = sv.cast_column("audio_filepath", Audio(decode=False))

    paths = []
    for ch in sv.data.column("audio_filepath").chunks:
        paths += ch.field("path").to_pylist()
    cols = ["gender", "age-group", "primary_language", "native_place_state",
            "native_place_district", "highest_qualification", "job_category", "occupation_domain"]

    d = pd.DataFrame({
        "idx": range(len(sv)), "dur": sv["duration"], "path": paths,
        "text": [clean_en(x) for x in sv["text"]],
        "prof": ["|".join(map(str, p)) for p in zip(*[sv[c] for c in cols])],
    })
    d["grp"] = d.path.str.split("_").str[0]          # first token of the file name = recording group

    log = {"clips_total": len(d)}
    d = d[(d.dur >= 1.5) & (d.dur <= 15)];                    log["after_length_1.5-15s"] = len(d)
    d = d[d.text.str.fullmatch(r"[A-Za-z0-9 ,.'?!\-]+")];     log["after_charset"] = len(d)
    d = d[d.text.str.split().str.len() >= 3];                 log["after_min_3_words"] = len(d)
    cps = d.text.str.len() / d.dur
    d = d[(cps >= 5) & (cps <= 25)];                          log["after_chars_per_sec_5-25"] = len(d)
    log["pool_hours"] = round(d.dur.sum() / 3600, 2)

    # Svarah has no speaker id. Speaker = unique demographic profile (gender, age group, mother tongue, ...)
    hrs = d.groupby("prof")["dur"].sum().sort_values(ascending=False)
    d["speaker"] = d.prof.map({p: f"spk{i:03d}" for i, p in enumerate(hrs.index)})
    d["utt"] = d.speaker + "_" + d.idx.astype(str).str.zfill(5)

    tr, va = group_split(d, rng)
    va = va[va.speaker.isin(set(tr.speaker))]                         # every val speaker is seen in training
    va = va[~va.text.str.lower().isin(set(tr.text.str.lower()))]      # no identical sentence in both sets
    assert not (set(tr.utt) & set(va.utt)) and not (set(tr.grp) & set(va.grp))

    log.update(train_h=round(tr.dur.sum() / 3600, 2), val_h=round(va.dur.sum() / 3600, 2),
               train_clips=len(tr), val_clips=len(va), speakers=int(tr.speaker.nunique()), seed=SEED)
    print("ENGLISH", log)
    save_split(tr, raw_sv, "audio_filepath", "English", out)
    save_split(va, raw_sv, "audio_filepath", "English", out)
    os.makedirs(f"{out}/English", exist_ok=True)
    pd.concat([tr, va])[["utt", "speaker", "split", "dur", "text", "idx"]].to_csv(f"{out}/English/manifest.csv", index=False)
    json.dump(log, open(f"{out}/English/selection_summary.json", "w"), indent=2)


# ----------------------------------------------------------------------------- Telugu
def clean_te(x):
    x = unicodedata.normalize("NFC", x).replace("\u200c", "").replace("\u200d", "")
    return re.sub(r"\s+", " ", x).strip()


def telugu(out, tok):
    rng = np.random.default_rng(SEED)
    te = load_dataset("SPRINGLab/IndicTTS_Telugu", split="train", token=tok)
    raw_te = te.cast_column("audio", Audio(decode=False))

    durs = np.array([sf.info(io.BytesIO(ex["audio"]["bytes"])).duration for ex in raw_te])
    gt = np.array(te["gender"])

    t = pd.DataFrame({"idx": range(len(te)), "dur": durs, "label": gt,
                      "text": [clean_te(x) for x in te["text"]]})
    logt = {"clips_total": len(t)}
    t = t[t.label == 1];                                      logt["label1_male"] = len(t)   # one voice only
    t = t[(t.dur >= 2) & (t.dur <= 15)];                      logt["after_length_2-15s"] = len(t)
    t = t[t.text.str.fullmatch(r"[\u0C00-\u0C7F .,?!]+")];    logt["after_telugu_charset"] = len(t)
    cps = t.text.str.len() / t.dur
    t = t[(cps >= 5) & (cps <= 30)];                          logt["after_chars_per_sec_5-30"] = len(t)
    t = t.drop_duplicates("text");                            logt["after_dedupe"] = len(t)
    logt["pool_hours"] = round(t.dur.sum() / 3600, 2)

    t["speaker"] = "tel_m"; t["grp"] = t.idx                  # every utterance is its own group
    t["utt"] = "tel_m_" + t.idx.astype(str).str.zfill(5)
    trt, vat = group_split(t, rng)
    assert not (set(trt.utt) & set(vat.utt)) and not (set(trt.text) & set(vat.text))

    logt.update(train_h=round(trt.dur.sum() / 3600, 2), val_h=round(vat.dur.sum() / 3600, 2),
                train_clips=len(trt), val_clips=len(vat), seed=SEED)
    print("TELUGU", logt)
    save_split(trt, raw_te, "audio", "Telugu", out)
    save_split(vat, raw_te, "audio", "Telugu", out)
    os.makedirs(f"{out}/Telugu", exist_ok=True)
    pd.concat([trt, vat])[["utt", "speaker", "split", "dur", "text", "idx"]].to_csv(f"{out}/Telugu/manifest.csv", index=False)
    json.dump(logt, open(f"{out}/Telugu/selection_summary.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--lang", choices=["English", "Telugu", "both"], default="both")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN")
    if a.lang in ("English", "both"):
        english(a.out, tok)
    if a.lang in ("Telugu", "both"):
        telugu(a.out, tok)
    print("done")
