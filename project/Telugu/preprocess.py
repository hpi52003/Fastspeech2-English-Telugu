"""
preprocess.py (Telugu) - data selection -> alignment -> feature extraction.

Follows the notebooks (fastspeech2-assignment.ipynb, telugu-fastspeech.ipynb, telugu-training.ipynb):
  1. select ~5 h train + ~1 h validation        (scripts/data_selection.py)
  2. Telugu dictionary from telugu_prep.py + MFA aligner trained from scratch (scripts/align_telugu.py)
  3. point the core config at the local folders and run the core preprocess.py
     (mel, pitch with PyWorld, energy, durations from the TextGrids, stats.json, train.txt / val.txt)

Run from this folder:
    python preprocess.py                 # everything
    python preprocess.py --from-hf       # restore the data and alignments I already produced from the Hub
    python preprocess.py --skip-selection --skip-align
"""
import argparse, glob, json, os, shutil, subprocess, sys, zipfile
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
LANG = CFG["language"]
P = {k: os.path.abspath(os.path.join(HERE, v)) for k, v in CFG["paths"].items()}
CORE = os.path.join(HERE, CFG["core_dir"])
CORE_CFG = os.path.join(HERE, CFG["core_config"])
OUT = os.path.join(P["work_root"], "output", LANG)          # ckpt/ and log/ written by the core train.py


def hf_get(name):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(CFG["hf_repo"], name, repo_type="dataset", token=os.environ.get("HF_TOKEN"))


def patch_core_config():
    """Same edit the Kaggle/Colab notebooks did: point the core config at local folders."""
    txt = open(CORE_CFG, encoding="utf-8").read()
    old = CFG["kaggle_paths_in_core_config"]
    txt = txt.replace(old["data"], P["data_dir"]).replace(old["preprocessed"], P["preprocessed_dir"])
    txt = txt.replace("/kaggle/working", P["work_root"])      # checkpoints / logs / any other leftover path
    open(CORE_CFG, "w", encoding="utf-8").write(txt)


def run(cmd, **kw):
    print("$", " ".join(cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed with exit code {r.returncode}")


def restore_from_hub():
    """What the training notebook did: unpack data + alignments from the Hub repo."""
    tmp = os.path.join(P["work_root"], "align_tmp")
    os.makedirs(P["data_dir"], exist_ok=True); os.makedirs(tmp, exist_ok=True)
    zipfile.ZipFile(hf_get(f"{LANG}_data.zip")).extractall(P["data_dir"])        # raw/, manifest.csv
    zipfile.ZipFile(hf_get(f"{LANG}_align.zip")).extractall(tmp)                 # TextGrid/, lab/
    shutil.copytree(f"{tmp}/TextGrid", f"{P['data_dir']}/TextGrid", dirs_exist_ok=True)
    shutil.copytree(f"{tmp}/lab", f"{P['data_dir']}/raw", dirs_exist_ok=True)    # transcripts the aligner used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-hf", action="store_true", help="download prepared data + alignments instead of recomputing")
    ap.add_argument("--skip-selection", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    a = ap.parse_args()
    data = P["data_dir"]
    root = os.path.dirname(HERE)

    if a.from_hf:
        restore_from_hub()
    else:
        if not a.skip_selection:
            run([sys.executable, os.path.join(root, "scripts", "data_selection.py"),
                 "--out", os.path.dirname(data), "--lang", LANG])
        if not a.skip_align:
            run([sys.executable, os.path.join(root, "scripts", "align_telugu.py"), data,
                 os.path.join(CORE, "telugu_prep.py")], cwd=root)

    tgs = glob.glob(f"{data}/TextGrid/*/*.TextGrid"); wavs = glob.glob(f"{data}/raw/*/*.wav")
    print("TextGrids:", len(tgs), "| wavs:", len(wavs), "(I had 2762 and 2762)")

    patch_core_config()
    os.makedirs(P["preprocessed_dir"], exist_ok=True)
    run([sys.executable, "preprocess.py", "config.yaml"], cwd=CORE)
    assert os.path.exists(os.path.join(P["preprocessed_dir"], "train.txt")), "preprocessing failed"
    print("features written to", P["preprocessed_dir"])


if __name__ == "__main__":
    main()
