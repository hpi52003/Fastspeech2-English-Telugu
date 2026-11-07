"""
inference.py (Telugu) 
Run from this folder:
    python inference.py
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


def main():
    inf = CFG["inference"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, inf["checkpoint"]))
    ap.add_argument("--sentences", default=os.path.join(HERE, inf["sentences"]))
    ap.add_argument("--extra", default=os.path.join(HERE, inf["extra_sentences"]))
    ap.add_argument("--out", default=P["results_dir"])
    ap.add_argument("--speaker_id", default=str(inf["speaker_id"]))
    ap.add_argument("--device", default=inf["device"])
    a = ap.parse_args()

    if not os.path.exists(a.ckpt):
        sys.exit(f"checkpoint not found: {a.ckpt} (see tools/download_from_hf.py)")
    if not os.path.exists(os.path.join(P["preprocessed_dir"], "stats.json")):
        sys.exit("stats.json / speakers.json missing - run preprocess.py first (pitch/energy statistics are needed)")

    patch_core_config()
    os.makedirs(a.out, exist_ok=True)
    run([sys.executable, "inference.py", "-c", "config.yaml", "--ckpt", os.path.abspath(a.ckpt),
         "--sentences", os.path.abspath(a.sentences), "--extra", os.path.abspath(a.extra),
         "--out", a.out, "--preprocessed_dir", P["preprocessed_dir"],
         "--speaker_id", a.speaker_id, "--device", a.device], cwd=CORE)

    best = os.path.join(os.path.dirname(a.ckpt), "best.json")
    if os.path.exists(best):
        shutil.copy(best, os.path.join(a.out, "best_checkpoint.json"))
    print("results written to", a.out)


if __name__ == "__main__":
    main()
