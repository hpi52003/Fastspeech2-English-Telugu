"""
train.py (Telugu) - trains the Telugu FastSpeech 2 model with the core training script.

What the training notebook (telugu-training.ipynb) did, as a script:
  * point the core config at local folders
  * optionally continue an earlier run (checkpoints + loss CSVs) from the Hub
  * run the core train.py with --resume (15000 steps in total, validation every 1,000 steps,
    best checkpoint chosen by validation loss, checkpoints/logs pushed to the Hub when HF_TOKEN is set)
  * copy best.pth.tar, best.json and the loss CSVs next to this script / into Results

Run from this folder (GPU needed):
    python train.py
    python train.py --restore-from-hf     # continue a run stored in the Hub repo
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


def restore_run_from_hub():
    from huggingface_hub import HfApi
    files = [f for f in HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(CFG["hf_repo"], repo_type="dataset")
             if f.startswith(CFG["run_name"] + "/")]
    for f in files:
        name = f.split("/", 1)[1]
        sub = "ckpt" if name.endswith(".pth.tar") or name == "best.json" else "log"
        shutil.copy(hf_get(f), f"{OUT}/{sub}/{name}")
    print("restored", len(files), "files from an earlier run")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore-from-hf", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="start from scratch")
    a = ap.parse_args()
    assert os.path.exists(os.path.join(P["preprocessed_dir"], "train.txt")), "run preprocess.py first"

    patch_core_config()
    os.makedirs(f"{OUT}/ckpt", exist_ok=True); os.makedirs(f"{OUT}/log", exist_ok=True)
    if a.restore_from_hf:
        restore_run_from_hub()

    env = dict(os.environ)
    if env.get("HF_TOKEN"):                                   # sync checkpoints/logs to the Hub while training
        env["HF_UPLOAD_REPO"] = CFG["hf_repo"]
        env["HF_UPLOAD_PREFIX"] = CFG["run_name"]
    run([sys.executable, "-u", "train.py", "-c", "config.yaml"] + ([] if a.no_resume else ["--resume"]), cwd=CORE, env=env)

    ck = os.path.join(HERE, "checkpoints"); os.makedirs(ck, exist_ok=True)
    for f in ("best.pth.tar", "best.json"):
        if os.path.exists(f"{OUT}/ckpt/{f}"):
            shutil.copy(f"{OUT}/ckpt/{f}", ck)
    logs = os.path.join(P["results_dir"], "logs"); os.makedirs(logs, exist_ok=True)
    for f in ("train_loss.csv", "val_loss.csv"):
        for src in (f"{OUT}/log/{f}", f"{OUT}/ckpt/{f}"):
            if os.path.exists(src):
                shutil.copy(src, logs); break
    print("best checkpoint copied to", ck)


if __name__ == "__main__":
    main()
