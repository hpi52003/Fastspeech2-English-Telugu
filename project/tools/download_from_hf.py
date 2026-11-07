"""
Download code bundles and the trained checkpoints from the Hugging Face dataset repo

    export HF_TOKEN=hf_xxx
    python tools/download_from_hf.py
"""
import os, shutil
from huggingface_hub import hf_hub_download

REPO, TOK = "akshaya148/tts-prepared-data", os.environ.get("HF_TOKEN")
get = lambda name: hf_hub_download(REPO, name, repo_type="dataset", token=TOK)

for lang, last in (("English", "30000"), ("Telugu", "15000")):
    shutil.copy(get(f"{lang}_code.zip"), f"{lang}_code.zip")
    os.makedirs(f"{lang}/checkpoints", exist_ok=True)
    for f in ("best.pth.tar", "best.json", f"{last}.pth.tar", "train_loss.csv", "val_loss.csv"):
        shutil.copy(get(f"{lang}_run/{f}"), f"{lang}/checkpoints/{f}")
print("done - now run: python tools/merge_code_zips.py English_code.zip Telugu_code.zip")
