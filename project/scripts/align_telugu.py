"""
python scripts/align_telugu.py data/Telugu Telugu/telugu_prep.py
"""
import glob, os, subprocess, sys
import numpy as np
import tgt

D = sys.argv[1] if len(sys.argv) > 1 else "data/Telugu"
PREP = sys.argv[2] if len(sys.argv) > 2 else "Telugu/telugu_prep.py"

# 1) transcripts without punctuation + dictionary
r = subprocess.run([sys.executable, os.path.basename(PREP), D, f"{D}/telugu.dict"], cwd=os.path.dirname(PREP) or ".")
assert r.returncode == 0, "telugu_prep failed"

# 2) train the acoustic model on this corpus and write TextGrids
MM = os.path.abspath("bin/micromamba")
MFA = f"MFA_ROOT_DIR=/tmp/mfa MAMBA_ROOT_PREFIX=/tmp/mm {MM} run -p /tmp/mfaenv mfa"
cmd = (f"{MFA} train --clean -j {os.cpu_count()} --output_directory {D}/TextGrid "
       f"{D}/raw {D}/telugu.dict {D}/telugu_acoustic.zip")
print(cmd)
print("mfa exit code:", subprocess.run(cmd, shell=True).returncode)

# 3) quick quality check of the alignments
tgs = glob.glob(f"{D}/TextGrid/*/*.TextGrid")
spn, maxd = 0, []
for p in tgs:
    iv = tgt.io.read_textgrid(p).get_tier_by_name("phones").intervals
    spn += any(i.text == "spn" for i in iv)
    maxd.append(max(i.end_time - i.start_time for i in iv if i.text not in ("", "sil", "sp", "spn")))
print("TextGrids:", len(tgs), "| files with 'spn':", spn,
      "| longest phone per file: median %.2fs, 99th pct %.2fs, max %.2fs" %
      (np.median(maxd), np.percentile(maxd, 99), max(maxd)),
      "| files with a phone > 1.5 s:", int((np.array(maxd) > 1.5).sum()))
