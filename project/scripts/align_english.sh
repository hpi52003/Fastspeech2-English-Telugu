#!/usr/bin/env bash
# English forced alignment with Montreal Forced Aligner (pretrained english_us_arpa).
# Steps taken from the notebook: normalise numbers -> find OOV words and G2P them -> align.
# Usage: bash scripts/align_english.sh data/English
set -euo pipefail
D=${1:-data/English}
export MFA_ROOT_DIR=${MFA_ROOT_DIR:-/tmp/mfa}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/tmp/mm}

# 0) one-off: get micromamba and install MFA into /tmp/mfaenv
if [ ! -x bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
fi
[ -d /tmp/mfaenv ] || ./bin/micromamba create -y -p /tmp/mfaenv -c conda-forge montreal-forced-aligner
MFA="./bin/micromamba run -p /tmp/mfaenv mfa"

# 1) spell out numbers in the .lab files
python scripts/normalize_english_labels.py "$D"

# 2) pretrained models + pronunciations for words that are not in the dictionary
$MFA model download acoustic   english_us_arpa
$MFA model download dictionary english_us_arpa
$MFA model download g2p        english_us_arpa
$MFA g2p "$D/raw" english_us_arpa "$D/oov_g2p.txt" --dictionary_path english_us_arpa
$MFA model add_words english_us_arpa "$D/oov_g2p.txt"

# 3) align -> TextGrid/<speaker>/*.TextGrid
$MFA align "$D/raw" english_us_arpa english_us_arpa "$D/TextGrid" \
     -j "$(nproc)" --clean --beam 100 --retry_beam 400
find "$D/TextGrid" -name "*.TextGrid" | wc -l
