"""
Unpack the two FastSpeech 2 code bundles into English/fastspeech2/ and Telugu/fastspeech2/.

The core code (model, core train/preprocess/inference scripts, core config.yaml, telugu_prep.py ...) is in
English_code.zip and Telugu_code.zip in the Hugging Face repo `akshaya148/tts-prepared-data`.
English/ and Telugu/ contain my pipeline scripts (preprocess.py, train.py, inference.py, config.yaml) which call
that core code. Download the zips (Files tab, or tools/download_from_hf.py) and run:

    python tools/merge_code_zips.py English_code.zip Telugu_code.zip
"""
import sys, zipfile

if len(sys.argv) != 3:
    sys.exit(__doc__)
for z, lang in zip(sys.argv[1:3], ("English", "Telugu")):
    zipfile.ZipFile(z).extractall(f"{lang}/fastspeech2")
    print("extracted", z, "->", f"{lang}/fastspeech2/")
