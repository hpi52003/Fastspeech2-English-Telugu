"""
Unpack the two FastSpeech 2 code bundles into English/fastspeech2/ and Telugu/fastspeech2/
 python tools/merge_code_zips.py English_code.zip Telugu_code.zip
"""
import sys, zipfile

if len(sys.argv) != 3:
    sys.exit(__doc__)
for z, lang in zip(sys.argv[1:3], ("English", "Telugu")):
    zipfile.ZipFile(z).extractall(f"{lang}/fastspeech2")
    print("extracted", z, "->", f"{lang}/fastspeech2/")
