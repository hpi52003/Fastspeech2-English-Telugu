# FastSpeech 2 TTS for English and Telugu

Two separate FastSpeech 2 models: English (trained on Svarah) and Telugu (trained on IndicTTS Telugu), about 5 h of training audio and 1 h of validation audio each. This is my submission for the FastSpeech 2 assignment (deadline 07-11-2026).

The full process is in "Technical_Report.pdf".

## What is in this repository

```
.
├── English/
│   ├── preprocess.py   train.py   inference.py   config.yaml
│   ├── fastspeech2/            core FastSpeech 2 code for English (from English_code.zip)
│   └── checkpoints/            best.pth.tar = step 12,000
├── Telugu/
│   ├── preprocess.py   train.py   inference.py   config.yaml
│   ├── fastspeech2/            core code for Telugu incl. telugu_prep.py (from Telugu_code.zip)
│   └── checkpoints/            best.pth.tar = step 15,000
├── Results/
│   ├── English/  training_loss.png  validation_loss.png  test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
│   └── Telugu/   (same layout)
├── Technical_Report.pdf
├── scripts/    data selection + forced alignment (shared by both languages)
├── tools/      helpers to fetch / unpack the code bundles and checkpoints from Hugging Face
├── requirements.txt
└── README.md
```

`preprocess.py`, `train.py` and `inference.py` in each language folder are the pipeline from my Kaggle/Colab notebooks as scripts; `config.yaml` holds the run settings (paths, run name, steps, inference options). The model and feature settings are in `fastspeech2/config.yaml`.

`Results/<Language>/logs/` has the raw loss CSVs and `variance_adaptor_plots/data/` has the numbers behind every variance-adaptor plot.

## Test sentences

| Language | Files in `test_generated_audios/` |
|---|---|
| English | `E1`, `E2a`, `E2b`, `E3`, `E4` (+ `E3_fixed`) |
| Telugu  | `T1`, `T2a`, `T2b`, `T3`, `T4` (+ `T3_joined`) |

Two extra files are not part of the required five. `E3` is synthesised from the text exactly as written in the assignment (`rhyth m`, with the stray space) and `E3_fixed` is the same sentence with the space removed. In the Telugu list `T3` is the text as printed in the PDF (`అడ విలో`) and `T3_joined` is the same with the word joined (`అడవిలో`). I kept both to show what the broken input does.

## Reproducing the results

You need a GPU for training, Python 3, `pip install -r requirements.txt`, and Montreal Forced Aligner (conda-forge, see the comment at the end of `requirements.txt`).

```bash
python tools/merge_code_zips.py English_code.zip Telugu_code.zip   # once, puts the core code in <Lang>/fastspeech2/
export HF_TOKEN=hf_xxx                                              # for the datasets / Hugging Face repo

cd English            # then the same inside Telugu/
python preprocess.py  # data selection -> alignment -> features   (python preprocess.py --from-hf reuses my prepared data)
python train.py       # 30,000 steps (Telugu: 15,000), resumes automatically
python inference.py   # test sentences + variance-adaptor plots -> ../Results/English
```

The scripts rewrite the Kaggle paths (`/kaggle/working/...`) in `fastspeech2/config.yaml` to local folders, using the paths in each `config.yaml`. Data selection uses seed 42.

## Checkpoints

English_run/best.pth.tar (step 12,000) and Telugu_run/best.pth.tar (step 15,000), about 142 MB each, are stored with Git LFS and are also publicly available on Hugging Face (English_run/best.pth.tar and Telugu_run/best.pth.tar): https://huggingface.co/datasets/akshaya148/tts-prepared-data/tree/main

## Notes

* English has 115 speaker ids, but Svarah has no speaker column. A speaker here is a unique demographic profile, so it is an approximation.
* Test audio uses the best-validation checkpoint (English step 12,000, Telugu step 15,000), English speaker id 0.
* Hours after preprocessing: English 4.65 h train / 0.92 h val, Telugu 4.27 h train / 0.86 h val.
