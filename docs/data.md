# Data

DEAF audio data is **not shipped in this repo**. You must place the three
task directories under your local `$DEAF_DATA_DIR` before running the
pipeline. The expected layout is:

```
$DEAF_DATA_DIR/
├── Task1/
│   ├── audios_EMIS/                # ESC clips (.wav, 16 kHz)
│   └── text_samples_EMIS.csv       # text/emotion/mention table
├── Task2/
│   ├── noisy_speech/               # BSC clips (.wav)
│   └── Text.xlsx                   # sentence_id -> sentence text
└── Task3/
    ├── SIC_clips/                  # SIC clips (.wav)
    └── SIC.xlsx                    # code -> sentence text
```

`parse_metadata.py` extracts structured fields out of the filenames; the
exact filename grammars are described below.

---

## Task 1 — ESC (Emotion–Semantic Conflict)

Filename pattern (mismatched clips):
```
{row_id}_{text_emotion}_{mention_type}_{speech_emotion}_{speaker_id}_{tts}.wav
```
Example: `01_angry_explicit_angry_0011_STYLE.wav`

Filename pattern (neutral text — no `mention_type`):
```
{row_id}_neutral_{speech_emotion}_{speaker_id}_{tts}.wav
```
Example: `26_neutral_angry_0016_F5TTS.wav`

| Field          | Values                                |
|----------------|---------------------------------------|
| text_emotion   | happy / sad / angry / neutral         |
| mention_type   | explicit / implicit (skipped if neutral) |
| speech_emotion | happy / sad / angry / neutral         |
| tts            | STYLE / COSY / F5TTS / ...            |

`text_samples_EMIS.csv` has columns `happy_explicit`, `happy_implicit`, …,
`neutral`. Each row corresponds to one `row_id`.

## Task 2 — BSC (Background–Sound Conflict)

Filename pattern:
```
{text_env}_{sentence_id}_{bg_env}_{snr}.wav
```
Example: `DKITCHEN_E01_DLIVING_-10.wav` (text describes a kitchen scene; the
real background is a living room; SNR = −10 dB).

Environment codes (see `parse_metadata.py:ENV_TO_CATEGORY`):

| Prefix | Examples            | Category        |
|--------|---------------------|-----------------|
| D…     | DWASHING DKITCHEN   | domestic        |
| N…     | NFIELD NRIVER NPARK | nature          |
| O…     | OOFFICE OHALLWAY    | office          |
| P…     | PSTATION PCAFETER   | public          |
| S…     | STRAFFIC SPSQUARE   | street          |
| T…     | TMETRO TBUS TCAR    | transportation  |

`mismatch_type` is derived from the categories:
- `matched`: text_env == bg_env
- `within`:  same category, different environment
- `cross`:   different category

## Task 3 — SIC (Speaker Identity Conflict)

Filename pattern (note the **double underscore** before the voice block):
```
{sub_dim}_{mention}_{text_id}_{num}__{voice_age}_{voice_gender}.wav
```
Example: `AGE_EX_EL_01__elderly_male.wav`

| Field        | Values                                       |
|--------------|----------------------------------------------|
| sub_dim      | AGE / GDR / CMB / NEU                        |
| mention      | EX (explicit) / IM (implicit) / NT (neutral) |
| text_id      | EL / YG (AGE) · F / M (GDR) · EF / YF / YM / EM (CMB) · NA (NEU) |
| voice_age    | elderly / young                              |
| voice_gender | male / female                                |

`is_matched` rules:
- AGE: `semantic_age == voice_age`
- GDR: `semantic_gender == voice_gender`
- CMB: both must match
- NEU: always True (no semantic identity cue)

CMB clips emit **two** eval rows each (one for age, one for gender) so the
benchmark can score the two attributes independently.

---

## Where to download

The DEAF audio set is not yet a public corpus; until the official release,
contact the authors. Reproducibility material for the rebuttal cycle (notably
the RAVDESS-1056 natural-speech replication) lives in a separate Zenodo
deposit referenced in the paper.

If you only have your own audio and want to run the *pipeline* against it,
hand-write a `data/metadata/<task>_clips.csv` with the same columns
`parse_metadata.py` produces and skip step 1.
