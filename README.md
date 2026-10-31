# TrustGuard AI — Fake Job / Task-Scam Detector

TrustGuard AI is a Streamlit app that screens job postings and recruiter messages for signs of fraud. It combines a trained XGBoost classifier (with TF-IDF text features and SHAP-based explanations) with a separate rule-based keyword layer and a text-consistency "offer intelligence" check (emails, phone numbers, domains, payment language). Job postings can be submitted as pasted text or as an uploaded screenshot/photo, which is read with Tesseract OCR.

> ⚠️ This is a hackathon prototype, not a certified fraud-detection service. It is a screening aid only — it cannot independently verify an employer, a recruiter's identity, or whether a vacancy exists. Always verify offers independently before paying any fee or sharing personal information.

## Features

- **ML scam classification** — XGBoost model trained on job-posting data, using TF-IDF text vectorization, categorical encoding, and scaled ordinal/numerical features.
- **Explainable AI** — Local TreeSHAP contributions show which features pushed a prediction toward or away from "scam," both by feature group and by individual term.
- **Rule-based warning flags** — Keyword checks for deposit/fee requests, urgency language, platform redirects (Telegram/WhatsApp), guaranteed-income claims, and requests for sensitive information.
- **Offer intelligence** — Extracts emails, URLs, phone numbers, LinkedIn links, and money mentions from the posting, and compares them against reference details (official domain, phone, recruiter name) you optionally supply.
- **Image OCR input** — Upload a screenshot or photo of a job posting; text is extracted with Tesseract and is editable before analysis.

## Tech stack

| Component | Technology |
|---|---|
| App framework | Streamlit |
| ML model | XGBoost (loaded from `xgb_model.ubj`) |
| Preprocessing | scikit-learn (TF-IDF, encoder, scaler) — bundled in `other_artifacts.joblib` |
| Explainability | XGBoost native TreeSHAP (`pred_contribs`) |
| OCR | Tesseract OCR via `pytesseract`, with `Pillow` for image preprocessing |
| Data handling | pandas, NumPy, SciPy (sparse matrices) |

## Project structure

```
.
├── app.py                  # Streamlit app (UI, inference, SHAP explanations, offer checks)
├── xgb_model.ubj            # Trained XGBoost model (UBJSON format)
├── other_artifacts.joblib   # Bundled preprocessing artifacts (vectorizers, encoder, scaler, mappings)
└── requirements.txt          # Python dependencies
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (required for screenshot/photo input)

Tesseract is a separate system application — `pytesseract` is only a Python wrapper around it.

- **Windows:** Install from the [UB Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki). If it's not on your `PATH`, set the `TESSERACT_CMD` environment variable to the full path of `tesseract.exe`.
- **macOS:** `brew install tesseract`
- **Linux (Debian/Ubuntu):** `sudo apt-get install tesseract-ocr`

### 3. Keep model files alongside `app.py`

Make sure these three files are in the same folder:
- `app.py`
- `xgb_model.ubj`
- `other_artifacts.joblib`

### 4. Run the app

```bash
streamlit run app.py
```

## How it works

1. **Input** — Paste a job posting/message, or upload a screenshot/photo (extracted with OCR into an editable text box).
2. **Feature extraction** — The text is passed through the trained TF-IDF vectorizers, categorical encoder, and ordinal/numerical feature pipeline to rebuild the exact feature vector used during training.
3. **Prediction** — The XGBoost model outputs a scam/not-scam verdict and probability.
4. **Explanation** — TreeSHAP contributions are computed for the single prediction and grouped by feature type (text terms, categorical, structured signals).
5. **Rule-based flags** — The raw text is independently scanned for known scam-indicator phrases.
6. **Offer intelligence** — Emails, URLs, phone numbers, and payment-related language are extracted and optionally cross-checked against reference company details you provide.

## Limitations

- Does not browse the web, query company registries, or validate domain ownership/recruiter identity.
- Model predictions and rule-based flags are decision support, not proof of fraud or legitimacy.
- OCR output can misread names, numbers, and symbols — always review extracted text before analysis.
- SHAP values explain model behavior for this specific input; they do not establish real-world causation.

## Reporting cyber fraud (India)

Suspected cyber fraud can be reported at [cybercrime.gov.in](https://cybercrime.gov.in) or by calling **1930**.

## AI and Open-Source Attribution

In line with our hackathon's AI and Open-Source Attribution Policy, all open-source libraries, pre-trained components, and AI tools used in building this project are listed below.

### Open-source libraries

| Library | Version | Licence | Link | Used for |
|---|---|---|---|---|
| Streamlit | >=1.37.0 | Apache-2.0 | https://github.com/streamlit/streamlit | Web app UI |
| scikit-learn | 1.6.1 | BSD-3-Clause | https://github.com/scikit-learn/scikit-learn | Preprocessing (TF-IDF, encoding, scaling) |
| pandas | >=2.0.0 | BSD-3-Clause | https://github.com/pandas-dev/pandas | Data handling |
| NumPy | >=1.26.0 | BSD-3-Clause | https://github.com/numpy/numpy | Numerical operations |
| SciPy | >=1.11.0 | BSD-3-Clause | https://github.com/scipy/scipy | Sparse matrix operations |
| joblib | >=1.3.0 | BSD-3-Clause | https://github.com/joblib/joblib | Loading preprocessing artifacts |
| XGBoost | >=2.0.0 | Apache-2.0 | https://github.com/dmlc/xgboost | Classification model + TreeSHAP explanations |
| pytesseract | — | Apache-2.0 | https://github.com/madmaze/pytesseract | Python wrapper for Tesseract OCR |
| Tesseract OCR | — | Apache-2.0 | https://github.com/tesseract-ocr/tesseract | OCR engine used to read uploaded images |
| Pillow | — | HPND (Historical Permission Notice and Disclaimer) | https://github.com/python-pillow/Pillow | Image preprocessing (grayscale, contrast) |

### Pre-trained models / datasets

- `xgb_model.ubj` — XGBoost classifier trained by the team on job-posting data. *(Add your training dataset's name, source, and licence here, e.g. the dataset used for the ~18K+ postings referenced in-app.)*
- `other_artifacts.joblib` — Bundled preprocessing objects (TF-IDF vectorizers, categorical encoder, ordinal scaler, feature mappings) fitted by the team on the same training data.

### AI models used inside the product

- None at inference time — the app does not call any external AI model (e.g. Claude, GPT) while running; all predictions come from the locally loaded XGBoost model.

### AI tools used to build the project

*(Fill in with a one-line note on which parts each tool helped with, per the policy format — e.g.:)*
- `[AI tool], [provider] — via [how accessed]: helped with [what it did]; reviewed and edited by the team.`

## Team

*(Add team name / members here.)*
