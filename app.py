"""
TrustGuard AI — Fake Job / Task-Scam Detector
Run: streamlit run app.py

Keep these files in the same folder:
- app.py
- xgb_model.ubj
- other_artifacts.joblib
"""

import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix


def extract_text_from_image(uploaded_image):
    """Extract English text from an uploaded job-posting image using Tesseract."""
    from PIL import Image, ImageOps, ImageEnhance
    import pytesseract

    image = Image.open(uploaded_image).convert("RGB")
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image)
    image = ImageEnhance.Contrast(image).enhance(1.5)

    # Optional override for Windows installations outside the default PATH.
    tesseract_cmd = os.environ.get("TESSERACT_CMD")
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    return pytesseract.image_to_string(
        image, lang="eng", config="--psm 6"
    ).strip()



# ---------------------------------------------------------------------
# 1. Page configuration — must be called only once, before UI commands
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="TrustGuard AI | Job Scam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------
# 2. Visual styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --tg-bg: #0b1220;
        --tg-panel: #111c2e;
        --tg-panel-2: #162338;
        --tg-border: #26364d;
        --tg-text: #edf4ff;
        --tg-muted: #9aabc2;
        --tg-blue: #4ea5ff;
        --tg-cyan: #42d3e8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(38, 110, 190, .18), transparent 34%),
            radial-gradient(ellipse at 92% 12%, rgba(25, 181, 200, .10), transparent 28%),
            var(--tg-bg);
        color: var(--tg-text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 18, 32, .82);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--tg-text) !important;
        letter-spacing: -0.5px;
    }

    p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
        color: var(--tg-text);
    }

    [data-testid="stCaptionContainer"] p,
    .muted {
        color: var(--tg-muted) !important;
    }

    .tg-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 4px 0 22px 0;
        border-bottom: 1px solid rgba(154, 171, 194, .16);
        margin-bottom: 34px;
    }

    .tg-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        color: #f5f9ff;
        font-family: 'Manrope', sans-serif;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: .3px;
    }

    .tg-brand-icon {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 13px;
        background: linear-gradient(145deg, #1c5eaa, #15324f);
        border: 1px solid rgba(105, 184, 255, .35);
        font-size: 22px;
    }

    .tg-status {
        color: #83e6bd;
        background: rgba(34, 197, 128, .10);
        border: 1px solid rgba(34, 197, 128, .25);
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .tg-hero {
        text-align: center;
        padding: 8px 0 30px 0;
    }

    .tg-eyebrow {
        display: inline-block;
        color: #8bc7ff;
        background: rgba(54, 143, 224, .12);
        border: 1px solid rgba(78, 165, 255, .25);
        border-radius: 999px;
        padding: 7px 13px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.4px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .tg-hero h1 {
        font-size: clamp(34px, 5vw, 54px) !important;
        line-height: 1.12;
        font-weight: 800 !important;
        margin: 0 0 13px 0;
    }

    .tg-gradient {
        background: linear-gradient(90deg, #77baff, #50d7e8);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .tg-hero p {
        max-width: 720px;
        margin: 0 auto;
        color: var(--tg-muted);
        font-size: 16px;
        line-height: 1.7;
    }

    .tg-section-kicker {
        color: #7dbdff;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .tg-panel {
        background: linear-gradient(145deg, rgba(22, 35, 56, .96), rgba(15, 27, 44, .96));
        border: 1px solid var(--tg-border);
        border-radius: 18px;
        padding: 23px;
        box-shadow: 0 16px 40px rgba(0, 0, 0, .16);
        height: 100%;
    }

    .tg-panel h3 {
        margin-top: 0;
        font-size: 20px;
    }

    .tg-feature {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        padding: 13px 0;
        border-bottom: 1px solid rgba(154, 171, 194, .12);
    }

    .tg-feature:last-child {
        border-bottom: none;
    }

    .tg-feature-icon {
        width: 34px;
        height: 34px;
        flex: 0 0 34px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: rgba(78, 165, 255, .12);
        color: #8bc7ff;
    }

    .tg-feature-title {
        color: #eaf2ff;
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 3px;
    }

    .tg-feature-desc {
        color: var(--tg-muted);
        font-size: 12px;
        line-height: 1.5;
    }

    /* Streamlit input fields */
    .stTextArea textarea {
        background: #0d1829 !important;
        color: #edf4ff !important;
        border: 1px solid #344761 !important;
        border-radius: 12px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #4ea5ff !important;
        box-shadow: 0 0 0 1px #4ea5ff !important;
    }

    div[data-baseweb="select"] > div {
        background: #0d1829 !important;
        border: 1px solid #344761 !important;
        border-radius: 10px !important;
        color: #edf4ff !important;
    }

    .stButton > button {
        background: linear-gradient(100deg, #2878df, #16a6c4) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 11px !important;
        min-height: 49px;
        font-weight: 700 !important;
        letter-spacing: .2px;
        transition: transform .18s ease, box-shadow .18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(40, 120, 223, .28);
    }

    [data-testid="stMetric"] {
        background: rgba(13, 24, 41, .72);
        border: 1px solid var(--tg-border);
        border-radius: 13px;
        padding: 15px 17px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--tg-muted) !important;
    }

    [data-testid="stMetricValue"] {
        color: #f4f8ff !important;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2878df, #16c4c4);
    }

    div[data-testid="stExpander"] {
        background: rgba(17, 28, 46, .72);
        border: 1px solid var(--tg-border);
        border-radius: 12px;
    }

    .tg-result {
        border-radius: 16px;
        padding: 22px;
        border: 1px solid;
        margin: 12px 0 18px 0;
    }

    .tg-result-danger {
        background: rgba(127, 29, 29, .17);
        border-color: rgba(248, 113, 113, .42);
    }

    .tg-result-safe {
        background: rgba(6, 95, 70, .17);
        border-color: rgba(52, 211, 153, .38);
    }

    .tg-result-title {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .tg-result-subtitle {
        color: #b6c4d8;
        font-size: 13px;
    }

    .tg-flag {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: rgba(13, 24, 41, .78);
        border: 1px solid rgba(248, 113, 113, .22);
        border-radius: 11px;
        padding: 13px 14px;
        margin: 8px 0;
        color: #f2e8e8;
        font-size: 13px;
    }

    .tg-footer {
        border-top: 1px solid rgba(154, 171, 194, .16);
        margin-top: 42px;
        padding-top: 20px;
        text-align: center;
        color: #8193aa;
        font-size: 12px;
        line-height: 1.7;
    }

    @media (max-width: 700px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .tg-topbar {
            margin-bottom: 24px;
        }
        .tg-hero {
            padding-bottom: 22px;
        }
        .tg-panel {
            padding: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 3. Load model artifacts
# ---------------------------------------------------------------------
MODEL_PATH = "xgb_model.ubj"
OTHER_ARTIFACTS_PATH = "other_artifacts.joblib"


@st.cache_resource
def load_artifacts():
    missing = [
        path for path in (MODEL_PATH, OTHER_ARTIFACTS_PATH)
        if not os.path.exists(path)
    ]

    if missing:
        st.error(
            f"Missing file(s): {', '.join(missing)}. "
            "Place both model files in the same folder as app.py."
        )
        st.stop()

    # Load XGBoost from its native UBJSON format.
    from xgboost import XGBClassifier

    loaded_model = XGBClassifier()
    loaded_model.load_model(MODEL_PATH)

    other = joblib.load(OTHER_ARTIFACTS_PATH)
    other["model"] = loaded_model
    return other


artifacts = load_artifacts()

model = artifacts["model"]
tfidf_vectorizers = artifacts["tfidf_vectorizers"]
text_features = artifacts["text_features"]
encoder = artifacts["encoder"]
nominal_features = artifacts["nominal_features"]
ordinal_scaler = artifacts["ordinal_scaler"]
experience_mapping = artifacts["experience_mapping"]
numerical_features = artifacts["numerical_features"]


# ---------------------------------------------------------------------
# 4. Rule-based explainability
# ---------------------------------------------------------------------
RULE_PATTERNS = {
    "Deposit / registration fee requested": [
        "registration fee", "deposit", "pay to start", "security deposit",
        "processing fee", "refundable fee",
    ],
    "Urgency / pressure language": [
        "urgent", "limited time", "act now", "hurry", "immediate joining",
        "apply immediately",
    ],
    "Platform redirect (Telegram/WhatsApp)": [
        "telegram", "whatsapp only", "contact us on whatsapp", "join our group",
    ],
    "Guaranteed / unrealistic income claims": [
        "guaranteed income", "guaranteed earning", "earn up to",
        "no experience needed", "work from home and earn",
    ],
    "Requests sensitive info upfront": [
        "bank details", "otp", "aadhar number", "pan card number",
        "credit card number",
    ],
}


RULE_EXPLANATIONS = {
    "Deposit / registration fee requested":
        "The message mentions a payment or deposit before work begins. Legitimate employers generally should not require candidates to pay to get a job.",
    "Urgency / pressure language":
        "Pressure tactics can discourage candidates from verifying the employer or reading the offer carefully.",
    "Platform redirect (Telegram/WhatsApp)":
        "The posting redirects communication to a messaging platform. Verify the recruiter and company independently.",
    "Guaranteed / unrealistic income claims":
        "Promises of guaranteed or unusually easy earnings may be used to attract applicants. Check whether the pay matches the work.",
    "Requests sensitive info upfront":
        "Requests for OTPs, bank details or identity information can expose you to financial or identity theft.",
}


def get_rule_flags(text: str):
    text_lower = text.lower()
    flags = []

    for label, keywords in RULE_PATTERNS.items():
        if any(keyword in text_lower for keyword in keywords):
            flags.append(label)

    return flags


# ---------------------------------------------------------------------
# 5. Extract structured signals from pasted text
# ---------------------------------------------------------------------
def extract_structured_fields(text: str):
    text_lower = text.lower()

    telecommuting = int(
        any(word in text_lower for word in
            ["work from home", "remote", "wfh", "from home"])
    )

    has_salary_range = int(bool(
        re.search(
            r"(₹|rs\.?|inr)\s?[\d,]+|salary|per month|per day|per week",
            text_lower,
        )
    ))

    has_benefits = int(
        any(word in text_lower for word in
            ["benefits", "perks", "insurance", "bonus", "incentive"])
    )

    # These fields cannot reliably be inferred from pasted text alone.
    has_company_logo = 0
    has_questions = 0

    return {
        "telecommuting": telecommuting,
        "has_company_logo": has_company_logo,
        "has_questions": has_questions,
        "has_salary_range": has_salary_range,
        "has_benefits": has_benefits,
    }


# ---------------------------------------------------------------------
# 6. Build the same feature vector used by the trained model
# ---------------------------------------------------------------------
def build_feature_vector(description_text: str, required_experience_choice: str):
    # The same pasted text is supplied to each trained text vectorizer,
    # matching the approach in the original app.
    text_values = {column: description_text for column in text_features}

    tfidf_blocks = []
    for column in text_features:
        vector = tfidf_vectorizers[column].transform([text_values[column]])
        tfidf_blocks.append(vector)

    X_text = hstack(tfidf_blocks)

    # Use the trained "Unknown" category for fields not provided by text.
    nominal_row = {column: "Unknown" for column in nominal_features}
    nominal_df = pd.DataFrame([nominal_row])[nominal_features]
    X_nominal = encoder.transform(nominal_df)

    # Ordinal experience feature.
    exp_encoded = experience_mapping.get(required_experience_choice, -1)
    X_ordinal = ordinal_scaler.transform(np.array([[exp_encoded]]))

    # Numeric / boolean signals.
    structured = extract_structured_fields(description_text)
    X_num = np.array([
        [structured[column] for column in numerical_features]
    ])

    X_final = hstack([
        X_text,
        X_nominal,
        csr_matrix(X_ordinal),
        csr_matrix(X_num),
    ]).tocsr()

    return X_final, structured


# ---------------------------------------------------------------------
# 7. Prediction helper
# ---------------------------------------------------------------------
def get_prediction_and_probability(X):
    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else -1
        scam_probability = float(probabilities[positive_index])
    else:
        score = float(model.decision_function(X)[0])
        scam_probability = float(1 / (1 + np.exp(-score)))

    return int(prediction), float(np.clip(scam_probability, 0, 1))


# ---------------------------------------------------------------------
# 7B. Local XGBoost explanation using native TreeSHAP contributions
# ---------------------------------------------------------------------
def build_model_feature_names():
    """Recreate feature names in exactly the same order as build_feature_vector."""
    names = []
    groups = []

    # TF-IDF blocks: one block per text field, in text_features order.
    for column in text_features:
        terms = tfidf_vectorizers[column].get_feature_names_out()
        names.extend([f"{column}: {term}" for term in terms])
        groups.extend([f"Text: {column}" for _ in terms])

    # One-hot encoded nominal columns.
    try:
        nominal_names = encoder.get_feature_names_out(nominal_features)
    except TypeError:
        nominal_names = encoder.get_feature_names_out()
    names.extend([str(name) for name in nominal_names])
    groups.extend(["Categorical features"] * len(nominal_names))

    # One ordinal feature followed by numeric features.
    names.append("required_experience")
    groups.append("Experience")

    names.extend([str(name) for name in numerical_features])
    groups.extend(["Structured signals"] * len(numerical_features))

    # Validate reconstructed feature labels against the uploaded joblib metadata.
    expected_count = artifacts.get("feature_count")
    if expected_count is not None and len(names) != int(expected_count):
        raise ValueError(
            "Preprocessing artifact feature-count mismatch: "
            f"reconstructed {len(names)} features, but joblib records "
            f"{expected_count}. Refusing to display potentially mislabeled SHAP values."
        )

    return names, groups


def explain_xgboost_prediction(X, top_n=12):
    """Return local SHAP contributions in raw-margin units for class 1 (scam)."""
    import xgboost as xgb

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X)
    contributions = booster.predict(dmatrix, pred_contribs=True)
    contributions = np.asarray(contributions)

    # For binary classification, expected shape is (1, n_features + 1).
    # The final column is the expected/base value.
    if contributions.ndim == 3:
        contributions = contributions[0, 0, :]
    else:
        contributions = contributions.reshape(-1, contributions.shape[-1])[0]

    feature_values = contributions[:-1]
    base_value = float(contributions[-1])
    feature_names, feature_groups = build_model_feature_names()

    # Guard against an artifact/preprocessing mismatch rather than mislabeling features.
    if len(feature_names) != len(feature_values):
        raise ValueError(
            "Feature-name count does not match XGBoost input feature count: "
            f"{len(feature_names)} names vs {len(feature_values)} contributions. "
            "Check that the saved vectorizers/encoder match the trained model."
        )

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Feature group": feature_groups,
        "SHAP contribution": feature_values,
    })
    explanation = explanation[explanation["SHAP contribution"].abs() > 1e-8].copy()
    explanation["Direction"] = np.where(
        explanation["SHAP contribution"] > 0,
        "Pushes toward scam",
        "Pushes away from scam",
    )
    explanation["Absolute impact"] = explanation["SHAP contribution"].abs()
    explanation = explanation.sort_values("Absolute impact", ascending=False)

    group_summary = (
        explanation.groupby("Feature group", as_index=False)["SHAP contribution"]
        .sum()
    )
    group_summary["Absolute impact"] = group_summary["SHAP contribution"].abs()
    group_summary = group_summary.sort_values("Absolute impact", ascending=False)

    # Aggregate identical text terms across the five duplicated TF-IDF slots.
    text_rows = explanation[
        explanation["Feature group"].str.startswith("Text:")
    ].copy()
    if not text_rows.empty:
        text_rows["Term"] = text_rows["Feature"].str.split(": ", n=1).str[-1]
        text_term_summary = (
            text_rows.groupby("Term", as_index=False)["SHAP contribution"].sum()
        )
        text_term_summary["Absolute impact"] = (
            text_term_summary["SHAP contribution"].abs()
        )
        text_term_summary["Direction"] = np.where(
            text_term_summary["SHAP contribution"] > 0,
            "Pushes toward scam",
            "Pushes away from scam",
        )
        text_term_summary = text_term_summary.sort_values(
            "Absolute impact", ascending=False
        )
    else:
        text_term_summary = pd.DataFrame(
            columns=["Term", "SHAP contribution", "Absolute impact", "Direction"]
        )

    return (
        base_value,
        explanation.head(top_n),
        group_summary,
        explanation,
        text_term_summary,
    )




# ---------------------------------------------------------------------
# 7C. Job-offer investigation signals (separate from the ML prediction)
# ---------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\w)")
LINKEDIN_RE = re.compile(r"https?://(?:www\.)?linkedin\.com/(?:in|company)/[^\s<>()]+", re.I)
MONEY_RE = re.compile(
    r"(₹|rs\.?|inr)\s?([\d,]+(?:\.\d{1,2})?)|"
    r"([\d,]+(?:\.\d{1,2})?)\s?(?:rupees|rs\.?|inr)",
    re.I,
)

def _clean_domain(value):
    value = value.lower().strip()
    value = re.sub(r"^https?://", "", value)
    value = re.sub(r"^www\.", "", value)
    return value.split("/")[0].split(":")[0]

def _company_tokens(name):
    ignored = {"private", "limited", "ltd", "pvt", "inc", "llp", "company", "co", "technologies"}
    return {
        token for token in re.findall(r"[a-z0-9]+", (name or "").lower())
        if len(token) > 2 and token not in ignored
    }

def investigate_offer(
    text,
    claimed_company="",
    recruiter_name="",
    official_website="",
    official_email_domain="",
    official_phone="",
):
    """
    Extract visible details and compare them with reference details supplied by
    the user. This is a consistency check, not an independent identity check.
    """
    emails = sorted(set(EMAIL_RE.findall(text)))
    urls = sorted(set(u.rstrip(".,;:!?)]}") for u in URL_RE.findall(text)))
    phones = sorted(set(PHONE_RE.findall(text)))
    linkedin_profiles = sorted(set(
        u.rstrip(".,;:!?)]}") for u in LINKEDIN_RE.findall(text)
    ))

    money_mentions = []
    for match in MONEY_RE.finditer(text):
        raw = match.group(0).strip()
        amount_text = match.group(2) or match.group(3) or ""
        try:
            amount = float(amount_text.replace(",", ""))
        except ValueError:
            amount = None
        start, end = max(0, match.start() - 65), min(len(text), match.end() + 65)
        context = text[start:end].replace("\n", " ")
        money_mentions.append({"mention": raw, "amount": amount, "context": context})

    payment_terms = (
        "registration fee", "security deposit", "training fee", "processing fee",
        "pay to start", "payment required", "refundable deposit", "joining fee",
        "application fee", "verification fee", "equipment fee",
    )
    lower = text.lower()
    payment_contexts = []
    for term in payment_terms:
        pos = lower.find(term)
        if pos >= 0:
            payment_contexts.append(
                text[max(0, pos-65):min(len(text), pos+len(term)+90)].replace("\n", " ")
            )

    email_domains = sorted({
        _clean_domain(email.split("@", 1)[1]) for email in emails
    })
    url_domains = sorted({_clean_domain(url) for url in urls})

    company_tokens = _company_tokens(claimed_company)
    domain_mismatch = []
    if company_tokens:
        for domain in email_domains:
            domain_tokens = set(re.findall(r"[a-z0-9]+", domain.split(".")[0]))
            if domain_tokens and not (company_tokens & domain_tokens):
                domain_mismatch.append(domain)

    official_domain = _clean_domain(official_email_domain or official_website)
    reference_checks = []
    if official_domain:
        matching = [d for d in email_domains if d == official_domain or d.endswith("." + official_domain)]
        if matching:
            reference_checks.append({
                "status": "Match",
                "title": "Email domain matches the reference domain",
                "detail": f"Found {', '.join(matching)}. This only confirms a text-level domain match; it does not prove control of the domain or recruiter identity.",
            })
        elif email_domains:
            reference_checks.append({
                "status": "Mismatch",
                "title": "Email domain differs from the reference domain",
                "detail": f"Reference domain: {official_domain}. Offer domain(s): {', '.join(email_domains)}. A legitimate third-party recruiter may use another domain, so investigate before concluding.",
            })
        else:
            reference_checks.append({
                "status": "Unknown",
                "title": "No email found to compare with the reference domain",
                "detail": f"Reference domain entered: {official_domain}.",
            })

    reference_phone = re.sub(r"\D", "", official_phone or "")
    if reference_phone:
        normalized_offer_phones = [re.sub(r"\D", "", p) for p in phones]
        reference_tail = reference_phone[-10:]
        if any(p[-10:] == reference_tail for p in normalized_offer_phones if len(p) >= 10):
            reference_checks.append({
                "status": "Match",
                "title": "Phone number matches the supplied reference",
                "detail": "A matching number appears in the offer text. This does not establish who controls the number.",
            })
        elif phones:
            reference_checks.append({
                "status": "Mismatch",
                "title": "Phone number differs from the supplied reference",
                "detail": "The offer contains phone number(s) that do not match the reference number entered.",
            })
        else:
            reference_checks.append({
                "status": "Unknown",
                "title": "No phone number found to compare",
                "detail": "A reference phone number was supplied, but no phone number was extracted from the offer.",
            })

    recruiter_present = bool(recruiter_name.strip() and recruiter_name.lower() in lower)
    if recruiter_name.strip():
        reference_checks.append({
            "status": "Match" if recruiter_present else "Not found",
            "title": "Recruiter name text check",
            "detail": (
                "The supplied recruiter name appears in the offer text. This does not verify the person's identity."
                if recruiter_present else
                "The supplied recruiter name was not found in the offer text."
            ),
        })

    findings = []
    if payment_contexts:
        findings.append({
            "level": "High attention",
            "title": "Payment-related language detected",
            "detail": "The text contains a fee/deposit phrase. Review whether money is requested before employment.",
        })
    if domain_mismatch:
        findings.append({
            "level": "Needs verification",
            "title": "Recruiter email may not match the claimed company name",
            "detail": "The email domain does not visibly contain the company-name keywords. This is a heuristic mismatch, not proof of impersonation; third-party recruiters may use different domains.",
        })
    if not emails and not urls:
        findings.append({
            "level": "Unknown",
            "title": "No email address or website found",
            "detail": "The offer does not expose a contact domain that this prototype can compare.",
        })
    if not claimed_company.strip():
        findings.append({
            "level": "Unknown",
            "title": "Company identity not supplied for comparison",
            "detail": "Enter the company name claimed in the offer to run a basic domain-name consistency check.",
        })

    return {
        "emails": emails,
        "email_domains": email_domains,
        "websites": urls,
        "website_domains": url_domains,
        "phone_numbers": phones,
        "linkedin_profiles": linkedin_profiles,
        "money_mentions": money_mentions,
        "payment_contexts": payment_contexts,
        "domain_mismatch": domain_mismatch,
        "reference_domain": official_domain,
        "reference_checks": reference_checks,
        "findings": findings,
    }


# ---------------------------------------------------------------------
# 8. Header and hero
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class="tg-topbar">
        <div class="tg-brand">
            <div class="tg-brand-icon">🛡️</div>
            <div>TRUSTGUARD <span style="color:#62b8ff;">AI</span></div>
        </div>
        <div class="tg-status">● DETECTION SYSTEM</div>
    </div>

    <div class="tg-hero">
        <div class="tg-eyebrow">AI-powered job safety screening</div>
        <h1>Detect scams before<br><span class="tg-gradient">they catch you.</span></h1>
        <p>
            Screen job postings, recruiter messages and task-based earning offers
            for suspicious patterns using machine learning and rule-based checks.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# Small trust / capability indicators.
stat1, stat2, stat3 = st.columns(3)
with stat1:
    st.metric("Training dataset", "18K+ postings")
with stat2:
    st.metric("Detection engine", "ML + Investigation")
with stat3:
    st.metric("Input options", "Text + Image OCR")


st.write("")


# ---------------------------------------------------------------------
# 9. Input form and information panel
# ---------------------------------------------------------------------
input_col, info_col = st.columns([1.75, 1], gap="large")

with input_col:
    st.markdown('<div class="tg-section-kicker">01 / Job screening</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Analyze a job posting")
        st.caption(
            "Paste the complete job description or recruiter message below."
        )

        # OCR text is stored in session state and remains editable.
        if "description_text" not in st.session_state:
            st.session_state["description_text"] = ""

        input_method = st.radio(
            "How would you like to add the job posting?",
            ["Paste text", "Upload screenshot / photo"],
            horizontal=True,
            key="input_method",
        )

        if input_method == "Upload screenshot / photo":
            uploaded_image = st.file_uploader(
                "Upload a screenshot or photo of the job posting",
                type=["png", "jpg", "jpeg", "webp"],
                help="Use a clear, high-resolution image for better text extraction.",
                key="job_image_upload",
            )

            if uploaded_image is not None:
                st.image(
                    uploaded_image,
                    caption="Uploaded job-posting image",
                    use_container_width=True,
                )
                if st.button("Extract text from image", use_container_width=True):
                    try:
                        with st.spinner("Reading text from the image..."):
                            extracted_text = extract_text_from_image(uploaded_image)

                        if extracted_text:
                            st.session_state["description_text"] = extracted_text
                            st.success(
                                "Text extracted. Review and edit it below before analysis."
                            )
                        else:
                            st.warning(
                                "No readable text found. Try a clearer or higher-resolution image."
                            )
                    except ModuleNotFoundError as error:
                        st.error(
                            "OCR packages are missing. Run: pip install pytesseract Pillow"
                        )
                    except Exception as error:
                        if type(error).__name__ == "TesseractNotFoundError":
                            st.error(
                                "Tesseract OCR application was not found. Install Tesseract "
                                "and restart Streamlit. If it is not on PATH, set the "
                                "TESSERACT_CMD environment variable to the full path "
                                "of tesseract.exe."
                            )
                        else:
                            st.error(f"Could not extract text from this image: {error}")

            st.caption(
                "Extracted text appears below. Please check OCR spelling and numbers."
            )
        else:
            st.caption("Paste the complete job description or recruiter message below.")

        with st.form("job_analysis_form"):
            description_text = st.text_area(
                "Job posting or message (editable)",
                height=235,
                placeholder=(
                    "Paste the job posting here...\\n\\n"
                    "Or upload a screenshot above and extract its text.\\n\\n"
                    "Example: Earn ₹5000/day working from home! "
                    "No experience needed. Pay a registration fee..."
                ),
                label_visibility="visible",
                key="description_text",
            )

            claimed_company = st.text_input(
                "Company name claimed in the offer (optional)",
                placeholder="e.g., ABC Technologies",
                help="Used for text-level consistency checks only; it does not independently verify the company.",
            )

            st.markdown("**Optional reference details for comparison**")
            recruiter_name = st.text_input(
                "Recruiter name (optional)",
                placeholder="e.g., Rahul Sharma",
                help="Checks whether this name appears in the submitted text; it does not verify the person's identity.",
            )
            official_website = st.text_input(
                "Company's official website/domain (optional)",
                placeholder="e.g., https://www.example.com",
                help="Enter a domain you independently obtained from a trusted source. TrustGuard compares it with the email domain in the offer.",
            )
            official_email_domain = st.text_input(
                "Known official email domain (optional)",
                placeholder="e.g., example.com",
                help="Optional; takes priority over the domain parsed from the official website.",
            )
            official_phone = st.text_input(
                "Known official phone number (optional)",
                placeholder="e.g., +91 9876543210",
                help="Compared with phone numbers extracted from the offer. A match does not prove ownership.",
            )

            experience_options = ["Not specified"] + list(experience_mapping.keys())
            experience_selection = st.selectbox(
                "Required experience (optional)",
                experience_options,
            )

            submitted = st.form_submit_button(
                "🔎  Analyze job posting",
                use_container_width=True,
            )

with info_col:
    st.markdown('<div class="tg-section-kicker">02 / What we inspect</div>',
                unsafe_allow_html=True)

    st.markdown(
        """
        <div class="tg-panel">
            <h3>Screen suspicious signals</h3>

            <div class="tg-feature">
                <div class="tg-feature-icon">📝</div>
                <div>
                    <div class="tg-feature-title">Job description patterns</div>
                    <div class="tg-feature-desc">Textual signals learned by the machine-learning model.</div>
                </div>
            </div>

            <div class="tg-feature">
                <div class="tg-feature-icon">💸</div>
                <div>
                    <div class="tg-feature-title">Payment requests</div>
                    <div class="tg-feature-desc">Registration fees, deposits and suspicious payment language.</div>
                </div>
            </div>

            <div class="tg-feature">
                <div class="tg-feature-icon">📲</div>
                <div>
                    <div class="tg-feature-title">Communication redirects</div>
                    <div class="tg-feature-desc">References to Telegram, WhatsApp and external groups.</div>
                </div>
            </div>

            <div class="tg-feature">
                <div class="tg-feature-icon">🔐</div>
                <div>
                    <div class="tg-feature-title">Sensitive information</div>
                    <div class="tg-feature-desc">Requests for OTPs, bank details or identity information.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This is a screening aid. It cannot independently verify an employer "
        "or guarantee that a posting is safe."
    )


# ---------------------------------------------------------------------
# 10. Run analysis and display the result dashboard
# ---------------------------------------------------------------------
if submitted:
    if not description_text.strip():
        st.warning("Please paste a job posting or message before analyzing.")
    else:
        selected_experience = (
            "Unknown"
            if experience_selection == "Not specified"
            else experience_selection
        )

        try:
            with st.spinner("Scanning the posting for suspicious signals..."):
                X_final, structured = build_feature_vector(
                    description_text,
                    selected_experience,
                )
                prediction, scam_probability = get_prediction_and_probability(X_final)
                rule_flags = get_rule_flags(description_text)
                investigation = investigate_offer(
                    description_text,
                    claimed_company,
                    recruiter_name,
                    official_website,
                    official_email_domain,
                    official_phone,
                )
                (
                    base_value,
                    top_explanations,
                    group_summary,
                    all_explanations,
                    text_term_summary,
                ) = explain_xgboost_prediction(X_final)

            st.write("")
            st.markdown("---")
            st.markdown('<div class="tg-section-kicker">03 / Analysis report</div>',
                        unsafe_allow_html=True)

            result_col, probability_col = st.columns([1.35, 1], gap="large")

            if prediction == 1:
                verdict = "Potential scam detected"
                verdict_detail = "The model classified this posting as fraudulent."
                result_class = "tg-result-danger"
                verdict_icon = "🚨"
            else:
                verdict = "No scam detected by the model"
                verdict_detail = "The model classified this posting as legitimate."
                result_class = "tg-result-safe"
                verdict_icon = "🛡️"

            with result_col:
                st.markdown(
                    f"""
                    <div class="tg-result {result_class}">
                        <div style="font-size:12px; letter-spacing:1.2px;
                                    text-transform:uppercase; color:#aebed2;
                                    font-weight:700;">Model verdict</div>
                        <div class="tg-result-title">{verdict_icon} {verdict}</div>
                        <div class="tg-result-subtitle">{verdict_detail}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with probability_col:
                st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
                st.metric("Model-estimated scam probability", f"{scam_probability:.1%}")
                st.progress(scam_probability)
                st.caption(
                    "This is the model's estimated probability for the scam class, "
                    "not a guarantee or independently verified risk score."
                )

            # -----------------------------------------------------------------
            # Local model explanation (SHAP / TreeSHAP)
            # -----------------------------------------------------------------
            st.write("")
            st.markdown('<div class="tg-section-kicker">04 / Explainable AI</div>',
                        unsafe_allow_html=True)
            st.subheader("Why did the model make this prediction?")
            st.caption(
                "These are local TreeSHAP contributions for this specific posting. "
                "Positive values push the model toward class 1 (assumed to mean scam); "
                "negative values push it away from class 1. Values are raw-score (margin) "
                "contributions, not percentage-point changes in probability."
            )

            explain_left, explain_right = st.columns([1.15, 1], gap="large")

            with explain_left:
                st.markdown("**Feature groups influencing this prediction**")
                if not group_summary.empty:
                    chart_data = group_summary.set_index("Feature group")[
                        ["SHAP contribution"]
                    ]
                    st.bar_chart(chart_data, height=270)
                else:
                    st.info("No non-zero feature contributions were returned.")

                st.caption(
                    "Group totals combine contributions from features in the same input block. "
                    "A positive total supports the scam class; a negative total supports the genuine class."
                )

            with explain_right:
                st.markdown("**Top individual feature contributions**")
                if not top_explanations.empty:
                    display = top_explanations[[
                        "Feature", "SHAP contribution", "Direction"
                    ]].copy()
                    display["SHAP contribution"] = display["SHAP contribution"].map(
                        lambda value: f"{value:+.4f}"
                    )
                    st.dataframe(
                        display,
                        use_container_width=True,
                        hide_index=True,
                        height=340,
                    )
                else:
                    st.info("No individual feature contributions were returned.")

            # Separate evidence by direction for an easier-to-read explanation.
            st.write("")
            st.markdown("**What pushed the prediction in each direction?**")
            toward_scam = all_explanations[
                all_explanations["SHAP contribution"] > 0
            ].head(8)
            away_from_scam = all_explanations[
                all_explanations["SHAP contribution"] < 0
            ].sort_values("SHAP contribution", ascending=True).head(8)

            direction_left, direction_right = st.columns(2, gap="large")
            with direction_left:
                st.markdown("🟥 **Pushed toward scam**")
                if toward_scam.empty:
                    st.caption("No positive feature contributions were returned.")
                else:
                    for _, row in toward_scam.iterrows():
                        st.markdown(
                            f"- `{row['Feature']}` — **+{row['SHAP contribution']:.4f}**"
                        )

            with direction_right:
                st.markdown("🟩 **Pushed away from scam**")
                if away_from_scam.empty:
                    st.caption("No negative feature contributions were returned.")
                else:
                    for _, row in away_from_scam.iterrows():
                        st.markdown(
                            f"- `{row['Feature']}` — **{row['SHAP contribution']:.4f}**"
                        )

            if not text_term_summary.empty:
                with st.expander("Text terms combined across duplicated TF-IDF fields"):
                    st.caption(
                        "The same pasted text is fed into five text vectorizers. "
                        "This table adds the SHAP values for identical terms across "
                        "those five input slots to reduce duplicate rows."
                    )
                    term_display = text_term_summary[
                        ["Term", "SHAP contribution", "Direction"]
                    ].head(20).copy()
                    term_display["SHAP contribution"] = term_display[
                        "SHAP contribution"
                    ].map(lambda value: f"{value:+.6f}")
                    st.dataframe(
                        term_display,
                        use_container_width=True,
                        hide_index=True,
                        height=320,
                    )

            with st.expander("How to read the model explanation"):
                st.markdown(
                    f"""
                    - **Base value:** `{base_value:+.4f}` is the model's expected raw score before this posting's feature contributions.
                    - **Positive contribution:** moves the raw model score toward class `1` (scam), assuming class `1` is the scam label in your training data.
                    - **Negative contribution:** moves the raw score away from class `1` and toward class `0` (genuine).
                    - **Magnitude:** a larger absolute contribution means a larger influence on this particular prediction.
                    - **Important:** these values explain the model's output; they do not prove that a phrase caused fraud. Correlated words can share or redistribute attribution.
                    - Text features are TF-IDF terms, not complete sentence-level explanations. Your current inference pipeline places the same pasted text into all five trained text fields, so a term may appear under multiple field names; these are duplicated model input slots, not five independent pieces of evidence.
                    """
                )

            with st.expander("View all non-zero feature contributions"):
                all_display = all_explanations[[
                    "Feature", "Feature group", "SHAP contribution", "Direction"
                ]].copy()
                all_display["SHAP contribution"] = all_display["SHAP contribution"].map(
                    lambda value: f"{value:+.6f}"
                )
                st.dataframe(
                    all_display,
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )

            # Rule-based explanations.
            st.write("")
            st.markdown('<div class="tg-section-kicker">05 / Rule-based checks</div>',
                        unsafe_allow_html=True)
            st.subheader("🚩 Rule-based warning signals")

            if rule_flags:
                st.caption(
                    f"{len(rule_flags)} rule group(s) matched phrases in the submitted text."
                )

                for flag in rule_flags:
                    explanation = RULE_EXPLANATIONS.get(flag, "")
                    st.markdown(
                        f"""
                        <div class="tg-flag">
                            <span style="font-size:17px;">⚠️</span>
                            <div>
                                <div style="font-weight:700; margin-bottom:4px;">{flag}</div>
                                <div style="color:#aebed2; line-height:1.5;">{explanation}</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.success(
                    "No phrases from the current rule-based warning list were detected. "
                    "This does not prove that the posting is genuine."
                )

            # -----------------------------------------------------------------
            # Offer intelligence: extracted details + transparent verification status
            # -----------------------------------------------------------------
            st.write("")
            st.markdown('<div class="tg-section-kicker">06 / Offer intelligence</div>',
                        unsafe_allow_html=True)
            st.subheader("Extracted details and investigation signals")
            st.caption(
                "These checks inspect information visible in the submitted text. "
                "They do not establish that a company, recruiter, website or vacancy is authentic."
            )

            detail_col, contact_col = st.columns(2, gap="large")
            with detail_col:
                st.markdown("**Extracted contact information**")
                st.write("**Claimed company:**", claimed_company.strip() or "Not provided")
                st.write("**Email addresses:**", ", ".join(investigation["emails"]) or "Not found")
                st.write("**Email domains:**", ", ".join(investigation["email_domains"]) or "Not found")
                st.write("**Websites:**", ", ".join(investigation["websites"]) or "Not found")
                st.write("**Phone numbers:**", ", ".join(investigation["phone_numbers"]) or "Not found")
                st.write("**LinkedIn profile/company links:**")
                if investigation["linkedin_profiles"]:
                    for profile_url in investigation["linkedin_profiles"]:
                        st.write(profile_url)
                else:
                    st.write("Not found")

            with contact_col:
                st.markdown("**Payment-related details**")
                if investigation["money_mentions"]:
                    for item in investigation["money_mentions"][:6]:
                        st.write(f"- **{item['mention']}** — `{item['context'][:150]}`")
                else:
                    st.write("No currency amounts detected.")
                if investigation["payment_contexts"]:
                    st.warning("Payment/fee language found. Check whether payment is demanded before employment.")
                else:
                    st.success("No configured upfront-fee phrases detected. This does not prove the offer is safe.")

            st.markdown("**Company & recruiter consistency checks**")
            st.caption(
                "Comparisons use the reference details you entered. A match is not proof of authenticity, "
                "and a mismatch is not proof of fraud."
            )
            if investigation["reference_checks"]:
                for check in investigation["reference_checks"]:
                    if check["status"] == "Match":
                        st.success(f"**{check['status']}: {check['title']}** — {check['detail']}")
                    elif check["status"] == "Mismatch":
                        st.warning(f"**{check['status']}: {check['title']}** — {check['detail']}")
                    else:
                        st.info(f"**{check['status']}: {check['title']}** — {check['detail']}")
            else:
                st.info(
                    "No independent reference details were entered. Add a trusted company domain "
                    "or phone number to compare it with the offer."
                )

            st.markdown("**Investigation findings**")
            if investigation["findings"]:
                for finding in investigation["findings"]:
                    if finding["level"] == "High attention":
                        st.warning(f"**{finding['title']}** — {finding['detail']}")
                    elif finding["level"] == "Needs verification":
                        st.warning(f"**{finding['title']}** — {finding['detail']}")
                    else:
                        st.info(f"**{finding['title']}** — {finding['detail']}")
            else:
                st.info("No additional extraction findings were generated.")

            with st.expander("View structured investigation data"):
                st.json({
                    "claimed_company": claimed_company.strip() or None,
                    "emails": investigation["emails"],
                    "email_domains": investigation["email_domains"],
                    "websites": investigation["websites"],
                    "website_domains": investigation["website_domains"],
                    "phone_numbers": investigation["phone_numbers"],
                    "linkedin_profiles": investigation["linkedin_profiles"],
                    "payment_contexts": investigation["payment_contexts"],
                    "possible_domain_mismatch": investigation["domain_mismatch"],
                    "reference_domain": investigation["reference_domain"],
                    "reference_checks": investigation["reference_checks"],
                })

            # Recommended next action.
            st.write("")
            st.subheader("Recommended next step")

            if prediction == 1 or rule_flags:
                st.warning(
                    "Pause before proceeding. Do not pay registration or training fees, "
                    "share OTPs, or provide sensitive financial information. Verify the "
                    "company using independently located contact details."
                )
            else:
                st.info(
                    "No strong warning was returned by this screening. Still verify the "
                    "company, recruiter identity, job offer and payment terms independently."
                )

            # Technical details for transparency.
            with st.expander("View auto-detected text signals"):
                st.json(structured)

            with st.expander("How to interpret this result"):
                st.markdown(
                    """
                    - **Model verdict:** the class predicted by the trained XGBoost model.
                    - **Scam probability:** the model's estimated probability for class `1`.
                    - **SHAP contributions:** show how individual model input features moved this one prediction toward or away from the scam class.
                    - **Rule-based warnings:** keyword-pattern checks that run separately from the machine-learning model; they do not change the XGBoost prediction.
                    - SHAP values explain model behavior, not real-world causation. A missing warning does not mean a posting is safe.
                    """
                )

            st.markdown(
                """
                <div class="tg-footer">
                    <b>Stay alert. Verify before you trust.</b><br>
                    Suspected cyber fraud in India can be reported at
                    <a href="https://cybercrime.gov.in" target="_blank"
                       style="color:#75baff;">cybercrime.gov.in</a>
                    or by calling <b>1930</b>.<br>
                    TrustGuard AI is a hackathon prototype and not a certified
                    fraud-detection service.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as error:
            st.error(
                "The analysis could not be completed. Check that the model files "
                "match the preprocessing artifacts exported from your notebook."
            )
            with st.expander("Technical error details"):
                st.exception(error)


# ---------------------------------------------------------------------
# 11. About / limitations
# ---------------------------------------------------------------------
with st.expander("About TrustGuard AI and its limitations"):
    st.write(
        "TrustGuard AI combines a trained text-and-structured-feature classifier "
        "with a separate keyword-based warning layer. It screens the text supplied "
        "by the user; it does not browse the web to verify a company, validate a "
        "recruiter's identity, or confirm whether a job vacancy exists. "
        "It extracts visible contact and payment details, identifies LinkedIn URLs, "
        "and compares offer email/phone/name text against reference details entered by "
        "the user. These are consistency checks only: the app does not browse the web, "
        "query company registries, validate domain ownership, or authenticate a recruiter. "
        "Treat every result as decision support and verify independently. "
        "Image uploads are converted to text with Tesseract OCR. Review extracted "
        "text because OCR can misread names, numbers and symbols."
    )
