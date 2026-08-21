# -*- coding: utf-8 -*-
"""
STUDENT RISK PREDICTION API
----------------------------
A REST API wrapping the trained models from ml_pipeline.py, for a
mobile app (Flutter/Android/iOS) to call directly: the app sends one
student's known academic data, the API returns risk predictions
immediately - no batch files, no retraining.

RUN:
    pip install flask flask-cors pandas numpy scikit-learn joblib
    python api_server.py
    (starts on http://0.0.0.0:5000)

ENDPOINTS:
    GET  /health            - check the server + models are loaded
    POST /predict            - the main endpoint (see below)

REQUEST BODY for POST /predict:
{
  "student_id": "220999",
  "year1": {                     <- required (minimum data needed)
    "fall_gpa": 3.1, "spring_gpa": 2.9,
    "total_courses": 12, "passed_courses": 10, "failed_courses": 2,
    "credits": 30, "points": 93.0,
    "avg_grade": 3.0, "max_grade": 4.0, "min_grade": 1.0
  },
  "year2": { ... same shape ... },   <- optional
  "year3": { ... same shape ... }    <- optional
}

The API automatically uses the MOST RECENT year provided for each
prediction category (e.g. if year1+year2 are sent, warning risk for
year 3 is returned, not year 2 - always the furthest genuinely
predictive stage available). Sending only year1 is enough to get a
first set of predictions; add year2/year3 as the student progresses
for sharper ones.

RESPONSE:
{
  "student_id": "220999",
  "years_provided": ["year1", "year2"],
  "predictions": {
    "warning_risk": {
      "available": true, "probability": 0.73, "predicts": "warned_in_year3",
      "based_on": "years1-2", "confidence_note": "well-validated (recall ~89% in testing)"
    },
    "not_promoted_risk": { ... },
    "delayed_progression_risk": { ... },
    "predicted_next_year_gpa": { ... }
  }
}
"""

from pathlib import Path
import os
import json
import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    from dotenv import load_dotenv
    # Looks for a .env file in the same folder as this script (not the
    # current working directory) and loads any KEY=value lines into
    # os.environ - so OPENROUTER_API_KEY (and anything else) just
    # needs to be written once in .env instead of retyped into every
    # new PowerShell window. Safe to skip silently if python-dotenv
    # isn't installed or no .env file exists - manual $env: still
    # works exactly as before either way.
    load_dotenv(Path(__file__).resolve().parent / "key.env")
except ImportError:
    pass

BASE = Path(__file__).resolve().parent
MODELS = BASE / "ml_models"

app = Flask(__name__)
CORS(app)  # allows the Flutter app (or a browser during testing) to call this from any origin

YEAR_FIELDS = [
    "fall_gpa", "spring_gpa", "total_courses", "passed_courses",
    "failed_courses", "credits", "points", "avg_grade", "max_grade", "min_grade",
]

FIELD_TO_COLUMN = {
    "fall_gpa": "fall_gpa", "spring_gpa": "spring_gpa",
    "total_courses": "total_courses", "passed_courses": "passed_courses",
    "failed_courses": "failed_courses", "credits": "credits",
    "points": "points", "avg_grade": "avg_grade",
    "max_grade": "max_grade", "min_grade": "min_grade",
}

# Each entry: which saved model file to use, what it predicts, a
# short honesty note carried over from ml_pipeline.py's own
# cross-validated results (see README section at the bottom of this
# file) - the app should show this alongside the number, not just
# the raw probability.
WARNING_MODELS = {
    1: ("warning_year2_model.joblib", "warned_in_year2", "years1", "well-validated (recall ~91% in testing)"),
    2: ("warning_year3_model.joblib", "warned_in_year3", "years1-2", "well-validated (recall ~89% in testing)"),
    3: ("warning_year4_model.joblib", "warned_in_year4", "years1-3", "validated (recall ~73% in testing)"),
}
DELAYED_MODELS = {
    1: ("delayed_progression_cutoff1_model.joblib", "years1", "weak signal (recall ~19% in testing) - treat as a soft hint, not a verdict"),
    2: ("delayed_progression_cutoff2_model.joblib", "years1-2", "weak signal (recall ~0% in testing at default threshold) - not reliable yet"),
    3: ("delayed_progression_cutoff3_model.joblib", "years1-3", "moderate signal (PR-AUC 0.39, recall improves at lower thresholds) - use as a flag to review, not a verdict"),
}
GPA_MODELS = {
    1: ("gpa_predict_year2_model.joblib", "year2_avg_gpa", "years1", "moderate fit (typical error ~0.31 GPA points). NOTE: LinearRegression scored slightly better on average cross-validated MAE, but was tested and rejected (2026-08-19) - it produced backwards predictions (higher GPA forecast for weaker students) due to multicollinearity between fall/spring/avg-grade features. RandomForest is slightly less accurate on average but doesn't extrapolate to nonsensical per-student values."),
    2: ("gpa_predict_year3_model.joblib", "year3_avg_gpa", "years1-2", "moderate fit (typical error ~0.31 GPA points). Same LinearRegression-instability finding as the year2 stage - see note there."),
    3: ("gpa_predict_year4_model.joblib", "year4_avg_gpa", "years1-3", "good fit (typical error ~0.17 GPA points). Same LinearRegression-instability finding as the year2 stage - see note there."),
}
PROMOTION_MODEL = ("promotion_year3_model.joblib", "years1", "well-validated via LogisticRegression (recall ~86%, ROC-AUC 0.91) - swapped from RandomForest (recall ~60%) after a real 3-model comparison confirmed it performs meaningfully better here, AND verified stable/correctly-directional on individual test cases (2026-08-19)")

# UNRELIABLE - included because it was explicitly requested, not
# because it's trustworthy. Trained on only 27 non-graduate examples
# out of 495 total - even the FULL-feature version of this model
# (using all 4 years) only caught 25% of actual non-graduates in
# testing; this years-1-3-only version is weaker still. Framed as
# 'graduation_probability' (not risk) since that's the literal
# metric being asked for, but every response carries this warning
# and the app UI must not hide it.
GRADUATION_MODEL = (
    "graduation_status_model_bundled.joblib", "years1-3",
    "LOW CONFIDENCE - trained on only 27 non-graduate examples; even "
    "the best version of this model caught just 25% of actual "
    "non-graduates in testing. Treat as a rough indicator only, "
    "never as a decision on its own."
)

_loaded_models = {}

# --- Clustering ("where does this student stand among their peers")
# was trained (ml_pipeline.py) but never wired into this API - adding
# it now. IMPORTANT LIMITATION: the K-means model was trained on each
# student's COMPLETE profile (all 4 years + career totals like
# academic_hours, total credits, military status - 59 columns). A
# NEW student who has only submitted year1 (or year1-3) data is
# missing most of those columns; the imputer fills gaps with the
# TRAINING SET's median, which keeps the model from crashing but
# means the assignment leans heavily on whatever partial data was
# provided plus population averages for everything else. This is
# fundamentally different from the other predictions here (which
# were each trained and validated specifically for a given data-
# completeness stage) - cluster confidence genuinely degrades with
# less data, and the response says so explicitly rather than
# presenting a partial-data cluster assignment with the same
# confidence as a complete one.
CLUSTER_FEATURES_FILE = BASE / "cluster_features.json"
CLUSTER_PROFILES_FILE = BASE / "cluster_profiles.csv"

CLUSTER_DESCRIPTIONS = {
    0: "أداء أعلى من المتوسط: معدل تراكمي أعلى، إنذارات أقل بكتير (0.11 مقابل 1.3 في المتوسط)، نسبة رسوب أقل من 1%.",
    1: "أداء أقل من المتوسط: معدل تراكمي أقل، إنذارات أكتر (1.3 في المتوسط)، نسبة رسوب أعلى (~4%).",
}

# Real, computed from the 495-student training set (not invented) -
# see build notes 2026-08-21. 'recovery_rate' is deliberately absent:
# we never defined or measured what 'recovery' means for a cluster,
# so making up a percentage for it would be exactly the kind of
# invented stat we've been removing elsewhere.
CLUSTER_STATS = {
    0: {"student_count": 281, "avg_cumulative_gpa": 2.89, "avg_warnings": 0.114},
    1: {"student_count": 214, "avg_cumulative_gpa": 2.433, "avg_warnings": 1.313},
}

# Display label only - the model still works internally with 0/1
# (that's what kmeans.predict() returns and what CLUSTER_DESCRIPTIONS
# is keyed on). A = the higher-performing group (cluster 0), B = the
# other one, matching the familiar "A is better than B" convention.
CLUSTER_LABELS = {0: "A", 1: "B"}

STUDENT_DATA_DIR = BASE / "student_data"
COURSES_FILE = STUDENT_DATA_DIR / "courses_detailed.csv"
PREREQ_RISK_FILE = BASE / "prerequisite_risk_lookup.json"
NO_PREREQ_FILE = BASE / "confirmed_no_prereq.json"

_courses_df_cache = None
_prereq_risk_cache = None
_no_prereq_cache = None


def get_courses_df():
    global _courses_df_cache
    if _courses_df_cache is None and COURSES_FILE.exists():
        _courses_df_cache = pd.read_csv(COURSES_FILE, encoding="utf-8-sig", dtype={"student_id": str})
    return _courses_df_cache


def get_prereq_risk_lookup():
    global _prereq_risk_cache
    if _prereq_risk_cache is None:
        if PREREQ_RISK_FILE.exists():
            _prereq_risk_cache = json.loads(PREREQ_RISK_FILE.read_text(encoding="utf-8"))
        else:
            _prereq_risk_cache = {}
    return _prereq_risk_cache


def get_no_prereq_courses():
    global _no_prereq_cache
    if _no_prereq_cache is None:
        if NO_PREREQ_FILE.exists():
            _no_prereq_cache = set(json.loads(NO_PREREQ_FILE.read_text(encoding="utf-8")))
        else:
            _no_prereq_cache = set()
    return _no_prereq_cache


def get_student_course_history(student_id):
    """Tier 1 - always available: every course this student has actually
    taken, with their real grade. Pure reporting, no model involved."""
    courses = get_courses_df()
    if courses is None:
        return []
    rows = courses[courses["student_id"] == str(student_id).strip()]
    history = []
    for _, r in rows.iterrows():
        history.append({
            "course_code": r["course_code"],
            "course_name": r["course_name"],
            "academic_year": int(r["academic_year"]) if pd.notna(r["academic_year"]) else None,
            "semester": r["semester"],
            "numeric_grade": float(r["numeric_grade"]) if pd.notna(r["numeric_grade"]) else None,
            "letter_grade": r["letter_grade"] if pd.notna(r["letter_grade"]) else None,
            "passed": bool(r["passed"]) if pd.notna(r["passed"]) else None,
        })
    return history


def get_student_course_risk_predictions(student_id):
    """Tier 2 - only for the 16 (prerequisite -> target) pairs with a
    confirmed official prerequisite AND enough historical students
    (15+) to compute a real fail-rate split. For a course the student
    HASN'T taken yet, but whose prerequisite they HAVE completed,
    returns the historical fail rate for students whose prerequisite
    grade was in the same range as theirs - genuine data, not a
    trained model, since most of these course pairs have too few
    students for a proper classifier but plenty for a direct
    historical comparison.

    IMPORTANT SCOPE LIMIT: only 18 of the 53 courses in our data have
    an officially confirmed prerequisite (from the IS-program lائحة
    and the shared H/BS/CS core - see build notes 2026-08-21). The 23
    CS3xx/4xx specialized courses (CS301-CS316, CS402, CS404-CS418)
    have NO confirmed prerequisite structure available, so they are
    deliberately excluded here rather than guessed - they still
    appear in get_student_course_history() with real past grades,
    just without a forward risk prediction.
    """
    history = get_student_course_history(student_id)
    if not history:
        return []

    taken_codes = {h["course_code"]: h for h in history}
    risk_lookup = get_prereq_risk_lookup()
    predictions = []

    for target_code, info in risk_lookup.items():
        if target_code in taken_codes:
            continue  # already taken - this is a forward-looking prediction only
        prereq_code = info["prerequisite"]
        prereq_record = taken_codes.get(prereq_code)
        if not prereq_record or prereq_record["numeric_grade"] is None:
            continue  # hasn't completed the prerequisite yet - not eligible/predictable

        prereq_grade = prereq_record["numeric_grade"]
        if prereq_grade < 2.4:
            fail_rate, bucket_n = info["low_prereq_fail_rate"], info["low_prereq_n"]
            bucket_label = "below C (< 2.4)"
        else:
            fail_rate, bucket_n = info["high_prereq_fail_rate"], info["high_prereq_n"]
            bucket_label = "C or above (>= 2.4)"

        if fail_rate is None:
            continue  # bucket had too few students to be trustworthy

        predictions.append({
            "course_code": target_code,
            "prerequisite_course": prereq_code,
            "your_prerequisite_grade": round(prereq_grade, 2),
            "prerequisite_grade_bucket": bucket_label,
            "historical_fail_rate": fail_rate,
            "based_on_n_students": bucket_n,
            "note": f"Of {bucket_n} past students who scored {bucket_label} in {prereq_code}, {fail_rate*100:.0f}% went on to fail {target_code}.",
        })

    return predictions


@app.route("/student/<student_id>/courses", methods=["GET"])
def student_courses(student_id):
    history = get_student_course_history(student_id)
    if not history:
        return jsonify({
            "error": f"No course records found for student {student_id}.",
            "student_id": student_id,
        }), 404
    return jsonify({
        "student_id": student_id,
        "course_history": history,
        "upcoming_course_risks": get_student_course_risk_predictions(student_id),
        "coverage_note": (
            "upcoming_course_risks only covers courses with an officially "
            "confirmed prerequisite and enough historical students to "
            "compute a real fail rate (16 of 53 courses in our data). "
            "Specialized CS3xx/4xx electives are not yet covered - "
            "course_history still shows their real past grades."
        ),
    })


_cluster_features_cache = None
_cluster_profiles_cache = None


def get_cluster_features():
    global _cluster_features_cache
    if _cluster_features_cache is None:
        import json
        if CLUSTER_FEATURES_FILE.exists():
            _cluster_features_cache = json.loads(CLUSTER_FEATURES_FILE.read_text(encoding="utf-8"))
        else:
            _cluster_features_cache = []
    return _cluster_features_cache


def get_cluster_profiles():
    global _cluster_profiles_cache
    if _cluster_profiles_cache is None and CLUSTER_PROFILES_FILE.exists():
        _cluster_profiles_cache = pd.read_csv(CLUSTER_PROFILES_FILE, encoding="utf-8-sig").set_index("cluster")
    return _cluster_profiles_cache


def predict_cluster(df):
    """Returns (cluster_id, cluster_label, confidence_note, description)
    or None if the clustering artifacts aren't available. `df` should
    already have whatever year1/2/3 columns the caller submitted -
    missing columns (year4, career totals, etc.) are imputed
    automatically."""
    cluster_features = get_cluster_features()
    if not cluster_features:
        return None

    preprocessor = load_model_raw("clustering_preprocessor.joblib")
    kmeans = load_model_raw("kmeans_model.joblib")
    if preprocessor is None or kmeans is None:
        return None

    row = {col: df[col].iloc[0] if col in df.columns else None for col in cluster_features}
    X = pd.DataFrame([row])[cluster_features]

    known_fraction = X.notna().sum(axis=1).iloc[0] / len(cluster_features)

    Xt = preprocessor.transform(X)
    cluster_id = int(kmeans.predict(Xt)[0])
    cluster_label = CLUSTER_LABELS.get(cluster_id, str(cluster_id))

    if known_fraction >= 0.7:
        confidence = f"موثوق نسبياً ({known_fraction*100:.0f}% من البيانات معروفة، الباقي بمتوسط الفوج)"
    elif known_fraction >= 0.3:
        confidence = f"تقريبي ({known_fraction*100:.0f}% من البيانات معروفة بس) - مؤشر أولي وليس تصنيفاً نهائياً"
    else:
        confidence = f"ضعيف جداً ({known_fraction*100:.0f}% من البيانات معروفة) - معظم التصنيف مبني على متوسط الفوج لا بيانات الطالب"

    return cluster_id, cluster_label, confidence, CLUSTER_DESCRIPTIONS.get(cluster_id, "غير معروف"), CLUSTER_STATS.get(cluster_id, {})




import re


def extract_target_year(predicts_str):
    """'warned_in_year2' -> 2, 'not_reaching_year3_on_schedule' -> 3.
    Returns None if no year pattern is found."""
    m = re.search(r"year(\d)", predicts_str)
    return int(m.group(1)) if m else None


def load_model(filename):
    if filename not in _loaded_models:
        path = MODELS / filename
        if not path.exists():
            return None
        _loaded_models[filename] = joblib.load(path)
    return _loaded_models[filename]


def load_model_raw(filename):
    """Same as load_model() but for the clustering artifacts, which
    ml_pipeline.py's clustering() saved as bare sklearn objects
    (joblib.dump(preprocessor, ...) / joblib.dump(model, ...)),
    unlike every other model here which is wrapped in a
    {"pipeline": ..., "features": [...]} dict. Kept as a separate
    function rather than special-casing load_model() so a caller can
    never accidentally treat a bare object as if it had that shape."""
    key = f"raw:{filename}"
    if key not in _loaded_models:
        path = MODELS / filename
        if not path.exists():
            return None
        _loaded_models[key] = joblib.load(path)
    return _loaded_models[key]


def build_feature_row(payload):
    """Turns the request JSON into the same engineered-feature shape
    ml_pipeline.py's feature_engineering() produces, reusing the
    exact same derivation logic (year{n}_avg_gpa, gpa_trend_yA_yB)
    so the API can never silently drift out of sync with how the
    models were trained."""
    row = {}
    years_provided = []

    for year_num in [1, 2, 3]:
        key = f"year{year_num}"
        if key not in payload or not isinstance(payload[key], dict):
            continue
        years_provided.append(key)
        y = payload[key]
        for field in YEAR_FIELDS:
            col = f"year{year_num}_{FIELD_TO_COLUMN[field]}"
            row[col] = y.get(field)

    df = pd.DataFrame([row])

    # Same derivation as ml_pipeline.py's feature_engineering() -
    # kept in sync manually since this is a standalone API process;
    # see that function if the engineered-feature logic ever changes.
    for year in [1, 2, 3]:
        cols = [c for c in [f"year{year}_fall_gpa", f"year{year}_spring_gpa"] if c in df.columns]
        if cols:
            df[f"year{year}_avg_gpa"] = df[cols].mean(axis=1)

    for a, b in [(1, 2), (2, 3)]:
        x, y_ = f"year{a}_avg_gpa", f"year{b}_avg_gpa"
        if x in df.columns and y_ in df.columns:
            df[f"gpa_trend_y{a}_y{b}"] = df[y_] - df[x]

    return df, years_provided


import shap

FEATURE_NAME_AR = {
    "fall_gpa": "معدل الفصل الأول",
    "spring_gpa": "معدل الفصل الثاني",
    "total_courses": "عدد المقررات الكلي",
    "passed_courses": "عدد المقررات الناجحة",
    "failed_courses": "عدد المقررات الراسبة",
    "credits": "الساعات المعتمدة",
    "points": "مجموع النقاط",
    "avg_grade": "متوسط الدرجات",
    "max_grade": "أعلى درجة",
    "min_grade": "أقل درجة",
    "avg_gpa": "متوسط المعدل التراكمي للسنة",
}


def readable_feature_name(col):
    """'year1_fall_gpa' -> 'معدل الفصل الأول (سنة 1)'. Falls back to
    the raw column name for anything not in the map (e.g. gpa_trend_*)
    so a new feature never crashes the explanation, just looks less
    polished until translated."""
    if col.startswith("gpa_trend_y"):
        # gpa_trend_y1_y2 -> "اتجاه المعدل من سنة 1 إلى سنة 2"
        try:
            a, b = col.replace("gpa_trend_y", "").split("_y")
            return f"اتجاه تغيّر المعدل من سنة {a} إلى سنة {b}"
        except Exception:
            return col
    if col.startswith("year"):
        year_num, _, rest = col.partition("_")
        year_num = year_num.replace("year", "")
        ar = FEATURE_NAME_AR.get(rest)
        if ar:
            return f"{ar} (سنة {year_num})"
    return col


def explain_prediction(model_bundle, df, is_regression=False, top_n=4, positive_label="increases_risk", negative_label="decreases_risk"):
    """Per-STUDENT explanation using SHAP - not 'which features matter
    for the model in general' but 'which of THIS student's specific
    numbers pushed THIS prediction up or down, and by how much'. Runs
    on the exact post-imputer/scaler input the model actually sees
    (SHAP needs the transformed space to be exact), then maps
    contributions back to the original feature names for display,
    since scaling doesn't change which value belongs to which
    feature.

    Two model families are in play here (2026-08-19): most systems
    still use RandomForest (TreeExplainer - fast, exact for trees),
    but promotion and all 3 GPA-regression stages were swapped to
    LogisticRegression/LinearRegression after a genuine 3-model
    comparison showed they perform meaningfully better there.
    TreeExplainer cannot explain a linear model - it needs
    LinearExplainer instead, which also needs reference/background
    data (saved alongside the model at training time, in
    model_bundle['background']) to establish what 'average' looks
    like. model_bundle['model_type'] ('tree' or 'linear', defaulting
    to 'tree' for older bundles saved before this distinction
    existed) picks the right path. The two explainers also disagree
    on output shape for classifiers: TreeExplainer returns one array
    per class (index [1] = the positive/at-risk class), LinearExplainer
    returns a single array directly (binary logistic contribution is
    inherently one-dimensional) - handled below rather than assumed.

    Guarantees at least one factor from EVERY year present in
    model_bundle['features'], not just a flat top-N-overall ranking.
    Confirmed (2026-08-12): year-1 data carries ~60% of this model
    family's total feature importance (a real, defensible finding -
    early performance is the strongest predictor of later academic
    risk) vs ~19% each for years 2 and 3. A flat top-4 list can
    therefore end up ALL year-1 factors purely because they dominate
    by magnitude, which reads as 'year 2/3 data was ignored' even
    though it was genuinely used in training and in this specific
    prediction - misleading despite being numerically accurate. This
    keeps the ranking honest (factors are still sorted by real
    impact) while making sure every year the caller actually
    submitted is visibly represented at least once.
    """
    pipe = model_bundle["pipeline"]
    features = model_bundle["features"]
    model_type = model_bundle.get("model_type", "tree")
    X = df[features]

    X_transformed = X.copy()
    for step_name in ["imputer", "scaler"]:
        X_transformed = pipe.named_steps[step_name].transform(X_transformed)

    model = pipe.named_steps["model"]

    if model_type == "linear":
        background = model_bundle["background"]
        explainer = shap.LinearExplainer(model, background)
        sv = explainer.shap_values(X_transformed)
        # LinearExplainer: single array regardless of classifier/regressor.
        contributions = np.array(sv)[0]
    else:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_transformed)
        if is_regression:
            contributions = np.array(sv)[0]
        else:
            # sv shape: (1, n_features, n_classes) - class 1 is the
            # 'positive'/at-risk class in every model here.
            contributions = np.array(sv)[0, :, 1]

    all_ranked = sorted(
        zip(features, contributions, X.iloc[0].tolist()),
        key=lambda t: abs(t[1]), reverse=True
    )

    def year_of(col):
        for y in (1, 2, 3):
            if col.startswith(f"year{y}"):
                return y
        return None

    years_present = sorted({year_of(f) for f in features if year_of(f)})

    selected = list(all_ranked[:top_n])
    covered_years = {year_of(c) for c, _, _ in selected}
    for year in years_present:
        if year in covered_years:
            continue
        # add this year's single strongest factor, even if it didn't
        # make the flat top-N, so every submitted year is visible.
        best_for_year = max(
            (t for t in all_ranked if year_of(t[0]) == year),
            key=lambda t: abs(t[1]), default=None
        )
        if best_for_year:
            selected.append(best_for_year)
            covered_years.add(year)

    factors = []
    for col, contribution, raw_value in selected:
        if is_regression:
            effect = "pushes_prediction_up" if contribution > 0 else "pushes_prediction_down"
        else:
            effect = positive_label if contribution > 0 else negative_label
        factors.append({
            "factor": readable_feature_name(col),
            "student_value": round(float(raw_value), 3),
            "effect": effect,
            "impact": round(float(abs(contribution)), 4),
        })
    return factors


def predict_with_model(model_bundle, df):
    pipe = model_bundle["pipeline"]
    needed = model_bundle["features"]
    missing = [f for f in needed if f not in df.columns]
    if missing:
        return None, missing
    X = df[needed]
    if hasattr(pipe, "predict_proba"):
        prob = float(pipe.predict_proba(X)[:, 1][0])
        return prob, None
    else:
        val = float(pipe.predict(X)[0])
        return val, None


STUDENT_DATA_DIR = BASE / "student_data"

_student_lookup = None


def build_student_lookup():
    """Loads the real, already-extracted student records from CSV and
    reshapes them into the same {year1: {...}, year2: {...}, year3: {...}}
    structure the API's own request format uses - so a lookup result
    can be handed straight back to the client and re-submitted to
    /predict unchanged. Checks student_ml_dataset.csv FIRST (the
    year-4-anchored file - most complete, years 1-4) and falls back to
    promotion_cohort_dataset.csv for the 35 students who never reached
    year 4 and so are absent from every other file (see
    build_dataset.py's promotion_prediction_system docstring)."""
    lookup = {}

    sources = [
        STUDENT_DATA_DIR / "student_ml_dataset.csv",
        STUDENT_DATA_DIR / "promotion_cohort_dataset.csv",
        STUDENT_DATA_DIR / "student_ml_dataset_anchor_y3.csv",
    ]

    for path in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"student_id": str})
        for _, row in df.iterrows():
            sid = str(row["student_id"]).strip()
            if sid in lookup:
                continue  # earlier (more complete) source already has this student
            record = {}
            for year_num in [1, 2, 3]:
                year_data = {}
                for field, col_suffix in FIELD_TO_COLUMN.items():
                    col = f"year{year_num}_{col_suffix}"
                    if col in row and pd.notna(row[col]):
                        year_data[field] = float(row[col]) if not isinstance(row[col], str) else row[col]
                if year_data:
                    record[f"year{year_num}"] = year_data
            if record:
                lookup[sid] = record

    return lookup


def get_student_lookup():
    global _student_lookup
    if _student_lookup is None:
        _student_lookup = build_student_lookup()
        print(f"Student lookup loaded: {len(_student_lookup)} students from {STUDENT_DATA_DIR}")
    return _student_lookup


@app.route("/student/<student_id>", methods=["GET"])
def get_student(student_id):
    lookup = get_student_lookup()
    record = lookup.get(str(student_id).strip())
    if record is None:
        return jsonify({
            "error": f"Student {student_id} not found in stored records.",
            "student_id": student_id,
        }), 404
    return jsonify({
        "student_id": student_id,
        **record,
    })


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "message": "Student Risk Prediction API is running.",
        "endpoints": {
            "GET /health": "check server + models + advisor + clustering status",
            "GET /student/<id>": "look up a student's stored year1/2/3 academic data",
            "GET /student/<id>/courses": "student's course history + upcoming-course risk predictions",
            "POST /predict": "send student data, get risk predictions back",
            "POST /chat": "AI academic advisor chat (needs OPENROUTER_API_KEY)",
        },
    })


ADVISOR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
ADVISOR_MODEL = os.environ.get("ADVISOR_MODEL", "deepseek/deepseek-chat")
KNOWLEDGE_FILE = BASE / "knowledge_base.txt"

_knowledge_cache = None


def load_knowledge():
    global _knowledge_cache
    if _knowledge_cache is None:
        if KNOWLEDGE_FILE.exists():
            _knowledge_cache = KNOWLEDGE_FILE.read_text(encoding="utf-8")
        else:
            _knowledge_cache = ""
            print(f"WARNING: {KNOWLEDGE_FILE} not found - advisor will have no curriculum knowledge.")
    return _knowledge_cache


def analyze_gpa(gpa):
    if gpa >= 3.5:
        return "ممتاز"
    elif gpa >= 3.0:
        return "جيد جدًا"
    elif gpa >= 2.5:
        return "جيد"
    elif gpa >= 2.0:
        return "متوسط"
    else:
        return "ضعيف"


def build_risk_context_for_advisor(student_id, submitted_years=None):
    """Same idea as the standalone advisor's fetch_risk_context(), but
    calling the local prediction logic directly (function calls, not
    an HTTP round-trip to itself) since both services now live in one
    process. Falls back to the student's OWN stored data if the
    caller didn't submit fresh year1/2/3 numbers with this chat
    message."""
    lookup = get_student_lookup()
    record = lookup.get(str(student_id).strip())
    if record is None:
        return None

    df, years_provided = build_feature_row({"student_id": student_id, **(submitted_years or record)})
    if "year1" not in years_provided:
        return None

    max_year = len(years_provided)
    lines = [f"بيانات الطالب الحقيقية (كود {student_id}) من نظام التنبؤ الأكاديمي:"]

    fname, predicts, based_on, note = WARNING_MODELS[max_year]
    model = load_model(fname)
    if model:
        prob, missing = predict_with_model(model, df)
        if not missing:
            lines.append(f"- خطر الإنذار الأكاديمي في سنة {extract_target_year(predicts)}: {prob*100:.0f}%")
            factors = explain_prediction(model, df)[:2]
            if factors:
                txt = "، ".join(f"{f['factor']} ({f['student_value']})" for f in factors)
                lines.append(f"  أهم العوامل: {txt}")

    fname, based_on, note = PROMOTION_MODEL
    model = load_model(fname)
    if model:
        prob, missing = predict_with_model(model, df)
        if not missing:
            lines.append(f"- خطر عدم الترقية لسنة 3: {prob*100:.0f}%")

    fname, predicts, based_on, note = GPA_MODELS[max_year]
    model = load_model(fname)
    if model:
        val, missing = predict_with_model(model, df)
        if not missing:
            lines.append(f"- توقع معدل السنة القادمة: {val:.2f}")

    if len(lines) == 1:
        return None
    return "\n".join(lines)


@app.route("/health", methods=["GET"])
def health():
    available = [f.name for f in MODELS.glob("*.joblib")] if MODELS.exists() else []
    knowledge = load_knowledge()
    return jsonify({
        "status": "ok" if available else "no models found",
        "models_available": available,
        "advisor_knowledge_loaded": len(knowledge) > 0,
        "advisor_knowledge_chars": len(knowledge),
        "advisor_api_key_configured": bool(ADVISOR_KEY),
        "clustering_available": bool(get_cluster_features()) and CLUSTER_PROFILES_FILE.exists(),
    })


@app.route("/chat", methods=["POST"])
def chat():
    if not ADVISOR_KEY:
        return jsonify({
            "error": "OPENROUTER_API_KEY is not set. Set it as an environment variable before starting the server."
        }), 500

    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400

    gpa = payload.get("gpa")
    student_id = payload.get("student_id")
    history = payload.get("history", [])
    user_message = payload.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "'message' is required."}), 400

    level = analyze_gpa(float(gpa)) if gpa is not None else None
    submitted_years = {k: payload[k] for k in ("year1", "year2", "year3") if k in payload}
    risk_context = build_risk_context_for_advisor(student_id, submitted_years or None) if student_id else None

    system_prompt = """أنت AI Academic Advisor داخل جامعة.

مهمتك:
- شرح المواد
- مساعدة الطلاب في الدراسة
- استخدام اللائحة والخطة الدراسية فقط كمرجع
- عدم اختلاق معلومات غير موجودة في المرجع

أسلوبك يتكيف مع مستوى الطالب:
- ضعيف → شرح بسيط جدًا، خطوة خطوة، بدون مصطلحات معقدة
- متوسط → شرح خطوة خطوة مع أمثلة
- جيد/جيد جدًا → شرح متوسط العمق
- ممتاز → شرح متقدم، يمكن الدخول في تفاصيل تقنية

قواعد الشكل والطول (مهمة جداً):
- خليكِ مختصرة - إجابة مركّزة على السؤال بس، مش مقال طويل. لو السؤال بسيط، جاوبي في فقرة أو فقرتين بحد أقصى.
- متكتبيش أي حاجة إلا لو ليها داعي فعلي للسؤال المطروح - من غير حشو أو مقدمات طويلة.
- استخدمي تنسيق Markdown بشكل معتدل ومفيد بس (## للعناوين لو محتاجة تقسيم، **بولد** للكلمات المهمة، قوائم - للنقاط) - مش كل رد محتاج عناوين، استخدميها بس لو الإجابة فعلاً فيها أقسام متعددة.
"""
    if level:
        system_prompt += f"\nبيانات الطالب: GPA={gpa}, المستوى={level}"

    if risk_context:
        system_prompt += f"\n\n{risk_context}\n"
        system_prompt += (
            "استخدم هذه البيانات لتخصيص نصائحك عند مناسبة السياق (مثلاً لو "
            "الطالب سأل عن كيفية تحسين أدائه) - لا تذكرها إلا إذا كانت "
            "مفيدة فعلاً للسؤال المطروح."
        )

    system_prompt += f"\n\nمرجع اللائحة والخطة الدراسية:\n{load_knowledge()[:12000]}"

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {ADVISOR_KEY}", "Content-Type": "application/json"},
            json={"model": ADVISOR_MODEL, "messages": messages},
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to reach OpenRouter: {e}"}), 502

    if response.status_code != 200:
        return jsonify({
            "error": f"OpenRouter API request failed with status {response.status_code}",
            "details": response.text,
        }), 502

    response_json = response.json()
    choices = response_json.get("choices", [])
    if not choices:
        return jsonify({"error": "Unexpected API response structure.", "details": response_json}), 502

    return jsonify({
        "reply": choices[0]["message"]["content"],
        "level": level,
        "risk_context_used": risk_context is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "Request body must be JSON."}), 400

    student_id = payload.get("student_id", "unknown")
    df, years_provided = build_feature_row(payload)

    if "year1" not in years_provided:
        return jsonify({
            "error": "year1 data is required (minimum). year2/year3 are optional and sharpen the predictions."
        }), 400

    max_year = len(years_provided)  # 1, 2, or 3
    predictions = {}

    # --- warning risk: use the furthest stage the data supports ---
    fname, predicts, based_on, note = WARNING_MODELS[max_year]
    model = load_model(fname)
    if model:
        prob, missing = predict_with_model(model, df)
        if missing:
            predictions["warning_risk"] = {"available": False, "reason": f"missing fields: {missing}"}
        else:
            predictions["warning_risk"] = {
                "available": True, "probability": round(prob, 4),
                "predicts": predicts, "target_year": extract_target_year(predicts),
                "based_on": based_on, "confidence_note": note,
                "explanation": explain_prediction(model, df),
            }
    else:
        predictions["warning_risk"] = {"available": False, "reason": f"model file not found: {fname}"}

    # --- not-promoted-to-year3 risk: only meaningful with year1 (that's all the model uses) ---
    fname, based_on, note = PROMOTION_MODEL
    model = load_model(fname)
    if model:
        prob, missing = predict_with_model(model, df)
        if missing:
            predictions["not_promoted_risk"] = {"available": False, "reason": f"missing fields: {missing}"}
        else:
            predictions["not_promoted_risk"] = {
                "available": True, "probability": round(prob, 4),
                "predicts": "not_reaching_year3_on_schedule", "target_year": 3,
                "based_on": based_on,
                "confidence_note": note,
                "explanation": explain_prediction(model, df),
            }
    else:
        predictions["not_promoted_risk"] = {"available": False, "reason": f"model file not found: {fname}"}

    # --- delayed progression risk: use the furthest stage available ---
    fname, based_on, note = DELAYED_MODELS[max_year]
    model = load_model(fname)
    if model:
        prob, missing = predict_with_model(model, df)
        if missing:
            predictions["delayed_progression_risk"] = {"available": False, "reason": f"missing fields: {missing}"}
        else:
            predictions["delayed_progression_risk"] = {
                "available": True, "probability": round(prob, 4),
                "predicts": "not_reaching_year4_on_schedule", "target_year": 4,
                "based_on": based_on,
                "confidence_note": note,
                "explanation": explain_prediction(model, df),
            }
    else:
        predictions["delayed_progression_risk"] = {"available": False, "reason": f"model file not found: {fname}"}

    # --- next year's average GPA ---
    fname, predicts, based_on, note = GPA_MODELS[max_year]
    model = load_model(fname)
    if model:
        val, missing = predict_with_model(model, df)
        if missing:
            predictions["predicted_next_year_gpa"] = {"available": False, "reason": f"missing fields: {missing}"}
        else:
            predictions["predicted_next_year_gpa"] = {
                "available": True, "value": round(val, 3),
                "predicts": predicts, "target_year": extract_target_year(predicts),
                "based_on": based_on, "confidence_note": note,
                "explanation": explain_prediction(model, df, is_regression=True),
            }
    else:
        predictions["predicted_next_year_gpa"] = {"available": False, "reason": f"model file not found: {fname}"}

    # --- graduation probability: needs all 3 years, and even then is
    # low-confidence - see GRADUATION_MODEL comment above.
    if max_year == 3:
        fname, based_on, note = GRADUATION_MODEL
        model = load_model(fname)
        if model:
            prob, missing = predict_with_model(model, df)
            if missing:
                predictions["graduation_probability"] = {"available": False, "reason": f"missing fields: {missing}"}
            else:
                predictions["graduation_probability"] = {
                    "available": True, "probability": round(prob, 4),
                    "predicts": "graduates_on_schedule", "target_year": 4,
                    "based_on": based_on,
                    "confidence_note": note,
                    "explanation": explain_prediction(
                        model, df,
                        positive_label="increases_graduation_probability",
                        negative_label="decreases_graduation_probability",
                    ),
                }
        else:
            predictions["graduation_probability"] = {"available": False, "reason": f"model file not found: {fname}"}
    else:
        predictions["graduation_probability"] = {
            "available": False,
            "reason": "needs year1, year2, AND year3 data (this model was only trained at that exact stage)",
        }

    # --- cluster: where does this student stand among their peers ---
    cluster_result = predict_cluster(df)
    if cluster_result:
        cluster_id, cluster_label, confidence, description, stats = cluster_result
        predictions["peer_cluster"] = {
            "available": True,
            "cluster_id": cluster_id,
            "cluster_label": cluster_label,
            "description": description,
            "confidence_note": confidence,
            "cluster_stats": stats,
        }
    else:
        predictions["peer_cluster"] = {"available": False, "reason": "clustering model files not found"}

    return jsonify({
        "student_id": student_id,
        "years_provided": years_provided,
        "predictions": predictions,
    })


if __name__ == "__main__":
    print("Loading models from:", MODELS)
    if not MODELS.exists():
        print(f"WARNING: {MODELS} does not exist. Run ml_pipeline.py first to train and save models.")
    print("Loading advisor knowledge base from:", KNOWLEDGE_FILE)
    kb = load_knowledge()
    print(f"Advisor knowledge base: {len(kb)} characters")
    if not ADVISOR_KEY:
        print("WARNING: OPENROUTER_API_KEY is not set. /chat will return an error until it is.")
    # Cloud platforms (Render, Railway, etc.) inject the port to bind
    # to via the PORT environment variable - falls back to 5000 for
    # local runs, so nothing changes when running this on your own
    # machine. debug=True is also switched off outside local runs:
    # it's a real security risk in production (exposes a debugger
    # that can execute arbitrary code) and Flask's own dev server
    # warns against using it in production regardless.
    port = int(os.environ.get("PORT", 5000))
    is_production = "PORT" in os.environ
    # use_reloader=False: on this Windows setup, Flask's file-watcher
    # was spuriously flagging Python's own stdlib files (unittest,
    # argparse, fileinput...) as 'changed' - likely antivirus or
    # cloud-sync software touching file timestamps - and restarting
    # the server mid-request, which reset any in-flight connection
    # (confirmed cause of the ERR_CONNECTION_RESET on /predict).
    # We're not editing this file while it runs, so live-reload adds
    # no value here, only risk.
    # threaded=True: without this, Flask's dev server handles ONE
    # request at a time - while /chat is waiting on OpenRouter (which
    # can take several seconds), every other request (even a quick
    # /health check, or a second browser tab) blocks until that one
    # finishes. That's what looked like "the server needs restarting
    # after every question" - it wasn't crashing, it was just stuck
    # single-threaded behind the slow chat call. This lets it handle
    # several requests concurrently instead.
    app.run(host="0.0.0.0", port=port, debug=not is_production, use_reloader=False, threaded=True)
