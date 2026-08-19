# -*- coding: utf-8 -*-
"""
STUDENT RISK PREDICTION API
----------------------------

REST API wrapping the trained models from ml_pipeline.py.

This version also loads OPENROUTER_API_KEY automatically from:

    key.evn

The file must be beside this Python file:

    combined_app.py
    key.evn

Example key.evn:

    OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxx

RUN:
    pip install flask flask-cors pandas numpy scikit-learn joblib requests shap python-dotenv

    python combined_app.py

SERVER:
    http://127.0.0.1:5000
    http://localhost:5000

ENDPOINTS:
    GET  /
    GET  /health
    GET  /student/<student_id>
    POST /predict
    POST /chat
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path
import os
import re
import json

import joblib
import numpy as np
import pandas as pd
import requests
import shap

from flask import Flask, request, jsonify
from flask_cors import CORS

# dotenv is used only to load key.evn automatically
from dotenv import load_dotenv


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE = Path(__file__).resolve().parent


# ============================================================
# LOAD ENVIRONMENT FILE
# ============================================================
#
# IMPORTANT:
# The environment file is intentionally named:
#
#     key.evn
#
# NOT:
#
#     .env
#
# It must be in the same directory as combined_app.py.
#
# Example:
#
# combined_deploy_package/
#     combined_app.py
#     key.evn
#     knowledge_base.txt
#     ml_models/
#
# key.evn contents:
#
# OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
#
# ============================================================

ENV_FILE = BASE / "key.env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
    print(f"Environment file loaded: {ENV_FILE}")
else:
    print(f"WARNING: Environment file NOT FOUND: {ENV_FILE}")


# ============================================================
# OPENROUTER CONFIGURATION
# ============================================================

ADVISOR_KEY = os.environ.get("OPENROUTER_API_KEY", "")

ADVISOR_MODEL = os.environ.get(
    "ADVISOR_MODEL",
    "deepseek/deepseek-chat"
)


# ============================================================
# FLASK APP
# ============================================================

MODELS = BASE / "ml_models"

app = Flask(__name__)

# Allows Flutter / Android / browser clients to call the API.
CORS(app)


# ============================================================
# YEAR FIELDS
# ============================================================

YEAR_FIELDS = [
    "fall_gpa",
    "spring_gpa",
    "total_courses",
    "passed_courses",
    "failed_courses",
    "credits",
    "points",
    "avg_grade",
    "max_grade",
    "min_grade",
]


FIELD_TO_COLUMN = {
    "fall_gpa": "fall_gpa",
    "spring_gpa": "spring_gpa",
    "total_courses": "total_courses",
    "passed_courses": "passed_courses",
    "failed_courses": "failed_courses",
    "credits": "credits",
    "points": "points",
    "avg_grade": "avg_grade",
    "max_grade": "max_grade",
    "min_grade": "min_grade",
}


# ============================================================
# WARNING MODELS
# ============================================================

WARNING_MODELS = {
    1: (
        "warning_year2_model.joblib",
        "warned_in_year2",
        "years1",
        "well-validated (recall ~91% in testing)"
    ),

    2: (
        "warning_year3_model.joblib",
        "warned_in_year3",
        "years1-2",
        "well-validated (recall ~89% in testing)"
    ),

    3: (
        "warning_year4_model.joblib",
        "warned_in_year4",
        "years1-3",
        "validated (recall ~73% in testing)"
    ),
}


# ============================================================
# DELAYED PROGRESSION MODELS
# ============================================================

DELAYED_MODELS = {
    1: (
        "delayed_progression_cutoff1_model.joblib",
        "years1",
        "weak signal (recall ~19% in testing) - treat as a soft hint, not a verdict"
    ),

    2: (
        "delayed_progression_cutoff2_model.joblib",
        "years1-2",
        "weak signal (recall ~0% in testing at default threshold) - not reliable yet"
    ),

    3: (
        "delayed_progression_cutoff3_model.joblib",
        "years1-3",
        "moderate signal (PR-AUC 0.39, recall improves at lower thresholds) - use as a flag to review, not a verdict"
    ),
}


# ============================================================
# GPA MODELS
# ============================================================

GPA_MODELS = {
    1: (
        "gpa_predict_year2_model.joblib",
        "year2_avg_gpa",
        "years1",
        "moderate fit (typical error ~0.31 GPA points)"
    ),

    2: (
        "gpa_predict_year3_model.joblib",
        "year3_avg_gpa",
        "years1-2",
        "moderate fit (typical error ~0.31 GPA points)"
    ),

    3: (
        "gpa_predict_year4_model.joblib",
        "year4_avg_gpa",
        "years1-3",
        "good fit (typical error ~0.17 GPA points)"
    ),
}


# ============================================================
# PROMOTION MODEL
# ============================================================

PROMOTION_MODEL = (
    "promotion_year3_model.joblib",
    "years1",
    "well-validated (recall ~60%, ROC-AUC 0.89 in testing)"
)


# ============================================================
# GRADUATION MODEL
# ============================================================

GRADUATION_MODEL = (
    "graduation_status_model_bundled.joblib",
    "years1-3",
    "LOW CONFIDENCE - trained on only 27 non-graduate examples; even "
    "the best version of this model caught just 25% of actual "
    "non-graduates in testing. Treat as a rough indicator only, "
    "never as a decision on its own."
)


# ============================================================
# MODEL CACHE
# ============================================================

_loaded_models = {}


# ============================================================
# CLUSTERING CONFIGURATION
# ============================================================

CLUSTER_FEATURES_FILE = BASE / "cluster_features.json"

CLUSTER_PROFILES_FILE = BASE / "cluster_profiles.csv"


CLUSTER_DESCRIPTIONS = {
    0: (
        "أداء أعلى من المتوسط: معدل تراكمي أعلى، "
        "إنذارات أقل بكتير (0.11 مقابل 1.3 في المتوسط)، "
        "نسبة رسوب أقل من 1%."
    ),

    1: (
        "أداء أقل من المتوسط: معدل تراكمي أقل، "
        "إنذارات أكتر (1.3 في المتوسط)، "
        "نسبة رسوب أعلى (~4%)."
    ),
}


CLUSTER_LABELS = {
    0: "A",
    1: "B",
}


_cluster_features_cache = None

_cluster_profiles_cache = None


# ============================================================
# CLUSTER FEATURES
# ============================================================

def get_cluster_features():

    global _cluster_features_cache

    if _cluster_features_cache is None:

        if CLUSTER_FEATURES_FILE.exists():

            _cluster_features_cache = json.loads(
                CLUSTER_FEATURES_FILE.read_text(
                    encoding="utf-8"
                )
            )

        else:

            _cluster_features_cache = []

    return _cluster_features_cache


# ============================================================
# CLUSTER PROFILES
# ============================================================

def get_cluster_profiles():

    global _cluster_profiles_cache

    if (
        _cluster_profiles_cache is None
        and CLUSTER_PROFILES_FILE.exists()
    ):

        _cluster_profiles_cache = pd.read_csv(
            CLUSTER_PROFILES_FILE,
            encoding="utf-8-sig"
        ).set_index("cluster")

    return _cluster_profiles_cache


# ============================================================
# PREDICT CLUSTER
# ============================================================

def predict_cluster(df):

    """
    Returns:

        (
            cluster_id,
            cluster_label,
            confidence_note,
            description
        )

    or None if clustering artifacts are unavailable.
    """

    cluster_features = get_cluster_features()

    if not cluster_features:
        return None


    preprocessor = load_model_raw(
        "clustering_preprocessor.joblib"
    )

    kmeans = load_model_raw(
        "kmeans_model.joblib"
    )


    if preprocessor is None or kmeans is None:
        return None


    row = {
        col: df[col].iloc[0]
        if col in df.columns
        else None
        for col in cluster_features
    }


    X = pd.DataFrame([row])[cluster_features]


    known_fraction = (
        X.notna().sum(axis=1).iloc[0]
        / len(cluster_features)
    )


    Xt = preprocessor.transform(X)


    cluster_id = int(
        kmeans.predict(Xt)[0]
    )


    cluster_label = CLUSTER_LABELS.get(
        cluster_id,
        str(cluster_id)
    )


    if known_fraction >= 0.7:

        confidence = (
            f"موثوق نسبياً "
            f"({known_fraction*100:.0f}% من البيانات معروفة، "
            f"الباقي بمتوسط الفوج)"
        )

    elif known_fraction >= 0.3:

        confidence = (
            f"تقريبي "
            f"({known_fraction*100:.0f}% من البيانات معروفة بس) "
            f"- مؤشر أولي وليس تصنيفاً نهائياً"
        )

    else:

        confidence = (
            f"ضعيف جداً "
            f"({known_fraction*100:.0f}% من البيانات معروفة) "
            f"- معظم التصنيف مبني على متوسط الفوج "
            f"لا بيانات الطالب"
        )


    return (
        cluster_id,
        cluster_label,
        confidence,
        CLUSTER_DESCRIPTIONS.get(
            cluster_id,
            "غير معروف"
        )
    )


# ============================================================
# EXTRACT TARGET YEAR
# ============================================================

def extract_target_year(predicts_str):

    """
    Example:

        warned_in_year2 -> 2

        not_reaching_year3_on_schedule -> 3
    """

    m = re.search(
        r"year(\d)",
        predicts_str
    )

    return int(m.group(1)) if m else None


# ============================================================
# LOAD NORMAL MODEL
# ============================================================

def load_model(filename):

    if filename not in _loaded_models:

        path = MODELS / filename

        if not path.exists():
            return None

        _loaded_models[filename] = joblib.load(path)

    return _loaded_models[filename]


# ============================================================
# LOAD RAW MODEL
# ============================================================

def load_model_raw(filename):

    """
    Used for clustering artifacts saved as bare sklearn objects.
    """

    key = f"raw:{filename}"

    if key not in _loaded_models:

        path = MODELS / filename

        if not path.exists():
            return None

        _loaded_models[key] = joblib.load(path)

    return _loaded_models[key]


# ============================================================
# BUILD FEATURE ROW
# ============================================================

def build_feature_row(payload):

    """
    Turns request JSON into the engineered feature shape
    expected by the trained models.
    """

    row = {}

    years_provided = []


    for year_num in [1, 2, 3]:

        key = f"year{year_num}"

        if (
            key not in payload
            or not isinstance(payload[key], dict)
        ):
            continue


        years_provided.append(key)

        y = payload[key]


        for field in YEAR_FIELDS:

            col = (
                f"year{year_num}_"
                f"{FIELD_TO_COLUMN[field]}"
            )

            row[col] = y.get(field)


    df = pd.DataFrame([row])


    # --------------------------------------------------------
    # YEAR AVERAGE GPA
    # --------------------------------------------------------

    for year in [1, 2, 3]:

        cols = [
            c
            for c in [
                f"year{year}_fall_gpa",
                f"year{year}_spring_gpa"
            ]
            if c in df.columns
        ]


        if cols:

            df[f"year{year}_avg_gpa"] = (
                df[cols].mean(axis=1)
            )


    # --------------------------------------------------------
    # GPA TREND
    # --------------------------------------------------------

    for a, b in [(1, 2), (2, 3)]:

        x = f"year{a}_avg_gpa"

        y_ = f"year{b}_avg_gpa"


        if x in df.columns and y_ in df.columns:

            df[
                f"gpa_trend_y{a}_y{b}"
            ] = (
                df[y_] - df[x]
            )


    return df, years_provided


# ============================================================
# ARABIC FEATURE NAMES
# ============================================================

FEATURE_NAME_AR = {

    "fall_gpa":
        "معدل الفصل الأول",

    "spring_gpa":
        "معدل الفصل الثاني",

    "total_courses":
        "عدد المقررات الكلي",

    "passed_courses":
        "عدد المقررات الناجحة",

    "failed_courses":
        "عدد المقررات الراسبة",

    "credits":
        "الساعات المعتمدة",

    "points":
        "مجموع النقاط",

    "avg_grade":
        "متوسط الدرجات",

    "max_grade":
        "أعلى درجة",

    "min_grade":
        "أقل درجة",

    "avg_gpa":
        "متوسط المعدل التراكمي للسنة",
}


# ============================================================
# READABLE FEATURE NAME
# ============================================================

def readable_feature_name(col):

    """
    Converts model feature names into Arabic descriptions.
    """

    if col.startswith("gpa_trend_y"):

        try:

            a, b = (
                col
                .replace(
                    "gpa_trend_y",
                    ""
                )
                .split("_y")
            )

            return (
                f"اتجاه تغيّر المعدل "
                f"من سنة {a} إلى سنة {b}"
            )

        except Exception:

            return col


    if col.startswith("year"):

        year_num, _, rest = col.partition("_")

        year_num = year_num.replace(
            "year",
            ""
        )

        ar = FEATURE_NAME_AR.get(rest)

        if ar:

            return (
                f"{ar} "
                f"(سنة {year_num})"
            )


    return col


# ============================================================
# SHAP EXPLANATION
# ============================================================

def explain_prediction(
    model_bundle,
    df,
    is_regression=False,
    top_n=4,
    positive_label="increases_risk",
    negative_label="decreases_risk"
):

    pipe = model_bundle["pipeline"]

    features = model_bundle["features"]

    X = df[features]


    X_transformed = X.copy()


    for step_name in [
        "imputer",
        "scaler"
    ]:

        X_transformed = (
            pipe.named_steps[
                step_name
            ].transform(
                X_transformed
            )
        )


    model = pipe.named_steps["model"]


    explainer = shap.TreeExplainer(model)


    sv = explainer.shap_values(
        X_transformed
    )


    if is_regression:

        contributions = np.array(sv)[0]

    else:

        contributions = np.array(
            sv
        )[0, :, 1]


    all_ranked = sorted(
        zip(
            features,
            contributions,
            X.iloc[0].tolist()
        ),
        key=lambda t: abs(t[1]),
        reverse=True
    )


    def year_of(col):

        for y in (1, 2, 3):

            if col.startswith(
                f"year{y}"
            ):
                return y

        return None


    years_present = sorted(
        {
            year_of(f)
            for f in features
            if year_of(f)
        }
    )


    selected = list(
        all_ranked[:top_n]
    )


    covered_years = {
        year_of(c)
        for c, _, _ in selected
    }


    for year in years_present:

        if year in covered_years:
            continue


        best_for_year = max(
            (
                t
                for t in all_ranked
                if year_of(t[0]) == year
            ),
            key=lambda t: abs(t[1]),
            default=None
        )


        if best_for_year:

            selected.append(
                best_for_year
            )

            covered_years.add(year)


    factors = []


    for (
        col,
        contribution,
        raw_value
    ) in selected:

        if is_regression:

            effect = (
                "pushes_prediction_up"
                if contribution > 0
                else
                "pushes_prediction_down"
            )

        else:

            effect = (
                positive_label
                if contribution > 0
                else negative_label
            )


        try:

            student_value = round(
                float(raw_value),
                3
            )

        except Exception:

            student_value = raw_value


        factors.append({

            "factor":
                readable_feature_name(col),

            "student_value":
                student_value,

            "effect":
                effect,

            "impact":
                round(
                    float(
                        abs(contribution)
                    ),
                    4
                ),
        })


    return factors


# ============================================================
# PREDICT WITH MODEL
# ============================================================

def predict_with_model(
    model_bundle,
    df
):

    pipe = model_bundle["pipeline"]

    needed = model_bundle["features"]


    missing = [
        f
        for f in needed
        if f not in df.columns
    ]


    if missing:

        return None, missing


    X = df[needed]


    if hasattr(
        pipe,
        "predict_proba"
    ):

        prob = float(
            pipe.predict_proba(X)[:, 1][0]
        )

        return prob, None


    else:

        val = float(
            pipe.predict(X)[0]
        )

        return val, None


# ============================================================
# STUDENT DATA
# ============================================================

STUDENT_DATA_DIR = (
    BASE / "student_data"
)


_student_lookup = None


# ============================================================
# BUILD STUDENT LOOKUP
# ============================================================

def build_student_lookup():

    """
    Loads stored student records from CSV files.
    """

    lookup = {}


    sources = [

        STUDENT_DATA_DIR
        / "student_ml_dataset.csv",

        STUDENT_DATA_DIR
        / "promotion_cohort_dataset.csv",

        STUDENT_DATA_DIR
        / "student_ml_dataset_anchor_y3.csv",
    ]


    for path in sources:

        if not path.exists():
            continue


        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={
                "student_id": str
            }
        )


        for _, row in df.iterrows():

            sid = str(
                row["student_id"]
            ).strip()


            if sid in lookup:
                continue


            record = {}


            for year_num in [
                1,
                2,
                3
            ]:

                year_data = {}


                for (
                    field,
                    col_suffix
                ) in FIELD_TO_COLUMN.items():

                    col = (
                        f"year{year_num}_"
                        f"{col_suffix}"
                    )


                    if (
                        col in row
                        and pd.notna(row[col])
                    ):

                        value = row[col]


                        if isinstance(
                            value,
                            str
                        ):

                            year_data[
                                field
                            ] = value

                        else:

                            year_data[
                                field
                            ] = float(value)


                if year_data:

                    record[
                        f"year{year_num}"
                    ] = year_data


            if record:

                lookup[sid] = record


    return lookup


# ============================================================
# GET STUDENT LOOKUP
# ============================================================

def get_student_lookup():

    global _student_lookup


    if _student_lookup is None:

        _student_lookup = (
            build_student_lookup()
        )


        print(
            f"Student lookup loaded: "
            f"{len(_student_lookup)} students "
            f"from {STUDENT_DATA_DIR}"
        )


    return _student_lookup


# ============================================================
# STUDENT ENDPOINT
# ============================================================

@app.route(
    "/student/<student_id>",
    methods=["GET"]
)
def get_student(student_id):

    lookup = get_student_lookup()


    record = lookup.get(
        str(student_id).strip()
    )


    if record is None:

        return jsonify({

            "error":
                f"Student {student_id} "
                f"not found in stored records.",

            "student_id":
                student_id,

        }), 404


    return jsonify({

        "student_id":
            student_id,

        **record,

    })


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def root():

    return jsonify({

        "message":
            "Student Risk Prediction API is running.",

        "endpoints": {

            "GET /health":
                "check server + models status",

            "POST /predict":
                "send student data, get risk predictions back",

            "POST /chat":
                "AI Academic Advisor",

            "GET /student/<student_id>":
                "get stored student data",
        },
    })


# ============================================================
# ADVISOR
# ============================================================

KNOWLEDGE_FILE = (
    BASE / "knowledge_base.txt"
)


_knowledge_cache = None


# ============================================================
# LOAD KNOWLEDGE
# ============================================================

def load_knowledge():

    global _knowledge_cache


    if _knowledge_cache is None:

        if KNOWLEDGE_FILE.exists():

            _knowledge_cache = (
                KNOWLEDGE_FILE.read_text(
                    encoding="utf-8"
                )
            )

        else:

            _knowledge_cache = ""

            print(
                "WARNING: "
                f"{KNOWLEDGE_FILE} not found - "
                "advisor will have no curriculum knowledge."
            )


    return _knowledge_cache


# ============================================================
# GPA ANALYSIS
# ============================================================

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


# ============================================================
# BUILD RISK CONTEXT
# ============================================================

def build_risk_context_for_advisor(
    student_id,
    submitted_years=None
):

    lookup = get_student_lookup()


    record = lookup.get(
        str(student_id).strip()
    )


    if record is None:
        return None


    df, years_provided = (
        build_feature_row(
            {
                "student_id":
                    student_id,

                **(
                    submitted_years
                    or record
                )
            }
        )
    )


    if "year1" not in years_provided:
        return None


    max_year = len(
        years_provided
    )


    lines = [
        f"بيانات الطالب الحقيقية "
        f"(كود {student_id}) "
        f"من نظام التنبؤ الأكاديمي:"
    ]


    # --------------------------------------------------------
    # WARNING
    # --------------------------------------------------------

    fname, predicts, based_on, note = (
        WARNING_MODELS[max_year]
    )


    model = load_model(fname)


    if model:

        prob, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if not missing:

            lines.append(
                f"- خطر الإنذار الأكاديمي "
                f"في سنة "
                f"{extract_target_year(predicts)}: "
                f"{prob*100:.0f}%"
            )


            factors = (
                explain_prediction(
                    model,
                    df
                )[:2]
            )


            if factors:

                txt = "، ".join(
                    f"{f['factor']} "
                    f"({f['student_value']})"
                    for f in factors
                )


                lines.append(
                    f"  أهم العوامل: {txt}"
                )


    # --------------------------------------------------------
    # PROMOTION
    # --------------------------------------------------------

    fname, based_on, note = (
        PROMOTION_MODEL
    )


    model = load_model(fname)


    if model:

        prob, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if not missing:

            lines.append(
                f"- خطر عدم الترقية "
                f"لسنة 3: "
                f"{prob*100:.0f}%"
            )


    # --------------------------------------------------------
    # GPA
    # --------------------------------------------------------

    fname, predicts, based_on, note = (
        GPA_MODELS[max_year]
    )


    model = load_model(fname)


    if model:

        val, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if not missing:

            lines.append(
                f"- توقع معدل السنة القادمة: "
                f"{val:.2f}"
            )


    if len(lines) == 1:
        return None


    return "\n".join(lines)


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    available = (
        [
            f.name
            for f in MODELS.glob(
                "*.joblib"
            )
        ]
        if MODELS.exists()
        else []
    )


    knowledge = load_knowledge()


    return jsonify({

        "status":
            "ok"
            if available
            else "no models found",

        "models_available":
            available,

        "advisor_knowledge_loaded":
            len(knowledge) > 0,

        "advisor_knowledge_chars":
            len(knowledge),

        "advisor_api_key_configured":
            bool(ADVISOR_KEY),

        "clustering_available":
            bool(
                get_cluster_features()
            )
            and CLUSTER_PROFILES_FILE.exists(),

    })


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if not ADVISOR_KEY:

        return jsonify({

            "error":
                "OPENROUTER_API_KEY is not set. "
                "Make sure key.evn exists beside "
                "combined_app.py and contains "
                "OPENROUTER_API_KEY=your_key"

        }), 500


    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    payload = request.get_json(
        force=True,
        silent=True
    )


    if not payload:

        return jsonify({
            "error":
                "Request body must be JSON."
        }), 400


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    gpa = payload.get(
        "gpa"
    )


    student_id = payload.get(
        "student_id"
    )


    history = payload.get(
        "history",
        []
    )


    user_message = payload.get(
        "message",
        ""
    ).strip()


    if not user_message:

        return jsonify({

            "error":
                "'message' is required."

        }), 400


    # --------------------------------------------------------
    # LEVEL
    # --------------------------------------------------------

    level = (
        analyze_gpa(
            float(gpa)
        )
        if gpa is not None
        else None
    )


    # --------------------------------------------------------
    # SUBMITTED YEARS
    # --------------------------------------------------------

    submitted_years = {
        k: payload[k]
        for k in (
            "year1",
            "year2",
            "year3"
        )
        if k in payload
    }


    # --------------------------------------------------------
    # RISK CONTEXT
    # --------------------------------------------------------

    risk_context = (

        build_risk_context_for_advisor(
            student_id,
            submitted_years or None
        )

        if student_id

        else None
    )


    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    system_prompt = """
أنت AI Academic Advisor داخل جامعة.

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

        system_prompt += (
            f"\nبيانات الطالب: "
            f"GPA={gpa}, "
            f"المستوى={level}"
        )


    # ========================================================
    # RISK CONTEXT
    # ========================================================

    if risk_context:

        system_prompt += (
            f"\n\n{risk_context}\n"
        )


        system_prompt += (
            "استخدم هذه البيانات لتخصيص "
            "نصائحك عند مناسبة السياق "
            "(مثلاً لو الطالب سأل عن كيفية "
            "تحسين أدائه) - لا تذكرها إلا إذا "
            "كانت مفيدة فعلاً للسؤال المطروح."
        )


    # ========================================================
    # KNOWLEDGE BASE
    # ========================================================

    system_prompt += (
        f"\n\nمرجع اللائحة والخطة الدراسية:\n"
        f"{load_knowledge()[:12000]}"
    )


    # ========================================================
    # MESSAGES
    # ========================================================

    messages = [

        {
            "role":
                "system",

            "content":
                system_prompt
        }

    ]


    messages.extend(
        history
    )


    messages.append({

        "role":
            "user",

        "content":
            user_message

    })


    # ========================================================
    # OPENROUTER REQUEST
    # ========================================================

    try:

        response = requests.post(

            "https://openrouter.ai/api/v1/chat/completions",

            headers={

                "Authorization":
                    f"Bearer {ADVISOR_KEY}",

                "Content-Type":
                    "application/json",

            },

            json={

                "model":
                    ADVISOR_MODEL,

                "messages":
                    messages,

            },

            timeout=60,
        )


    except requests.exceptions.RequestException as e:

        return jsonify({

            "error":
                f"Failed to reach OpenRouter: {e}"

        }), 502


    # ========================================================
    # OPENROUTER ERROR
    # ========================================================

    if response.status_code != 200:

        return jsonify({

            "error":
                "OpenRouter API request failed "
                f"with status {response.status_code}",

            "details":
                response.text,

        }), 502


    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    response_json = response.json()


    choices = response_json.get(
        "choices",
        []
    )


    if not choices:

        return jsonify({

            "error":
                "Unexpected API response structure.",

            "details":
                response_json,

        }), 502


    # ========================================================
    # RETURN CHAT RESPONSE
    # ========================================================

    return jsonify({

        "reply":
            choices[0]["message"]["content"],

        "level":
            level,

        "risk_context_used":
            risk_context is not None,

    })


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # READ JSON
    # --------------------------------------------------------

    payload = request.get_json(
        force=True,
        silent=True
    )


    if not payload:

        return jsonify({

            "error":
                "Request body must be JSON."

        }), 400


    # --------------------------------------------------------
    # STUDENT ID
    # --------------------------------------------------------

    student_id = payload.get(
        "student_id",
        "unknown"
    )


    # --------------------------------------------------------
    # BUILD FEATURES
    # --------------------------------------------------------

    df, years_provided = (
        build_feature_row(payload)
    )


    # --------------------------------------------------------
    # YEAR 1 REQUIRED
    # --------------------------------------------------------

    if "year1" not in years_provided:

        return jsonify({

            "error":
                "year1 data is required "
                "(minimum). year2/year3 are "
                "optional and sharpen the predictions."

        }), 400


    # --------------------------------------------------------
    # MAX YEAR
    # --------------------------------------------------------

    max_year = len(
        years_provided
    )


    predictions = {}


    # ========================================================
    # WARNING RISK
    # ========================================================

    fname, predicts, based_on, note = (
        WARNING_MODELS[max_year]
    )


    model = load_model(fname)


    if model:

        prob, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if missing:

            predictions[
                "warning_risk"
            ] = {

                "available":
                    False,

                "reason":
                    f"missing fields: {missing}"

            }


        else:

            predictions[
                "warning_risk"
            ] = {

                "available":
                    True,

                "probability":
                    round(prob, 4),

                "predicts":
                    predicts,

                "target_year":
                    extract_target_year(
                        predicts
                    ),

                "based_on":
                    based_on,

                "confidence_note":
                    note,

                "explanation":
                    explain_prediction(
                        model,
                        df
                    ),
            }


    else:

        predictions[
            "warning_risk"
        ] = {

            "available":
                False,

            "reason":
                f"model file not found: {fname}"

        }


    # ========================================================
    # NOT PROMOTED RISK
    # ========================================================

    fname, based_on, note = (
        PROMOTION_MODEL
    )


    model = load_model(fname)


    if model:

        prob, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if missing:

            predictions[
                "not_promoted_risk"
            ] = {

                "available":
                    False,

                "reason":
                    f"missing fields: {missing}"

            }


        else:

            predictions[
                "not_promoted_risk"
            ] = {

                "available":
                    True,

                "probability":
                    round(prob, 4),

                "predicts":
                    "not_reaching_year3_on_schedule",

                "target_year":
                    3,

                "based_on":
                    based_on,

                "confidence_note":
                    note,

                "explanation":
                    explain_prediction(
                        model,
                        df
                    ),
            }


    else:

        predictions[
            "not_promoted_risk"
        ] = {

            "available":
                False,

            "reason":
                f"model file not found: {fname}"

        }


    # ========================================================
    # DELAYED PROGRESSION
    # ========================================================

    fname, based_on, note = (
        DELAYED_MODELS[max_year]
    )


    model = load_model(fname)


    if model:

        prob, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if missing:

            predictions[
                "delayed_progression_risk"
            ] = {

                "available":
                    False,

                "reason":
                    f"missing fields: {missing}"

            }


        else:

            predictions[
                "delayed_progression_risk"
            ] = {

                "available":
                    True,

                "probability":
                    round(prob, 4),

                "predicts":
                    "not_reaching_year4_on_schedule",

                "target_year":
                    4,

                "based_on":
                    based_on,

                "confidence_note":
                    note,

                "explanation":
                    explain_prediction(
                        model,
                        df
                    ),
            }


    else:

        predictions[
            "delayed_progression_risk"
        ] = {

            "available":
                False,

            "reason":
                f"model file not found: {fname}"

        }


    # ========================================================
    # NEXT YEAR GPA
    # ========================================================

    fname, predicts, based_on, note = (
        GPA_MODELS[max_year]
    )


    model = load_model(fname)


    if model:

        val, missing = (
            predict_with_model(
                model,
                df
            )
        )


        if missing:

            predictions[
                "predicted_next_year_gpa"
            ] = {

                "available":
                    False,

                "reason":
                    f"missing fields: {missing}"

            }


        else:

            predictions[
                "predicted_next_year_gpa"
            ] = {

                "available":
                    True,

                "value":
                    round(val, 3),

                "predicts":
                    predicts,

                "target_year":
                    extract_target_year(
                        predicts
                    ),

                "based_on":
                    based_on,

                "confidence_note":
                    note,

                "explanation":
                    explain_prediction(
                        model,
                        df,
                        is_regression=True
                    ),
            }


    else:

        predictions[
            "predicted_next_year_gpa"
        ] = {

            "available":
                False,

            "reason":
                f"model file not found: {fname}"

        }


    # ========================================================
    # GRADUATION PROBABILITY
    # ========================================================

    if max_year == 3:

        fname, based_on, note = (
            GRADUATION_MODEL
        )


        model = load_model(fname)


        if model:

            prob, missing = (
                predict_with_model(
                    model,
                    df
                )
            )


            if missing:

                predictions[
                    "graduation_probability"
                ] = {

                    "available":
                        False,

                    "reason":
                        f"missing fields: {missing}"

                }


            else:

                predictions[
                    "graduation_probability"
                ] = {

                    "available":
                        True,

                    "probability":
                        round(prob, 4),

                    "predicts":
                        "graduates_on_schedule",

                    "target_year":
                        4,

                    "based_on":
                        based_on,

                    "confidence_note":
                        note,

                    "explanation":
                        explain_prediction(

                            model,

                            df,

                            positive_label=
                                "increases_graduation_probability",

                            negative_label=
                                "decreases_graduation_probability",
                        ),
                }


        else:

            predictions[
                "graduation_probability"
            ] = {

                "available":
                    False,

                "reason":
                    f"model file not found: {fname}"

            }


    else:

        predictions[
            "graduation_probability"
        ] = {

            "available":
                False,

            "reason":
                "needs year1, year2, AND year3 data "
                "(this model was only trained at that exact stage)",

        }


    # ========================================================
    # PEER CLUSTER
    # ========================================================

    cluster_result = predict_cluster(
        df
    )


    if cluster_result:

        (
            cluster_id,
            cluster_label,
            confidence,
            description
        ) = cluster_result


        predictions[
            "peer_cluster"
        ] = {

            "available":
                True,

            "cluster_id":
                cluster_id,

            "cluster_label":
                cluster_label,

            "description":
                description,

            "confidence_note":
                confidence,

        }


    else:

        predictions[
            "peer_cluster"
        ] = {

            "available":
                False,

            "reason":
                "clustering model files not found"

        }


    # ========================================================
    # RETURN PREDICTION
    # ========================================================

    return jsonify({

        "student_id":
            student_id,

        "years_provided":
            years_provided,

        "predictions":
            predictions,

    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print(" STUDENT RISK PREDICTION API")
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    print(
        "Loading models from:",
        MODELS
    )


    if not MODELS.exists():

        print(
            "WARNING: "
            f"{MODELS} does not exist. "
            "Run ml_pipeline.py first to train "
            "and save models."
        )


    # --------------------------------------------------------
    # ENVIRONMENT FILE
    # --------------------------------------------------------

    print()

    print(
        "Environment file:",
        ENV_FILE
    )

    print(
        "Environment file exists:",
        ENV_FILE.exists()
    )


    # --------------------------------------------------------
    # API KEY
    # --------------------------------------------------------

    print()

    print(
        "OPENROUTER_API_KEY configured:",
        bool(ADVISOR_KEY)
    )


    if not ADVISOR_KEY:

        print()

        print(
            "WARNING: OPENROUTER_API_KEY is NOT SET."
        )

        print(
            "Make sure you have a file named:"
        )

        print(
            f"    {ENV_FILE}"
        )

        print(
            "with:"
        )

        print(
            "    OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx"
        )


    # --------------------------------------------------------
    # ADVISOR MODEL
    # --------------------------------------------------------

    print()

    print(
        "Advisor model:",
        ADVISOR_MODEL
    )


    # --------------------------------------------------------
    # KNOWLEDGE BASE
    # --------------------------------------------------------

    print()

    print(
        "Loading advisor knowledge base from:",
        KNOWLEDGE_FILE
    )


    kb = load_knowledge()


    print(
        f"Advisor knowledge base: "
        f"{len(kb)} characters"
    )


    # --------------------------------------------------------
    # SERVER
    # --------------------------------------------------------

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )


    is_production = (
        "PORT" in os.environ
    )


    print()

    print(
        "Server will start on:"
    )

    print(
        "    http://127.0.0.1:5000"
    )

    print(
        "    http://localhost:5000"
    )

    print()

    print("=" * 60)
    print()


    # --------------------------------------------------------
    # START FLASK
    # --------------------------------------------------------

    app.run(

        host="0.0.0.0",

        port=port,

        debug=not is_production,

        use_reloader=False,

        threaded=True
    )