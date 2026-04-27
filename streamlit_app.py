# -*- coding: utf-8 -*-
"""
NanoRelease-MCF7 Streamlit online prediction tool
Run locally:
    streamlit run streamlit_app_nanorelease.py
Deploy online:
    Upload this file, requirements.txt, best_model.pkl, and production_model_metadata.json to GitHub.
"""

from pathlib import Path
import json
import pickle
import math

import numpy as np
import pandas as pd
import streamlit as st

try:
    import tensorflow as tf
except Exception:
    tf = None

APP_TITLE = "NanoRelease-MCF7 AI"
APP_SUBTITLE = "Synthetic-polymeric anticancer nanoparticle release predictor for MCF-7 environment"
TARGET_NAME = "Drug release amount (%)"

LOCAL_WINDOWS_OUTPUT_DIR = Path(r"C:\Users\User\Desktop\PhD Projects\1.1st semester\AI Dataset\AI Dataset\6. Sixth dataset")
LOCAL_WINDOWS_REPORTS_DIR = LOCAL_WINDOWS_OUTPUT_DIR / "Reports"
APP_DIR = Path(__file__).resolve().parent

SEARCH_DIRS = [
    APP_DIR,
    APP_DIR / "Reports",
    APP_DIR / "model_artifacts",
    LOCAL_WINDOWS_REPORTS_DIR,
]

FEATURE_COLUMNS = [
    "size (DLS) of nanoparticle (nm)-mean",
    "Polydispersity Index (PDI) of nanoparticle-mean",
    "Zeta potential of nanoparticle (mV)-mean",
    "Drug loading capacity (%)-mean",
    "Entrapment efficiency of nanoparticle (%)-mean",
    "Temperature °C",
    "PH",
    "Time of Drug release (h)",
]

FEATURE_UI = {
    "size (DLS) of nanoparticle (nm)-mean": {
        "short": "Particle size by DLS",
        "unit": "nm",
        "help": "Enter the mean hydrodynamic size after drug loading.",
    },
    "Polydispersity Index (PDI) of nanoparticle-mean": {
        "short": "Polydispersity index",
        "unit": "PDI",
        "help": "Enter the mean PDI after drug loading.",
    },
    "Zeta potential of nanoparticle (mV)-mean": {
        "short": "Zeta potential",
        "unit": "mV",
        "help": "Enter the mean zeta potential after drug loading. Negative values are allowed.",
    },
    "Drug loading capacity (%)-mean": {
        "short": "Drug loading capacity",
        "unit": "%",
        "help": "Enter the mean drug loading capacity after drug loading.",
    },
    "Entrapment efficiency of nanoparticle (%)-mean": {
        "short": "Entrapment efficiency",
        "unit": "%",
        "help": "Enter the mean entrapment efficiency after drug loading.",
    },
    "Temperature °C": {
        "short": "Release temperature",
        "unit": "°C",
        "help": "Enter the release-study temperature.",
    },
    "PH": {
        "short": "Release medium pH",
        "unit": "pH",
        "help": "Enter the release-medium pH.",
    },
    "Time of Drug release (h)": {
        "short": "Release time",
        "unit": "h",
        "help": "Enter the drug-release time point.",
    },
}

SCIENTIFIC_NOTES = [
    "This application is intended specifically for the MCF-7 environment.",
    "The model is appropriate for synthetic polymeric nanoparticle systems and is not intended for natural or semi-synthetic polymer systems.",
    "All nanoparticle-related inputs should be entered using post-drug-loading values.",
    "Prediction reliability is highest when the entered values remain within the recommended range derived from the training dataset.",
    "This tool is intended for research support only and should not replace experimental validation.",
]

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --glass: rgba(255,255,255,0.78);
    --glass-strong: rgba(255,255,255,0.92);
    --ink: #102033;
    --muted: #5f6f85;
    --accent: #087f8c;
    --accent2: #b72175;
    --good: #0e8f67;
    --warn: #c47a00;
}
.stApp {
    background:
        radial-gradient(circle at 8% 8%, rgba(183,33,117,0.17), transparent 25%),
        radial-gradient(circle at 92% 16%, rgba(8,127,140,0.18), transparent 28%),
        radial-gradient(circle at 52% 95%, rgba(88,186,171,0.16), transparent 32%),
        linear-gradient(135deg, #eef7fb 0%, #f9fbff 45%, #f5eef8 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1320px;
}
.hero-card {
    border-radius: 32px;
    padding: 28px 34px;
    background: linear-gradient(145deg, rgba(7,21,38,0.95), rgba(16,50,77,0.88));
    color: white;
    box-shadow: 0 26px 80px rgba(20, 32, 54, 0.28);
    border: 1px solid rgba(255,255,255,0.22);
    margin-bottom: 20px;
}
.hero-title {
    font-size: 2.55rem;
    font-weight: 820;
    letter-spacing: -0.035em;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    color: #d8ecff;
    font-size: 1.08rem;
    line-height: 1.55;
    max-width: 720px;
    margin: 0;
}
.soft-card {
    border-radius: 24px;
    padding: 22px 24px;
    background: var(--glass);
    box-shadow: 0 15px 45px rgba(18, 38, 63, 0.10);
    border: 1px solid rgba(255,255,255,0.72);
    margin-bottom: 16px;
}
.note-card {
    border-radius: 24px;
    padding: 22px 24px;
    background: linear-gradient(145deg, rgba(255,255,255,0.86), rgba(242,250,252,0.82));
    border-left: 7px solid var(--accent);
    box-shadow: 0 15px 42px rgba(18, 38, 63, 0.10);
    margin-bottom: 18px;
}
.note-card h3, .soft-card h3 {
    margin-top: 0;
    color: var(--ink);
}
.note-card li {
    margin-bottom: 7px;
    color: #26374d;
    line-height: 1.55;
}
.range-chip {
    display: inline-block;
    border-radius: 999px;
    padding: 5px 11px;
    margin: 3px 6px 3px 0;
    font-size: 0.82rem;
    color: #124052;
    background: rgba(8,127,140,0.10);
    border: 1px solid rgba(8,127,140,0.18);
}
.reliability-high {
    border-radius: 18px;
    padding: 14px 18px;
    background: rgba(14,143,103,0.10);
    border: 1px solid rgba(14,143,103,0.22);
    color: #075b43;
    font-weight: 650;
}
.reliability-caution {
    border-radius: 18px;
    padding: 14px 18px;
    background: rgba(196,122,0,0.10);
    border: 1px solid rgba(196,122,0,0.25);
    color: #7a4b00;
    font-weight: 650;
}
.prediction-card {
    border-radius: 28px;
    padding: 28px;
    background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(240,250,250,0.92));
    box-shadow: 0 24px 70px rgba(18, 38, 63, 0.13);
    border: 1px solid rgba(255,255,255,0.78);
}
.big-number {
    font-size: 3.0rem;
    font-weight: 850;
    color: var(--accent2);
    letter-spacing: -0.04em;
    margin-bottom: 0;
}
.small-muted {
    color: var(--muted);
    font-size: 0.92rem;
}
.stNumberInput label, .stSlider label {
    font-weight: 680 !important;
    color: #21324a !important;
}
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.94), rgba(245,250,252,0.94));
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

HERO_HTML = """
<div class="hero-card">
  <div style="display:flex; gap:28px; align-items:center; justify-content:space-between; flex-wrap:wrap;">
    <div style="flex:1; min-width:310px;">
      <div class="hero-title">NanoRelease-MCF7 AI</div>
      <p class="hero-subtitle">
        A scientific decision-support interface for predicting anticancer drug release from synthetic polymeric nanoparticles in the MCF-7 environment.
      </p>
      <div style="margin-top:18px; display:flex; flex-wrap:wrap; gap:9px;">
        <span class="range-chip" style="background:rgba(118,225,212,0.16); color:#dffefa; border-color:rgba(118,225,212,0.35);">Post-drug-loading inputs</span>
        <span class="range-chip" style="background:rgba(255,139,196,0.15); color:#ffe3f2; border-color:rgba(255,139,196,0.30);">Synthetic polymeric systems</span>
        <span class="range-chip" style="background:rgba(207,231,255,0.13); color:#e8f5ff; border-color:rgba(207,231,255,0.25);">Dataset-bounded prediction</span>
      </div>
    </div>
    <div style="flex:0 0 360px; max-width:100%;">
      <svg width="360" height="230" viewBox="0 0 360 230" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <radialGradient id="tumor" cx="50%" cy="50%" r="58%">
            <stop offset="0%" stop-color="#ff9acc" stop-opacity="0.98"/>
            <stop offset="52%" stop-color="#c83282" stop-opacity="0.88"/>
            <stop offset="100%" stop-color="#5b1748" stop-opacity="0.72"/>
          </radialGradient>
          <radialGradient id="nano" cx="50%" cy="50%" r="52%">
            <stop offset="0%" stop-color="#ffffff"/>
            <stop offset="67%" stop-color="#76e1d4"/>
            <stop offset="100%" stop-color="#087f8c"/>
          </radialGradient>
          <filter id="glow"><feGaussianBlur stdDeviation="4" result="coloredBlur"/><feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
        </defs>
        <rect x="0" y="0" width="360" height="230" fill="rgba(255,255,255,0.03)" rx="28"/>
        <circle cx="218" cy="112" r="64" fill="url(#tumor)" filter="url(#glow)"/>
        <circle cx="190" cy="85" r="22" fill="rgba(255,255,255,0.15)"/>
        <circle cx="238" cy="125" r="18" fill="rgba(255,255,255,0.15)"/>
        <circle cx="214" cy="148" r="13" fill="rgba(255,255,255,0.12)"/>
        <path d="M36,70 C88,28 132,48 180,92" stroke="#76e1d4" stroke-width="3" fill="none" stroke-dasharray="7 7" opacity="0.95"/>
        <path d="M62,170 C104,128 138,132 180,124" stroke="#76e1d4" stroke-width="3" fill="none" stroke-dasharray="7 7" opacity="0.82"/>
        <path d="M286,58 C270,78 257,91 244,102" stroke="#ffb7dd" stroke-width="2.5" fill="none" stroke-dasharray="5 6" opacity="0.7"/>
        <circle cx="36" cy="70" r="15" fill="url(#nano)"/>
        <circle cx="62" cy="170" r="13" fill="url(#nano)"/>
        <circle cx="102" cy="48" r="10" fill="url(#nano)" opacity="0.92"/>
        <circle cx="120" cy="151" r="9" fill="url(#nano)" opacity="0.9"/>
        <circle cx="286" cy="58" r="12" fill="url(#nano)" opacity="0.88"/>
        <circle cx="302" cy="165" r="9" fill="url(#nano)" opacity="0.82"/>
        <text x="88" y="210" fill="#cfe7ff" font-size="13" font-family="Arial">Nanoparticle–tumor release interaction</text>
      </svg>
    </div>
  </div>
</div>
"""
st.markdown(HERO_HTML, unsafe_allow_html=True)


def find_existing_file(names):
    if isinstance(names, str):
        names = [names]
    candidates = []
    for directory in SEARCH_DIRS:
        for name in names:
            candidates.append(directory / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_artifact_path(path_or_name, fallback_names=None):
    names = []
    if path_or_name:
        path = Path(path_or_name)
        if path.exists():
            return path
        names.append(path.name)
    if fallback_names:
        if isinstance(fallback_names, str):
            names.append(fallback_names)
        else:
            names.extend(fallback_names)
    return find_existing_file(names)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    metadata_path = find_existing_file("production_model_metadata.json")
    metadata = None
    if metadata_path is not None:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if metadata is None:
        pickle_path = find_existing_file(["best_model.pkl", "best_traditional_model.pkl"])
        if pickle_path is None:
            raise FileNotFoundError(
                "Model files were not found. Upload best_model.pkl and production_model_metadata.json, "
                "or place them in a Reports or model_artifacts folder."
            )
        with open(pickle_path, "rb") as f:
            bundle = pickle.load(f)
        feature_stats = bundle.get("feature_descriptive_statistics")
        if not feature_stats:
            raise ValueError(
                "The pickle file does not contain feature descriptive statistics. "
                "Run train_nanorelease_models_v2.py first, then upload the new best_model.pkl and production_model_metadata.json."
            )
        return {
            "production_model_type": "sklearn_pipeline",
            "metadata": {"feature_descriptive_statistics": feature_stats},
            "bundle": bundle,
            "model_name": bundle.get("model_name", "Best trained ML model"),
            "artifact_source": str(pickle_path),
        }

    production_type = metadata.get("production_model_type", "sklearn_pipeline")
    model_files = metadata.get("model_files", {})
    feature_stats = metadata.get("feature_descriptive_statistics")

    if production_type == "sklearn_pipeline":
        pickle_path = resolve_artifact_path(model_files.get("production_pickle"), ["best_model.pkl", "best_traditional_model.pkl"])
        if pickle_path is None:
            raise FileNotFoundError("Could not find the sklearn production pickle file.")
        with open(pickle_path, "rb") as f:
            bundle = pickle.load(f)
        if not feature_stats:
            feature_stats = bundle.get("feature_descriptive_statistics")
        if not feature_stats:
            raise ValueError("Feature descriptive statistics are missing from metadata and pickle bundle.")
        return {
            "production_model_type": "sklearn_pipeline",
            "metadata": metadata,
            "bundle": bundle,
            "model_name": bundle.get("model_name", metadata.get("best_overall_model", "Best trained ML model")),
            "artifact_source": str(pickle_path),
        }

    if production_type == "keras_dnn":
        keras_path = resolve_artifact_path(model_files.get("production_keras"), "best_deep_learning_model.keras")
        preprocess_path = resolve_artifact_path(model_files.get("production_preprocessors_pickle"), "best_deep_learning_preprocessors.pkl")
        if tf is not None and keras_path is not None and preprocess_path is not None:
            model = tf.keras.models.load_model(keras_path)
            with open(preprocess_path, "rb") as f:
                preprocessors = pickle.load(f)
            if not feature_stats:
                feature_stats = preprocessors.get("feature_descriptive_statistics")
            if not feature_stats:
                raise ValueError("Feature descriptive statistics are missing from metadata and Keras preprocessors.")
            return {
                "production_model_type": "keras_dnn",
                "metadata": metadata,
                "model": model,
                "preprocessors": preprocessors,
                "model_name": preprocessors.get("model_name", metadata.get("best_overall_model", "Deep learning model")),
                "artifact_source": str(keras_path),
            }

        fallback_path = resolve_artifact_path(
            model_files.get("best_model_pickle_traditional_fallback"),
            ["best_model.pkl", "best_traditional_model.pkl"],
        )
        if fallback_path is None:
            raise FileNotFoundError(
                "The selected production model is Keras, but TensorFlow/Keras artifacts were not available, "
                "and no sklearn fallback pickle was found."
            )
        with open(fallback_path, "rb") as f:
            bundle = pickle.load(f)
        if not feature_stats:
            feature_stats = bundle.get("feature_descriptive_statistics")
        if not feature_stats:
            raise ValueError("Feature descriptive statistics are missing from metadata and fallback pickle.")
        return {
            "production_model_type": "sklearn_pipeline",
            "metadata": metadata,
            "bundle": bundle,
            "model_name": bundle.get("model_name", "Best traditional ML fallback model"),
            "artifact_source": str(fallback_path),
            "fallback_warning": "The metadata indicates that the best overall model was a Keras DNN, but this app is using the sklearn pickle fallback. Add TensorFlow and the Keras artifacts if you want the DNN online.",
        }

    raise ValueError(f"Unknown production_model_type: {production_type}")


def clean_float(value, fallback=0.0):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return float(fallback)


def get_feature_stats(artifacts):
    metadata = artifacts.get("metadata", {})
    stats = metadata.get("feature_descriptive_statistics", {})
    if not stats and "bundle" in artifacts:
        stats = artifacts["bundle"].get("feature_descriptive_statistics", {})
    if not stats and "preprocessors" in artifacts:
        stats = artifacts["preprocessors"].get("feature_descriptive_statistics", {})
    missing = [f for f in FEATURE_COLUMNS if f not in stats]
    if missing:
        raise ValueError("The following feature ranges are missing from the metadata: " + ", ".join(missing))
    return stats


def smart_step(min_value, max_value):
    span = abs(float(max_value) - float(min_value))
    if span <= 1:
        return 0.001, "%.4f"
    if span <= 20:
        return 0.01, "%.3f"
    if span <= 200:
        return 0.1, "%.2f"
    return 1.0, "%.2f"


def format_number(value):
    value = clean_float(value)
    if abs(value) >= 100:
        return f"{value:,.2f}"
    if abs(value) >= 10:
        return f"{value:,.3f}"
    return f"{value:,.4f}"


def render_notes():
    items = "".join([f"<li>{note}</li>" for note in SCIENTIFIC_NOTES])
    st.markdown(
        f"""
        <div class="note-card">
            <h3>Scientific Notes</h3>
            <ol>{items}</ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(artifacts, feature_stats):
    st.sidebar.title("Model information")
    st.sidebar.write(f"**Model used:** {artifacts.get('model_name', 'Trained model')}")
    st.sidebar.write(f"**Artifact type:** {artifacts.get('production_model_type', 'unknown')}")
    if artifacts.get("fallback_warning"):
        st.sidebar.warning(artifacts["fallback_warning"])
    metadata = artifacts.get("metadata", {})
    if metadata.get("created_at"):
        st.sidebar.write(f"**Training metadata created:** {metadata['created_at']}")
    st.sidebar.caption("Inputs are limited by observed min/max values from the training database. Reliability notes use the IQR-based recommended range.")
    with st.sidebar.expander("Input range policy", expanded=False):
        st.write(metadata.get(
            "input_range_policy",
            "Hard limits use observed training-dataset min/max. Recommended ranges use IQR-based descriptive statistics.",
        ))
    with st.sidebar.expander("Quick feature ranges", expanded=False):
        for feature in FEATURE_COLUMNS:
            stat = feature_stats[feature]
            ui = FEATURE_UI[feature]
            st.caption(
                f"{ui['short']}: {format_number(stat['input_limit_min'])} to {format_number(stat['input_limit_max'])} {ui['unit']}"
            )


def prediction_from_values(artifacts, values):
    X = np.array([values], dtype=float)
    if artifacts["production_model_type"] == "sklearn_pipeline":
        estimator = artifacts["bundle"]["estimator"]
        return float(estimator.predict(X)[0])
    preprocessors = artifacts["preprocessors"]
    model = artifacts["model"]
    X_i = preprocessors["x_imputer"].transform(X)
    X_s = preprocessors["x_scaler"].transform(X_i)
    pred_s = model.predict(X_s, verbose=0).reshape(-1, 1)
    pred = preprocessors["y_scaler"].inverse_transform(pred_s).ravel()[0]
    return float(pred)


def make_stats_table(feature_stats):
    rows = []
    for feature in FEATURE_COLUMNS:
        stat = feature_stats[feature]
        ui = FEATURE_UI[feature]
        rows.append({
            "Variable": ui["short"],
            "Unit": ui["unit"],
            "Mean": clean_float(stat.get("mean")),
            "SD": clean_float(stat.get("sd")),
            "Median": clean_float(stat.get("median")),
            "Q1": clean_float(stat.get("q1")),
            "Q3": clean_float(stat.get("q3")),
            "Observed Min": clean_float(stat.get("input_limit_min", stat.get("min"))),
            "Observed Max": clean_float(stat.get("input_limit_max", stat.get("max"))),
            "Recommended Min": clean_float(stat.get("recommended_min", stat.get("input_limit_min"))),
            "Recommended Max": clean_float(stat.get("recommended_max", stat.get("input_limit_max"))),
        })
    return pd.DataFrame(rows)


try:
    artifacts = load_artifacts()
    feature_stats = get_feature_stats(artifacts)
except Exception as exc:
    st.error("The online tool could not load the trained model or its descriptive-statistics metadata.")
    st.exception(exc)
    st.stop()

render_sidebar(artifacts, feature_stats)
render_notes()

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.markdown('<div class="soft-card"><h3>Enter nanoparticle and release-condition values</h3><p class="small-muted">Each input is bounded by the observed minimum and maximum in the training database. Median values are used as defaults.</p></div>', unsafe_allow_html=True)
    values = []
    reliability_flags = []
    display_rows = []
    cols = st.columns(2, gap="large")
    for i, feature in enumerate(FEATURE_COLUMNS):
        stat = feature_stats[feature]
        ui = FEATURE_UI[feature]
        min_value = clean_float(stat.get("input_limit_min", stat.get("min")))
        max_value = clean_float(stat.get("input_limit_max", stat.get("max")))
        recommended_min = clean_float(stat.get("recommended_min", min_value), min_value)
        recommended_max = clean_float(stat.get("recommended_max", max_value), max_value)
        default_value = clean_float(stat.get("default_value", stat.get("median", (min_value + max_value) / 2)), (min_value + max_value) / 2)
        default_value = min(max(default_value, min_value), max_value)
        step, number_format = smart_step(min_value, max_value)
        label = f"{ui['short']} ({ui['unit']})"
        with cols[i % 2]:
            value = st.number_input(
                label,
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(default_value),
                step=float(step),
                format=number_format,
                help=(
                    f"{ui['help']} Observed range: {format_number(min_value)} to {format_number(max_value)} {ui['unit']}. "
                    f"Recommended range: {format_number(recommended_min)} to {format_number(recommended_max)} {ui['unit']}."
                ),
                key=f"input_{i}",
            )
            st.markdown(
                f"<span class='range-chip'>Observed: {format_number(min_value)}–{format_number(max_value)} {ui['unit']}</span>"
                f"<span class='range-chip'>Recommended: {format_number(recommended_min)}–{format_number(recommended_max)} {ui['unit']}</span>",
                unsafe_allow_html=True,
            )
        values.append(float(value))
        within_recommended = (float(value) >= recommended_min) and (float(value) <= recommended_max)
        reliability_flags.append(within_recommended)
        display_rows.append({
            "Input variable": ui["short"],
            "Entered value": float(value),
            "Unit": ui["unit"],
            "Within recommended range": "Yes" if within_recommended else "No",
        })

with right:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("Prediction panel")
    st.caption("Click the button after confirming that values correspond to post-drug-loading nanoparticles.")
    predict_clicked = st.button("Predict drug release amount (%)", type="primary", use_container_width=True)
    if predict_clicked:
        try:
            pred = prediction_from_values(artifacts, values)
            st.markdown(f"<div class='big-number'>{pred:,.3f}%</div>", unsafe_allow_html=True)
            st.caption(f"Predicted {TARGET_NAME} using {artifacts.get('model_name', 'trained model')}.")
            progress_value = max(0.0, min(1.0, pred / 100.0))
            st.progress(progress_value)
            if all(reliability_flags):
                st.markdown(
                    "<div class='reliability-high'>Reliability status: high relative to the training domain. All values are within the IQR-based recommended ranges.</div>",
                    unsafe_allow_html=True,
                )
            else:
                outside = [FEATURE_UI[FEATURE_COLUMNS[i]]["short"] for i, flag in enumerate(reliability_flags) if not flag]
                st.markdown(
                    "<div class='reliability-caution'>Reliability status: caution. The input is still inside the observed training-dataset limits, "
                    "but at least one value is outside the IQR-based recommended range.</div>",
                    unsafe_allow_html=True,
                )
                st.warning("Outside recommended range: " + ", ".join(outside))
        except Exception as exc:
            st.error("Prediction failed. Please check the trained model files and input values.")
            st.exception(exc)
    else:
        st.info("Enter values on the left, then run the prediction.")
    st.markdown('</div>', unsafe_allow_html=True)

st.divider()

tab1, tab2, tab3 = st.tabs(["Entered values", "Training descriptive statistics", "Scientific scope"])

with tab1:
    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)

with tab2:
    stats_table = make_stats_table(feature_stats)
    st.dataframe(stats_table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download descriptive input ranges as CSV",
        data=stats_table.to_csv(index=False).encode("utf-8"),
        file_name="nanorelease_input_ranges.csv",
        mime="text/csv",
    )

with tab3:
    st.markdown("### Scientific Notes")
    for note in SCIENTIFIC_NOTES:
        st.markdown(f"- {note}")
    st.markdown(
        "The app prevents entries outside the observed training-dataset min/max range. "
        "Predictions inside the observed range but outside the IQR-based recommended range should be interpreted with extra caution."
    )
