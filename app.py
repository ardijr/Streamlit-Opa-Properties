import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import __main__

# ── Required custom function for unpickling the pipeline ─────────────────────
# The trained pipeline's ColumnTransformer references this function by name
# (it was defined at notebook/__main__ scope during training). It must exist
# in __main__ *before* joblib.load() is called, or unpickling will fail.
def to_string_array(x):
    """Cast a column (possibly mixed int/str, e.g. 0 mixed with 'A'/'D' codes)
    to a uniform string dtype before one-hot/ordinal encoding."""
    return x.astype(str)

__main__.to_string_array = to_string_array

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Philadelphia Property Value Predictor",
    page_icon="🏠",
    layout="wide",
)

MODEL_PATH = "xgboost_tuned_pipeline.pkl"
METADATA_PATH = "xgboost_tuned_pipeline_metadata.pkl"


@st.cache_resource
def load_model():
    pipeline = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipeline, metadata


pipeline, metadata = load_model()

ALL_FEATURES = metadata["all_features"]
NUM_FEATURES = metadata["num_features"]
CAT_OHE = metadata["cat_features_ohe"]
CAT_ORD = metadata["cat_features_ord"]

# Known category values learned by the model during training (for reference).
KNOWN_CATEGORIES = {
    "basements": ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Unknown"],
    "garage_type": ["0", "A", "B", "C", "F", "S", "T", "Unknown"],
    "general_construction": ["0", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "Unknown"],
    "topography": ["A", "B", "C", "D", "E", "F"],
    "view_type": ["0", "A", "B", "C", "D", "E", "H", "I"],
    "type_heater": ["0", "A", "B", "C", "D", "E", "G", "H", "Unknown"],
    "parcel_shape": ["A", "B", "C", "D", "E"],
    "building_era": [
        "Historic (<1900)", "Early 20th (1900-49)", "Mid-century (1950-79)",
        "Modern (1980-99)", "Contemporary (2000+)",
    ],
    "zoning": [
        "RSA1", "RSA2", "RSA3", "RSA4", "RSA5", "RSD1", "RSD2", "RSD3",
        "RM1", "RM2", "RM3", "RM4", "RTA1", "RMX1", "RMX2", "RMX3",
        "CA1", "CA2", "CMX1", "CMX2", "CMX2.5", "CMX3", "CMX4", "CMX5",
        "I1", "I2", "I3", "ICMX", "IP", "IRMX",
        "SPAIR", "SPINS", "SPPOA", "SPPOP", "SPSTA",
    ],
    "category_code": ["1", "2", "3", "4", "5", "6"],
}

# Example row used to build the downloadable CSV template.
EXAMPLE_ROW = {
    "log_total_livable_area": 7.31,
    "log_total_area": 7.60,
    "frontage": 16.0,
    "depth": 90.0,
    "livable_area_ratio": 0.83,
    "number_of_bathrooms": 2,
    "number_of_bedrooms": 3,
    "number_of_rooms": 7,
    "number_stories": 2,
    "bath_bed_ratio": 0.67,
    "fireplaces": 0,
    "garage_spaces": 1,
    "off_street_open": 0,
    "building_age": 45,
    "exterior_condition": 3,
    "interior_condition": 3,
    "geographic_ward": 21,
    "has_central_air": 1,
    "has_garage": 1,
    "has_fireplace": 0,
    "has_basement": 1,
    "sale_year": 2021,
    "sale_month": 6,
    "basements": "A",
    "garage_type": "B",
    "general_construction": "A",
    "topography": "A",
    "view_type": "0",
    "type_heater": "A",
    "parcel_shape": "C",
    "building_era": "Mid-century (1950-79)",
    "zoning": "RSA5",
    "category_code": "1",
}


def make_template_csv():
    df_template = pd.DataFrame([EXAMPLE_ROW])[ALL_FEATURES]
    buf = io.StringIO()
    df_template.to_csv(buf, index=False)
    return buf.getvalue()


# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏠 Philadelphia Property Value Predictor")
st.caption("Batch prediction — upload a CSV and get market value estimates for every row at once.")

col1, col2, col3 = st.columns(3)
col1.metric("Model", metadata["model_name"])
col2.metric("Test R²", f"{metadata['test_r2']:.3f}")
col3.metric("Test MAPE", f"{metadata['test_mape_pct']:.1f}%")

st.divider()

# ── Instructions + template ───────────────────────────────────────────────
with st.expander("📋 How to prepare your CSV (click to expand)", expanded=False):
    st.markdown(f"""
Your CSV needs these **{len(ALL_FEATURES)} columns**. Download the template below,
fill in your rows, and re-upload it.

**Numeric columns ({len(NUM_FEATURES)}):**
`{"`, `".join(NUM_FEATURES)}`

**Categorical columns ({len(CAT_OHE) + len(CAT_ORD)}):**
`{"`, `".join(CAT_OHE + CAT_ORD)}`

Notes:
- `log_total_livable_area` = `log(1 + total_livable_area)`, same for `log_total_area`.
- `has_central_air`, `has_garage`, `has_fireplace`, `has_basement` are `1`/`0` flags.
- `exterior_condition` / `interior_condition` are Philadelphia OPA condition codes (numeric).
- Empty/missing cells are fine — the model imputes them automatically.
- Unrecognized category values are also fine — the model treats them as "unknown".
    """)

    for col, cats in KNOWN_CATEGORIES.items():
        st.write(f"**{col}**: {', '.join(cats)}")

    st.download_button(
        "⬇️ Download CSV template",
        data=make_template_csv(),
        file_name="property_template.csv",
        mime="text/csv",
    )

# ── File upload & batch prediction ────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your properties CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read the CSV file: {e}")
        st.stop()

    st.write(f"**{len(df_input):,} rows** loaded.")
    st.dataframe(df_input.head(10), use_container_width=True)

    missing_cols = [c for c in ALL_FEATURES if c not in df_input.columns]
    if missing_cols:
        st.error(
            "Your CSV is missing these required columns:\n\n"
            + ", ".join(f"`{c}`" for c in missing_cols)
        )
        st.stop()

    if st.button("🚀 Predict all rows", type="primary"):
        with st.spinner(f"Predicting market value for {len(df_input):,} properties..."):
            df_pred = df_input.copy()

            # Coerce numeric feature columns; leave categoricals as-is (the
            # pipeline's own imputer + to_string_array step handles them).
            for col in NUM_FEATURES:
                df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

            X = df_pred[ALL_FEATURES]

            log_preds = pipeline.predict(X)
            preds_usd = np.expm1(log_preds)

            df_result = df_input.copy()
            df_result["predicted_market_value"] = preds_usd

        st.success(f"Done — predicted {len(df_result):,} properties.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total portfolio value", f"${preds_usd.sum():,.0f}")
        m2.metric("Average value", f"${preds_usd.mean():,.0f}")
        m3.metric("Median value", f"${np.median(preds_usd):,.0f}")
        m4.metric("Value range", f"${preds_usd.min():,.0f} – ${preds_usd.max():,.0f}")

        display_df = df_result.copy()
        display_df["predicted_market_value"] = display_df["predicted_market_value"].apply(
            lambda x: f"${x:,.0f}"
        )
        st.dataframe(display_df, use_container_width=True)

        csv_buf = io.StringIO()
        df_result.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download results as CSV",
            data=csv_buf.getvalue(),
            file_name="predicted_property_values.csv",
            mime="text/csv",
        )
else:
    st.info("👆 Upload a CSV file to get started, or grab the template above first.")
