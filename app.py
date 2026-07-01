import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import __main__

# ── Required custom function for unpickling the pipeline ─────────────────────
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
st.caption("Predict property market values — upload a CSV for batch prediction or fill in fields manually.")

col1, col2, col3 = st.columns(3)
col1.metric("Model", metadata["model_name"])
col2.metric("Test R²", f"{metadata['test_r2']:.3f}")
col3.metric("Test MAPE", f"{metadata['test_mape_pct']:.1f}%")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_csv, tab_manual = st.tabs(["📂 Batch CSV Upload", "✏️ Manual Input"])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — BATCH CSV UPLOAD
# ═══════════════════════════════════════════════════════════════════════════
with tab_csv:
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

        if st.button("🚀 Predict all rows", type="primary", key="btn_csv"):
            with st.spinner(f"Predicting market value for {len(df_input):,} properties..."):
                df_pred = df_input.copy()
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


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — MANUAL INPUT
# ═══════════════════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("Fill in property details")
    st.caption("All fields are pre-filled with example values. Adjust any field and click Predict.")

    # ── Section: Area & Lot ────────────────────────────────────────────────
    with st.expander("📐 Area & Lot", expanded=True):
        c1, c2, c3 = st.columns(3)
        total_livable_area = c1.number_input(
            "Total Livable Area (sqft)", min_value=1, value=1500,
            help="Raw livable area — will be log-transformed automatically."
        )
        total_area = c2.number_input(
            "Total Area (sqft)", min_value=1, value=1998,
            help="Total parcel area — will be log-transformed automatically."
        )
        frontage = c3.number_input("Frontage (ft)", min_value=0.0, value=16.0)

        c4, c5, c6 = st.columns(3)
        depth = c4.number_input("Depth (ft)", min_value=0.0, value=90.0)
        livable_area_ratio = c5.number_input(
            "Livable Area Ratio", min_value=0.0, max_value=1.0, value=0.83, step=0.01,
            help="livable_area / total_area"
        )
        parcel_shape = c6.selectbox(
            "Parcel Shape", KNOWN_CATEGORIES["parcel_shape"],
            index=KNOWN_CATEGORIES["parcel_shape"].index("C")
        )
        topography = st.selectbox(
            "Topography", KNOWN_CATEGORIES["topography"],
            index=KNOWN_CATEGORIES["topography"].index("A")
        )

    # ── Section: Rooms & Layout ────────────────────────────────────────────
    with st.expander("🛏️ Rooms & Layout", expanded=True):
        c1, c2, c3 = st.columns(3)
        number_of_bathrooms = c1.number_input("Bathrooms", min_value=0, value=2)
        number_of_bedrooms  = c2.number_input("Bedrooms",  min_value=0, value=3)
        number_of_rooms     = c3.number_input("Total Rooms", min_value=0, value=7)

        c4, c5, c6 = st.columns(3)
        number_stories  = c4.number_input("Stories", min_value=1, value=2)
        bath_bed_ratio  = c5.number_input(
            "Bath/Bed Ratio", min_value=0.0, value=0.67, step=0.01,
            help="number_of_bathrooms / number_of_bedrooms"
        )
        fireplaces = c6.number_input("Fireplaces", min_value=0, value=0)

    # ── Section: Garage & Parking ──────────────────────────────────────────
    with st.expander("🚗 Garage & Parking", expanded=False):
        c1, c2, c3 = st.columns(3)
        garage_spaces   = c1.number_input("Garage Spaces", min_value=0, value=1)
        off_street_open = c2.number_input("Off-Street Open Spaces", min_value=0, value=0)
        garage_type     = c3.selectbox(
            "Garage Type", KNOWN_CATEGORIES["garage_type"],
            index=KNOWN_CATEGORIES["garage_type"].index("B")
        )
        has_garage = st.checkbox("Has Garage", value=True)

    # ── Section: Building Details ──────────────────────────────────────────
    with st.expander("🏗️ Building Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        building_age        = c1.number_input("Building Age (years)", min_value=0, value=45)
        exterior_condition  = c2.selectbox(
            "Exterior Condition (1–7)", [1, 2, 3, 4, 5, 6, 7], index=2,
            help="Philadelphia OPA condition code"
        )
        interior_condition  = c3.selectbox(
            "Interior Condition (1–7)", [1, 2, 3, 4, 5, 6, 7], index=2,
        )

        c4, c5, c6 = st.columns(3)
        general_construction = c4.selectbox(
            "General Construction", KNOWN_CATEGORIES["general_construction"],
            index=KNOWN_CATEGORIES["general_construction"].index("A")
        )
        building_era = c5.selectbox(
            "Building Era", KNOWN_CATEGORIES["building_era"],
            index=KNOWN_CATEGORIES["building_era"].index("Mid-century (1950-79)")
        )
        type_heater = c6.selectbox(
            "Heater Type", KNOWN_CATEGORIES["type_heater"],
            index=KNOWN_CATEGORIES["type_heater"].index("A")
        )

        c7, c8 = st.columns(2)
        basements = c7.selectbox(
            "Basement Type", KNOWN_CATEGORIES["basements"],
            index=KNOWN_CATEGORIES["basements"].index("A")
        )
        view_type = c8.selectbox(
            "View Type", KNOWN_CATEGORIES["view_type"],
            index=KNOWN_CATEGORIES["view_type"].index("0")
        )

        c9, c10, c11 = st.columns(3)
        has_central_air = c9.checkbox("Has Central Air", value=True)
        has_fireplace   = c10.checkbox("Has Fireplace",  value=False)
        has_basement    = c11.checkbox("Has Basement",   value=True)

    # ── Section: Location & Zoning ─────────────────────────────────────────
    with st.expander("📍 Location & Zoning", expanded=True):
        c1, c2, c3 = st.columns(3)
        geographic_ward = c1.number_input("Geographic Ward", min_value=1, max_value=66, value=21)
        zoning = c2.selectbox(
            "Zoning", KNOWN_CATEGORIES["zoning"],
            index=KNOWN_CATEGORIES["zoning"].index("RSA5")
        )
        category_code = c3.selectbox(
            "Category Code", KNOWN_CATEGORIES["category_code"],
            index=KNOWN_CATEGORIES["category_code"].index("1")
        )

    # ── Section: Sale Info ─────────────────────────────────────────────────
    with st.expander("📅 Sale Information", expanded=False):
        c1, c2 = st.columns(2)
        sale_year  = c1.number_input("Sale Year",  min_value=2000, max_value=2030, value=2021)
        sale_month = c2.number_input("Sale Month", min_value=1,    max_value=12,   value=6)

    # ── Predict button ─────────────────────────────────────────────────────
    st.divider()
    if st.button("🔮 Predict Market Value", type="primary", key="btn_manual"):
        # Build the input row — log-transform area columns
        input_data = {
            "log_total_livable_area": np.log1p(total_livable_area),
            "log_total_area":         np.log1p(total_area),
            "frontage":               frontage,
            "depth":                  depth,
            "livable_area_ratio":     livable_area_ratio,
            "number_of_bathrooms":    number_of_bathrooms,
            "number_of_bedrooms":     number_of_bedrooms,
            "number_of_rooms":        number_of_rooms,
            "number_stories":         number_stories,
            "bath_bed_ratio":         bath_bed_ratio,
            "fireplaces":             fireplaces,
            "garage_spaces":          garage_spaces,
            "off_street_open":        off_street_open,
            "building_age":           building_age,
            "exterior_condition":     exterior_condition,
            "interior_condition":     interior_condition,
            "geographic_ward":        geographic_ward,
            "has_central_air":        int(has_central_air),
            "has_garage":             int(has_garage),
            "has_fireplace":          int(has_fireplace),
            "has_basement":           int(has_basement),
            "sale_year":              sale_year,
            "sale_month":             sale_month,
            "basements":              basements,
            "garage_type":            garage_type,
            "general_construction":   general_construction,
            "topography":             topography,
            "view_type":              view_type,
            "type_heater":            type_heater,
            "parcel_shape":           parcel_shape,
            "building_era":           building_era,
            "zoning":                 zoning,
            "category_code":          category_code,
        }

        df_manual = pd.DataFrame([input_data])[ALL_FEATURES]

        for col in NUM_FEATURES:
            df_manual[col] = pd.to_numeric(df_manual[col], errors="coerce")

        log_pred = pipeline.predict(df_manual)[0]
        pred_usd = np.expm1(log_pred)

        st.success(f"### 🏷️ Estimated Market Value: **${pred_usd:,.0f}**")

        # Show the input summary
        with st.expander("📊 Input summary", expanded=False):
            summary_df = pd.DataFrame([input_data]).T.rename(columns={0: "Value"})
            st.dataframe(summary_df, use_container_width=True)
