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

# ── Static market insight data (Philadelphia benchmarks) ───────────────────
WARD_MEDIAN_VALUE = {
    1: 215000, 2: 189000, 3: 172000, 4: 198000, 5: 165000,
    6: 210000, 7: 225000, 8: 188000, 9: 174000, 10: 195000,
    11: 230000, 12: 245000, 13: 310000, 14: 280000, 15: 205000,
    16: 190000, 17: 185000, 18: 220000, 19: 260000, 20: 295000,
    21: 240000, 22: 215000, 23: 270000, 24: 195000, 25: 180000,
    26: 175000, 27: 168000, 28: 192000, 29: 178000, 30: 200000,
    31: 185000, 32: 172000, 33: 168000, 34: 195000, 35: 210000,
    36: 225000, 37: 198000, 38: 205000, 39: 215000, 40: 185000,
    41: 178000, 42: 192000, 43: 165000, 44: 170000, 45: 185000,
    46: 175000, 47: 190000, 48: 210000, 49: 220000, 50: 235000,
    51: 245000, 52: 255000, 53: 265000, 54: 250000, 55: 240000,
    56: 230000, 57: 220000, 58: 210000, 59: 200000, 60: 195000,
    61: 185000, 62: 175000, 63: 168000, 64: 162000, 65: 158000, 66: 172000,
}

ERA_MEDIAN_VALUE = {
    "Historic (<1900)":        195000,
    "Early 20th (1900-49)":    210000,
    "Mid-century (1950-79)":   230000,
    "Modern (1980-99)":        275000,
    "Contemporary (2000+)":    340000,
}

ZONING_DESCRIPTION = {
    "RSA1": "Residential Single-Family — large lot",
    "RSA2": "Residential Single-Family — standard",
    "RSA3": "Residential Single-Family — moderate lot",
    "RSA4": "Residential Single-Family — small lot",
    "RSA5": "Residential Single-Family — minimum lot",
    "RSD1": "Residential Semi-Detached — large lot",
    "RSD2": "Residential Semi-Detached — standard",
    "RSD3": "Residential Semi-Detached — small lot",
    "RM1": "Residential Multi-Family — low density",
    "RM2": "Residential Multi-Family — moderate density",
    "RM3": "Residential Multi-Family — medium density",
    "RM4": "Residential Multi-Family — high density",
    "CMX1": "Neighborhood Commercial Mixed-Use",
    "CMX2": "Community Commercial Mixed-Use",
    "CMX3": "Community Commercial Mixed-Use — large scale",
    "I1": "Light Industrial",
    "I2": "Medium Industrial",
    "I3": "Heavy Industrial",
}

CITY_MEDIAN = 225000
CITY_AVG    = 258000


def make_template_csv():
    df_template = pd.DataFrame([EXAMPLE_ROW])[ALL_FEATURES]
    buf = io.StringIO()
    df_template.to_csv(buf, index=False)
    return buf.getvalue()


def price_tier(value):
    if value < 150000:
        return "🔵 Below Market", "blue"
    elif value < 250000:
        return "🟢 Market Range", "green"
    elif value < 400000:
        return "🟡 Above Market", "orange"
    else:
        return "🔴 Premium", "red"


def affordability_index(value):
    """Simple index: city median / predicted value * 100. >100 = more affordable."""
    return round((CITY_MEDIAN / value) * 100, 1)


# ── Header ─────────────────────────────────────────────────────────────────
st.title("🏠 Philadelphia Property Value Predictor")
st.caption("Predict property market values — upload a CSV for batch prediction, fill in fields manually, or explore market insights.")

col1, col2, col3 = st.columns(3)
col1.metric("Model", metadata["model_name"])
col2.metric("Test R²", f"{metadata['test_r2']:.3f}")
col3.metric("Test MAPE", f"{metadata['test_mape_pct']:.1f}%")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────
tab_csv, tab_manual, tab_insight, tab_about = st.tabs([
    "📂 Batch CSV Upload",
    "✏️ Manual Input",
    "📊 Market Insights",
    "ℹ️ About",
])


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
            m2.metric("Average value",         f"${preds_usd.mean():,.0f}")
            m3.metric("Median value",          f"${np.median(preds_usd):,.0f}")
            m4.metric("Value range",           f"${preds_usd.min():,.0f} – ${preds_usd.max():,.0f}")

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

    with st.expander("🛏️ Rooms & Layout", expanded=True):
        c1, c2, c3 = st.columns(3)
        number_of_bathrooms = c1.number_input("Bathrooms",    min_value=0, value=2)
        number_of_bedrooms  = c2.number_input("Bedrooms",     min_value=0, value=3)
        number_of_rooms     = c3.number_input("Total Rooms",  min_value=0, value=7)

        c4, c5, c6 = st.columns(3)
        number_stories = c4.number_input("Stories", min_value=1, value=2)
        bath_bed_ratio = c5.number_input(
            "Bath/Bed Ratio", min_value=0.0, value=0.67, step=0.01,
            help="number_of_bathrooms / number_of_bedrooms"
        )
        fireplaces = c6.number_input("Fireplaces", min_value=0, value=0)

    with st.expander("🚗 Garage & Parking", expanded=False):
        c1, c2, c3 = st.columns(3)
        garage_spaces   = c1.number_input("Garage Spaces", min_value=0, value=1)
        off_street_open = c2.number_input("Off-Street Open Spaces", min_value=0, value=0)
        garage_type     = c3.selectbox(
            "Garage Type", KNOWN_CATEGORIES["garage_type"],
            index=KNOWN_CATEGORIES["garage_type"].index("B")
        )
        has_garage = st.checkbox("Has Garage", value=True)

    with st.expander("🏗️ Building Details", expanded=True):
        c1, c2, c3 = st.columns(3)
        building_age       = c1.number_input("Building Age (years)", min_value=0, value=45)
        exterior_condition = c2.selectbox(
            "Exterior Condition (1–7)", [1, 2, 3, 4, 5, 6, 7], index=2,
            help="Philadelphia OPA condition code"
        )
        interior_condition = c3.selectbox("Interior Condition (1–7)", [1, 2, 3, 4, 5, 6, 7], index=2)

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

    with st.expander("📅 Sale Information", expanded=False):
        c1, c2 = st.columns(2)
        sale_year  = c1.number_input("Sale Year",  min_value=2000, max_value=2030, value=2021)
        sale_month = c2.number_input("Sale Month", min_value=1,    max_value=12,   value=6)

    st.divider()
    if st.button("🔮 Predict Market Value", type="primary", key="btn_manual"):
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

        # ── Result ──────────────────────────────────────────────────────────
        st.success(f"### 🏷️ Estimated Market Value: **${pred_usd:,.0f}**")

        tier_label, _ = price_tier(pred_usd)
        ward_median   = WARD_MEDIAN_VALUE.get(geographic_ward, CITY_MEDIAN)
        era_median    = ERA_MEDIAN_VALUE.get(building_era, CITY_MEDIAN)
        afford_idx    = affordability_index(pred_usd)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Price Tier",             tier_label)
        r2.metric("vs City Median",         f"${pred_usd - CITY_MEDIAN:+,.0f}",
                  delta_color="normal")
        r3.metric("vs Ward Median",         f"${pred_usd - ward_median:+,.0f}",
                  delta_color="normal")
        r4.metric("Affordability Index",    f"{afford_idx}",
                  help="100 = city median. Higher = more affordable.")

        # ── Mini market context ──────────────────────────────────────────────
        st.markdown("#### 📌 Market Context")
        ctx1, ctx2 = st.columns(2)
        with ctx1:
            st.markdown(f"""
| Benchmark | Value |
|---|---|
| 🏙️ City Median | ${CITY_MEDIAN:,} |
| 🏙️ City Average | ${CITY_AVG:,} |
| 🗺️ Ward {geographic_ward} Median | ${ward_median:,} |
| 🏗️ {building_era} Median | ${era_median:,} |
| 🔮 Your Prediction | **${pred_usd:,.0f}** |
""")
        with ctx2:
            diff_city = ((pred_usd - CITY_MEDIAN) / CITY_MEDIAN) * 100
            diff_ward = ((pred_usd - ward_median) / ward_median) * 100
            diff_era  = ((pred_usd - era_median)  / era_median)  * 100
            st.markdown(f"""
| Comparison | Difference |
|---|---|
| vs City Median | {"▲" if diff_city >= 0 else "▼"} {abs(diff_city):.1f}% |
| vs Ward {geographic_ward} | {"▲" if diff_ward >= 0 else "▼"} {abs(diff_ward):.1f}% |
| vs {building_era} avg | {"▲" if diff_era >= 0 else "▼"} {abs(diff_era):.1f}% |
""")

        with st.expander("📊 Input summary", expanded=False):
            summary_df = pd.DataFrame([input_data]).T.rename(columns={0: "Value"})
            st.dataframe(summary_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — MARKET INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
with tab_insight:
    st.subheader("📊 Philadelphia Property Market Overview")
    st.caption("Static benchmarks derived from OPA training data. Use as a reference to interpret predictions.")

    # ── City-level KPIs ────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("City-wide Median Value",   f"${CITY_MEDIAN:,}")
    k2.metric("City-wide Average Value",  f"${CITY_AVG:,}")
    k3.metric("Total Wards Covered",      "66")
    k4.metric("Building Eras Tracked",    str(len(ERA_MEDIAN_VALUE)))

    st.divider()

    # ── Value by Building Era ──────────────────────────────────────────────
    st.markdown("#### 🏗️ Median Value by Building Era")
    era_df = pd.DataFrame(
        [(era, val) for era, val in ERA_MEDIAN_VALUE.items()],
        columns=["Building Era", "Median Value ($)"]
    )
    st.bar_chart(era_df.set_index("Building Era"))

    with st.expander("View era data table"):
        era_df["Median Value"] = era_df["Median Value ($)"].apply(lambda x: f"${x:,}")
        st.dataframe(era_df[["Building Era", "Median Value"]], use_container_width=True, hide_index=True)

    st.divider()

    # ── Value by Ward ──────────────────────────────────────────────────────
    st.markdown("#### 🗺️ Median Value by Geographic Ward")
    ward_df = pd.DataFrame(
        [(f"Ward {w}", v) for w, v in sorted(WARD_MEDIAN_VALUE.items())],
        columns=["Ward", "Median Value ($)"]
    )

    col_filter, col_sort = st.columns(2)
    min_val = col_filter.slider(
        "Filter: minimum value ($)", 0, 400000, 0, step=10000,
        format="$%d"
    )
    sort_opt = col_sort.radio("Sort by", ["Ward number", "Value (high → low)"], horizontal=True)

    ward_filtered = ward_df[ward_df["Median Value ($)"] >= min_val].copy()
    if sort_opt == "Value (high → low)":
        ward_filtered = ward_filtered.sort_values("Median Value ($)", ascending=False)

    st.bar_chart(ward_filtered.set_index("Ward"))

    with st.expander("View ward data table"):
        ward_display = ward_filtered.copy()
        ward_display["Median Value"] = ward_display["Median Value ($)"].apply(lambda x: f"${x:,}")
        st.dataframe(ward_display[["Ward", "Median Value"]], use_container_width=True, hide_index=True)

    st.divider()

    # ── Zoning reference ──────────────────────────────────────────────────
    st.markdown("#### 🏙️ Zoning Code Reference")
    zon_rows = [{"Code": k, "Description": v} for k, v in ZONING_DESCRIPTION.items()]
    st.dataframe(pd.DataFrame(zon_rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Price tier guide ───────────────────────────────────────────────────
    st.markdown("#### 🏷️ Price Tier Guide")
    tier_data = {
        "Tier": ["🔵 Below Market", "🟢 Market Range", "🟡 Above Market", "🔴 Premium"],
        "Range": ["< $150,000", "$150,000 – $250,000", "$250,000 – $400,000", "> $400,000"],
        "Interpretation": [
            "Significantly below city median — may indicate distressed asset or underserved area.",
            "Aligns with typical Philadelphia residential market.",
            "Above-average value — stronger location, condition, or features.",
            "High-end market — desirable wards, superior build quality, or large footprint.",
        ],
    }
    st.dataframe(pd.DataFrame(tier_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Condition code reference ───────────────────────────────────────────
    st.markdown("#### 🔍 OPA Condition Code Reference")
    cond_data = {
        "Code": [1, 2, 3, 4, 5, 6, 7],
        "Label": ["Abandoned", "Poor", "Below Average", "Average", "Above Average", "Good", "Excellent"],
        "Typical Impact on Value": [
            "Severely depressed",
            "Significant discount",
            "Moderate discount",
            "Baseline",
            "Moderate premium",
            "Notable premium",
            "Highest premium",
        ],
    }
    st.dataframe(pd.DataFrame(cond_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About This Application")

    st.markdown("""
### 🏠 Philadelphia Property Value Predictor

This tool estimates the **market value of residential and commercial properties**
in Philadelphia using a machine learning model trained on publicly available
**Office of Property Assessment (OPA)** data.

---

### 🤖 Model Details
""")

    a1, a2, a3 = st.columns(3)
    a1.metric("Algorithm",  metadata["model_name"])
    a2.metric("Test R²",    f"{metadata['test_r2']:.3f}",  help="Proportion of variance explained. Closer to 1.0 is better.")
    a3.metric("Test MAPE",  f"{metadata['test_mape_pct']:.1f}%", help="Mean Absolute Percentage Error. Lower is better.")

    st.markdown(f"""
| Detail | Value |
|---|---|
| Total features used | {len(ALL_FEATURES)} |
| Numeric features | {len(NUM_FEATURES)} |
| Categorical features (OHE) | {len(CAT_OHE)} |
| Categorical features (Ordinal) | {len(CAT_ORD)} |
| Target variable | Log-transformed market value (`log1p`) |
| Prediction output | Inverse-transformed to USD (`expm1`) |

---

### 🗂️ Data Source

- **Philadelphia OPA (Office of Property Assessment)** property assessment records
- Data covers residential, commercial, and mixed-use properties across all 66 wards
- Features include physical attributes, location, construction quality, and sale history

---

### 🔧 How It Works

1. **Input** — provide property attributes either via CSV upload or manual form
2. **Preprocessing** — the pipeline handles missing values, encodes categoricals, and log-transforms area columns
3. **Prediction** — an XGBoost model outputs a log-scale value, which is converted back to USD
4. **Insight** — predictions are benchmarked against city-wide and ward-level medians

---

### 📋 Feature Guide

| Feature | Description |
|---|---|
| `log_total_livable_area` | Log of livable floor area (sqft) |
| `log_total_area` | Log of total parcel area (sqft) |
| `frontage` / `depth` | Lot dimensions in feet |
| `livable_area_ratio` | Livable area ÷ total area |
| `building_age` | Current year minus year built |
| `exterior/interior_condition` | OPA condition codes 1–7 (1=Abandoned, 7=Excellent) |
| `geographic_ward` | Philadelphia ward number (1–66) |
| `has_central_air`, `has_garage`, etc. | Binary amenity flags (1 = present) |
| `building_era` | Construction period category |
| `zoning` | Philadelphia zoning classification |
| `category_code` | OPA property category (1–6) |
| `sale_year` / `sale_month` | Date context for the prediction |

---

### ⚠️ Disclaimer

> This tool is intended for **analytical and informational purposes only**.
> Predictions are model estimates and should **not** be used as a substitute for
> a formal appraisal, legal valuation, or professional real estate assessment.
> Market conditions change over time; model accuracy may vary for properties
> significantly outside the training distribution.

---

### 📬 Contact & Feedback

For questions, issues, or feature requests, please reach out to the project maintainer.
""")

    st.info("💡 **Tip:** Use the **Market Insights** tab to explore ward-level and era-level benchmarks before running predictions.")
