import os
import glob

import pandas as pd
import streamlit as st

from src.pipeline.prediction_pipeline import PredictionPipeline


EXPLAINABILITY_DIR = os.path.join("artifacts", "explainability")


def get_latest_file(keyword):
    files = glob.glob(os.path.join(EXPLAINABILITY_DIR, f"*{keyword}*"))
    if not files:
        return None
    files.sort(reverse=True)
    return files[0]


st.set_page_config(
    page_title="GMAA Explainability Dashboard",
    layout="wide"
)

st.title("🌍 Global Mobility Application Analyzer")
st.subheader("Explainability Dashboard — SHAP, LIME, Feature Importance")

if st.button("🔄 Refresh Latest Explainability Files"):
    st.rerun()

shap_summary = get_latest_file("shap_summary")
shap_force = get_latest_file("shap_force")
lime_file = get_latest_file("lime_explanation")
feature_file = get_latest_file("feature_importance")

st.markdown("---")

# ==========================
# Prediction Section
# ==========================
st.header("🎯 Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        continent = st.selectbox(
            "Continent",
            ["Asia", "Africa", "North America", "Europe", "South America", "Oceania"]
        )

        education_of_employee = st.selectbox(
            "Education of Employee",
            ["High School", "Bachelor's", "Master's", "Doctorate"]
        )

        has_job_experience = st.selectbox(
            "Has Job Experience",
            ["Y", "N"]
        )

        requires_job_training = st.selectbox(
            "Requires Job Training",
            ["Y", "N"]
        )

        no_of_employees = st.number_input(
            "Number of Employees",
            min_value=1,
            value=1000
        )

    with col2:
        yr_of_estab = st.number_input(
            "Year of Establishment",
            min_value=1800,
            max_value=2026,
            value=2000
        )

        region_of_employment = st.selectbox(
            "Region of Employment",
            ["Northeast", "South", "West", "Midwest", "Island"]
        )

        prevailing_wage = st.number_input(
            "Prevailing Wage",
            min_value=0.0,
            value=50000.0
        )

        unit_of_wage = st.selectbox(
            "Unit of Wage",
            ["Hour", "Week", "Month", "Year"]
        )

        full_time_position = st.selectbox(
            "Full Time Position",
            ["Y", "N"]
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        "continent": continent,
        "education_of_employee": education_of_employee,
        "has_job_experience": has_job_experience,
        "requires_job_training": requires_job_training,
        "no_of_employees": no_of_employees,
        "yr_of_estab": yr_of_estab,
        "region_of_employment": region_of_employment,
        "prevailing_wage": prevailing_wage,
        "unit_of_wage": unit_of_wage,
        "full_time_position": full_time_position,
    }

    try:
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_data)

        st.success("Prediction completed successfully")

        col1, col2, col3 = st.columns(3)

        col1.metric("Prediction Result", result["result"])
        col2.metric("Certified Probability", result["certified_probability"])
        col3.metric("Denied Probability", result["denied_probability"])

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")

# ==========================
# SHAP Summary
# ==========================
st.header("📌 SHAP Summary Plot")

if shap_summary:
    st.image(shap_summary, caption="SHAP Summary Plot", use_container_width=True)
else:
    st.warning("No SHAP summary plot found.")

st.markdown("---")

# ==========================
# Feature Importance
# ==========================
st.header("📊 Feature Importance")

if feature_file:
    df = pd.read_csv(feature_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Feature Importance Table")
        st.dataframe(df, use_container_width=True)

    with col2:
        st.subheader("Feature Importance Bar Chart")
        chart_df = df.head(15).set_index("feature")
        st.bar_chart(chart_df["importance"])
else:
    st.warning("No feature importance file found.")

st.markdown("---")

# ==========================
# Individual Explanation
# ==========================
st.header("🧠 Individual Prediction Explanations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("SHAP Force Plot")

    if shap_force:
        with open(shap_force, "r", encoding="utf-8") as file:
            shap_html = file.read()

        st.components.v1.html(shap_html, height=400, scrolling=True)
    else:
        st.warning("No SHAP force plot found.")

with col2:
    st.subheader("LIME Explanation")

    if lime_file:
        with open(lime_file, "r", encoding="utf-8") as file:
            lime_html = file.read()

        st.components.v1.html(lime_html, height=400, scrolling=True)
    else:
        st.warning("No LIME explanation found.")

st.markdown("---")

st.info(
    "Note: The prediction form gives live model predictions. "
    "The SHAP and LIME panels currently show the latest saved explainability files from the pipeline."
)