import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="centered",
)


st.title("🩺 Breast Cancer Prediction")
st.write(
    "Enter the selected features below and send them to the "
    "FastAPI prediction service."
)

st.caption(f"API: {API_URL}")


# ---------------------------------------------------------
# API Health Check
# ---------------------------------------------------------

st.subheader("API Status")

try:
    response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    if response.status_code == 200:
        health = response.json()

        if health.get("status") == "healthy":
            st.success("API is healthy")
        else:
            st.warning("API is running, but the model is not ready.")
    else:
        st.error(f"API returned status code: {response.status_code}")

except requests.exceptions.RequestException:
    st.error("Cannot connect to the FastAPI service.")


st.divider()


# ---------------------------------------------------------
# Prediction Form
# ---------------------------------------------------------

st.subheader("Prediction")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        mean_radius = st.number_input(
            "Mean Radius",
            min_value=0.0,
            value=14.0,
            format="%.6f",
        )

        mean_concavity = st.number_input(
            "Mean Concavity",
            min_value=0.0,
            value=0.1,
            format="%.6f",
        )

        mean_concave_points = st.number_input(
            "Mean Concave Points",
            min_value=0.0,
            value=0.05,
            format="%.6f",
        )

        radius_error = st.number_input(
            "Radius Error",
            min_value=0.0,
            value=0.5,
            format="%.6f",
        )

    with col2:
        worst_perimeter = st.number_input(
            "Worst Perimeter",
            min_value=0.0,
            value=100.0,
            format="%.6f",
        )

        worst_concavity = st.number_input(
            "Worst Concavity",
            min_value=0.0,
            value=0.2,
            format="%.6f",
        )

        worst_concave_points = st.number_input(
            "Worst Concave Points",
            min_value=0.0,
            value=0.1,
            format="%.6f",
        )

        worst_symmetry = st.number_input(
            "Worst Symmetry",
            min_value=0.0,
            value=0.25,
            format="%.6f",
        )

    predict_button = st.form_submit_button(
        "Predict",
        use_container_width=True,
    )


# ---------------------------------------------------------
# Send Prediction Request
# ---------------------------------------------------------

if predict_button:

    payload = {
        "mean_radius": mean_radius,
        "mean_concavity": mean_concavity,
        "mean_concave_points": mean_concave_points,
        "radius_error": radius_error,
        "worst_perimeter": worst_perimeter,
        "worst_concavity": worst_concavity,
        "worst_concave_points": worst_concave_points,
        "worst_symmetry": worst_symmetry,
    }

    try:
        with st.spinner("Getting prediction..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            class_name = result["class_name"]

            st.success("Prediction completed")

            if class_name == "malignant":
                st.error(f"Prediction: {class_name.upper()}")
            else:
                st.success(f"Prediction: {class_name.upper()}")

            st.write(f"Prediction value: `{prediction}`")

        elif response.status_code == 422:

            st.error("Invalid input. Please check the entered values.")

            try:
                st.json(response.json())
            except ValueError:
                pass

        elif response.status_code == 503:

            st.error("The prediction model is currently unavailable.")

        else:

            st.error(f"Prediction failed. HTTP status: {response.status_code}")

    except requests.exceptions.Timeout:

        st.error("The request timed out. Please try again.")

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the FastAPI service. " "Make sure the API is running."
        )

    except requests.exceptions.RequestException as error:

        st.error(f"Request failed: {error}")
