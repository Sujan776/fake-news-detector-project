import re
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================
# APP CONFIG
# =========================

MODEL_PATH = Path("model/fake_news_model.joblib")

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)


# =========================
# CUSTOM STYLE
# =========================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 18px;
        color: #4b5563;
        margin-bottom: 25px;
    }
    .result-card {
        padding: 22px;
        border-radius: 14px;
        background-color: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-top: 15px;
    }
    .true-box {
        background-color: #dcfce7;
        color: #166534;
        padding: 18px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
    }
    .false-box {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 18px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
    }
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 30px;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HELPER FUNCTIONS
# =========================

def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def get_model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)

    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("classifier")
        if clf is not None and hasattr(clf, "classes_"):
            return list(clf.classes_)

    return []


def predict_news(model, text):
    text = clean_text(text)

    prediction = model.predict([text])[0]
    prediction_upper = str(prediction).upper()

    classes = get_model_classes(model)
    probability_data = {}
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        confidence = max(probabilities) * 100

        for label, prob in zip(classes, probabilities):
            probability_data[str(label).upper()] = prob * 100

    if prediction_upper in ["REAL", "TRUE", "1"]:
        final_result = "TRUE NEWS"
        status = "TRUE"
    else:
        final_result = "FALSE NEWS"
        status = "FALSE"

    return {
        "news": text,
        "raw_prediction": prediction_upper,
        "final_result": final_result,
        "status": status,
        "confidence": confidence,
        "probability_data": probability_data,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def display_prediction(result):
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if result["status"] == "TRUE":
        st.markdown('<div class="true-box">✅ TRUE NEWS</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="false-box">❌ FALSE NEWS</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Final Result", result["final_result"])

    with col2:
        st.metric("Model Label", result["raw_prediction"])

    with col3:
        if result["confidence"] is not None:
            st.metric("Confidence", f"{result['confidence']:.2f}%")
        else:
            st.metric("Confidence", "N/A")

    if result["probability_data"]:
        st.subheader("Prediction Probability")

        prob_df = pd.DataFrame({
            "Class": list(result["probability_data"].keys()),
            "Probability (%)": list(result["probability_data"].values())
        })

        st.bar_chart(prob_df.set_index("Class"))

        st.dataframe(prob_df, use_container_width=True)

    with st.expander("View Entered News Text"):
        st.write(result["news"])


# =========================
# LOAD MODEL
# =========================

model = load_model()


# =========================
# SIDEBAR
# =========================


st.sidebar.title("📌 Project Info")
st.sidebar.write("**Project:** Fake News Detector")
st.sidebar.write("**Developer:** Sri Sujan")
st.sidebar.write("**Domain:** Machine Learning + NLP")
st.sidebar.write("**Algorithm:** TF-IDF + Logistic Regression")

st.sidebar.markdown("---")

if model is None:
    st.sidebar.error("Model not loaded")
else:
    st.sidebar.success("Model loaded successfully")

st.sidebar.markdown("---")
st.sidebar.subheader("App Features")
st.sidebar.write("✅ Single news detection")
st.sidebar.write("✅ TRUE / FALSE output")
st.sidebar.write("✅ Confidence score")
st.sidebar.write("✅ Probability chart")
st.sidebar.write("✅ Batch CSV prediction")
st.sidebar.write("✅ Download results")
st.sidebar.write("✅ Prediction history")


# =========================
# MAIN TITLE
# =========================

st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Detect whether a news headline or article is TRUE or FALSE using Machine Learning.</div>',
    unsafe_allow_html=True
)

if model is None:
    st.error("Model file not found.")
    st.write("Please train the model first:")
    st.code("python train_model.py")
    st.stop()


# =========================
# SESSION STATE
# =========================

if "history" not in st.session_state:
    st.session_state.history = []


# =========================
# TABS
# =========================

tab1, tab2, tab3 = st.tabs([
    "🔍 Single News Detection",
    "📁 Batch CSV Detection",
    "📊 Prediction History"
])


# =========================
# TAB 1: SINGLE NEWS DETECTION
# =========================

with tab1:
    st.subheader("Check Single News")

    examples = {
        "Select sample news": "",
        "TRUE example - Scholarship": "Government announces a new scholarship scheme for engineering students to support higher education.",
        "TRUE example - NASA": "NASA published satellite data showing changes in global climate patterns.",
        "TRUE example - University": "The university released the semester examination timetable on its official website.",
        "FALSE example - Magic tablet": "One magic tablet can cure all diseases overnight without any medical treatment.",
        "FALSE example - Alien city": "Breaking news: a secret alien city has been found under the ocean with unlimited gold.",
        "FALSE example - 200 years life": "Scientists discovered that drinking hot water once can make humans live for 200 years."
    }

    selected_example = st.selectbox("Choose a sample or type your own:", list(examples.keys()))

    news_text = st.text_area(
        "Enter news headline or article:",
        value=examples[selected_example],
        height=220,
        placeholder="Paste any news headline or article here..."
    )

    col_a, col_b = st.columns([1, 1])

    with col_a:
        detect_clicked = st.button("🔍 Detect News", use_container_width=True)

    with col_b:
        clear_clicked = st.button("🧹 Clear Text", use_container_width=True)

    if clear_clicked:
        st.info("Select another sample or manually clear the text box.")

    if detect_clicked:
        cleaned_text = clean_text(news_text)

        if len(cleaned_text) < 10:
            st.warning("Please enter at least 10 characters of news text.")
        else:
            result = predict_news(model, cleaned_text)
            display_prediction(result)

            st.session_state.history.insert(0, {
                "Time": result["time"],
                "News": result["news"][:150] + "..." if len(result["news"]) > 150 else result["news"],
                "Prediction": result["final_result"],
                "Confidence": f"{result['confidence']:.2f}%" if result["confidence"] is not None else "N/A"
            })


# =========================
# TAB 2: BATCH CSV DETECTION
# =========================

with tab2:
    st.subheader("Batch News Detection using CSV")

    st.write("Upload a CSV file that contains news headlines or articles.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            if batch_df.empty:
                st.error("The uploaded CSV file is empty.")
            else:
                st.write("CSV Preview")
                st.dataframe(batch_df.head(), use_container_width=True)

                text_column = st.selectbox(
                    "Select the column containing news text:",
                    batch_df.columns
                )

                if st.button("🚀 Run Batch Detection", use_container_width=True):
                    results = []
                    progress_bar = st.progress(0)

                    total_rows = len(batch_df)

                    for index, text in enumerate(batch_df[text_column].fillna("")):
                        cleaned_text = clean_text(text)

                        if len(cleaned_text) < 10:
                            results.append({
                                "news_text": cleaned_text,
                                "prediction": "INVALID INPUT",
                                "confidence": "N/A"
                            })
                        else:
                            prediction_result = predict_news(model, cleaned_text)

                            results.append({
                                "news_text": cleaned_text,
                                "prediction": prediction_result["final_result"],
                                "confidence": f"{prediction_result['confidence']:.2f}%" if prediction_result["confidence"] is not None else "N/A"
                            })

                        progress_bar.progress((index + 1) / total_rows)

                    result_df = pd.DataFrame(results)

                    st.success("Batch prediction completed.")
                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=csv_data,
                        file_name="fake_news_detection_results.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as error:
            st.error("Unable to process the uploaded CSV file.")
            st.write(error)


# =========================
# TAB 3: HISTORY
# =========================

with tab3:
    st.subheader("Prediction History")

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True)

        history_csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download History",
            data=history_csv,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.button("Clear Prediction History"):
            st.session_state.history = []
            st.success("History cleared.")
    else:
        st.info("No prediction history available yet.")


# =========================
# FOOTER
# =========================

st.markdown(
    '<div class="footer">Fake News Detector | Internship Project | Python | Machine Learning | NLP | Streamlit</div>',
    unsafe_allow_html=True
)
