import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import joblib

from huggingface_hub import hf_hub_download

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.text_preprocessing import (  # noqa: E402
    preprocess, MODEL_TEXT_COLUMN, TEST_SIZE, RANDOM_STATE
)


MODEL_REPO = "Maruf39237/imdb-sentiment-model"
DATASET_REPO = "Maruf39237/imdb-sentiment-app-datasets"


MODEL_FILES = {
    ("CountVectorizer", "Logistic Regression"): "cv_logistic_regression.joblib",
    ("CountVectorizer", "MultinomialNB"): "cv_multinomial_nb.joblib",
    ("TF-IDF", "Logistic Regression"): "tfidf_logistic_regression.joblib",
    ("TF-IDF", "MultinomialNB"): "tfidf_multinomial_nb.joblib",
}



@st.cache_resource(show_spinner="Loading ML models from Hugging Face...")
def load_vectorizers_and_models():

    count_vectorizer_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="vectorizer/count_vectorizer.joblib",
        repo_type="model"
    )

    tfidf_vectorizer_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="vectorizer/tfidf_vectorizer.joblib",
        repo_type="model"
    )

    vectorizers = {
        "CountVectorizer": joblib.load(count_vectorizer_path),
        "TF-IDF": joblib.load(tfidf_vectorizer_path),
    }

    models = {}

    for key, filename in MODEL_FILES.items():
        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=f"models/{filename}",
            repo_type="model"
        )

        models[key] = joblib.load(model_path)

    return vectorizers, models



@st.cache_data(show_spinner="Downloading cleaned dataset from Hugging Face...")
def load_data():

    file_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="Cleaned IMDB Dataset.csv",
        repo_type="dataset"
    )

    return pd.read_csv(file_path)





@st.cache_data
def get_test_data():
    """Recreate the exact test split used in Model_Run.ipynb."""
    data = load_data()
    _, test_df = train_test_split(
        data, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=data["sentiment"]
    )
    return test_df.reset_index(drop=True)


@st.cache_data
def evaluate_model(vectorizer_name, model_name):
    """Compute test-set metrics once per model (cached)."""
    vectorizers, models = load_vectorizers_and_models()
    test_df = get_test_data()
    X_test_vec = vectorizers[vectorizer_name].transform(test_df[MODEL_TEXT_COLUMN].fillna(""))
    y_test = test_df["sentiment"]

    model = models[(vectorizer_name, model_name)]
    y_pred = model.predict(X_test_vec)
    pos_idx = list(model.classes_).index("positive")
    y_scores = model.predict_proba(X_test_vec)[:, pos_idx]
    y_true_bin = (y_test == "positive").astype(int)

    fpr, tpr, _ = roc_curve(y_true_bin, y_scores)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_scores)

    return {
        "acc": accuracy_score(y_test, y_pred),
        "prec": precision_score(y_test, y_pred, pos_label="positive"),
        "rec": recall_score(y_test, y_pred, pos_label="positive"),
        "f1": f1_score(y_test, y_pred, pos_label="positive"),
        "cm": confusion_matrix(y_test, y_pred, labels=["negative", "positive"]),
        "fpr": fpr, "tpr": tpr, "roc_auc": auc(fpr, tpr),
        "precision": precision, "recall": recall,
        "ap": average_precision_score(y_true_bin, y_scores),
    }



st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 25px;
        font-size: 17px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.03);
        box-shadow: 0px 5px 18px rgba(0,0,0,.25);
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])
with col1:
    st.title("IMDB::Movie Review Sentiment Prediction System 🎬")
    st.write(
        "#### Enter a movie review and get an instant sentiment prediction "
        "with confidence scores. Powered by classical NLP models trained on 40K IMDB reviews."
    )
with col2:
    st.image(
        "https://images.pexels.com/photos/6633007/pexels-photo-6633007.jpeg",
        width=230,
        caption="Will the model agree with you? 🍿"
    )

st.divider()

with st.expander("🗂️ View sample of cleaned data"):
    st.dataframe(load_data().head(), use_container_width=True)

with st.expander("🔗 Prediction Pipeline"):
    st.code("""
User Review (raw text)
    ↓
Preprocessing (same as training): lowercase → remove HTML/URLs/punctuation
→ expand negations (wasn't → was not) → remove stopwords → lemmatize
    ↓
Selected Vectorizer (Count / TF-IDF, unigrams + bigrams)
    ↓
Selected Model (Logistic Regression / MultinomialNB)
    ↓
Predicted Sentiment + Probabilities
    ↓
Optional: Reveal True Label (for reviews loaded from the test set)
    """, language="text")

st.subheader("⚙️ Model Configuration")

col_v, col_m = st.columns(2)
with col_v:
    vectorizer_choice = st.selectbox(
        "Vectorizer",
        ["TF-IDF", "CountVectorizer"],
        index=0,
        help="TF-IDF usually gives better-calibrated probabilities on this dataset"
    )
with col_m:
    model_choice = st.selectbox(
        "Algorithm",
        ["Logistic Regression", "MultinomialNB"],
        index=0,
        help="Logistic Regression + TF-IDF is the recommended combination"
    )

st.info(
    """**Recommended combination:** TF-IDF + Logistic Regression  
    (highest F1-score and better calibrated probabilities on the IMDB test set)""")

with st.expander("📊 Show Model Performance on Test Set", expanded=False):
    vectorizers, models = load_vectorizers_and_models()
    test_df = get_test_data()
    
    st.markdown(f"#### Performance of **{vectorizer_choice} + {model_choice}**")
    st.caption(f"Evaluated on the held-out test set ({len(test_df):,} reviews never seen during training).")

    perf = evaluate_model(vectorizer_choice, model_choice)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{perf['acc']:.4f}")
    m2.metric("Precision", f"{perf['prec']:.4f}")
    m3.metric("Recall", f"{perf['rec']:.4f}")
    m4.metric("F1-Score", f"{perf['f1']:.4f}")

    st.divider()

    st.markdown("##### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(perf["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"], ax=ax_cm)
    ax_cm.set_xlabel("Predicted", color = 'red')
    ax_cm.set_ylabel("Actual", color = 'red')
    ax_cm.set_title("Confusion Matrix", color = 'red', fontsize = 18)
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    st.markdown("##### ROC & Precision-Recall Curves")
    fig_curves, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(perf["fpr"], perf["tpr"], color="darkorange", lw=2, label=f"AUC = {perf['roc_auc']:.4f}")
    axes[0].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", color = 'red')
    axes[0].set_ylabel("True Positive Rate", color = 'red')
    axes[0].set_title("ROC Curve", color = 'red', fontsize = 18)
    axes[0].legend(loc="lower right")

    axes[1].plot(perf["recall"], perf["precision"], color="blue", lw=2, label=f"AP = {perf['ap']:.4f}")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall", color = 'red')
    axes[1].set_ylabel("Precision", color = 'red')
    axes[1].set_title("Precision-Recall Curve", color = 'red', fontsize = 18)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    st.pyplot(fig_curves)
    plt.close(fig_curves)


if "current_review" not in st.session_state:
    st.session_state.current_review = ""
if "true_sentiment" not in st.session_state:
    st.session_state.true_sentiment = None


def set_review(text, true_label):
    """Load a new review and clear the previous prediction."""
    st.session_state.current_review = text
    st.session_state.true_sentiment = true_label
    st.session_state.pop("prediction_result", None)


tab1, tab2, tab3 = st.tabs([
    "✍️ Manual Entry",
    "🎲 Load Random Test Review",
    "🔢 Load Test Set Row"
])

with tab1:
    review_text = st.text_area(
        "Paste or type a movie review here",
        height=180,
        placeholder="Example: This movie was absolutely fantastic, the acting was superb...",
        key="manual_review"
    )
    if st.button("Use this review", key="use_manual"):
        set_review(review_text, None)
        st.success("Review loaded!")

with tab2:
    st.caption("Reviews come from the test set, so the model has never seen them.")
    if st.button("🎲 Load Random Review"):
        sample = test_df.sample(1).iloc[0]
        set_review(sample["review"], sample["sentiment"])
        st.success("Random review loaded!")

with tab3:
    row_idx = st.number_input("Test set row index", min_value=0, max_value=len(test_df) - 1, value=0)
    if st.button("Load this row"):
        sample = test_df.iloc[int(row_idx)]
        set_review(sample["review"], sample["sentiment"])
        st.success(f"Row {row_idx} loaded!")

st.divider()
st.subheader("📝 Current Review")

current = st.session_state.current_review
if current:
    st.write(current[:800] + ("..." if len(current) > 800 else ""))
    with st.expander("🧹 See the preprocessed text the model receives"):
        st.write(preprocess(current) or "_(nothing left after preprocessing)_")
else:
    st.info("No review loaded yet. Use one of the tabs above.")

col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    predict_clicked = st.button("🚀 Predict Sentiment", type="primary", use_container_width=True)

if predict_clicked:
    processed = preprocess(current)
    if not processed.strip():
        st.warning("Please load or type a review first (it must contain real words).")
    else:
        vectorizer = vectorizers[vectorizer_choice]
        model = models[(vectorizer_choice, model_choice)]

        X = vectorizer.transform([processed])
        proba = model.predict_proba(X)[0]
        pred_label = model.classes_[np.argmax(proba)]

        pos_prob = proba[list(model.classes_).index("positive")] * 100
        neg_prob = proba[list(model.classes_).index("negative")] * 100

        st.session_state.prediction_result = {
            "label": pred_label,
            "pos": pos_prob,
            "neg": neg_prob,
            "confidence": max(pos_prob, neg_prob),
            "model_name": f"{vectorizer_choice} + {model_choice}",
        }

if "prediction_result" in st.session_state:
    res = st.session_state.prediction_result

    conf = res["confidence"]
    if conf >= 80:
        certainty, adverb = "High", "highly"
    elif conf >= 60:
        certainty, adverb = "Medium", "moderately"
    else:
        certainty, adverb = "Low", "only slightly"

    interpretation = (
        f"The model is {adverb} confident that this review expresses a "
        f"**{res['label']}** sentiment."
    )

    st.markdown("### Prediction Result")
    st.caption(f"Model used: **{res['model_name']}**")

    result_df = pd.DataFrame({
        "Metric": [
            "Predicted Sentiment",
            "Positive Probability",
            "Negative Probability",
            "Confidence",
            "Certainty Level"
        ],
        "Value": [
            res["label"].upper(),
            f"{res['pos']:.1f}%",
            f"{res['neg']:.1f}%",
            f"{res['confidence']:.1f}%",
            certainty
        ]
    })

    st.dataframe(result_df, use_container_width=True, hide_index=True, height=210)

    st.info(interpretation)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    labels = ["Positive", "Negative"]
    values = [res["pos"], res["neg"]]
    colors = ["#2ecc71", "#e74c3c"]

    bars = axes[0].bar(labels, values, color=colors)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Probability (%)", color = 'red')
    axes[0].set_title("Class Probabilities", fontsize=12, fontweight="bold", color = 'red')
    axes[0].bar_label(bars, fmt="%.1f%%", padding=3)

    axes[1].pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.04, 0.04)
    )
    axes[1].set_title("Probability Split", fontsize=12, fontweight="bold", color = 'red')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if st.session_state.true_sentiment is not None:
        st.write("")

        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            reveal_clicked = st.button("🔍 Reveal True Sentiment", use_container_width=True)

        if reveal_clicked:
            true = st.session_state.true_sentiment
            pred = res["label"]

            if true == pred:
                st.success(f"True Sentiment: **{true.upper()}** — Model was correct! ✅")
            else:
                st.error(f"True Sentiment: **{true.upper()}** — Model was wrong ❌")
