import json
from pathlib import Path
import pandas as pd
import streamlit as st

METRICS_PATH = Path(__file__).resolve().parent.parent / "Model" / "model_metrics.json"

st.divider()

col1, col2 = st.columns([4, 1.8])

with col1 :
    st.title("IMDB Movie Review Sentiment Analysis 🎬")
    st.header("NLP-Based Binary Sentiment Classification System")
    st.subheader("Analyze Movie Reviews Using Natural Language Processing (Classical NLP) & Machine Learning to Classify Polarity")

    
with col2 :
    st.image("https://images.pexels.com/photos/18501410/pexels-photo-18501410.jpeg", width = 200)
    st.markdown("""###### This application analyzes IMDB movie reviews and predicts whether a review expresses a positive or negative sentiment.🎥🍿""")
    

st.divider()

st.markdown(
    """
    ### 📖 Project Description

    ###### The ***IMDB Movie Review Sentiment Analysis*** project is an end-to-end Natural Language Processing and Machine Learning application that classifies movie reviews as positive or negative.

    ###### The project follows a complete ***Classical NLP*** workflow, including exploratory data analysis, text cleaning, lemmatization, feature extraction, model training, model evaluation, and deployment through an interactive Streamlit application.
""")

st.markdown("### 🎯 Project Highlights")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Dataset Size",
        "49,582"
    )

with c2:
    st.metric(
        "Sentiment Classes",
        "2"
    )

with c3:
    st.metric(
        "NLP Approach",
        "Classical"
    )

with c4:
    st.metric(
        "Model Combinations",
        "4"
    )

st.markdown("### Project Objective")
st.write(
    """
    The objective of this project is to develop a Classical NLP-based
    sentiment classification system that can automatically determine
    whether an IMDB movie review expresses a positive or negative sentiment.
    """
)

st.markdown("### 🧠 Models Used")

c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    #### Logistic Regression

    * CountVectorizer + Logistic Regression
    * TF-IDF + Logistic Regression
    """)

with c2:
    st.markdown("""
    #### Naive Bayes

    * CountVectorizer + Multinomial Naive Bayes
    * TF-IDF + Multinomial Naive Bayes
    """)

if METRICS_PATH.exists():
    st.markdown("#### 📊 Test Set Results")
    metrics_df = pd.DataFrame(json.loads(METRICS_PATH.read_text()))
    st.dataframe(
        metrics_df.style.format({c: "{:.4f}" for c in ["Accuracy", "Precision", "Recall", "F1"]}),
        use_container_width=True,
        hide_index=True
    )




st.markdown(
    """
    ### 🔄 NLP & ML Workflow

    ```text
                        IMDB Movie Reviews
                                │
                                ▼
                    Exploratory Data Analysis
                                │
                                ▼
                       Text Preprocessing
                      • Text Cleaning
                      • Negation Handling
                      • Stopword Removal
                      • Lemmatization
                                │
                                ▼
                        Feature Extraction
                      • Count Vectorizer
                      • TF-IDF Vectorizer
                      (unigrams + bigrams)
                                │
                                ▼
                         Train-Test Split
                            (80% / 20%)
                                │
                                ▼
                         Model Training
                     • Logistic Regression
                     • Multinomial Naive Bayes
                                │
                                ▼
                       Model Evaluation
                     • Accuracy
                     • Precision
                     • Recall
                     • F1-Score
                     • Classification Report
                     • Confusion Matrix
                     • ROC-AUC
                                │
                                ▼
                     Streamlit Web Application
                                │
                                ▼
                       Sentiment Prediction
    ```
    """
)



st.markdown(
    """
    ### 📊 Dataset Overview

    This project is built on the ***IMDB Movie Review Dataset***, a widely recognized benchmark in Natural Language Processing for binary sentiment classification.

    **Key characteristics of the dataset:**
    - **Size**: 50,000 movie reviews
    - **Labels**: Balanced binary sentiment — 25,000 **positive** and 25,000 **negative** reviews
    - **Content**: Real user-written reviews collected from the Internet Movie Database (IMDB)
    - **Average review length**: ~1,300 characters (roughly 230 words)
    - **Range**: Reviews vary from very short (~30 characters) to long-form critiques (up to ~13,600 characters)

    The dataset provides a rich, real-world collection of natural language text that captures the diversity of opinions, writing styles, and emotional tones found in actual movie reviews. It serves as an excellent foundation for developing and evaluating sentiment analysis models using classical NLP techniques such as text cleaning, tokenization, lemmatization, Bag-of-Words / TF-IDF features and machine learning classifiers.
    """
)

st.markdown("### Key Components")

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        **-Text Representations-**
        * **Review** – Original raw movie review text
        * **Cleaned Review** – Preprocessed text (lowercased, HTML/URLs/special characters removed, negations expanded, stopwords removed)
        * **Lemmatized Text** – Words reduced to their base/lemma form (this column is used to train the models)
        """
    )

with c2:
    st.markdown(
        """
        **-Feature Engineering Techniques-**
        * **CountVectorizer** – Bag-of-Words feature representation
        * **TF-IDF** – Term Frequency–Inverse Document Frequency weighting
        * **N-Grams** – Unigram + bigram model features (e.g. "not good"); bigrams and trigrams are also explored in EDA
        """
    )

st.markdown(
    """
    ### Target Variable

    * **Sentiment** – Positive or Negative
    """
)


st.markdown(
    "### 📈 Data Pipeline Dashboard"
)

hdr_1, hdr_2, hdr_3, hdr_4 = st.columns(
    [2, 2, 2, 3]
)

hdr_1.markdown("#### Stage")
hdr_2.markdown("#### Rows")
hdr_3.markdown("#### Columns")
hdr_4.markdown("#### Details")

st.divider()

row1_1, row1_2, row1_3, row1_4 = st.columns(
    [2, 2, 2, 3]
)

row1_1.markdown("### 📁 Raw")

row1_2.metric(
    label="Total Rows",
    value="50,000"
)

row1_3.metric(
    label="Columns",
    value="2"
)

row1_4.text(
    "Original IMDB dataset"
)

st.divider()

row2_1, row2_2, row2_3, row2_4 = st.columns(
    [2, 2, 2, 3]
)

row2_1.markdown("### ✨ Cleaned")

row2_2.metric(
    label="Final Rows",
    value="49,582",
    delta="-418"
)

row2_3.metric(
    label="Columns",
    value="8"
)

with row2_4:

    st.markdown(
        """
        **Preprocessing:**
        * Cleaning
        * Negation handling
        * Lemmatization
        """)

    st.markdown(
        "**Deployment:** Streamlit 🚀"
    )



st.markdown(
    """
    ### 🛠 Technologies Used
    """
)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        """
        #### Programming Language
        * Python

        #### Data Manipulation & Utilities
        * Pandas
        * Pathlib
        * Joblib

        #### Natural Language Processing
        * NLTK (Stopwords, WordNetLemmatizer)
        * Regular Expressions (`re`)

        #### Data Visualization
        * Matplotlib
        * Seaborn
        * WordCloud

        #### Web Application
        * Streamlit
        """
    )

with c2:
    st.markdown(
        """
        #### Machine Learning & Feature Engineering
        * Scikit-learn
            - Train-Test Split
            - CountVectorizer
            - TfidfVectorizer
            - Logistic Regression
            - Multinomial Naive Bayes
            - Classification Metrics (Accuracy, Classification Report, Confusion Matrix, ROC-AUC, Precision-Recall)

        #### Development Environment
        * Jupyter Notebook
        * Visual Studio Code

        #### Version Control
        * Git
        * GitHub
        """
    )



st.markdown("### 🧭 Application Features")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    #### 📊 Exploratory Data Analysis
    * Dataset overview & basic statistics
    * Sentiment distribution
    * Text length analysis
    * Word count analysis
    * Distribution of Sentence Count
    * Most frequent words analysis
    * Positive WordCloud
    * Negative WordCloud
    * Bigram analysis
    * Trigram analysis
    """)

with c2:
    st.markdown("""
    #### 🎯 Sentiment Prediction
    * Enter a custom movie review
    * Apply the same preprocessing pipeline (cleaning + negation handling + lemmatization)
    * Select vectorizer
        - CountVectorizer
        - TF-IDF
    * Choose between trained models:
        - Logistic Regression
        - Multinomial Naive Bayes
    * Get sentiment prediction with confidence score
    """)
