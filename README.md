---
# IMDB Movie Review Sentiment Analysis

Classical NLP binary sentiment classification system built with Logistic Regression and Multinomial Naive Bayes, deployed as an interactive Streamlit application.

> **Live Demo** → https://huggingface.co/spaces/Maruf39237/imdb-sentiment-app
---

## Overview

This project implements a complete **Classical NLP** pipeline for binary sentiment classification on the IMDB Movie Review dataset. It covers everything from exploratory data analysis and text preprocessing to model training, evaluation, and an interactive web application for real-time predictions.

The system classifies movie reviews as **positive** or **negative** using two classic algorithms:

- Logistic Regression
- Multinomial Naive Bayes

Each algorithm is paired with two feature extraction methods:

- CountVectorizer (Bag-of-Words)
- TF-IDF

**Recommended combination:** TF-IDF + Logistic Regression

---

## Features

### Home Page

- Project overview and objectives
- Model performance metrics table
- Full NLP + ML workflow diagram
- Dataset characteristics
- Technology stack

### Exploratory Data Analysis (EDA)

- Dataset explorer (raw vs cleaned)
- Sentiment distribution
- Text length statistics (character count, word count, sentence count)
- Word frequency analysis (overall / positive / negative)
- Interactive WordClouds
- Bigram and Trigram analysis

### Sentiment Prediction

- Manual review entry
- Load random or specific reviews from the held-out test set
- Choose vectorizer (CountVectorizer or TF-IDF)
- Choose model (Logistic Regression or Multinomial Naive Bayes)
- Real-time prediction with confidence scores
- Probability visualization (bar chart + pie chart)
- Option to reveal the true label for test-set reviews
- Full model performance section (Accuracy, Precision, Recall, F1, Confusion Matrix, ROC & Precision-Recall curves)

---

## Project Structure

```text
imdb-sentiment-app/

├── Pages/
│   ├── Home.py
│   ├── EDA.py
│   └── Prediction.py

├── utils/
│   ├── __init__.py
│   └── text_preprocessing.py

├── main.py                         # Streamlit entry point
├── app.py                          # Hugging Face Space launcher
├── requirements.txt                # Deployment dependencies
├── pyproject.toml                  # uv project configuration
├── uv.lock                         # Locked dependency versions
├── LICENSE
└── README.md
```

> **Note:** The large CSV files and trained model artifacts are not included in the GitHub/Space application repository to avoid unnecessary repository size and deployment overhead.
>
> Dataset files are hosted on Hugging Face Dataset Hub, while trained models, vectorizers, and model metrics are hosted on Hugging Face Model Hub. The application downloads the required files at runtime.

---

## Deployment Architecture

The project uses a decoupled architecture for easier deployment and smaller application repositories.

```text
GitHub / Hugging Face Space
        │
        │ Application source code
        ▼
   Streamlit App
        │
        ├──────────────► Hugging Face Dataset Hub
        │                    │
        │                    ├── IMDB Dataset.csv
        │                    └── Cleaned IMDB Dataset.csv
        │
        └──────────────► Hugging Face Model Hub
                             │
                             ├── CountVectorizer
                             ├── TF-IDF Vectorizer
                             ├── Logistic Regression models
                             ├── Multinomial Naive Bayes models
                             └── model_metrics.json
```

The Streamlit application uses `huggingface_hub` and `hf_hub_download()` to retrieve the required datasets and ML artifacts at runtime.

The Hugging Face Space uses a lightweight `app.py` launcher while the main application remains the original Streamlit project.

---

## Dataset

The IMDB datasets are hosted on Hugging Face and are automatically downloaded by the application when needed.

- **Dataset repository:** [Maruf39237/imdb-sentiment-app-datasets](https://huggingface.co/datasets/Maruf39237/imdb-sentiment-app-datasets)

The repository contains:

- `IMDB Dataset.csv`
- `Cleaned IMDB Dataset.csv`

The cleaned dataset is used by the modeling and prediction workflow, while the raw dataset is available for EDA comparison.

---

## Model Artifacts

The trained machine-learning artifacts are hosted separately on Hugging Face Model Hub.

- **Model repository:** [Maruf39237/imdb-sentiment-model](https://huggingface.co/Maruf39237/imdb-sentiment-model)

The repository contains:

```text
imdb-sentiment-model/

├── models/
│   ├── cv_logistic_regression.joblib
│   ├── cv_multinomial_nb.joblib
│   ├── tfidf_logistic_regression.joblib
│   └── tfidf_multinomial_nb.joblib
│
├── vectorizer/
│   ├── count_vectorizer.joblib
│   └── tfidf_vectorizer.joblib
│
└── model_metrics.json
```

The application downloads these artifacts when the Prediction page requires them.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Maruf39237/imdb-sentiment-app.git

cd imdb-sentiment-app
```

### 2. Install dependencies

**Using uv (recommended):**

```bash
uv sync
```

**Or using pip:**

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run main.py
```

or:

```bash
uv run streamlit run main.py
```

The application will open in your browser.

The required dataset, models, and vectorizers are downloaded automatically from Hugging Face when the corresponding application features are used.

---

## Model Performance (Test Set)

| Model                                 | Accuracy | Precision | Recall | F1-Score |
| ------------------------------------- | -------- | --------- | ------ | -------- |
| CountVectorizer + Logistic Regression | –        | –         | –      | –        |
| CountVectorizer + MultinomialNB       | –        | –         | –      | –        |
| TF-IDF + Logistic Regression          | –        | –         | –      | –        |
| TF-IDF + MultinomialNB                | –        | –         | –      | –        |

> Exact model metrics are stored in `model_metrics.json` in the Hugging Face Model repository and are used by the application where required.

---

## Text Preprocessing Pipeline

All text (training and prediction) goes through the same pipeline defined in `utils/text_preprocessing.py`:

1. Lowercasing
2. HTML tag and URL removal
3. Expansion of contractions / negations (`wasn't` → `was not`)
4. Removal of punctuation and non-letter characters
5. Stopword removal while **keeping** important negation words such as `no`, `not`, and `never`
6. Lemmatization (noun + verb)

The final processed text used for model training is stored in the `lemmatized_text` column.

---

## Technologies Used

- **Python** 3.10+
- **Streamlit** – interactive web application
- **scikit-learn** – CountVectorizer, TfidfVectorizer, LogisticRegression, MultinomialNB
- **NLTK** – stopwords and lemmatization
- **pandas**, **numpy**
- **matplotlib**, **seaborn**, **wordcloud**
- **plotly** – interactive visualizations
- **joblib** – model serialization
- **huggingface-hub** – runtime access to Hugging Face datasets and model artifacts
- **uv** – dependency and environment management

---

## Deployment

The application can be run locally with Streamlit and deployed using the same application source code.

For the Hugging Face Space deployment:

```text
app.py
    ↓
launches Streamlit
    ↓
main.py
    ↓
Pages/
    ├── Home.py
    ├── EDA.py
    └── Prediction.py
```

The Space uses a Gradio-based runtime launcher while the user-facing application itself remains Streamlit. The Hugging Face Space configuration is defined in the YAML metadata at the top of this README.

---

## License

This project is licensed under the MIT License.

See the [LICENSE](https://github.com/Maruf39237/imdb-sentiment-app/blob/main/LICENSE) file for details.

---

## Acknowledgements

- IMDB Movie Review Dataset
- scikit-learn and NLTK communities
- Streamlit team
- Hugging Face for hosting the dataset and model artifacts
- Python open-source community

---
