import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
import gc
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from huggingface_hub import hf_hub_download

# Hugging Face dataset URLs
DATASET_REPO = "Maruf39237/imdb-sentiment-app-datasets"


SENTIMENT_ORDER = ["negative", "positive"]
SENTIMENT_COLORS = {"negative": "#e74c3c", "positive": "#2ecc71"}
TEXT_STAT_COLS = ["char_count", "word_count", "char_count_no_spaces", "sentence_count"]



@st.cache_data(show_spinner="Downloading cleaned dataset...")
def load_cleaned():

    file_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="Cleaned IMDB Dataset.csv",
        repo_type="dataset"
    )

    return pd.read_csv(file_path)


@st.cache_data(show_spinner="Downloading raw dataset...")
def load_raw():

    file_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename="IMDB Dataset.csv",
        repo_type="dataset"
    )

    return pd.read_csv(file_path)




@st.cache_data
def get_texts(sentiment="all"):
    data = load_cleaned()
    if sentiment != "all":
        data = data[data["sentiment"] == sentiment]
    return data["clean_review"].fillna("")


@st.cache_data
def get_top_words(sentiment="all", n=20):
    all_words = " ".join(get_texts(sentiment)).split()
    return Counter(all_words).most_common(n)


@st.cache_data
def get_top_ngrams(sentiment="all", ngram_range=(2, 2), n=15):
    vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=5)
    X = vectorizer.fit_transform(get_texts(sentiment))
    counts = X.sum(axis=0).A1
    ngram_df = pd.DataFrame({"ngram": vectorizer.get_feature_names_out(), "count": counts})
    return ngram_df.sort_values("count", ascending=False).head(n).reset_index(drop=True)


@st.cache_data
def create_wordcloud(sentiment="all", bg_color="white", colormap="viridis"):
    return WordCloud(
        width=1200,
        height=600,
        background_color=bg_color,
        max_words=120,
        collocations=False,
        colormap=colormap
    ).generate(" ".join(get_texts(sentiment)))


df = load_cleaned()


c1, c2 = st.columns(2)

with c1:
    st.title("NLP Exploratory Data Analysis📝")
    st.write("#### Discover insights from ~50k IMDB movie reviews for classical NLP text classification")

with c2:
    st.image("https://images.pexels.com/photos/38933571/pexels-photo-38933571.jpeg", width=280, caption="Understanding text is the first step to understanding sentiment 🎬")
st.write("##### Explore the IMDB Movie Reviews dataset through interactive visualizations. Discover text statistics, word frequencies, sentiment patterns, and key phrases that will help us build strong classical NLP models.")

st.divider()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Dataset",
        "📈 Text Statistics",
        "🔤 Word Frequency",
        "☁️ WordClouds",
        "🔗 N-grams"
    ]
)


with tab1:
    st.subheader("📋 Dataset Explorer")

    view_option = st.radio(
        "Select Dataset Version",
        options=["Raw Dataset", "Cleaned / Preprocessed Dataset"],
        horizontal=True,
        index=1,
        key="dataset_version"
    )

    if view_option == "Raw Dataset":
        current_df = load_raw()
        st.info("Showing the **original** IMDB reviews (before any cleaning).")
    else:
        current_df = df
        st.success("Showing the **cleaned & preprocessed** dataset used for modeling.")

    st.write("###### Preview of the selected dataset")
    st.dataframe(current_df.head(), use_container_width=True)

    with st.expander("🔍 View Dataset Sample"):
        st.dataframe(
            current_df.head(100),
            use_container_width=True
        )

    st.subheader("📊 Dataset Summary")

    rows = current_df.shape[0]
    columns = current_df.shape[1]
    missing = int(current_df.isnull().sum().sum())
    duplicates = int(current_df.duplicated().sum())

    pos_count = (current_df["sentiment"] == "positive").sum()
    neg_count = (current_df["sentiment"] == "negative").sum()

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Reviews", f"{rows:,}")
    col2.metric("Columns", columns)
    col3.metric("Missing Values", missing)
    col4.metric("Duplicates", duplicates)
    col5.metric("Positive", f"{pos_count:,}")
    col6.metric("Negative", f"{neg_count:,}")

    st.divider()

    st.subheader("🎯 Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=current_df,
        x="sentiment",
        hue="sentiment",
        order=SENTIMENT_ORDER,
        palette=SENTIMENT_COLORS,
        legend=False,
        ax=ax
    )
    for container in ax.containers:
        ax.bar_label(container, color="black", fontsize=12, fontweight="bold")
    ax.set_title("Positive vs Negative Reviews", color="blue", fontsize=18, fontweight="bold")
    ax.set_xlabel("Sentiment", color="red", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", color="red", fontsize=14, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)
    

    if view_option == "Cleaned / Preprocessed Dataset":
        st.divider()
        st.subheader("📑 Text Length Statistics")
        st.dataframe(df[TEXT_STAT_COLS].describe().round(2), use_container_width=True)
    else:
        st.markdown(
            """
            > 💡 **Note**  
            > Text length columns are created during preprocessing. Switch to
            > **Cleaned / Preprocessed Dataset** to see them. The other tabs always
            > use the cleaned dataset.
            """
        )


with tab2:
    st.subheader("📈 Text Length Distributions")
    st.caption("Based on the cleaned dataset (HTML tags removed before counting).")

    stat_option = st.selectbox(
        "Select Text Feature to Explore",
        options=TEXT_STAT_COLS,
        index=1
    )

    bins = st.slider("Number of bins", min_value=10, max_value=100, value=15, key="stat_bins")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=stat_option,
        bins=bins,
        kde=False,
        color="#6366F1",
        ax=ax
    )
    ax.set_title(f"Distribution of {stat_option}", color="blue", fontsize=18, fontweight="bold")
    ax.set_xlabel(stat_option, color="red", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", color="red", fontsize=14, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)
    st.divider()

    st.subheader("📦 Review Length by Sentiment")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="sentiment",
        y="word_count",
        hue="sentiment",
        order=SENTIMENT_ORDER,
        palette=SENTIMENT_COLORS,
        legend=False,
        ax=ax2
    )
    ax2.set_title("Word Count Distribution by Sentiment", color="blue", fontsize=18, fontweight="bold")
    ax2.set_xlabel("Sentiment", color="red", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Word Count", color="red", fontsize=14, fontweight="bold")
    st.pyplot(fig2)
    plt.close(fig2)


with tab3:
    st.subheader("🔤 Most Frequent Words")

    top_n = st.slider("Show Top N words", min_value=5, max_value=50, value=5, key="top_n_words")

    st.markdown("#### Overall Top Words")
    words, counts = zip(*get_top_words("all", top_n))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=list(counts), y=list(words), hue=list(words), palette="viridis", legend=False, ax=ax)
    ax.set_title("Top Words (All Reviews)", color="blue", fontsize=16, fontweight="bold")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Positive vs Negative")
    fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

    words_p, counts_p = zip(*get_top_words("positive", top_n))
    sns.barplot(x=list(counts_p), y=list(words_p), hue=list(words_p), palette="Greens_r", legend=False, ax=axes[0])
    axes[0].set_title("Top Positive Words", color="green", fontsize=14, fontweight="bold")

    words_n, counts_n = zip(*get_top_words("negative", top_n))
    sns.barplot(x=list(counts_n), y=list(words_n), hue=list(words_n), palette="Reds_r", legend=False, ax=axes[1])
    axes[1].set_title("Top Negative Words", color="red", fontsize=14, fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


with tab4:
    st.subheader("☁️ Word Clouds")

    wc_option = st.radio(
        "Choose WordCloud",
        options=["Overall", "Positive Reviews", "Negative Reviews"],
        horizontal=True,
        key="wc_radio"
    )

    generate = st.button("🔄 Generate WordCloud", type="primary")

    if generate:
        with st.spinner("Generating WordCloud... this may take a few seconds the first time"):
            if wc_option == "Overall":
                wordcloud = create_wordcloud("all", "white", "viridis")
                title = "Overall WordCloud"
            elif wc_option == "Positive Reviews":
                wordcloud = create_wordcloud("positive", "navy", "Greens")
                title = "Positive Reviews WordCloud"
            else:
                wordcloud = create_wordcloud("negative", "black", "Reds")
                title = "Negative Reviews WordCloud"

            fig, ax = plt.subplots(figsize=(12, 7))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(title, color="red", fontsize=20, fontweight="bold")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("👆 Click the button above to generate the WordCloud.")




with tab5:
    st.subheader("🔗 Most Common N-grams")

    if "load_ngrams" not in st.session_state:
        st.session_state.load_ngrams = False
    if "load_sentiment_ngrams" not in st.session_state:
        st.session_state.load_sentiment_ngrams = False

    if st.button("🚀 Load N-gram Analysis", key="btn_load_ngrams"):
        st.session_state.load_ngrams = True

    if st.session_state.load_ngrams:
        ngram_type = st.radio(
            "Select N-gram",
            options=["Bigrams (2 words)", "Trigrams (3 words)"],
            horizontal=True,
            key="ngram_type",
        )

        top_n_ngrams = st.slider("Top N N-grams", 5, 20, 5, key="ngram_slider")

        n_range = (2, 2) if ngram_type == "Bigrams (2 words)" else (3, 3)
        title = "Top Bigrams" if n_range == (2, 2) else "Top Trigrams"

        ngram_df = get_top_ngrams("all", ngram_range=n_range, n=top_n_ngrams)

        fig = px.bar(
            ngram_df,
            x="count",
            y="ngram",
            orientation="h",
            color="count",
            color_continuous_scale="magma",
            title=title,
            labels={"count": "Frequency", "ngram": "N-gram"},
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=400,
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.divider()

        if st.button(
            "📊 Compare Positive vs Negative Bigrams",
            key="btn_sentiment_ngrams",
        ):
            st.session_state.load_sentiment_ngrams = True

        if st.session_state.load_sentiment_ngrams:
            col_a, col_b = st.columns(2)

            with col_a:
                pos_bigrams = get_top_ngrams(
                    "positive", ngram_range=(2, 2), n=12
                )
                fig_p = px.bar(
                    pos_bigrams,
                    x="count",
                    y="ngram",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Greens",
                    title="Positive Bigrams",
                    labels={"count": "Frequency", "ngram": ""},
                )
                fig_p.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_p, use_container_width=True)        
    

            with col_b:
                neg_bigrams = get_top_ngrams(
                    "negative", ngram_range=(2, 2), n=12
                )
                fig_n = px.bar(
                    neg_bigrams,
                    x="count",
                    y="ngram",
                    orientation="h",
                    color="count",
                    color_continuous_scale="Reds",
                    title="Negative Bigrams",
                    labels={"count": "Frequency", "ngram": ""},
                )
                fig_n.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_n, use_container_width=True)       

        gc.collect()