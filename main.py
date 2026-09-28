import streamlit as st


st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬"
)


home_page = st.Page(
    "Pages/Home.py",
    title="Home",
    icon="🏠"
)

eda_page = st.Page(
    "Pages/EDA.py",
    title="EDA",
    icon="📊"
)

prediction_page = st.Page(
    "Pages/Prediction.py",
    title="Prediction",
    icon="🎯"
)


pg = st.navigation(
    [
        home_page,
        eda_page,
        prediction_page
    ]
)

pg.run()