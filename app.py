from io import StringIO

import streamlit as st

from nlp import analyze_sentiment_vader, display_sentiment_plotly

with st.sidebar:
    st.title("Sentiment Analyzer")

    selected_option = st.radio(
        "Please select one option",
        ["Text Fields", "File Upload"],
        captions=[
            "Enter three texts",
            "Upload a file with texts",
        ],
        index=None,
    )

if selected_option == "Text Fields":
    st.cache_data.clear()
    st.cache_resource.clear()
    text1 = st.text_input("Text Input 1", placeholder="Enter text")
    text2 = st.text_input("Text Input 2", placeholder="Enter text")
    text3 = st.text_input("Text Input 3", placeholder="Enter text")

    if st.button("Analyze"):
        for text in [text1, text2, text3]:
            text = text.strip()  # Remove any extra whitespace/newline
            sentiment, scores = analyze_sentiment_vader(text)
            fig = display_sentiment_plotly(sentiment, scores)

            st.write(f"Text: {text}")
            st.write(f"Sentiment: {sentiment}, Scores: {scores}")

            st.plotly_chart(fig, use_container_width=True)

if selected_option == "File Upload":
    st.cache_data.clear()
    st.cache_resource.clear()
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        # To read file as string:
        string_data = stringio.read()
        texts = string_data.split("\n")

        for text in texts:
            text = text.strip()  # Remove any extra whitespace/newline
            sentiment, scores = analyze_sentiment_vader(text)
            fig = display_sentiment_plotly(sentiment, scores)

            st.write(f"Text: {text}")
            st.write(f"Sentiment: {sentiment}, Scores: {scores}")

            st.plotly_chart(fig, use_container_width=True)
