import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    st.error("Please set your GEMINI_API_KEY in the .env file")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

st.title("Sentiment Analysis Dashboard")

uploaded_file = st.file_uploader("Upload CSV file with 'review' column", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'review' not in df.columns:
            st.error("CSV must contain a 'review' column")
            st.stop()

        st.info("Processing reviews... This may take a while for large files.")

        sentiments = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, row in df.iterrows():
            review = str(row['review'])
            prompt = f"Analyze the sentiment of this review: '{review}'. Respond with only one word: positive, negative, or neutral."
            try:
                response = model.generate_content(prompt)
                sentiment = response.text.strip().lower()
                # Ensure it's one of the expected values
                if sentiment not in ['positive', 'negative', 'neutral']:
                    sentiment = 'neutral'
                sentiments.append(sentiment)
            except Exception as e:
                st.warning(f"Error analyzing review {i+1}: {e}")
                sentiments.append('neutral')  # Default to neutral on error

            progress_bar.progress((i + 1) / len(df))
            status_text.text(f"Processed {i + 1} of {len(df)} reviews")

        df['sentiment'] = sentiments

        st.success("Analysis complete!")

        # Display results table
        st.subheader("Sentiment Analysis Results")
        st.dataframe(df)

        # Sentiment distribution chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts()
        fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, title="Overall Sentiment Distribution")
        st.plotly_chart(fig)

        # Additional stats
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            st.metric("Positive", sentiment_counts.get('positive', 0))
        with col3:
            st.metric("Negative", sentiment_counts.get('negative', 0))

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or invalid.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it's a valid CSV.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")