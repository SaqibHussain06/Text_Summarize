import streamlit as st
from transformers import pipeline

# Check for PyTorch availability
try:
    import torch
except ImportError:
    st.error("PyTorch is not installed. Please install it using 'pip install torch'.")

# Load the summarization pipeline
@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

pipe = load_model()

# Streamlit app interface
st.title("Text Summarization App")
st.write("This app summarizes text using the **facebook/bart-large-cnn** model.")

# Input text
input_text = st.text_area("Enter the text you want to summarize:", height=300)

# Summary button
if st.button("Summarize"):
    if input_text.strip():
        st.write("Summarizing...")
        try:
            # Perform summarization
            summary = pipe(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("Summary:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.error("Please enter text to summarize!")
