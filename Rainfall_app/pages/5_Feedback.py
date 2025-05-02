import streamlit as st
import os
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="Rainfall App - Feedback", layout="centered")

# Custom CSS for attractive and responsive styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f8ff;
        padding: 20px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        max-width: 600px;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
        box-sizing: border-box;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stTextInput>div>label, .stTextArea>div>label {
        font-weight: 500;
        color: #333;
        margin-bottom: 8px;
    }
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
        transition: border-color 0.3s;
        width: 100%;
        box-sizing: border-box;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
        border-color: #4CAF50;
        outline: none;
    }
    h1 {
        color: #333;
        font-weight: 700;
        text-align: center;
        font-size: clamp(1.8rem, 6vw, 2.2rem);
    }
    .stSuccess {
        background-color: #e6ffe6;
        border-radius: 8px;
        padding: 12px;
        color: #2e7d32;
        width: 100%;
        box-sizing: border-box;
    }
    /* Mobile adjustments */
    @media (max-width: 600px) {
        .main {
            padding: 10px;
        }
        .card {
            padding: 16px;
            max-width: 100%;
        }
        .stButton>button {
            padding: 10px 20px;
            font-size: 0.9rem;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            padding: 8px;
            font-size: 0.9rem;
        }
        .stSuccess {
            padding: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Main content
with st.container():
    st.title("📝 Provide Feedback")
    st.markdown('<div class="card" role="region" aria-label="Feedback Form Section">', unsafe_allow_html=True)
    with st.form("feedback_form"):
        name = st.text_input("Name (optional)", placeholder="Enter your name (optional)", label_visibility="visible")
        feedback = st.text_area("Your Feedback", placeholder="Share your thoughts about the Rainfall App", label_visibility="visible")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with open(os.path.join('data', 'feedback.csv'), 'a') as f:
                # Escape commas and newlines in feedback to prevent CSV issues
                safe_name = name.replace(',', '').replace('\n', ' ')
                safe_feedback = feedback.replace(',', '').replace('\n', ' ')
                f.write(f"{datetime.now()},{safe_name},{safe_feedback}\n")
            st.success("Thank you for your feedback!")
    st.markdown('</div>', unsafe_allow_html=True)