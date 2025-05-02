import streamlit as st
from utils.data_utils import load_model_evaluation_results

# Set page configuration
st.set_page_config(page_title="Rainfall Prediction App", layout="centered", initial_sidebar_state="expanded")

# Custom CSS for attractive and responsive styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        padding: 20px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }
    .card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        width: 100%;
        box-sizing: border-box;
    }
    .metric-box {
        background-color: #eff6ff;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        width: 100%;
        box-sizing: border-box;
    }
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        font-size: clamp(1.8rem, 5vw, 2.5rem);
    }
    h2 {
        color: #1e40af;
        font-weight: 600;
        font-size: clamp(1.4rem, 4vw, 1.8rem);
    }
    .stMetric {
        background-color: #eff6ff;
        padding: 16px;
        border-radius: 8px;
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
        }
        .metric-box {
            padding: 12px;
            margin: 8px 0;
        }
        .stButton>button {
            padding: 8px 16px;
            font-size: 0.9rem;
        }
        .stMetric {
            padding: 12px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Main title and introduction
with st.container():
    st.title("🌧️ Rainfall Prediction App for Eastern Nepal")
    st.markdown('<div class="card" role="region" aria-label="Introduction Section">', unsafe_allow_html=True)
    st.markdown("""
        Welcome to the Rainfall App, designed to predict rainfall and extreme weather events in Eastern Nepal using advanced machine learning models. This app also provides insights from news articles analyzed through natural language processing (NLP).

        ### Features
        - **Predictions**: View historical predictions or input new data for rainfall forecasts.
        - **Regional Analysis**: Explore model performance across different stations.
        - **News Insights**: Discover trends and sentiments from rainfall-related news.
        - **Feedback**: Share your thoughts to improve the app.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Key Metrics Section
with st.container():
    st.markdown('<div class="card" role="region" aria-label="Key Metrics Section">', unsafe_allow_html=True)
    st.subheader("📊 Key Metrics")
    results = load_model_evaluation_results()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Regression R2 Score", results[results['Metric'] == 'R2']['Value'].iloc[0])
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric("Classification F1 Score", results[results['Metric'] == 'F1']['Value'].iloc[0])
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)