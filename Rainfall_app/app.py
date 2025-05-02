import streamlit as st

# Set page configuration
st.set_page_config(page_title="Rainfall App", layout="centered")

# Custom CSS for attractive and responsive styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        padding: 20px;
    }
    .hero {
        background: linear-gradient(135deg, #3b82f6 0%, #1e3a8a 100%);
        padding: 40px;
        border-radius: 12px;
        text-align: center;
        color: white;
        margin-bottom: 20px;
    }
    .hero h1 {
        font-size: clamp(1.8rem, 5vw, 2.5rem);
        font-weight: 700;
        margin-bottom: 16px;
    }
    .hero p {
        font-size: clamp(1.0rem, 4vw, 1.2rem);
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
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
    .footer {
        background-color: #1e3a8a;
        color: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-size: 1.0rem;
        margin-top: 40px;
    }
    .footer a {
        color: #93c5fd;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    hr {
        border: 0;
        height: 1px;
        background: #e2e8f0;
        margin: 20px 0;
    }
    /* Mobile adjustments */
    @media (max-width: 600px) {
        .hero {
            padding: 20px;
        }
        .main {
            padding: 10px;
        }
        .card {
            padding: 16px;
        }
        .footer {
            padding: 15px;
            font-size: 0.9rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Hero section
with st.container():
    st.markdown('<div class="hero" role="banner" aria-label="Rainfall App Hero Section">', unsafe_allow_html=True)
    st.title("🌧️ Rainfall App for Eastern Nepal")
    st.markdown("Navigate through the pages using the sidebar to explore rainfall predictions, regional analysis, news insights, and provide feedback.")
    st.markdown('</div>', unsafe_allow_html=True)

# Horizontal line
st.markdown('<hr>', unsafe_allow_html=True)

# Main content
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
        Welcome to the Rainfall App, your tool for exploring rainfall predictions and insights in Eastern Nepal. 
        Use the sidebar to access:
        - **Predictions**: View historical and real-time rainfall forecasts.
        - **Regional Analysis**: Analyze model performance across stations.
        - **News Insights**: Discover trends from rainfall-related news.
        - **Feedback**: Share your thoughts to improve the app.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Horizontal line
st.markdown('<hr>', unsafe_allow_html=True)

# Footer with data source and author information
with st.container():
    st.markdown('<div class="footer" role="contentinfo" aria-label="Footer with Data Source and Author Information">', unsafe_allow_html=True)
    st.markdown("""
        **Data Source**: https://cdhm.tu.edu.np" Central Department of Hydrology and Meteorology,
          
                Tribhuvan University, Nepal.
        **Application Developed By**: Rangit Sapkota  
               Kathmandu, Nepal
    """)
    st.markdown('</div>', unsafe_allow_html=True)