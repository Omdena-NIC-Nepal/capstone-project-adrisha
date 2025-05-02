import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_folium import st_folium
from utils.data_utils import load_regional_performance_regression, load_regional_performance_classification, load_feature_data
from utils.visualization_utils import plot_station_map
import folium

# Set page configuration
st.set_page_config(page_title="Regional Analysis Dashboard", layout="centered", initial_sidebar_state="expanded")

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
        margin: 10px 0;
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
    .folium-map {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        .folium-map {
            height: 300px !important;
        }
        .stPlotlyChart {
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Main title
with st.container():
    st.title("📍 Regional Analysis Dashboard")
    st.markdown("Explore regression and classification performance across stations with interactive visualizations.")

# Load performance data
reg_perf = load_regional_performance_regression()
clf_perf = load_regional_performance_classification()

# Load feature data
feature_data = load_feature_data()

# Data Preview Section
with st.container():
    st.markdown('<div class="card" role="region" aria-label="Data Preview Section">', unsafe_allow_html=True)
    st.subheader("📊 Data Preview")
    with st.expander("Feature Data Preview"):
        st.write("Feature Data Preview (before processing):")
        st.dataframe(feature_data.head()[['station_id', 'station_name_x', 'lat(deg)', 'lon(deg)', 'rainfall_sum']], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Locations Data Processing
if 'station_id' not in feature_data.columns:
    with st.container():
        st.markdown('<div class="card" role="region" aria-label="Error Section">', unsafe_allow_html=True)
        st.error("station_id column missing in feature_data")
        st.write("Feature Data Columns:", feature_data.columns.tolist())
        st.markdown('</div>', unsafe_allow_html=True)
else:
    locations_df = feature_data[['station_id', 'station_name_x', 'lat(deg)', 'lon(deg)']].drop_duplicates().set_index('station_id')
    
    # Debug Information Section
    with st.container():
        st.markdown('<div class="card" role="region" aria-label="Debug Information Section">', unsafe_allow_html=True)
        st.subheader("🔍 Debug Information")
        with st.expander("DataFrame Details"):
            st.write("Locations DataFrame Preview:")
            st.dataframe(locations_df.head()[['station_name_x', 'lat(deg)', 'lon(deg)']], use_container_width=True)
            st.write("Regression DataFrame Preview:")
            st.dataframe(reg_perf.head(), use_container_width=True)
            reg_ids = set(reg_perf.index)
            loc_ids = set(locations_df.index)
            common_ids = reg_ids.intersection(loc_ids)
            st.write(f"Number of common station_ids: {len(common_ids)}")
            st.write(f"Sample common station_ids: {list(common_ids)[:5]}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Regression Performance Section
    with st.container():
        st.markdown('<div class="card" role="region" aria-label="Regression Performance Section">', unsafe_allow_html=True)
        st.subheader("📈 Regression Performance by Station")
        if not reg_perf.empty:
            fig_reg = px.bar(reg_perf.reset_index(), x='station_id', y='R2', title="R2 Score by Station",
                            color='R2', color_continuous_scale='Blues')
            fig_reg.update_layout(
                font=dict(size=12),
                xaxis_tickangle=45,
                margin=dict(l=10, r=10, t=50, b=50)
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.warning("No regression performance data available.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Classification Performance Section
    with st.container():
        st.markdown('<div class="card" role="region" aria-label="Classification Performance Section">', unsafe_allow_html=True)
        st.subheader("📉 Classification Performance by Station")
        if not clf_perf.empty:
            fig_clf = px.bar(clf_perf.reset_index(), x='station_id', y='F1', title="F1 Score by Station",
                            color='F1', color_continuous_scale='Blues')
            fig_clf.update_layout(
                font=dict(size=12),
                xaxis_tickangle=45,
                margin=dict(l=10, r=10, t=50, b=50)
            )
            st.plotly_chart(fig_clf, use_container_width=True)
        else:
            st.warning("No classification performance data available.")
        st.markdown('</div>', unsafe_allow_html=True)

# Station Performance Map Section
with st.container():
    st.markdown('<div class="card" role="region" aria-label="Station Performance Map Section">', unsafe_allow_html=True)
    st.subheader("🗺️ Station Performance Map")
    
    # List of predefined locations
    locations_data = {
        'Rajbiraj': {'lat': 26.5419, 'lon': 86.7567},
        'Siraha': {'lat': 26.6397, 'lon': 86.1853},
        'Tarahara': {'lat': 26.7056, 'lon': 87.2569},
        'Taplejung': {'lat': 27.3540, 'lon': 87.6680}
    }

    # Create a map centered around Nepal
    map_center = [27.5, 86.5]
    map_fig = folium.Map(location=map_center, zoom_start=8, tiles="CartoDB Positron")

    # Add markers for each station
    for station, coords in locations_data.items():
        folium.Marker(
            location=[coords['lat'], coords['lon']],
            popup=station,
            icon=folium.Icon(color='blue', icon='cloud', icon_size=(30, 30))
        ).add_to(map_fig)

    # Show the map
    st_folium(map_fig, width='100%', height=500, key="folium_map")
    st.markdown('</div>', unsafe_allow_html=True)