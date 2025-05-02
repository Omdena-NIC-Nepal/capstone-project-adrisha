import folium
from streamlit_folium import st_folium
import plotly.express as px

def plot_station_map(reg_perf, locations_df, metric='R2'):
    """
    Generate a Folium map with markers for each station displaying the specified metric.

    Parameters:
    - reg_perf (pd.DataFrame): Regression performance dataframe with 'station_id' and metric columns.
    - locations_df (pd.DataFrame): Locations dataframe with 'station_id', 'lat(deg)', 'lon(deg)', and 'station_name_x' columns.
    - metric (str, optional): The metric to display on the map. Defaults to 'R2'.

    Returns:
    - folium.Map: The generated Folium map.
    """
    # Ensure station_id is the index in both dataframes
    reg_perf = reg_perf.set_index('station_id')
    locations_df = locations_df.set_index('station_id')
    
    # Merge the dataframes on station_id
    merged_df = reg_perf.join(locations_df, how='inner')
    
    # Create a map centered around the mean latitude and longitude
    mean_lat = merged_df['lat(deg)'].mean()
    mean_lon = merged_df['lon(deg)'].mean()
    map_center = [mean_lat, mean_lon]
    
    # Create a Folium map
    m = folium.Map(location=map_center, zoom_start=6)
    
    # Add markers for each station
    for idx, row in merged_df.iterrows():
        folium.Marker(
            [row['lat(deg)'], row['lon(deg)']],
            popup=f"{row['station_name_x']} - {metric}: {row[metric]:.4f}",
            icon=folium.Icon(color='blue')
        ).add_to(m)
    
    return m

def plot_time_series(data, y_columns=['rainfall_sum', 'pred_rainfall'], title="Rainfall Over Time"):
    """
    Create a time series line chart using Plotly Express.

    Parameters:
    - data (pd.DataFrame): Dataframe containing date and y_columns data.
    - y_columns (list, optional): List of column names to plot on the y-axis. Defaults to ['rainfall_sum', 'pred_rainfall'].
    - title (str, optional): Title of the chart. Defaults to "Rainfall Over Time".

    Returns:
    - plotly.graph_objs.Figure: The generated Plotly figure.
    """
    fig = px.line(data, x='date', y=y_columns, title=title)
    return fig