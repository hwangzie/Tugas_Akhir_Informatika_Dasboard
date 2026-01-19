import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Monitoring Titik Panas Kabupaten Kuburaya",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fungsi untuk load data real
@st.cache_data
def load_real_data():
    """Load real data dari CSV files"""
    # Load historical data
    historical_df = pd.read_csv('monthly_hotspot_sum.csv')
    
    # Load forecast data
    # forecast_df = pd.read_csv('monthly_hotspot_forecasts_2025.csv')
    # forecast_df = pd.read_csv('better_LSTM_monthly_hotspot_forecasts_2025.csv')
    forecast_df = pd.read_csv('improved_monthly_hotspot_forecasts_2025.csv')
    
    # Load categorical forecast data
    categorical_df = pd.read_csv('categorical_forecasts_2025.csv')
    
    # Load tile boundaries
    tiles_df = pd.read_csv('pontianak_tile_boundaries.csv')
    
    # Load real weather data from Kuburaya
    weather_df = pd.read_csv('Kuburaya Dalam angka 2014-2024.csv')
    
    # Clean weather data
    weather_df['Time'] = pd.to_datetime(weather_df['Time'])
    weather_df = weather_df.rename(columns={
        'Time': 'year_month',
        'penyinaran matahari': 'solar_radiation_raw',
        'avg kecepatan angin(knot)': 'wind_speed_knot',
        'arah angin terbanyak': 'wind_direction_raw',
        'curah hujan(mm)': 'rainfall_raw'
    })
    
    # Convert knots to m/s (1 knot = 0.514444 m/s)
    weather_df['wind_speed_ms'] = pd.to_numeric(weather_df['wind_speed_knot'], errors='coerce') * 0.514444
    
    # Clean rainfall data (remove commas and convert)
    weather_df['rainfall'] = weather_df['rainfall_raw'].astype(str).str.replace(',', '.').apply(pd.to_numeric, errors='coerce')
    
    # Clean solar radiation data
    weather_df['solar_radiation'] = pd.to_numeric(weather_df['solar_radiation_raw'], errors='coerce')
    
    # Convert solar radiation percentage to W/m² (approximate conversion)
    # Assuming max solar radiation ~1000 W/m² at 100%
    weather_df['solar_radiation_wm2'] = weather_df['solar_radiation'] * 10
    
    # Create weather lookup dict
    weather_lookup = {}
    for _, row in weather_df.iterrows():
        if pd.notna(row['year_month']):
            weather_lookup[row['year_month'].strftime('%Y-%m')] = {
                'rainfall': row['rainfall'] if pd.notna(row['rainfall']) else None,
                'solar_radiation': row['solar_radiation_wm2'] if pd.notna(row['solar_radiation_wm2']) else None,
                'wind_speed': row['wind_speed_ms'] if pd.notna(row['wind_speed_ms']) else None
            }
    
    # Create categorical risk lookup for 2025
    categorical_lookup = {}
    for _, row in categorical_df.iterrows():
        year_month_key = pd.to_datetime(row['year_month']).strftime('%Y-%m')
        categorical_lookup[year_month_key] = {}
        for tile_num in range(1, 26):
            tile_col = f'tile_{tile_num}'
            risk_category = row[tile_col]
            # Map English to Indonesian
            if risk_category == 'High':
                risk_level = 'Tinggi'
            elif risk_category == 'Medium':
                risk_level = 'Sedang'
            elif risk_category == 'Low':
                risk_level = 'Rendah'
            else:
                risk_level = 'Rendah'
            categorical_lookup[year_month_key][tile_num] = risk_level
    
    # Combine historical and forecast
    combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
    
    # Convert to long format
    data_list = []
    for _, row in combined_df.iterrows():
        year_month = row['year_month']
        date = pd.to_datetime(year_month)
        year_month_key = date.strftime('%Y-%m')
        
        # Get month for season determination
        month = date.month
        is_dry_season = month in [4, 5, 6, 7, 8, 9, 10]
        
        # Get real weather data if available
        real_weather = weather_lookup.get(year_month_key, {})
        
        for tile_num in range(1, 26):  # 25 tiles
            tile_col = f'tile_{tile_num}'
            hotspot_count = row[tile_col]
            
            # Get tile info
            tile_info = tiles_df[tiles_df['id'] == tile_num].iloc[0]
            lat = (tile_info['lat_top_left'] + tile_info['lat_bottom_left']) / 2
            lon = (tile_info['lon_top_left'] + tile_info['lon_bottom_left']) / 2
            
            # Generate area name based on grid position
            area_name = f"Tile {tile_num}"
            
            # Use real weather data if available, otherwise simulate
            if real_weather.get('rainfall') is not None:
                # Use real data
                rainfall = real_weather['rainfall']
                solar_radiation = real_weather.get('solar_radiation')
                if solar_radiation is None:
                    solar_radiation = 450 if is_dry_season else 350
                    
                wind_speed = real_weather.get('wind_speed')
                if wind_speed is None:
                    wind_speed = 3.5 if is_dry_season else 2.8
                
                # Estimate temperature based on rainfall and season
                if is_dry_season:
                    temperature = 28 - (rainfall / 100) + np.random.normal(0, 0.5)
                else:
                    temperature = 27 - (rainfall / 150) + np.random.normal(0, 0.5)
                
                # Estimate humidity based on rainfall
                humidity = min(95, max(60, 75 + (rainfall / 20) + np.random.normal(0, 3)))
                
                # Wind direction based on season (real data often shows T/A - not available)
                if is_dry_season:
                    wind_direction = 120 + np.random.normal(0, 30)  # Southeast
                else:
                    wind_direction = 240 + np.random.normal(0, 30)  # Southwest
            else:
                # Simulate for 2025 or missing data
                if is_dry_season:
                    rainfall = max(0, 120 - hotspot_count * 10 + np.random.normal(0, 30))
                    temperature = 28 + hotspot_count * 0.1 + np.random.normal(0, 1)
                    humidity = max(60, 75 - hotspot_count * 0.5 + np.random.normal(0, 5))
                    solar_radiation = 450 + np.random.normal(0, 40)
                    wind_speed = 3.5 + np.random.normal(0, 1)
                    wind_direction = 120 + np.random.normal(0, 30)
                else:
                    rainfall = max(0, 280 - hotspot_count * 5 + np.random.normal(0, 50))
                    temperature = 27 + hotspot_count * 0.05 + np.random.normal(0, 1)
                    humidity = max(70, 85 - hotspot_count * 0.3 + np.random.normal(0, 5))
                    solar_radiation = 350 + np.random.normal(0, 40)
                    wind_speed = 2.8 + np.random.normal(0, 1)
                    wind_direction = 240 + np.random.normal(0, 30)
            
            wind_direction = wind_direction % 360
            
            # Ensure all values are numeric (not None)
            rainfall = rainfall if rainfall is not None else 0
            solar_radiation = solar_radiation if solar_radiation is not None else (450 if is_dry_season else 350)
            wind_speed = wind_speed if wind_speed is not None else (3.5 if is_dry_season else 2.8)
            temperature = temperature if temperature is not None else (28 if is_dry_season else 27)
            humidity = humidity if humidity is not None else (75 if is_dry_season else 85)
            
            # FFMC calculation
            ffmc = max(20, min(95, 60 + (hotspot_count * 2) - (rainfall * 0.1)))
            
            # Check if this is 2025 forecast data with categorical risk
            year = date.year
            if year == 2025 and year_month_key in categorical_lookup:
                # Use categorical risk from CSV
                risk_level = categorical_lookup[year_month_key].get(tile_num, 'Rendah')
                # Assign risk score based on categorical level
                if risk_level == 'Tinggi':
                    risk_score = 60  # High risk score
                elif risk_level == 'Sedang':
                    risk_score = 40  # Medium risk score
                else:
                    risk_score = 20  # Low risk score
            else:
                # Calculate risk for historical data
                risk_score = (
                    hotspot_count * 5 + 
                    max(0, (100 - rainfall/3)) * 0.25 +
                    max(0, (temperature - 26)) * 0.15 +
                    max(0, (ffmc - 40)) * 0.15 +
                    max(0, (wind_speed - 2)) * 0.10
                )
                
                if risk_score > 70:
                    risk_level = "Sangat Tinggi"
                elif risk_score > 50:
                    risk_level = "Tinggi"
                elif risk_score > 30:
                    risk_level = "Sedang"
                else:
                    risk_level = "Rendah"
            
            # ISPU calculation
            ispu = max(0, int(45 + (hotspot_count * 3) + np.random.normal(0, 10)))
            
            data_list.append({
                'tanggal': date,
                'area': area_name,
                'tile_id': tile_num,
                'latitude': lat,
                'longitude': lon,
                'titik_panas': hotspot_count,
                'curah_hujan': rainfall,
                'sinaran_matahari': solar_radiation,
                'kecepatan_angin': wind_speed,
                'arah_angin': wind_direction,
                'suhu': temperature,
                'kelembaban': humidity,
                'ffmc': ffmc,
                'ispu': ispu,
                'tingkat_risiko': risk_level,
                'skor_risiko': risk_score,
                'musim': 'Kemarau' if is_dry_season else 'Hujan'
            })
    
    return pd.DataFrame(data_list)

# Load real data
df = load_real_data()


# Sidebar
st.sidebar.title("🔥 Dashboard Forecasting")
st.sidebar.markdown("**Prediksi Titik Panas Kabupaten Kuburaya**")
st.sidebar.markdown("*Kalimantan Barat*")
st.sidebar.markdown("---")

# Area filter
st.sidebar.subheader("📍 Filter Area")
all_areas = sorted(df['area'].unique())
selected_areas = st.sidebar.multiselect(
    "Pilih Area/Tile:",
    options=all_areas,
    default=all_areas
)

# Date range filter - Month based
st.sidebar.subheader("📅 Rentang Waktu")
years = sorted(df['tanggal'].dt.year.unique())
months = list(range(1, 13))
month_names = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
               'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.selectbox("Tahun Awal:", years, index=0)
    start_month = st.selectbox("Bulan Awal:", months, format_func=lambda x: month_names[x-1], index=0)
with col2:
    end_year = st.selectbox("Tahun Akhir:", years, index=len(years)-1)
    end_month = st.selectbox("Bulan Akhir:", months, format_func=lambda x: month_names[x-1], index=11)

# Apply filters
filtered_df = df.copy()

# Filter by areas
if selected_areas:
    filtered_df = filtered_df[filtered_df['area'].isin(selected_areas)]

# Filter by date range
start_date = pd.Timestamp(year=start_year, month=start_month, day=1)
end_date = pd.Timestamp(year=end_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
filtered_df = filtered_df[(filtered_df['tanggal'] >= start_date) & (filtered_df['tanggal'] <= end_date)]

# Header
st.title("🔥 Dashboard Forecasting Titik Panas Kabupaten Kuburaya")
st.markdown("**Prediksi Titik Panas Kabupaten Kuburaya, Kalimantan Barat**")
st.markdown("---")

# Metrics utama - simplified
col1, col2, col3 = st.columns(3)

with col1:
    total_hotspots = filtered_df['titik_panas'].sum()
    st.metric(
        label="Total Titik Panas",
        value=f"{total_hotspots:,.0f}"
    )

with col2:
    avg_hotspots = filtered_df.groupby('tanggal')['titik_panas'].sum().mean()
    st.metric(
        label="Rata-rata per Bulan",
        value=f"{avg_hotspots:.1f}"
    )

with col3:
    max_month = filtered_df.groupby('tanggal')['titik_panas'].sum().idxmax()
    st.metric(
        label="Bulan Puncak",
        value=max_month.strftime('%B')
    )

st.markdown("---")

# Separate historical and forecast data
historical_df = filtered_df[filtered_df['tanggal'].dt.year < 2025]
forecast_df = filtered_df[filtered_df['tanggal'].dt.year == 2025]

# Historical Trend Section
if len(historical_df) > 0:
    st.subheader("📊 Trend Titik Panas Historis (Januari 2014 - Desember 2024)")
    
    # Monthly aggregation for historical
    monthly_historical = historical_df.groupby('tanggal').agg({
        'titik_panas': 'sum',
        'curah_hujan': 'mean'
    }).reset_index()
    
    # Historical line chart
    fig_historical = go.Figure()
    
    fig_historical.add_trace(go.Scatter(
        x=monthly_historical['tanggal'],
        y=monthly_historical['titik_panas'],
        mode='lines+markers',
        name='Titik Panas Historis',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig_historical.update_layout(
        title="Data Historis Titik Panas (2014-2024)",
        xaxis_title="Tahun",
        yaxis_title="Jumlah Titik Panas",
        height=400,
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig_historical, use_container_width=True)
    st.markdown("*Data historis digunakan sebagai dasar untuk prediksi 2025*")

st.markdown("---")

# Forecast Section (Main Focus)
if len(forecast_df) > 0:
    st.subheader("🔮 Prediksi Titik Panas 2025 (Januari - Desember 2025)")
    
    # Monthly aggregation for forecast
    monthly_forecast = forecast_df.groupby('tanggal').agg({
        'titik_panas': 'sum',
        'curah_hujan': 'mean'
    }).reset_index()
    
    # Forecast line chart
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=monthly_forecast['tanggal'],
        y=monthly_forecast['titik_panas'],
        mode='lines+markers',
        name='Prediksi Titik Panas',
        line=dict(color='red', width=3),
        marker=dict(size=10),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))
    
    fig_forecast.update_layout(
        title="Prediksi Titik Panas Bulanan 2025",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Titik Panas",
        height=500,
        hovermode='x unified',
        showlegend=False
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("Data prediksi 2025 tidak tersedia dalam filter yang dipilih")

st.markdown("---")

# Monthly breakdown table - for forecast only
if len(forecast_df) > 0:

    st.info(
        "**📌 Catatan Sumber Data:**\n\n"
        "• **Titik Panas**: Hasil prediksi model LSTM berdasarkan data historis MODIS/VIIRS 2014-2024\n\n"
        "• **Curah Hujan**: Data disimulasikan berdasarkan pola musiman historis Kabupaten Kuburaya. "
        "Simulasi menggunakan rata-rata curah hujan musim kemarau (~120 mm) dan musim hujan (~280 mm) "
        "dengan mempertimbangkan korelasi terhadap jumlah titik panas prediksi."
    )

    st.subheader("📋 Detail Bulanan Prediksi 2025")
    monthly_summary = forecast_df.groupby('tanggal').agg({
        'titik_panas': 'sum',
        'curah_hujan': 'mean'
    }).reset_index()
    monthly_summary['Bulan'] = monthly_summary['tanggal'].dt.strftime('%B %Y')
    monthly_summary['Titik Panas'] = monthly_summary['titik_panas'].round(0).astype(int)
    monthly_summary['Curah Hujan (mm)'] = monthly_summary['curah_hujan'].round(1)
    
    st.dataframe(
        monthly_summary[['Bulan', 'Titik Panas', 'Curah Hujan (mm)']].set_index('Bulan'),
        use_container_width=True
    )
else:
    st.info("Pilih tahun 2025 untuk melihat detail prediksi bulanan")

st.markdown("---")

# Map visualization
st.subheader("Peta Distribusi Spasial")

# If forecast data is available, allow month selection for 2025
if len(forecast_df) > 0:
    # Get available months in 2025
    available_months_2025 = sorted(forecast_df['tanggal'].unique())
    
    # Month selector for map
    col_map1, col_map2 = st.columns([3, 1])
    with col_map2:
        selected_map_month = st.selectbox(
            "Pilih Bulan:",
            options=available_months_2025,
            format_func=lambda x: x.strftime('%B %Y'),
            index=len(available_months_2025)-1  # Default to last month
        )
    
    # Filter data for selected month
    map_data = forecast_df[forecast_df['tanggal'] == selected_map_month]
    
    fig_map = px.scatter_mapbox(
        map_data,
        lat='latitude',
        lon='longitude',
        size='titik_panas',
        color='tingkat_risiko',
        hover_name='area',
        hover_data={'titik_panas': True, 'latitude': False, 'longitude': False},
        color_discrete_map={
            'Rendah': 'green',
            'Sedang': 'yellow', 
            'Tinggi': 'orange',
            'Sangat Tinggi': 'red'
        },
        title=f"Prediksi Sebaran Titik Panas - {selected_map_month.strftime('%B %Y')}",
        mapbox_style="open-street-map",
        zoom=10,
        center={"lat": -0.35, "lon": 109.2}
    )
else:
    # Fallback to latest available data if no forecast
    latest_date = filtered_df['tanggal'].max()
    map_data = filtered_df[filtered_df['tanggal'] == latest_date]
    
    fig_map = px.scatter_mapbox(
        map_data,
        lat='latitude',
        lon='longitude',
        size='titik_panas',
        color='tingkat_risiko',
        hover_name='area',
        hover_data={'titik_panas': True, 'latitude': False, 'longitude': False},
        color_discrete_map={
            'Rendah': 'green',
            'Sedang': 'yellow', 
            'Tinggi': 'orange',
            'Sangat Tinggi': 'red'
        },
        title=f"Sebaran Titik Panas - {latest_date.strftime('%B %Y')}",
        mapbox_style="open-street-map",
        zoom=10,
        center={"lat": -0.35, "lon": 109.2}
    )

fig_map.update_layout(height=700)
st.plotly_chart(fig_map, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
**ℹ️ Informasi Data:**
- Data historis: MODIS/VIIRS 2014-2024
- Data cuaca: Kabupaten Kuburaya  
- Prediksi: Model LSTM untuk tahun 2025
- Cakupan: 25 tiles di Kabupaten Kuburaya
- Sumber: MODIS/VIIRS & Kuburaya Dalam Angka

""")

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "**📊 Tentang Dashboard**\n\n"
    "Dashboard forecasting dengan data:\n"
    "• 25 tiles area Kabupaten Kuburaya\n"
    "• Historis 2014-2024\n"
    "• Prediksi 2025\n\n"
    "**Sumber Data:**\n"
    "• Titik panas: MODIS/VIIRS\n"
    "• Cuaca: Kuburaya Dalam Angka"
)
