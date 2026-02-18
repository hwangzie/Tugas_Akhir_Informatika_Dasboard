import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Monitoring Titik Panas Kabupaten Kuburaya",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Mapping tile ID to location names
TILE_LOCATION_MAP = {
    1: "Blok SK 1", 2: "Blok SK 2", 3: "Blok SK 3",
    4: "Blok SK 4", 5: "Blok SK 5",
    6: "Blok TP 1", 7: "Blok TP 2", 8: "Blok TP 3",
    9: "Blok TP 4", 10: "Blok TP 5",
    11: "Blok SR 1", 12: "Blok SR 2", 13: "Blok SR 3",
    14: "Blok SR 4", 15: "Blok SR 5",
    16: "Blok BA 1", 17: "Blok BA 2", 18: "Blok BA 3",
    19: "Blok BA 4", 20: "Blok BA 5",
    21: "Blok KB 1", 22: "Blok KB 2", 23: "Blok KB 3",
    24: "Blok KB 4", 25: "Blok KB 5"
}

# Fungsi untuk load data real
@st.cache_data
def load_real_data():
    """Load real data dari CSV files"""
    # Load historical data
    historical_df = pd.read_csv('monthly_hotspot_sum.csv')
    
    # Load forecast data
    # forecast_df = pd.read_csv('monthly_hotspot_forecasts_2025.csv')
    # forecast_df = pd.read_csv('better_LSTM_monthly_hotspot_forecasts_2025.csv')
    # forecast_df = pd.read_csv('improved_monthly_hotspot_forecasts_2025.csv')
    forecast_df = pd.read_csv('monthly_hotspot_forecasts_2025_new.csv')
    
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
            
            # Use location name mapping
            area_name = TILE_LOCATION_MAP.get(tile_num, f"Tile {tile_num}")
            
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

@st.cache_data
def load_validation_data():
    """Load data realisasi/aktual tahun 2025 untuk validasi"""
    try:
        # Membaca file CSV data asli 2025
        val_df = pd.read_csv('real_monthly_hotspot_sum2025.csv')
        
        # Mengubah format data dari lebar (wide) ke panjang (long) agar cocok dengan data forecast
        val_melted = val_df.melt(
            id_vars=['year_month'], 
            var_name='tile_str', 
            value_name='titik_panas_aktual'
        )
        
        # Membersihkan kolom tile_id (mengubah 'tile_1' menjadi angka 1)
        val_melted['tile_id'] = val_melted['tile_str'].str.replace('tile_', '').astype(int)
        
        # Mengubah format tanggal
        val_melted['tanggal'] = pd.to_datetime(val_melted['year_month'])
        
        return val_melted[['tanggal', 'tile_id', 'titik_panas_aktual']]
    except FileNotFoundError:
        return None

# Load real data
df = load_real_data()

# Page Navigation
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Ringkasan Eksekutif", "📋 Detail Data"]
)
st.sidebar.markdown("---")

# Filter Panel
st.sidebar.subheader("Panel Filter")

st.sidebar.info("""
    **Legenda Kode Blok:**
    - **SK** = Sungai Kakap
    - **TP** = Teluk Pakedai
    - **SR** = Sungai Raya
    - **BA** = Batu Ampar
    - **KB** = Kubu Raya
    """)
# Area filter

st.sidebar.markdown("""
<style>
    /* Limit multiselect height */
    div[data-baseweb="select"] > div {
        max-height: 100px;
        overflow-y: auto;
    }
    
    /* Style the selected items container */
    .stMultiSelect [data-baseweb="tag"] {
        font-size: 12px;
        padding: 2px 6px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("**Filter Area/Lokasi**")
all_areas = sorted(df['area'].unique())
selected_areas = st.sidebar.multiselect(
    "Pilih Lokasi:",
    options=all_areas,
    default=all_areas
)

# Date range filter - Month based
st.sidebar.markdown("**Rentang Waktu**")
# Filter to show only 2020 onwards for more relevant data
years = sorted([y for y in df['tanggal'].dt.year.unique() if y >= 2020])
months = list(range(1, 13))
month_names = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni',
               'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

# Default to 2024 if available, otherwise use first/last year
default_year = 2024
default_start_idx = years.index(default_year) if default_year in years else 0
default_end_idx = years.index(default_year+1) if default_year in years else len(years)-1

col1, col2 = st.sidebar.columns(2)
with col1:
    start_year = st.selectbox("Tahun Awal:", years, index=default_start_idx)
    start_month = st.selectbox("Bulan Awal:", months, format_func=lambda x: month_names[x-1], index=0)
with col2:
    end_year = st.selectbox("Tahun Akhir:", years, index=default_end_idx)
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

# Calculate YoY metrics for comparison
prev_year_start = start_date - pd.DateOffset(years=1)
prev_year_end = end_date - pd.DateOffset(years=1)
prev_year_df = df[(df['tanggal'] >= prev_year_start) & (df['tanggal'] <= prev_year_end)]
if selected_areas:
    prev_year_df = prev_year_df[prev_year_df['area'].isin(selected_areas)]

# Separate historical and forecast data
historical_df = filtered_df[filtered_df['tanggal'].dt.year <= 2026]
forecast_df = filtered_df[filtered_df['tanggal'].dt.year == 2025]

# ============================================================================
# PAGE: RINGKASAN EKSEKUTIF
# ============================================================================
if page == "📊 Ringkasan Eksekutif":
    # Header
    st.title("Dashboard Forecasting Titik Panas Kabupaten Kuburaya")
    st.markdown("**Sistem Prakiran dan Monitoring Titik Panas Kabupaten Kuburaya, Kalimantan Barat**")
    
    # Description and constraints
    st.info("""
    **Tentang Dashboard:**
    
    Dashboard ini menyediakan analisis dan prakiran titik panas (hotspot) untuk Kabupaten Kuburaya menggunakan model LSTM (Long Short-Term Memory). 
    Dashboard dirancang untuk membantu pengambilan keputusan dalam mitigasi kebakaran lahan dan hutan.
    
    **Fungsi Utama:**
    - Visualisasi tren historis titik panas (2020-2024)
    - Prakiran bulanan untuk tahun 2025
    - Analisis distribusi spasial per lokasi/blok
    - Kategorisasi tingkat risiko (Rendah, Sedang, Tinggi)
    
    **Batasan:**
    - Data prakiran berbasis model LSTM yang dilatih dengan data historis MODIS/VIIRS (2014-2024)
    - Akurasi prakiran dapat dipengaruhi oleh faktor eksternal yang tidak terprakiran (anomali cuaca ekstrem, dll)
    - Data cuaca 2025 merupakan estimasi berdasarkan pola musiman historis
    - Cakupan: 25 blok area di Kabupaten Kuburaya
    """)
    
    st.markdown("---")

    # KPI Cards with YoY comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        total_hotspots = filtered_df['titik_panas'].sum()
        prev_total_hotspots = prev_year_df['titik_panas'].sum()
        yoy_change = 0 if prev_total_hotspots == 0 else ((total_hotspots - prev_total_hotspots) / prev_total_hotspots) * 100
        
        st.metric(
            label="Total Titik Panas",
            value=f"{total_hotspots:,.0f}",
            delta=f"{yoy_change:+.1f}% YoY" if prev_total_hotspots > 0 else "N/A"
        )

    with col2:
        avg_hotspots = filtered_df.groupby('tanggal')['titik_panas'].sum().mean()
        prev_avg_hotspots = prev_year_df.groupby('tanggal')['titik_panas'].sum().mean()
        yoy_avg_change = 0 if prev_avg_hotspots == 0 else ((avg_hotspots - prev_avg_hotspots) / prev_avg_hotspots) * 100
        
        st.metric(
            label="Rata-rata Titik Panas per Bulan",
            value=f"{avg_hotspots:.1f}",
            delta=f"{yoy_avg_change:+.1f}% YoY" if prev_avg_hotspots > 0 else "N/A"
        )

    with col3:
        monthly_totals = filtered_df.groupby('tanggal')['titik_panas'].sum()
        max_month = monthly_totals.idxmax()
        max_month_value = monthly_totals.max()
        
        st.metric(
            label="Bulan Puncak Titik Panas",
            value=max_month.strftime('%B %Y'),
            delta=f"{max_month_value:,.0f} titik panas"
        )

    st.markdown("---")

    # Combined Historical and Forecast Chart
    st.subheader("Tren Titik Panas: Data Historis vs Prakiran")
    
    # Monthly aggregation
    monthly_historical = historical_df.groupby('tanggal').agg({
        'titik_panas': 'sum',
        'curah_hujan': 'mean'
    }).reset_index()
    
    monthly_forecast = forecast_df.groupby('tanggal').agg({
        'titik_panas': 'sum',
        'curah_hujan': 'mean'
    }).reset_index()
    
    # Combined chart
    fig_combined = go.Figure()
    
    # Historical data - solid line
    if len(monthly_historical) > 0:
        fig_combined.add_trace(go.Scatter(
            x=monthly_historical['tanggal'],
            y=monthly_historical['titik_panas'],
            mode='lines+markers',
            name='Realisasi (Historis)',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6),
            hovertemplate='<b>Realisasi</b><br>Tanggal: %{x|%B %Y}<br>Titik Panas: %{y:,.0f}<extra></extra>'
        ))
    
    # Forecast data - dashed line with different color
    if len(monthly_forecast) > 0:
        fig_combined.add_trace(go.Scatter(
            x=monthly_forecast['tanggal'],
            y=monthly_forecast['titik_panas'],
            mode='lines+markers',
            name='Prakiran (Forecast)',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, symbol='diamond'),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.1)',
            hovertemplate='<b>Prakiran</b><br>Tanggal: %{x|%B %Y}<br>Titik Panas: %{y:,.0f}<extra></extra>'
        ))
    
    fig_combined.update_layout(
        title="Perbandingan Data Realisasi dan Prakiran Titik Panas",
        xaxis_title="Periode",
        yaxis_title="Jumlah Titik Panas",
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_combined, use_container_width=True)
    
    st.info("""
    **Interpretasi Grafik:**
    - **Garis Biru Solid**: Data realisasi/historis dari pengamatan satelit MODIS/VIIRS
    - **Garis Oranye Putus-putus**: Prakiran model LSTM untuk periode mendatang
    - Pola musiman terlihat jelas dengan puncak pada bulan-bulan kemarau (Juli-Oktober)
    """)
    
    st.markdown("---")

    st.title("Evaluasi Akurasi Model Forecasting (2025)")
    st.markdown("**Perbandingan Data Prakiraan (Forecast) vs Realisasi (Aktual)**")
    
    # 1. Load Data
    validation_df = load_validation_data()
    
    if validation_df is None:
        st.error("File 'real_monthly_hotspot_sum2025.csv' tidak ditemukan. Mohon upload file tersebut.")
    else:
        # Ambil data forecast khusus tahun 2025 dari dataset utama
        forecast_2025 = df[df['tanggal'].dt.year == 2025].copy()
        
        # Gabungkan (Merge) data Forecast dan Aktual berdasarkan Tanggal dan Lokasi
        eval_df = pd.merge(
            forecast_2025,
            validation_df,
            on=['tanggal', 'tile_id'],
            how='inner',
            suffixes=('_pred', '_act')
        )
        
        # Filter berdasarkan area yang dipilih di sidebar
        if selected_areas:
            eval_df = eval_df[eval_df['area'].isin(selected_areas)]
            
        if len(eval_df) > 0:
            # 2. Perhitungan Error (MAPE & MAE)
            # Menangani pembagian dengan nol untuk MAPE:
            # Kita gunakan pendekatan "Safe MAPE" dimana jika nilai aktual 0, dianggap 1 untuk pembagi
            # agar tidak error infinity. Ini umum untuk data kejadian jarang (count data).
            
            y_true = eval_df['titik_panas_aktual']
            y_pred = eval_df['titik_panas']
            
            # Hitung MAE (Mean Absolute Error) - Rata-rata selisih mutlak
            mae = np.mean(np.abs(y_true - y_pred))
            
            # Hitung MAPE (Mean Absolute Percentage Error)
            # Rumus: Rata-rata dari |(Aktual - Prediksi) / Max(Aktual, 1)| * 100
            mape_per_point = np.abs((y_true - y_pred) / np.maximum(y_true, 1)) * 100
            mape = np.mean(mape_per_point)
            
            # Hitung Akurasi (100% - MAPE)
            accuracy = max(0, 100 - mape)
            
            # 3. Tampilkan KPI
            st.markdown("### 📊 Metrik Performa Model")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "MAPE (Tingkat Error)", 
                    f"{mape:.2f}%", 
                    help="Rata-rata persentase kesalahan. Semakin KECIL semakin baik."
                )
            
            with col2:
                st.metric(
                    "Akurasi Model", 
                    f"{accuracy:.2f}%", 
                    help="Estimasi ketepatan prediksi (100% - MAPE)."
                )
                
            with col3:
                st.metric(
                    "MAE (Rata-rata Selisih)", 
                    f"{mae:.2f} titik", 
                    help="Rata-rata selisih jumlah titik panas (prediksi vs aktual)."
                )
                
            st.markdown("---")
            
            # 4. Visualisasi Grafik Perbandingan
            st.subheader("Tren: Prediksi vs Aktual")
            
            # Agregasi per bulan untuk grafik garis
            monthly_eval = eval_df.groupby('tanggal').agg({
                'titik_panas': 'sum',
                'titik_panas_aktual': 'sum'
            }).reset_index()
            
            fig_comp = go.Figure()
            
            # Garis Data Aktual
            fig_comp.add_trace(go.Scatter(
                x=monthly_eval['tanggal'],
                y=monthly_eval['titik_panas_aktual'],
                mode='lines+markers',
                name='Aktual (Realisasi)',
                line=dict(color='#2ecc71', width=3), # Hijau
                marker=dict(size=8)
            ))
            
            # Garis Data Prediksi
            fig_comp.add_trace(go.Scatter(
                x=monthly_eval['tanggal'],
                y=monthly_eval['titik_panas'],
                mode='lines+markers',
                name='Prakiraan (Forecast)',
                line=dict(color='#e74c3c', width=2, dash='dash'), # Merah putus-putus
                marker=dict(size=6, symbol='x')
            ))
            
            fig_comp.update_layout(
                title="Perbandingan Jumlah Titik Panas Bulanan (2025)",
                xaxis_title="Bulan",
                yaxis_title="Jumlah Titik Panas",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # 5. Tabel Detail Error per Bulan
            st.subheader("Rincian Error per Bulan")
            
            # Hitung error per bulan
            monthly_eval['Selisih (Diff)'] = monthly_eval['titik_panas'] - monthly_eval['titik_panas_aktual']
            monthly_eval['MAPE Bulanan (%)'] = (
                np.abs(monthly_eval['Selisih (Diff)']) / 
                np.maximum(monthly_eval['titik_panas_aktual'], 1) * 100
            ).round(2)
            
            # Format tampilan tabel
            display_table = monthly_eval.rename(columns={
                'tanggal': 'Bulan',
                'titik_panas': 'Prediksi',
                'titik_panas_aktual': 'Aktual'
            })
            
            display_table['Bulan'] = display_table['Bulan'].dt.strftime('%B %Y')
            
            # Styling tabel
            st.dataframe(
                display_table.style.background_gradient(subset=['MAPE Bulanan (%)'], cmap='Reds'),
                use_container_width=True
            )
            
            st.info("""
            **Catatan Perhitungan MAPE:**
            Karena data titik panas sering bernilai 0 (nol), perhitungan MAPE menggunakan penyesuaian (Safe MAPE) 
            dimana pembagi minimal dianggap 1. Hal ini mencegah error pembagian dengan nol dan tetap memberikan 
            gambaran akurasi yang representatif.
            """)
            
        else:
            st.warning("Data untuk tahun 2025 tidak ditemukan dalam rentang filter yang dipilih.")

    # Map visualization
    st.subheader("Peta Distribusi Spasial Titik Panas")

    # Month selector for map
    if len(forecast_df) > 0:
        available_months_2025 = sorted(forecast_df['tanggal'].unique())
        
        col_map1, col_map2 = st.columns([3, 1])
        with col_map2:
            selected_map_month = st.selectbox(
                "Pilih Bulan:",
                options=available_months_2025,
                format_func=lambda x: x.strftime('%B %Y'),
                index=len(available_months_2025)-1
            )
        
        map_data = forecast_df[forecast_df['tanggal'] == selected_map_month]
    else:
        latest_date = filtered_df['tanggal'].max()
        map_data = filtered_df[filtered_df['tanggal'] == latest_date]
        selected_map_month = latest_date
    
    # Create map with better zoom settings
    fig_map = px.scatter_mapbox(
        map_data,
        lat='latitude',
        lon='longitude',
        size='titik_panas',
        color='tingkat_risiko',
        hover_name='area',
        hover_data=['titik_panas', 'tingkat_risiko'],
        color_discrete_map={
            'Rendah': '#2ecc71',
            'Sedang': '#f39c12', 
            'Tinggi': '#e74c3c',
            'Sangat Tinggi': '#c0392b'
        },
        title=f"Sebaran Risiko Titik Panas - {selected_map_month.strftime('%B %Y')}",
        mapbox_style="open-street-map",
        zoom=8.5,  # Adjusted for better view of entire region
        center={"lat": -0.35, "lon": 109.2},
        size_max=30
    )
    
    fig_map.update_layout(
        height=700,
        mapbox=dict(
            bearing=0,
            pitch=0
        )
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

            

# ============================================================================
# PAGE: DETAIL DATA
# ============================================================================
elif page == "📋 Detail Data":
    st.title("Detail Data Prakiraan Titik Panas 2025")
    st.markdown("**Analisis Detail dan Tabel Data Bulanan**")
    st.markdown("---")
    
    if len(forecast_df) > 0:
        st.subheader("Ringkasan Bulanan Prakiraan 2025")
        st.info(
            "**Catatan Sumber Data:**\n\n"
            "• **Titik Panas**: Hasil prakiran model LSTM berdasarkan data historis MODIS/VIIRS 2014-2024\n\n"
            "• **Curah Hujan**: Data disimulasikan berdasarkan pola musiman historis Kabupaten Kuburaya. "
            "Simulasi menggunakan rata-rata curah hujan musim kemarau (~120 mm) dan musim hujan (~280 mm) "
            "dengan mempertimbangkan korelasi terhadap jumlah titik panas prakiran.\n\n"
            "• **Kategori Risiko**: Dihitung berdasarkan threshold dari metode Quartile pada skor risiko prakiran titik panas."
        )
        # Monthly summary with risk categorization
        monthly_summary = forecast_df.groupby('tanggal').agg({
            'titik_panas': 'sum',
            'curah_hujan': 'mean',
            'tingkat_risiko': lambda x: x.mode()[0] if len(x) > 0 else 'Rendah'
        }).reset_index()
        
        monthly_summary['Bulan'] = monthly_summary['tanggal'].dt.strftime('%B %Y')
        monthly_summary['Titik Panas'] = monthly_summary['titik_panas'].round(0).astype(int)
        monthly_summary['Curah Hujan (mm)'] = monthly_summary['curah_hujan'].round(1)
        monthly_summary['Kategori Risiko'] = monthly_summary['tingkat_risiko']
        
        # Get min and max values for conditional formatting
        min_val = monthly_summary['Titik Panas'].min()
        max_val = monthly_summary['Titik Panas'].max()
        
        # Display table with styling
        display_df = monthly_summary[['Bulan', 'Titik Panas', 'Curah Hujan (mm)', 'Kategori Risiko']].copy()
        
        # Apply conditional formatting using Styler
        def highlight_values(row):
            styles = [''] * len(row)
            if row['Titik Panas'] == min_val:
                styles[1] = 'background-color: #d4edda; color: #155724; font-weight: bold'
            elif row['Titik Panas'] == max_val:
                styles[1] = 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            
            # Color code risk category
            if row['Kategori Risiko'] == 'Tinggi':
                styles[3] = 'background-color: #f8d7da; color: #721c24; font-weight: bold'
            elif row['Kategori Risiko'] == 'Sedang':
                styles[3] = 'background-color: #fff3cd; color: #856404; font-weight: bold'
            else:
                styles[3] = 'background-color: #d4edda; color: #155724; font-weight: bold'
            
            return styles
        
        styled_df = display_df.style.apply(highlight_values, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=500)
        
        st.markdown("""
        **Legenda Tabel:**
        - 🟢 **Hijau**: Nilai terendah atau Risiko Rendah
        - 🔴 **Merah**: Nilai tertinggi atau Risiko Tinggi
        - 🟡 **Kuning**: Risiko Sedang
        """)
        
        st.markdown("---")
        
        # Detailed forecast chart
        st.subheader("Grafik Detail Prakiraan 2025")
        
        fig_detail = go.Figure()
        
        fig_detail.add_trace(go.Bar(
            x=monthly_summary['tanggal'],
            y=monthly_summary['titik_panas'],
            name='Prakiraan Titik Panas',
            marker=dict(
                color=monthly_summary['titik_panas'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Jumlah")
            ),
            hovertemplate='<b>%{x|%B %Y}</b><br>Titik Panas: %{y:,.0f}<extra></extra>'
        ))
        
        fig_detail.update_layout(
            title="Prakiran Titik Panas per Bulan (2025)",
            xaxis_title="Bulan",
            yaxis_title="Jumlah Titik Panas",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_detail, use_container_width=True)
        
        st.markdown("---")
        
        # Area-wise breakdown
        st.subheader("Breakdown per Lokasi")
        
        area_summary = forecast_df.groupby('area').agg({
            'titik_panas': 'sum',
            'tingkat_risiko': lambda x: x.mode()[0] if len(x) > 0 else 'Rendah'
        }).reset_index()
        
        area_summary = area_summary.sort_values('titik_panas', ascending=False)
        area_summary.columns = ['Lokasi', 'Total Prakiran Titik Panas (2025)', 'Kategori Risiko Dominan']
        area_summary['Total Prakiran Titik Panas (2025)'] = area_summary['Total Prakiran Titik Panas (2025)'].round(0).astype(int)
        
        st.dataframe(area_summary, use_container_width=True, height=400)
        
        
    else:
        st.warning("Data prakiran 2025 tidak tersedia. Silakan sesuaikan filter rentang waktu.")
        st.info("Pilih tahun 2025 pada filter sidebar untuk melihat data prakiran.")



# Footer
st.markdown("---")
# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "**Informasi Dashboard**\n\n"
    "**Cakupan Data:**\n"
    "• 25 blok area Kabupaten Kuburaya\n"
    "• Historis: 2020-2024\n"
    "• Prakiran: 2025\n\n"
    "**Model:**\n"
    "• LSTM (Long Short-Term Memory)\n"
    "• Training: Data 2014-2024\n\n"
    "**Sumber Data:**\n"
    "• Titik panas: MODIS/VIIRS\n"
    "• Cuaca: Kuburaya Dalam Angka\n\n"
    "**Kategori Risiko:**\n"
    "🟢 Rendah\n"
    "🟡 Sedang\n"
    "🔴 Tinggi"
)
