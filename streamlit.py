import os
import zipfile
import gzip
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. UNIFORM KLEURENPALET & STYLING
# ==========================================
PRIMARY_COLOR = "#1E3A8A"       # Donkerblauw / Indigo
SECONDARY_COLOR = "#3B82F6"     # Helderblauw
ACCENT_COLOR = "#60A5FA"        # Lichtblauw
NEUTRAL_DARK = "#1E293B"        # Donkergrijs
NEUTRAL_LIGHT = "#F8FAFC"       # Heel lichtgrijs/wit

COLOR_DISCRETE_MAP = {
    "L": "#1E3A8A",
    "Inbound (L - Landing)": "#1E3A8A",
    "S": "#60A5FA",
    "Outbound (S - Start)": "#60A5FA"
}
COLOR_CONTINUOUS_SCALE = "Blues"

st.set_page_config(
    page_title="Zürich Airport (ZRH) - Vluchten & Vertraging Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Coördinaten Zürich Airport (ZRH / LSZH: Zürich-Kloten)
ZRH_LAT, ZRH_LON = 47.4582, 8.5554

# Header
st.title("✈️ Zürich Airport (ZRH) — Vluchten, Weer & Vertraging Dashboard")
st.markdown("""
**Case 3: Minor Data Science & Visual Analytics**  
*Een integraal dashboard waarin 323.461 vluchten van/naar Zürich Airport, Meteostat weeromstandigheden (station 06670), 
OpenFlights luchthavencoördinaten en hoge-resolutie vluchttelemetrie samenkomen.*
""")


# ==========================================
# 2. HELPER FUNCTIES (AFSTAND & IQR OUTLIER REMOVAL)
# ==========================================
def remove_outliers_iqr(df_in, column, factor=1.5):
    """Verwijder uitschieters uit een dataframe op basis van de Interquartile Range (IQR)."""
    if df_in.empty or column not in df_in.columns:
        return df_in
    q1 = df_in[column].quantile(0.25)
    q3 = df_in[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (factor * iqr)
    upper_bound = q3 + (factor * iqr)
    return df_in[(df_in[column] >= lower_bound) & (df_in[column] <= upper_bound)]


def haversine_distance_vectorized(lat1, lon1, lat2, lon2):
    """Snelle gevectoriseerde Haversine afstandsberekening in kilometers."""
    R = 6371.0 # Straal van de aarde in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def safe_read_csv_or_zip(file_path, header='infer'):
    """Beveiligde lezer die ZIP, CSV of GZ leest zonder te crashen."""
    if not os.path.exists(file_path):
        return None
    
    if zipfile.is_zipfile(file_path):
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                csv_files = [f for f in z.namelist() if f.endswith('.csv') and not f.startswith('__MACOSX')]
                if csv_files:
                    return pd.read_csv(z.open(csv_files[0]), header=header)
        except Exception:
            pass
            
    try:
        return pd.read_csv(file_path, header=header)
    except Exception:
        return None


# ==========================================
# 3. PIPELINE MET CACHING VOOR DATA PREPROCESSING
# ==========================================
@st.cache_data
def load_and_clean_data():
    # A. Schedule Airport Data (Zürich)
    df_sched = None
    candidate_paths = [
        'schedule_airport.csv', 'schedule_airport.zip', 'schedule_airport.csv.zip',
        'data/schedule_airport.csv', 'data/schedule_airport.zip'
    ]
    
    for path in candidate_paths:
        df_sched = safe_read_csv_or_zip(path)
        if df_sched is not None and not df_sched.empty:
            break
            
    if df_sched is None or df_sched.empty:
        st.error("⚠️ Het bestand 'schedule_airport.csv' kon niet worden ingelezen. Plaats de dataset in de hoofdmap van het project.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Datum- en Tijdconversie
    df_sched['STD_DT'] = pd.to_datetime(df_sched['STD'], format='%d/%m/%Y', errors='coerce')
    
    df_sched['Sched_Time'] = pd.to_datetime(
        df_sched['STD'].astype(str) + ' ' + df_sched['STA_STD_ltc'].astype(str), 
        format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )
    df_sched['Act_Time'] = pd.to_datetime(
        df_sched['STD'].astype(str) + ' ' + df_sched['ATA_ATD_ltc'].astype(str), 
        format='%d/%m/%Y %H:%M:%S', errors='coerce'
    )
    
    df_sched['Vertraging_Minuten'] = (df_sched['Act_Time'] - df_sched['Sched_Time']).dt.total_seconds() / 60.0
    
    df_sched.loc[df_sched['Vertraging_Minuten'] < -720, 'Vertraging_Minuten'] += 1440
    df_sched.loc[df_sched['Vertraging_Minuten'] > 720, 'Vertraging_Minuten'] -= 1440
    
    df_sched['Is_Vertraagd'] = df_sched['Vertraging_Minuten'] > 15
    df_sched['Uur'] = df_sched['Sched_Time'].dt.hour
    df_sched['Maand'] = df_sched['Sched_Time'].dt.strftime('%Y-%m')
    df_sched['DagVanWeek'] = df_sched['Sched_Time'].dt.day_name()
    df_sched['ICAO_clean'] = df_sched['Org/Des'].astype(str).str.strip().str.upper()

    # B. Airports Extended Data
    df_airports = None
    airport_paths = [
        'airports-extended-clean.csv', 'airports-extended.csv', 'airports-extended.csv.zip',
        'data/airports-extended-clean.csv', 'data/airports-extended.csv'
    ]
    
    for path in airport_paths:
        df_airports = safe_read_csv_or_zip(path)
        if df_airports is not None and not df_airports.empty:
            break
            
    if df_airports is not None and not df_airports.empty:
        expected_cols = [
            'Airport_ID', 'Name', 'City', 'Country', 'IATA', 'ICAO', 
            'Latitude', 'Longitude', 'Altitude', 'Timezone', 'DST', 'Tz_Database', 'Type', 'Source'
        ]
        if 'Latitude' not in df_airports.columns:
            df_airports.columns = expected_cols[:len(df_airports.columns)]
            
        df_airports['ICAO_clean'] = df_airports['ICAO'].astype(str).str.strip().str.upper()
        
        df_merged = pd.merge(
            df_sched, 
            df_airports[['ICAO_clean', 'Name', 'City', 'Country', 'Latitude', 'Longitude']], 
            on='ICAO_clean', 
            how='left'
        )
        
        df_merged['Afstand_km'] = haversine_distance_vectorized(
            ZRH_LAT, ZRH_LON, df_merged['Latitude'], df_merged['Longitude']
        )
    else:
        df_airports = pd.DataFrame()
        df_merged = df_sched
        df_merged['Afstand_km'] = np.nan

    # C. Meteostat Weerdata (Station 06670 Zürich-Kloten)
    df_weather = pd.DataFrame()
    weather_paths = ['06670.csv.gz', '06670.csv', 'data/06670.csv.gz', 'data/06670.csv']
    
    for wpath in weather_paths:
        if os.path.exists(wpath):
            try:
                if wpath.endswith('.gz'):
                    with gzip.open(wpath, 'rt') as f:
                        df_weather = pd.read_csv(f, header=None)
                else:
                    df_weather = pd.read_csv(wpath, header=None)
                break
            except Exception:
                continue

    if not df_weather.empty:
        cols = ['Datum_Str', 'TG', 'TN', 'TX', 'SQ', 'SP', 'DD', 'FH', 'FF', 'PX', 'VN']
        df_weather = df_weather.iloc[:, :len(cols)]
        df_weather.columns = cols[:df_weather.shape[1]]
        
        df_weather['Weather_Date'] = pd.to_datetime(df_weather['Datum_Str'].astype(str), errors='coerce')
        
        df_weather['Temperatuur_C'] = df_weather['TG'] / 10.0 if df_weather['TG'].max() > 100 else df_weather['TG']
        df_weather['Windsnelheid_m_s'] = df_weather['FH'] / 10.0 if df_weather['FH'].max() > 100 else df_weather['FH']
        
        df_merged = pd.merge(
            df_merged, 
            df_weather[['Weather_Date', 'Temperatuur_C', 'Windsnelheid_m_s']], 
            left_on='STD_DT', 
            right_on='Weather_Date', 
            how='left'
        )

    return df_merged, df_airports, df_weather


@st.cache_data
def load_telemetry_file(filename):
    zip_paths = ['flightdata.zip', 'data/flightdata.zip']
    for zpath in zip_paths:
        if os.path.exists(zpath) and zipfile.is_zipfile(zpath):
            try:
                with zipfile.ZipFile(zpath, 'r') as z:
                    if filename in z.namelist():
                        return pd.read_excel(z.open(filename))
            except Exception:
                pass
    return pd.DataFrame()


# Data inladen
df, df_airports, df_weather = load_and_clean_data()

# ==========================================
# 4. INTERACTIEVE SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Dynamic Dashboard Filters")

if not df.empty and 'STD_DT' in df.columns:
    min_d, max_d = df['STD_DT'].min().date(), df['STD_DT'].max().date()
    date_range = st.sidebar.date_input("Selecteer Datum Periode", [min_d, max_d])
    
    type_filter = st.sidebar.radio("Vluchttype (LSV)", ["Alles", "Inbound (L - Landing)", "Outbound (S - Start)"])
    
    act_list = ["Alles"] + sorted([str(x) for x in df['ACT'].dropna().unique()])
    act_filter = st.sidebar.selectbox("Vliegtuigtype (ACT)", act_list)
    
    rwy_list = ["Alles"] + sorted([str(x) for x in df['RWY'].dropna().unique()])
    rwy_filter = st.sidebar.selectbox("Start-/Landingsbaan (RWY)", rwy_list)
    
    min_delay = st.sidebar.slider("Minimale Vertraging Filter (minuten)", -30, 180, 0)
    
    hide_outliers = st.sidebar.checkbox("Verberg Extreme Uitschieters in Grafieken (IQR)", value=True)
    
    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['STD_DT'].dt.date >= date_range[0]) & 
            (filtered_df['STD_DT'].dt.date <= date_range[1])
        ]
    if type_filter != "Alles":
        code = "L" if "Inbound" in type_filter else "S"
        filtered_df = filtered_df[filtered_df['LSV'] == code]
    if act_filter != "Alles":
        filtered_df = filtered_df[filtered_df['ACT'].astype(str) == act_filter]
    if rwy_filter != "Alles":
        filtered_df = filtered 