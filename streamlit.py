import gzip
import os
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

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
    initial_sidebar_state="expanded",
)

# Coördinaten Zürich Airport (ZRH / LSZH: Zürich-Kloten)
ZRH_LAT, ZRH_LON = 47.4582, 8.5554

# Header
st.title("✈️ Zürich Airport (ZRH) — Vluchten, Weer & Vertraging Dashboard")
st.markdown(
    """
    **Case 3: Minor Data Science & Visual Analytics**  
    *Een integraal dashboard waarin 323.461 vluchten van/naar Zürich Airport, Meteostat weeromstandigheden (station 06670), 
    OpenFlights luchthavencoördinaten en hoge-resolutie vluchttelemetrie samenkomen.*
    """
)


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
    R = 6371.0  # Straal van de aarde in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return R * c


def safe_read_csv_or_zip(file_path, header="infer"):
    """Beveiligde lezer die ZIP, CSV of GZ leest zonder te crashen."""
    if not os.path.exists(file_path):
        return None

    if zipfile.is_zipfile(file_path):
        try:
            with zipfile.ZipFile(file_path, "r") as z:
                csv_files = [
                    f
                    for f in z.namelist()
                    if f.endswith(".csv") and not f.startswith("__MACOSX")
                ]
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
        "schedule_airport.csv",
        "schedule_airport.zip",
        "schedule_airport.csv.zip",
        "data/schedule_airport.csv",
        "data/schedule_airport.zip",
    ]

    for path in candidate_paths:
        df_sched = safe_read_csv_or_zip(path)
        if df_sched is not None and not df_sched.empty:
            break

    if df_sched is None or df_sched.empty:
        st.error(
            "⚠️ Het bestand 'schedule_airport.csv' kon niet worden ingelezen. "
            "Plaats de dataset in de hoofdmap van het project."
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Datum- en Tijdconversie
    df_sched["STD_DT"] = pd.to_datetime(
        df_sched["STD"], format="%d/%m/%Y", errors="coerce"
    )

    df_sched["Sched_Time"] = pd.to_datetime(
        df_sched["STD"].astype(str) + " " + df_sched["STA_STD_ltc"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )
    df_sched["Act_Time"] = pd.to_datetime(
        df_sched["STD"].astype(str) + " " + df_sched["ATA_ATD_ltc"].astype(str),
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    df_sched["Vertraging_Minuten"] = (
        df_sched["Act_Time"] - df_sched["Sched_Time"]
    ).dt.total_seconds() / 60.0

    df_sched.loc[df_sched["Vertraging_Minuten"] < -720, "Vertraging_Minuten"] += 1440
    df_sched.loc[df_sched["Vertraging_Minuten"] > 720, "Vertraging_Minuten"] -= 1440

    df_sched["Is_Vertraagd"] = df_sched["Vertraging_Minuten"] > 15
    df_sched["Uur"] = df_sched["Sched_Time"].dt.hour
    df_sched["Maand"] = df_sched["Sched_Time"].dt.strftime("%Y-%m")
    df_sched["DagVanWeek"] = df_sched["Sched_Time"].dt.day_name()
    df_sched["ICAO_clean"] = (
        df_sched["Org/Des"].astype(str).str.strip().str.upper()
    )

    # B. Airports Extended Data
    df_airports = None
    airport_paths = [
        "airports-extended-clean.csv",
        "airports-extended.csv",
        "airports-extended.csv.zip",
        "data/airports-extended-clean.csv",
        "data/airports-extended.csv",
    ]

    for path in airport_paths:
        df_airports = safe_read_csv_or_zip(path)
        if df_airports is not None and not df_airports.empty:
            break

    if df_airports is not None and not df_airports.empty:
        expected_cols = [
            "Airport_ID", "Name", "City", "Country", "IATA", "ICAO",
            "Latitude", "Longitude", "Altitude", "Timezone", "DST",
            "Tz_Database", "Type", "Source"
        ]
        if "Latitude" not in df_airports.columns:
            df_airports.columns = expected_cols[: len(df_airports.columns)]

        df_airports["ICAO_clean"] = (
            df_airports["ICAO"].astype(str).str.strip().str.upper()
        )

        df_merged = pd.merge(
            df_sched,
            df_airports[
                [
                    "ICAO_clean",
                    "Name",
                    "City",
                    "Country",
                    "Latitude",
                    "Longitude",
                ]
            ],
            on="ICAO_clean",
            how="left",
        )

        df_merged["Afstand_km"] = haversine_distance_vectorized(
            ZRH_LAT, ZRH_LON, df_merged["Latitude"], df_merged["Longitude"]
        )
    else:
        df_airports = pd.DataFrame()
        df_merged = df_sched
        df_merged["Afstand_km"] = np.nan

    # C. Meteostat Weerdata (Station 06670 Zürich-Kloten)
    df_weather = pd.DataFrame()
    weather_paths = [
        "06670.csv.gz",
        "06670.csv",
        "data/06670.csv.gz",
        "data/06670.csv",
    ]

    for wpath in weather_paths:
        if os.path.exists(wpath):
            try:
                if wpath.endswith(".gz"):
                    with gzip.open(wpath, "rt") as f:
                        df_weather = pd.read_csv(f, header=None)
                else:
                    df_weather = pd.read_csv(wpath, header=None)
                break
            except Exception:
                continue

    if not df_weather.empty:
        cols = [
            "Datum_Str", "TG", "TN", "TX", "SQ", "SP", "DD", "FH", "FF", "PX", "VN"
        ]
        df_weather = df_weather.iloc[:, : len(cols)]
        df_weather.columns = cols[: df_weather.shape[1]]

        df_weather["Weather_Date"] = pd.to_datetime(
            df_weather["Datum_Str"].astype(str), errors="coerce"
        )

        df_weather["Temperatuur_C"] = (
            df_weather["TG"] / 10.0 if df_weather["TG"].max() > 100 else df_weather["TG"]
        )
        df_weather["Windsnelheid_m_s"] = (
            df_weather["FH"] / 10.0 if df_weather["FH"].max() > 100 else df_weather["FH"]
        )

        df_merged = pd.merge(
            df_merged,
            df_weather[["Weather_Date", "Temperatuur_C", "Windsnelheid_m_s"]],
            left_on="STD_DT",
            right_on="Weather_Date",
            how="left",
        )

    return df_merged, df_airports, df_weather


@st.cache_data
def load_telemetry_file(filename):
    zip_paths = ["flightdata.zip", "data/flightdata.zip"]
    for zpath in zip_paths:
        if os.path.exists(zpath) and zipfile.is_zipfile(zpath):
            try:
                with zipfile.ZipFile(zpath, "r") as z:
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

if not df.empty and "STD_DT" in df.columns:
    min_d, max_d = df["STD_DT"].min().date(), df["STD_DT"].max().date()
    date_range = st.sidebar.date_input("Selecteer Datum Periode", [min_d, max_d])

    type_filter = st.sidebar.radio(
        "Vluchttype (LSV)",
        ["Alles", "Inbound (L - Landing)", "Outbound (S - Start)"],
    )

    act_list = ["Alles"] + sorted([str(x) for x in df["ACT"].dropna().unique()])
    act_filter = st.sidebar.selectbox("Vliegtuigtype (ACT)", act_list)

    rwy_list = ["Alles"] + sorted([str(x) for x in df["RWY"].dropna().unique()])
    rwy_filter = st.sidebar.selectbox("Start-/Landingsbaan (RWY)", rwy_list)

    min_delay = st.sidebar.slider(
        "Minimale Vertraging Filter (minuten)", -30, 180, 0
    )

    hide_outliers = st.sidebar.checkbox(
        "Verberg Extreme Uitschieters in Grafieken (IQR)", value=True
    )

    filtered_df = df.copy()
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["STD_DT"].dt.date >= date_range[0])
            & (filtered_df["STD_DT"].dt.date <= date_range[1])
        ]
    if type_filter != "Alles":
        code = "L" if "Inbound" in type_filter else "S"
        filtered_df = filtered_df[filtered_df["LSV"] == code]
    if act_filter != "Alles":
        filtered_df = filtered_df[filtered_df["ACT"].astype(str) == act_filter]
    if rwy_filter != "Alles":
        filtered_df = filtered_df[filtered_df["RWY"].astype(str) == rwy_filter]

    filtered_df = filtered_df[filtered_df["Vertraging_Minuten"] >= min_delay]
else:
    filtered_df = df
    hide_outliers = True

# Dataframe opgeschoond van uitschieters voor de grafieken
if hide_outliers and not filtered_df.empty:
    chart_df = remove_outliers_iqr(filtered_df, "Vertraging_Minuten", factor=1.5)
else:
    chart_df = filtered_df

# ==========================================
# 5. EXECUTIVE KPI HIGHLIGHT CARDS
# ==========================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(label="📊 Totaal Vluchten", value=f"{len(filtered_df):,}")
with k2:
    avg_d = chart_df["Vertraging_Minuten"].mean() if not chart_df.empty else 0
    st.metric(label="⏱️ Gem. Vertraging", value=f"{avg_d:.1f} min")
with k3:
    pct_d = (
        (filtered_df["Is_Vertraagd"].sum() / len(filtered_df) * 100)
        if len(filtered_df) > 0
        else 0
    )
    st.metric(label="🚨 Vertraagd (>15m)", value=f"{pct_d:.1f}%")
with k4:
    avg_dist = filtered_df["Afstand_km"].mean() if not filtered_df.empty else 0
    st.metric(label="🌐 Gem. Vluchtafstand", value=f"{avg_dist:.0f} km")
with k5:
    top_dest = (
        filtered_df["Org/Des"].mode()[0]
        if not filtered_df.empty and "Org/Des" in filtered_df.columns
        else "N/A"
    )
    st.metric(label="📍 Drukste Bestemming", value=top_dest)

st.divider()

# ==========================================
# 6. HOOFDDASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Tijdsanalyse & Trends",
        "🗺️ 3D Kaart & Afstanden",
        "🌤️ Weerinvloed (Meteostat 06670)",
        "🤖 Machine Learning Model",
        "🛫 Vluchttelemetrie & Data Inspectie",
    ]
)

# --- TAB 1: TIJDSANALYSE & TRENDS ---
with tab1:
    st.subheader("📈 Verloop en Piekuren van Vertragingen op Zürich Airport (Zonder Uitschieters)")

    if not chart_df.empty:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            df_m = (
                chart_df.groupby(["Maand", "LSV"])["Vertraging_Minuten"]
                .mean()
                .reset_index()
            )
            df_m["LSV_Name"] = df_m["LSV"].map(
                {"L": "Inbound (L - Landing)", "S": "Outbound (S - Start)"}
            )
            fig_m = px.line(
                df_m,
                x="Maand",
                y="Vertraging_Minuten",
                color="LSV_Name",
                title="Gemiddelde Vertraging per Maand (2019-2020)",
                markers=True,
                color_discrete_map=COLOR_DISCRETE_MAP,
                labels={
                    "Vertraging_Minuten": "Gem. Vertraging (min)",
                    "Maand": "Maand",
                    "LSV_Name": "Vluchttype"
                },
            )
            fig_m.update_layout(template="plotly_white")
            st.plotly_chart(fig_m, use_container_width=True)

        with col_t2:
            df_u = (
                chart_df.groupby("Uur")["Vertraging_Minuten"]
                .mean()
                .reset_index()
            )
            fig_u = px.bar(
                df_u,
                x="Uur",
                y="Vertraging_Minuten",
                color="Vertraging_Minuten",
                title="Gemiddelde Vertraging per Uur van de Dag (Piekuren)",
                color_continuous_scale=COLOR_CONTINUOUS_SCALE,
                labels={"Vertraging_Minuten": "Gem. Vertraging (min)", "Uur": "Uur van de dag"},
            )
            fig_u.update_layout(template="plotly_white")
            st.plotly_chart(fig_u, use_container_width=True)

        col_t3, col_t4 = st.columns(2)
        with col_t3:
            fig_rwy = px.box(
                chart_df.dropna(subset=["RWY"]),
                x="RWY",
                y="Vertraging_Minuten",
                title="Spreiding van Vertraging per Start-/Landingsbaan (Zonder Uitschieters)",
                points=False,  # Verbergt uitschieter-stippen
                color_discrete_sequence=[PRIMARY_COLOR],
                labels={"RWY": "Baan Nummer", "Vertraging_Minuten": "Vertraging (min)"},
            )
            fig_rwy.update_layout(template="plotly_white")
            st.plotly_chart(fig_rwy, use_container_width=True)

        with col_t4:
            df_dow = (
                chart_df.groupby("DagVanWeek")["Vertraging_Minuten"]
                .mean()
                .reindex(
                    [
                        "Monday", "Tuesday", "Wednesday",
                        "Thursday", "Friday", "Saturday", "Sunday"
                    ]
                )
                .reset_index()
            )
            fig_dow = px.bar(
                df_dow,
                x="DagVanWeek",
                y="Vertraging_Minuten",
                title="Gemiddelde Vertraging per Dag van de Week",
                color="Vertraging_Minuten",
                color_continuous_scale=COLOR_CONTINUOUS_SCALE,
                labels={"Vertraging_Minuten": "Gem. Vertraging (min)", "DagVanWeek": "Dag van de Week"}
            )
            fig_dow.update_layout(template="plotly_white")
            st.plotly_chart(fig_dow, use_container_width=True)

# --- TAB 2: 3D GEOGRAFISCHE KAART & AFSTANDEN ---
with tab2:
    st.subheader("🗺️ 3D Vluchtroutes & Geografische Spreiding vanaf Zürich Airport (ZRH)")
    st.markdown(
        "De afstanden worden gevectoriseerd berekend met de **Haversine afstandsformule** tussen Zürich Airport (`LSZH`) en de buitenlandse bestemmingen."
    )

    df_map = chart_df.dropna(subset=["Latitude", "Longitude"]).copy()

    if not df_map.empty:
        df_geo = (
            df_map.groupby(
                [
                    "Org/Des", "Name", "City", "Country",
                    "Latitude", "Longitude", "Afstand_km",
                ]
            )
            .agg(
                Aantal_Vluchten=("FLT", "count"),
                Gem_Vertraging=("Vertraging_Minuten", "mean"),
            )
            .reset_index()
        )

        view_state = pdk.ViewState(
            latitude=ZRH_LAT, longitude=ZRH_LON, zoom=3, pitch=45
        )

        layer_scatter = pdk.Layer(
            "ScatterplotLayer",
            data=df_geo,
            get_position="[Longitude, Latitude]",
            get_color="[30, 58, 138, 180]",  # Match met PRIMARY_COLOR (#1E3A8A)
            get_radius="Aantal_Vluchten * 120 + 15000",
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer_scatter],
                initial_view_state=view_state,
                tooltip={
                    "text": "Luchthaven: {Name} ({Org/Des})\nStad/Land: {City}, {Country}\nAfstand: {Afstand_km:.0f} km\nAantal Vluchten: {Aantal_Vluchten}\nGem. Vertraging: {Gem_Vertraging:.1f} min"
                },
            )
        )

        fig_dist = px.scatter(
            df_geo,
            x="Afstand_km",
            y="Gem_Vertraging",
            size="Aantal_Vluchten",
            color="Gem_Vertraging",
            color_continuous_scale=COLOR_CONTINUOUS_SCALE,
            hover_name="Name",
            title="Relatie tussen Vluchtafstand vanaf Zürich (km) en Vertraging (minuten)",
            trendline="ols",
            trendline_color_discrete=[NEUTRAL_DARK],
            labels={
                "Afstand_km": "Afstand vanaf Zürich (km)",
                "Gem_Vertraging": "Gemiddelde Vertraging (min)",
            },
        )
        fig_dist.update_layout(template="plotly_white")
        st.plotly_chart(fig_dist, use_container_width=True)

# --- TAB 3: WEERINVLOED (METEOSTAT STATION 06670 ZÜRICH-KLOTEN) ---
with tab3:
    st.subheader("🌤️ Invloed van Meteostat Weergegevens op Vertragingen")
    st.markdown(
        "Koppeling van dagelijkse weergegevens gemeten op station **06670 (Zürich-Kloten)**."
    )

    if (
        "Temperatuur_C" in chart_df.columns
        and "Windsnelheid_m_s" in chart_df.columns
    ):
        df_w_daily = (
            chart_df.groupby("STD_DT")
            .agg(
                Gem_Vertraging=("Vertraging_Minuten", "mean"),
                Temperatuur=("Temperatuur_C", "first"),
                Windsnelheid=("Windsnelheid_m_s", "first"),
            )
            .dropna()
            .reset_index()
        )

        col_w1, col_w2 = st.columns(2)
        with col_w1:
            fig_w1 = px.scatter(
                df_w_daily,
                x="Windsnelheid",
                y="Gem_Vertraging",
                title="Windsnelheid op Zürich (m/s) vs. Dagelijkse Vertraging",
                trendline="ols",
                color_discrete_sequence=[PRIMARY_COLOR],
                labels={
                    "Windsnelheid": "Windsnelheid (m/s)",
                    "Gem_Vertraging": "Gem. Vertraging (min)",
                },
            )
            fig_w1.update_layout(template="plotly_white")
            st.plotly_chart(fig_w1, use_container_width=True)

        with col_w2:
            fig_w2 = px.scatter(
                df_w_daily,
                x="Temperatuur",
                y="Gem_Vertraging",
                title="Temperatuur op Zürich (°C) vs. Dagelijkse Vertraging",
                trendline="ols",
                color_discrete_sequence=[SECONDARY_COLOR],
                labels={
                    "Temperatuur": "Temperatuur (°C)",
                    "Gem_Vertraging": "Gem. Vertraging (min)",
                },
            )
            fig_w2.update_layout(template="plotly_white")
            st.plotly_chart(fig_w2, use_container_width=True)

# --- TAB 4: MACHINE LEARNING MODEL ---
with tab4:
    st.subheader("🤖 Machine Learning: Vertraging Voorspellen op Zürich Airport")
    st.markdown(
        "Een **Scikit-Learn Gradient Boosting Regressor** getraind op historische vlucht- en weerparameters."
    )

    @st.cache_resource
    def train_ml_model(data):
        ml_data = (
            data[
                [
                    "Vertraging_Minuten",
                    "LSV",
                    "ACT",
                    "RWY",
                    "Afstand_km",
                    "Temperatuur_C",
                    "Windsnelheid_m_s",
                    "Uur",
                ]
            ]
            .dropna()
            .copy()
        )
        if len(ml_data) > 50000:
            ml_data = ml_data.sample(n=50000, random_state=42)

        cat_cols = ["LSV", "ACT", "RWY"]
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        ml_data[cat_cols] = encoder.fit_transform(ml_data[cat_cols].astype(str))

        X = ml_data.drop("Vertraging_Minuten", axis=1)
        y = ml_data["Vertraging_Minuten"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        return model, encoder, mae, rmse, r2, list(X.columns)

    if not df.empty:
        with st.spinner("Model wordt getraind op de data van Zürich..."):
            model, encoder, mae, rmse, r2, feature_names = train_ml_model(df)

        m1, m2, m3 = st.columns(3)
        m1.metric("Model MAE (Mean Absolute Error)", f"{mae:.2f} min")
        m2.metric("Model RMSE (Root Mean Sq. Error)", f"{rmse:.2f} min")
        m3.metric("Model R² Score", f"{r2:.4f}")

        st.divider()
        st.markdown("#### 🔮 Interactieve Vertragingsvoorspeller")

        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            in_lsv = st.selectbox(
                "Vluchttype", ["L (Landing/Inbound)", "S (Start/Outbound)"]
            )
            in_act = st.selectbox(
                "Vliegtuigtype",
                sorted([str(x) for x in df["ACT"].dropna().unique()]),
            )
            in_uur = st.slider("Vertrek-/Aankomstuur", 0, 23, 14)
        with col_p2:
            in_rwy = st.selectbox(
                "Landings/Startbaan",
                sorted([str(x) for x in df["RWY"].dropna().unique()]),
            )
            in_dist = st.number_input(
                "Vluchtafstand tot Zürich (km)", 100, 15000, 1200
            )
        with col_p3:
            in_wind = st.slider("Verwachte Windsnelheid (m/s)", 0, 30, 8)
            in_temp = st.slider("Verwachte Temperatuur (°C)", -10, 35, 18)

        if st.button("🚀 Bereken Voorspelde Vertraging"):
            lsv_code = "L" if "Landing" in in_lsv else "S"
            input_row = pd.DataFrame(
                [
                    [
                        lsv_code,
                        in_act,
                        in_rwy,
                        in_dist,
                        in_temp,
                        in_wind,
                        in_uur,
                    ]
                ],
                columns=feature_names,
            )
            input_row[["LSV", "ACT", "RWY"]] = encoder.transform(
                input_row[["LSV", "ACT", "RWY"]].astype(str)
            )

            pred_delay = model.predict(input_row)[0]
            st.success(
                f"Voorspelde Vertraging voor deze vlucht: **{pred_delay:.1f} minuten**"
            )

# --- TAB 5: VLUCHTTELEMETRIE & DATA INSPECTIE ---
with tab5:
    st.subheader("🛫 Hoge-Resolutie Vluchttelemetrie & Opschoning")
    st.markdown(
        "Analyse van de **7 gedetailleerde vluchtbestanden** uit `flightdata.zip`."
    )

    zip_paths = ["flightdata.zip", "data/flightdata.zip"]
    found_telemetry = False
    for zpath in zip_paths:
        if os.path.exists(zpath) and zipfile.is_zipfile(zpath):
            try:
                with zipfile.ZipFile(zpath, "r") as z:
                    f_list = sorted([f for f in z.namelist() if f.endswith(".xlsx")])

                if f_list:
                    sel_f = st.selectbox("Selecteer Telemetrie Bestand", f_list)
                    if sel_f:
                        df_tel = load_telemetry_file(sel_f)
                        st.write(
                            f"Datapunten in dit telemetriebestand: **{len(df_tel):,}**"
                        )

                        col_f1, col_f2 = st.columns(2)
                        with col_f1:
                            alt_col = [
                                c for c in df_tel.columns if "Altitude" in c
                            ]
                            time_col = [
                                c for c in df_tel.columns if "Time" in c
                            ]
                            if alt_col and time_col:
                                fig_alt = px.line(
                                    df_tel,
                                    x=time_col[0],
                                    y=alt_col[0],
                                    title="Hoogteverloop (Feet) gedurende de Vlucht",
                                    color_discrete_sequence=[PRIMARY_COLOR],
                                )
                                fig_alt.update_layout(template="plotly_white")
                                st.plotly_chart(
                                    fig_alt, use_container_width=True
                                )
                        with col_f2:
                            speed_col = [
                                c
                                for c in df_tel.columns
                                if "SPEED" in c.upper() or "SPEED" in c
                            ]
                            if speed_col and time_col:
                                fig_sp = px.line(
                                    df_tel,
                                    x=time_col[0],
                                    y=speed_col[0],
                                    title="Snelheid (Airspeed) over de Tijd",
                                    color_discrete_sequence=[SECONDARY_COLOR],
                                )
                                fig_sp.update_layout(template="plotly_white")
                                st.plotly_chart(
                                    fig_sp, use_container_width=True
                                )

                        st.dataframe(df_tel.head(100), use_container_width=True)
                found_telemetry = True
                break
            except Exception:
                pass

    if not found_telemetry:
        st.info(
            "Plaats `flightdata.zip` in de map als je de 7 telemetriebestanden wilt analyseren."
        )

    st.divider()
    st.markdown("#### 📋 Geconverteerde & Opschoonde Dataset Inspectie")
    st.dataframe(filtered_df.head(500), use_container_width=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Gefilterde & Opschoonde Dataset als CSV",
        data=csv_bytes,
        file_name="zurich_flight_data_cleaned.csv",
        mime="text/csv",
    )