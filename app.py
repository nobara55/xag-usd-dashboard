import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="XAG/USD Professional Analysis Dashboard",
    page_icon="🥈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para el tema plata
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #C0C0C0, #E5E5E5);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #C0C0C0;
    }
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
    }
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffecb5;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal con estilo
st.markdown("""
<div class="main-header">
    <h1>🥈 XAG/USD Professional Analysis Dashboard</h1>
    <h3>Análisis Profesional de Plata - Optimizado para Traders</h3>
</div>
""", unsafe_allow_html=True)

# Configuración específica para XAG/USD
XAG_SYMBOLS = ["SI=F", "SLV", "PSLV", "XAGUSD=X"]  # Futuros primero, luego ETFs
CORRELATION_ASSETS = {
    "DXY": "DX=F",            # US Dollar Index Futures  
    "Gold": "GC=F",           # Gold Futures
    "S&P500": "^GSPC",        # S&P 500
    "US10Y": "^TNX",          # Bonos 10 años
    "VIX": "^VIX",            # Volatilidad
    "Copper": "HG=F"          # Cobre (demanda industrial)
}

PSYCHOLOGICAL_LEVELS = [20.00, 22.50, 25.00, 27.50, 30.00, 32.50, 35.00, 40.00, 45.00, 50.00]

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Dashboard")

# Selector de sección
section = st.sidebar.selectbox(
    "📊 Seleccionar Sección de Análisis",
    ["🎯 Resumen Ejecutivo", 
     "📊 Volatilidad y Rangos", 
     "📅 Estacionalidad", 
     "🌅 Comportamiento de Apertura",
     "🔗 Correlaciones",
     "📰 Eventos Económicos",
     "🎭 Patrones de Comportamiento"]
)

# Configuración temporal
st.sidebar.subheader("📅 Período de Análisis")
years_back = st.sidebar.slider("Años de Historia", min_value=1, max_value=15, value=10, step=1)
start_date = datetime.now() - timedelta(days=years_back*365)
end_date = datetime.now()

st.sidebar.write(f"**Período:** {start_date.strftime('%Y-%m-%d')} a {end_date.strftime('%Y-%m-%d')}")

# Botón de actualización
if st.sidebar.button("🔄 Actualizar Datos", type="primary"):
    st.cache_data.clear()
    st.rerun()

# ======================= FUNCIONES AUXILIARES =======================

@st.cache_data(ttl=300)
def get_xag_data(start_date, end_date):
    """Obtiene datos de XAG/USD con múltiples símbolos como fallback"""
    for symbol in XAG_SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if not data.empty and len(data) > 50:  # Mínimo 50 días de datos
                st.sidebar.success(f"✅ Datos obtenidos de: {symbol}")
                data = data.dropna()
                return data, symbol
        except Exception as e:
            st.sidebar.warning(f"⚠️ Fallo {symbol}: {str(e)[:50]}...")
            continue
    
    st.sidebar.error("❌ No se pudieron obtener datos de ningún símbolo")
    return pd.DataFrame(), None

@st.cache_data(ttl=300) 
def get_correlation_data(start_date, end_date):
    """Obtiene datos de activos para correlaciones"""
    correlation_data = {}
    
    for name, symbol in CORRELATION_ASSETS.items():
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if not data.empty:
                correlation_data[name] = data['Close']
        except:
            st.sidebar.warning(f"⚠️ No se pudo obtener {name}")
    
    return correlation_data

def calculate_comprehensive_metrics(data):
    """Calcula todas las métricas de XAG/USD"""
    
    # Retornos y volatilidad
    data['Returns'] = data['Close'].pct_change()
    data['Abs_Returns'] = abs(data['Returns'])
    data['Daily_Range'] = ((data['High'] - data['Low']) / data['Close']) * 100
    data['Gap'] = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
    
    # Volatilidad rolling
    data['Vol_20d'] = data['Returns'].rolling(20).std() * np.sqrt(252) * 100
    data['Vol_60d'] = data['Returns'].rolling(60).std() * np.sqrt(252) * 100
    
    # Medias móviles
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Métricas temporales
    data['Day_of_Week'] = data.index.dayofweek
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    data['Hour'] = data.index.hour
    
    # Identificar días positivos
    data['Positive_Day'] = (data['Returns'] > 0).astype(int)
    
    return data

def calculate_monthly_patterns(data):
    """Análisis detallado de patrones mensuales"""
    monthly_stats = data.groupby('Month').agg({
        'Returns': ['mean', 'std', 'count'],
        'Daily_Range': 'mean',
        'Positive_Day': 'mean',
        'Vol_20d': 'mean'
    }).round(6)
    
    monthly_stats.columns = ['Avg_Return', 'Volatility', 'Count', 'Avg_Range', 'Positive_Days', 'Avg_Vol']
    
    # Convertir a porcentajes
    monthly_stats['Avg_Return_Pct'] = monthly_stats['Avg_Return'] * 100
    monthly_stats['Volatility_Pct'] = monthly_stats['Volatility'] * 100 * np.sqrt(252)
    monthly_stats['Positive_Days_Pct'] = monthly_stats['Positive_Days'] * 100
    
    # Agregar nombres de meses
    month_names = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
                   7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'}
    monthly_stats['Month_Name'] = monthly_stats.index.map(month_names)
    
    # Clasificaciones
    monthly_stats['Classification'] = monthly_stats['Avg_Return_Pct'].apply(
        lambda x: '🟢 Favorable' if x > 0.1 else '🔴 Desfavorable' if x < -0.05 else '🟡 Neutral'
    )
    
    return monthly_stats

def analyze_gaps(data):
    """Análisis específico de gaps de apertura"""
    data['Gap'] = ((data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)) * 100
    data['Gap_Size'] = abs(data['Gap'])
    data['Gap_Direction'] = np.where(data['Gap'] > 0, 'Alcista', 'Bajista')
    
    # Clasificar gaps por tamaño
    data['Gap_Category'] = pd.cut(
        data['Gap_Size'],
        bins=[0, 0.5, 1.0, 2.0, float('inf')],
        labels=['Normal', 'Moderado', 'Alto', 'Extremo']
    )
    
    # Calcular si el gap se cierra durante el día
    data['Gap_Filled'] = np.where(
        (data['Gap'] > 0) & (data['Low'] <= data['Close'].shift(1)),
        True,
        np.where(
            (data['Gap'] < 0) & (data['High'] >= data['Close'].shift(1)),
            True,
            False
        )
    )
    
    return data

def calculate_correlations(xag_data, correlation_data):
    """Calcula correlaciones específicas para XAG/USD"""
    correlations = {}
    
    for asset_name, asset_data in correlation_data.items():
        if len(asset_data) > 0:
            # Sincronizar fechas
            common_dates = xag_data.index.intersection(asset_data.index)
            if len(common_dates) > 50:
                xag_returns = xag_data.loc[common_dates, 'Returns']
                asset_returns = asset_data.loc[common_dates].pct_change()
                
                correlation = xag_returns.corr(asset_returns)
                correlations[asset_name] = correlation
    
    return correlations

def detect_patterns_and_alerts(data):
    """Detecta patrones y genera alertas específicas para XAG/USD"""
    current_price = data['Close'].iloc[-1]
    yesterday_price = data['Close'].iloc[-2]
    daily_change = ((current_price - yesterday_price) / yesterday_price) * 100
    
    current_vol = data['Vol_20d'].iloc[-1]
    avg_vol = data['Vol_20d'].mean()
    
    current_rsi = data['RSI'].iloc[-1]
    
    alerts = []
    patterns = []
    
    # Volatilidad extrema
    if current_vol > avg_vol * 1.5:
        alerts.append("⚠️ **ALTA VOLATILIDAD**: Volatilidad actual 50% superior al promedio")
    elif current_vol < avg_vol * 0.7:
        alerts.append("😴 **BAJA VOLATILIDAD**: Posible acumulación antes de movimiento")
    
    # Movimientos extremos
    if abs(daily_change) > 3.0:
        alerts.append(f"🔥 **MOVIMIENTO EXTREMO**: {daily_change:.2f}% - Revisar noticias")
    
    # Niveles psicológicos
    for level in PSYCHOLOGICAL_LEVELS:
        if abs(current_price - level) / level < 0.01:  # Dentro del 1%
            alerts.append(f"🎯 **NIVEL PSICOLÓGICO**: Precio cerca de ${level:.2f}")
    
    # RSI extremos
    if current_rsi > 70:
        patterns.append("📈 **SOBRECOMPRA**: RSI > 70 - Posible corrección")
    elif current_rsi < 30:
        patterns.append("📉 **SOBREVENTA**: RSI < 30 - Posible rebote")
    
    # Mean reversion
    ma20 = data['MA20'].iloc[-1]
    distance_from_ma = abs(current_price - ma20) / ma20 * 100
    
    if distance_from_ma > 2.5:
        patterns.append(f"🔄 **MEAN REVERSION**: {distance_from_ma:.1f}% alejado de MA20")
    
    return alerts, patterns

# ======================= OBTENER DATOS =======================

with st.spinner("📊 Descargando datos de XAG/USD y activos correlacionados..."):
    xag_data, symbol_used = get_xag_data(start_date, end_date)
    correlation_data = get_correlation_data(start_date, end_date)

if xag_data.empty:
    st.error("❌ No se pudieron cargar los datos. Verificar conexión.")
    st.stop()

# Procesar datos
xag_data = calculate_comprehensive_metrics(xag_data)
xag_data = analyze_gaps(xag_data)
monthly_stats = calculate_monthly_patterns(xag_data)
correlations = calculate_correlations(xag_data, correlation_data)
alerts, patterns = detect_patterns_and_alerts(xag_data)

# Información del dataset en sidebar
st.sidebar.success(f"✅ Datos cargados: {symbol_used}")
st.sidebar.info(f"""
**📈 Dataset Info:**
- **Registros:** {len(xag_data):,} días
- **Precio actual:** ${xag_data['Close'].iloc[-1]:.2f}
- **Cambio diario:** {((xag_data['Close'].iloc[-1] - xag_data['Close'].iloc[-2]) / xag_data['Close'].iloc[-2] * 100):.2f}%
""")

# ======================= SECCIONES DEL DASHBOARD =======================

if section == "🎯 Resumen Ejecutivo":
    st.header("🎯 Resumen Ejecutivo - XAG/USD")
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = xag_data['Close'].iloc[-1]
        yesterday_price = xag_data['Close'].iloc[-2]
        daily_change = ((current_price - yesterday_price) / yesterday_price) * 100
        st.metric("💰 Precio Actual", f"${current_price:.2f}", f"{daily_change:.2f}%")
    
    with col2:
        vol_20d = xag_data['Vol_20d'].iloc[-1]
        st.metric("📊 Volatilidad 20d", f"{vol_20d:.1f}%")
    
    with col3:
        rsi = xag_data['RSI'].iloc[-1]
        st.metric("📈 RSI (14)", f"{rsi:.1f}")
    
    with col4:
        avg_range = xag_data['Daily_Range'].mean()
        st.metric("📏 Rango Promedio", f"{avg_range:.2f}%")
    
    # Alertas y patrones
    st.subheader("🚨 Alertas Actuales")
    if alerts:
        for alert in alerts:
            st.warning(alert)
    else:
        st.success("✅ Sin alertas de riesgo extremo")
    
    if patterns:
        st.subheader("🎭 Patrones Detectados")
        for pattern in patterns:
            st.info(pattern)
    
    # Gráfico principal
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Precio XAG/USD con Medias Móviles', 'RSI'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Precio y MAs
    fig.add_trace(
        go.Scatter(x=xag_data.index, y=xag_data['Close'], name='XAG/USD', line=dict(color='gold', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=xag_data.index, y=xag_data['MA20'], name='MA20', line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=xag_data.index, y=xag_data['MA50'], name='MA50', line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    # Niveles psicológicos
    for level in PSYCHOLOGICAL_LEVELS:
        if xag_data['Close'].min() <= level <= xag_data['Close'].max():
            fig.add_hline(y=level, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    
    # RSI
    fig.add_trace(
        go.Scatter(x=xag_data.index, y=xag_data['RSI'], name='RSI', line=dict(color='purple')),
        row=2, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=600, title="📊 Vista General XAG/USD")
    st.plotly_chart(fig, use_container_width=True)

elif section == "📊 Volatilidad y Rangos":
    st.header("📊 Volatilidad y Rangos - La Huella Digital de XAG/USD")
    
    # Métricas clave
    daily_vol = xag_data['Returns'].std() * np.sqrt(252) * 100
    avg_daily_move = xag_data['Abs_Returns'].mean() * 100
    avg_daily_range = xag_data['Daily_Range'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Volatilidad Anualizada", f"{daily_vol:.1f}%")
    with col2:
        st.metric("📏 Movimiento Diario Promedio", f"{avg_daily_move:.2f}%")
    with col3:
        st.metric("📊 Rango Intradiario Promedio", f"{avg_daily_range:.2f}%")
    with col4:
        extreme_up = xag_data['Returns'].max() * 100
        extreme_down = xag_data['Returns'].min() * 100
        st.metric("⚡ Extremos", f"+{extreme_up:.1f}% / {extreme_down:.1f}%")
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95, 99]
    move_percentiles = np.percentile(xag_data['Abs_Returns'].dropna() * 100, percentiles)
    range_percentiles = np.percentile(xag_data['Daily_Range'].dropna(), percentiles)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Percentiles de Movimientos")
        percentile_df = pd.DataFrame({
            'Percentil': percentiles,
            'Movimiento (%)': [f"{v:.2f}%" for v in move_percentiles],
            'Interpretación': [
                "Día tranquilo", "Día normal", "Día activo", 
                "Alta volatilidad", "Evento significativo", "Crisis"
            ]
        })
        st.dataframe(percentile_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("📏 Percentiles de Rango")
        range_df = pd.DataFrame({
            'Percentil': percentiles,
            'Rango (%)': [f"{v:.2f}%" for v in range_percentiles],
            'Aplicación': [
                "Scalping", "Day trading", "Swing",
                "Alta volatilidad", "Excepcional", "Extremo"
            ]
        })
        st.dataframe(range_df, use_container_width=True, hide_index=True)
    
    # Gráfico de volatilidad
    fig1 = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Precio XAG/USD', 'Volatilidad Rolling 20 días'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    fig1.add_trace(
        go.Scatter(x=xag_data.index, y=xag_data['Close'], name='XAG/USD', line=dict(color='gold')),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(x=xag_data.index, y=xag_data['Vol_20d'], name='Vol 20d', 
                  line=dict(color='red'), fill='tonexty'),
        row=2, col=1
    )
    
    fig1.add_hline(y=daily_vol, line_dash="dash", line_color="orange", 
                   annotation_text=f"Promedio: {daily_vol:.1f}%", row=2, col=1)
    
    fig1.update_layout(height=600, title="📈 Evolución de Precio y Volatilidad")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Distribución de retornos
    fig2 = px.histogram(
        xag_data['Returns'].dropna() * 100,
        nbins=50,
        title="📊 Distribución de Retornos Diarios",
        labels={'value': 'Retorno (%)', 'count': 'Frecuencia'}
    )
    
    for p in [5, 25, 50, 75, 95]:
        val = np.percentile(xag_data['Returns'].dropna() * 100, p)
        fig2.add_vline(x=val, line_dash="dash", 
                       annotation_text=f"P{p}: {val:.2f}%")
    
    st.plotly_chart(fig2, use_container_width=True)

elif section == "📅 Estacionalidad":
    st.header("📅 Análisis de Estacionalidad - Patrones Temporales de XAG/USD")
    
    # Estadísticas mensuales
    st.subheader("📊 Rendimiento por Mes")
    
    display_monthly = monthly_stats[['Month_Name', 'Avg_Return_Pct', 'Volatility_Pct', 
                                   'Positive_Days_Pct', 'Classification']].copy()
    display_monthly.columns = ['Mes', 'Rendimiento (%)', 'Volatilidad (%)', 
                             'Días Positivos (%)', 'Clasificación']
    display_monthly = display_monthly.round(2)
    
    st.dataframe(display_monthly, use_container_width=True, hide_index=True)
    
    # Gráficos de estacionalidad
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = px.bar(
            monthly_stats.reset_index(),
            x='Month_Name',
            y='Avg_Return_Pct',
            title="📈 Rendimiento Promedio por Mes",
            color='Avg_Return_Pct',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        fig4 = px.bar(
            monthly_stats.reset_index(),
            x='Month_Name',
            y='Volatility_Pct',
            title="📊 Volatilidad por Mes",
            color='Volatility_Pct',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Análisis por día de la semana
    st.subheader("📅 Análisis por Día de la Semana")
    
    weekday_stats = xag_data.groupby('Day_of_Week').agg({
        'Returns': 'mean',
        'Positive_Day': 'mean',
        'Daily_Range': 'mean'
    }) * 100
    
    weekday_names = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 
                    3: 'Jueves', 4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
    weekday_stats['Day_Name'] = weekday_stats.index.map(weekday_names)
    
    fig5 = px.bar(
        weekday_stats.reset_index(),
        x='Day_Name',
        y='Returns',
        title="📈 Rendimiento Promedio por Día de la Semana",
        color='Returns',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig5, use_container_width=True)

elif section == "🌅 Comportamiento de Apertura":
    st.header("🌅 Análisis de Gaps de Apertura - XAG/USD")
    
    # Estadísticas de gaps
    total_gaps = len(xag_data.dropna(subset=['Gap']))
    positive_gaps = len(xag_data[xag_data['Gap'] > 0])
    negative_gaps = len(xag_data[xag_data['Gap'] < 0])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Gaps", f"{total_gaps:,}")
    with col2:
        pct_positive = (positive_gaps / total_gaps) * 100
        st.metric("📈 Gaps Alcistas", f"{pct_positive:.1f}%")
    with col3:
        pct_negative = (negative_gaps / total_gaps) * 100
        st.metric("📉 Gaps Bajistas", f"{pct_negative:.1f}%")
    with col4:
        avg_gap = xag_data['Gap'].mean()
        st.metric("📊 Gap Promedio", f"{avg_gap:.3f}%")
    
    # Análisis por categoría de gap
    st.subheader("📊 Análisis por Tamaño de Gap")
    
    gap_analysis = xag_data.groupby('Gap_Category').agg({
        'Gap': 'count',
        'Gap_Filled': 'mean'
    })
    gap_analysis.columns = ['Frecuencia', 'Prob_Cierre']
    gap_analysis['Prob_Cierre_Pct'] = gap_analysis['Prob_Cierre'] * 100
    gap_analysis['Frecuencia_Pct'] = (gap_analysis['Frecuencia'] / gap_analysis['Frecuencia'].sum()) * 100
    
    st.dataframe(gap_analysis, use_container_width=True)
    
    # Gráfico de gaps
    fig6 = px.histogram(
        xag_data['Gap'].dropna(),
        nbins=50,
        title="📊 Distribución de Gaps de Apertura",
        labels={'value': 'Gap Size (%)', 'count': 'Frecuencia'}
    )
    st.plotly_chart(fig6, use_container_width=True)

elif section == "🔗 Correlaciones":
    st.header("🔗 Análisis de Correlaciones - XAG/USD vs Mercados")
    
    if correlations:
        # Mostrar correlaciones
        corr_df = pd.DataFrame(list(correlations.items()), columns=['Activo', 'Correlación'])
        corr_df['Correlación'] = corr_df['Correlación'].round(3)
        corr_df['Fuerza'] = corr_df['Correlación'].abs().apply(
            lambda x: 'Muy Fuerte' if x > 0.7 else 'Fuerte' if x > 0.5 else 'Moderada' if x > 0.3 else 'Débil'
        )
        corr_df['Dirección'] = corr_df['Correlación'].apply(
            lambda x: 'Positiva' if x > 0 else 'Negativa'
        )
        
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
        
        # Gráfico de correlaciones
        fig7 = px.bar(
            corr_df,
            x='Activo',
            y='Correlación',
            title="📊 Correlaciones XAG/USD vs Otros Activos",
            color='Correlación',
            color_continuous_scale='RdBu_r'
        )
        st.plotly_chart(fig7, use_container_width=True)
        
        # Matriz de correlación
        if len(correlation_data) > 1:
            st.subheader("🔥 Matriz de Correlación")
            
            # Crear DataFrame con todos los activos
            all_data = pd.DataFrame()
            all_data['XAG/USD'] = xag_data['Returns']
            
            for name, data in correlation_data.items():
                if len(data) > 0:
                    returns = data.pct_change()
                    all_data[name] = returns
            
            corr_matrix = all_data.corr()
            
            fig8 = px.imshow(
                corr_matrix,
                title="🔥 Matriz de Correlación",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig8, use_container_width=True)
    
    else:
        st.warning("⚠️ No se pudieron calcular correlaciones. Verificar conectividad.")

elif section == "📰 Eventos Económicos":
    st.header("📰 Impacto de Eventos Económicos en XAG/USD")
    
    st.info("""
    📋 **Eventos de Mayor Impacto en XAG/USD:**
    
    **🔴 IMPACTO EXTREMO (3-8% movimientos):**
    - Decisiones FOMC (Fed)
    - Crisis geopolíticas
    
    **🟡 IMPACTO ALTO (2-5% movimientos):**
    - Datos CPI/PCE (inflación)
    - NFP (empleo)
    - Datos producción industrial
    
    **🟢 IMPACTO MODERADO (1-3% movimientos):**
    - PMI manufacturero
    - Inventarios de plata
    - Tensiones comerciales
    """)
    
    # Analizar volatilidad en días específicos (aproximación)
    # Identificar días con movimientos extremos
    extreme_days = xag_data[abs(xag_data['Returns']) > 0.03]  # >3%
    
    st.subheader("⚡ Días con Movimientos Extremos (>3%)")
    st.write(f"Total de días extremos: {len(extreme_days)} ({len(extreme_days)/len(xag_data)*100:.1f}% de todos los días)")
    
    if len(extreme_days) > 0:
        extreme_summary = extreme_days[['Returns', 'Daily_Range', 'Vol_20d']].copy()
        extreme_summary['Returns_Pct'] = extreme_summary['Returns'] * 100
        extreme_summary = extreme_summary.sort_values('Returns_Pct', ascending=False)
        
        st.dataframe(
            extreme_summary[['Returns_Pct', 'Daily_Range', 'Vol_20d']].head(10),
            column_config={
                'Returns_Pct': 'Retorno (%)',
                'Daily_Range': 'Rango (%)',
                'Vol_20d': 'Volatilidad 20d (%)'
            }
        )

elif section == "🎭 Patrones de Comportamiento":
    st.header("🎭 Patrones de Comportamiento - La Psicología de XAG/USD")
    
    # Mean Reversion Analysis
    st.subheader("🔄 Análisis de Mean Reversion")
    
    # Calcular distancia de MA20
    xag_data['Distance_MA20'] = ((xag_data['Close'] - xag_data['MA20']) / xag_data['MA20']) * 100
    
    # Identificar casos de mean reversion
    oversold_cases = xag_data[xag_data['Distance_MA20'] < -2.5]
    overbought_cases = xag_data[xag_data['Distance_MA20'] > 2.5]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📉 Casos Oversold (<-2.5%)", len(oversold_cases))
        if len(oversold_cases) > 0:
            # Analizar retornos futuros
            future_returns = []
            for idx in oversold_cases.index:
                try:
                    next_5_days = xag_data.loc[idx:idx + pd.Timedelta(days=5), 'Returns'].sum()
                    future_returns.append(next_5_days * 100)
                except:
                    pass
            
            if future_returns:
                avg_return = np.mean(future_returns)
                success_rate = len([r for r in future_returns if r > 0]) / len(future_returns) * 100
                st.metric("📈 Retorno promedio 5d", f"{avg_return:.2f}%")
                st.metric("✅ Tasa de éxito", f"{success_rate:.1f}%")
    
    with col2:
        st.metric("📈 Casos Overbought (>2.5%)", len(overbought_cases))
        if len(overbought_cases) > 0:
            future_returns = []
            for idx in overbought_cases.index:
                try:
                    next_5_days = xag_data.loc[idx:idx + pd.Timedelta(days=5), 'Returns'].sum()
                    future_returns.append(next_5_days * 100)
                except:
                    pass
            
            if future_returns:
                avg_return = np.mean(future_returns)
                success_rate = len([r for r in future_returns if r < 0]) / len(future_returns) * 100
                st.metric("📉 Retorno promedio 5d", f"{avg_return:.2f}%")
                st.metric("✅ Tasa de éxito", f"{success_rate:.1f}%")
    
    # Gráfico de distancia de MA20
    fig9 = px.line(
        x=xag_data.index,
        y=xag_data['Distance_MA20'],
        title="📊 Distancia de MA20 - Oportunidades de Mean Reversion"
    )
    fig9.add_hline(y=2.5, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig9.add_hline(y=-2.5, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig9.add_hline(y=0, line_color="blue", annotation_text="MA20")
    
    st.plotly_chart(fig9, use_container_width=True)
    
    # Análisis de niveles psicológicos
    st.subheader("🎯 Análisis de Niveles Psicológicos")
    
    current_price = xag_data['Close'].iloc[-1]
    nearest_levels = []
    
    for level in PSYCHOLOGICAL_LEVELS:
        distance = abs(current_price - level)
        distance_pct = (distance / current_price) * 100
        
        if distance_pct < 5:  # Dentro del 5%
            direction = "Resistencia" if level > current_price else "Soporte"
            nearest_levels.append({
                'Nivel': f"${level:.2f}",
                'Distancia': f"{distance:.2f}",
                'Distancia (%)': f"{distance_pct:.2f}%",
                'Tipo': direction
            })
    
    if nearest_levels:
        st.subheader("🎯 Niveles Psicológicos Cercanos")
        st.dataframe(pd.DataFrame(nearest_levels), hide_index=True)

# ======================= EXPORTAR DATOS =======================
st.sidebar.header("📥 Exportar Análisis")

if st.sidebar.button("📊 Exportar Datos Completos"):
    export_data = xag_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                           'Daily_Range', 'Vol_20d', 'RSI', 'Gap']].copy()
    
    csv_data = export_data.to_csv()
    st.sidebar.download_button(
        label="⬇️ Descargar CSV",
        data=csv_data,
        file_name=f"XAG_USD_complete_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

if st.sidebar.button("📋 Exportar Resumen Mensual"):
    csv_monthly = monthly_stats.to_csv()
    st.sidebar.download_button(
        label="⬇️ Descargar Resumen Mensual",
        data=csv_monthly,
        file_name=f"XAG_USD_monthly_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
*🥈 XAG/USD Professional Dashboard | Optimizado específicamente para análisis de plata*  
*Datos: Yahoo Finance | Actualización: Manual via botón*
""")
