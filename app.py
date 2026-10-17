# Dashboard GAC - Insights de Negocio
# Reestructuracion: 4 vistas de negocio
#   1. Panorama Ejecutivo   (KPIs, tendencia vs objetivo, tarjetas de insight)
#   2. Embudo de Ventas     (conversion por etapa, cuellos de botella)
#   3. Desempeno            (asesores de piso, BDC, marketing, productos)
#   4. Laboratorio          (evidencia del proceso estadistico archivado)
#
# Las 13 vistas originales (Etapas I-III, trazables a los notebooks del
# equipo) quedan guardadas como evidencia en:
#   archivo/app_etapas_original.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="GAC - Insights de Negocio", page_icon="🚗", layout="wide")

# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def load_data():
    df_bitacora = pd.read_csv("Bitacora_Piso_Limpia.csv")
    df_leads = pd.read_csv("Leads_Reales_Limpia.csv")
    df_bitacora["Periodo"] = pd.to_datetime(df_bitacora["Fecha"]).dt.strftime("%Y-%m")
    df_leads["Fecha"] = pd.to_datetime(df_leads["Periodo"] + "-01")
    return df_bitacora, df_leads

@st.cache_data
def load_regresion_data():
    df_resumen = pd.read_csv("base_resumen_limpio.csv").bfill().ffill()
    df_funnel = pd.read_csv("funnel_mensual.csv")
    try:
        df_bcs = pd.read_csv("bcs_modelos.csv")
    except FileNotFoundError:
        df_bcs = None
    return df_resumen, df_funnel, df_bcs

@st.cache_data
def load_citas():
    try:
        dfc = pd.read_csv("Citas_Digital.csv")
    except FileNotFoundError:
        return None
    dfc.columns = dfc.columns.str.strip()
    dfc["Potencial"] = pd.to_numeric(
        dfc["Potencial de compra"].astype(str)
        .str.replace("%", "", regex=False).str.strip(), errors="coerce")
    for col in ["PDM", "SDC", "Venta"]:
        dfc[col] = dfc[col].astype(str).str.strip().str.upper().eq("TRUE")

    def limpiar_bdc(x):
        x = str(x).strip().lower()
        if x.startswith("ana"): return "Ana"
        if x.startswith("antonio"): return "Antonio"
        if x.startswith(("ignacio", "igancio")): return "Ignacio"
        if x.startswith("saul"): return "Saul"
        return None

    dfc["BDC_L"] = dfc["BDC"].apply(limpiar_bdc)
    dfc = dfc[dfc["Potencial"].notna() & dfc["BDC_L"].notna()].copy()
    return dfc

@st.cache_data
def load_mkt():
    try:
        base = pd.read_csv("MKT_DIGITAL.csv", header=None, dtype=str,
                           encoding="utf-8-sig")
    except FileNotFoundError:
        return None

    def texto(x):
        return "" if pd.isna(x) else str(x).strip()

    def numero(x):
        s = texto(x).replace("$", "").replace(",", "").replace(" ", "")
        if s in ("", "-", "—"):
            return np.nan
        return pd.to_numeric(s, errors="coerce")

    meses = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
             "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9,
             "octubre": 10, "noviembre": 11, "diciembre": 12}
    marca = base.index[base[0].fillna("").str.strip()
                       .eq("Presupuesto mensual de MKT por canal")]
    if len(marca) == 0:
        return None
    inicio = marca[0]
    columnas_fecha = {}
    for col, valor in base.loc[inicio + 1].items():
        partes = texto(valor).lower().split(",")
        if len(partes) == 2 and partes[0].strip() in meses:
            anio = int(partes[1].strip())
            anio = anio + 2000 if anio < 100 else anio
            columnas_fecha[col] = pd.Timestamp(anio, meses[partes[0].strip()], 1)

    regs = []
    for canal in ["Plaza", "Digital", "Piso"]:
        candidatos = base.index[(base.index > inicio)
                                & base[0].fillna("").str.strip().eq(canal)]
        if len(candidatos) == 0:
            continue
        fila = candidatos[0]
        for col, fecha in columnas_fecha.items():
            regs.append({"Canal": canal, "Fecha": fecha,
                         "Leads": numero(base.loc[fila + 1, col]),
                         "Ventas": numero(base.loc[fila + 2, col]),
                         "Presupuesto": numero(base.loc[fila, col])})
    datos = pd.DataFrame(regs)
    completos = datos.dropna(subset=["Leads", "Ventas", "Presupuesto"])
    fc = completos.groupby("Fecha")["Canal"].nunique()
    fc = fc[fc.eq(3)].index
    dfm = (completos[completos["Fecha"].isin(fc)]
           .sort_values(["Fecha", "Canal"]).reset_index(drop=True))
    return dfm

df_bitacora, df_leads = load_data()
df_resumen, df_funnel, df_bcs = load_regresion_data()
df_citas = load_citas()
df_mkt = load_mkt()

# =========================================================
# TEMA VINO TINTO / BORGONA (presentacion GAC)
# =========================================================
GAC_AZUL       = "#800020"
GAC_AZUL_MED   = "#9B1B30"
GAC_AZUL_CLARO = "#DDA0A8"
GAC_PLATA      = "#C9747F"
GAC_OSCURO     = "#4A0414"
GAC_BLANCO     = "#FFFFFF"
REJILLA        = "#F3E4E7"
VERDE_OK       = "#2E7D32"
AMBAR          = "#B26A00"
ROJO_ALERTA    = "#B00020"

ESTILO_FUENTE = dict(family="Arial, sans-serif", size=13, color=GAC_OSCURO)

def estilo_fig(fig, titulo, alto=420):
    fig.update_layout(
        font=ESTILO_FUENTE, height=alto,
        title=dict(text=titulo, font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        yaxis=dict(gridcolor=REJILLA))
    return fig

def tarjeta_insight(icono, titulo, dato, detalle, color=GAC_AZUL):
    st.markdown(
        "<div style='border:1px solid #EBD3D9; border-left:5px solid " + color +
        "; border-radius:8px; padding:12px 14px; background:#FDF8F9; "
        "margin-bottom:10px;'>"
        "<div style='font-size:14px; font-weight:bold; color:" + GAC_OSCURO + ";'>"
        + icono + " " + titulo + "</div>"
        "<div style='font-size:21px; font-weight:bold; color:" + color +
        "; margin:4px 0;'>" + dato + "</div>"
        "<div style='font-size:13px; color:" + GAC_OSCURO + ";'>" + detalle +
        "</div></div>",
        unsafe_allow_html=True)

def tabla_bonita(df):
    return df.style.format(precision=1, thousands=",")

# =========================================================
# PRE-CALCULOS DE NEGOCIO
# =========================================================
# --- Embudo: tasas de transicion promedio ---
ETAPAS = ["Leads_Generados", "Leads_Efectivos", "Cita_Programada",
          "Cita_Efectiva", "Prueba_Manejo", "SDC_Generada",
          "SDC_Aprobada", "Ventas"]
NOMBRE_ETAPA = {
    "Leads_Generados": "Leads generados",
    "Leads_Efectivos": "Leads efectivos",
    "Cita_Programada": "Cita programada",
    "Cita_Efectiva": "Cita efectiva",
    "Prueba_Manejo": "Prueba de manejo",
    "SDC_Generada": "Solicitud de credito",
    "SDC_Aprobada": "SDC aprobada",
    "Ventas": "Ventas",
}
# benchmark minimo aceptable por transicion (nombre, %)
BENCH = {
    ("Leads_Generados", "Leads_Efectivos"): ("Contacto efectivo", 60),
    ("Leads_Efectivos", "Cita_Programada"): ("Agendamiento", 40),
    ("Cita_Programada", "Cita_Efectiva"): ("Show rate", 70),
    ("Cita_Efectiva", "Prueba_Manejo"): ("Prueba en piso", 60),
    ("Prueba_Manejo", "SDC_Generada"): ("Intencion de compra", 70),
    ("SDC_Generada", "SDC_Aprobada"): ("Aprobacion de credito", 65),
    ("SDC_Aprobada", "Ventas"): ("Cierre", 80),
}
prom_etapas = df_funnel[ETAPAS].mean()
transiciones = []
for a, b in zip(ETAPAS[:-1], ETAPAS[1:]):
    tasa = float((df_funnel[b] / df_funnel[a] * 100).mean())
    nombre, bench = BENCH[(a, b)]
    transiciones.append({"Transicion": nombre, "Desde": NOMBRE_ETAPA[a],
                         "Tasa %": round(tasa, 1), "Benchmark %": bench})
df_trans = pd.DataFrame(transiciones)
df_funnel["Conv_Lead_Venta"] = df_funnel["Ventas"] / df_funnel["Leads_Generados"] * 100

# --- Resumen mensual ---
df_resumen["Periodo"] = (df_resumen["Año"].astype(str) + "-"
                         + df_resumen["Mes"].astype(int).astype(str).str.zfill(2))

# --- Asesores de piso ---
g_asesor = df_bitacora.groupby("Asesor").agg(
    Visitas=("Cliente", "count"), PDM=("PDM", "sum"), SDC=("SDC", "sum"),
    Ventas=("Venta", "sum")).reset_index()
g_asesor["% PDM"] = (g_asesor["PDM"] / g_asesor["Visitas"] * 100).round(1)
g_asesor["% Venta/visita"] = (g_asesor["Ventas"] / g_asesor["Visitas"] * 100).round(1)
g_asesor = g_asesor.sort_values("Ventas", ascending=False)

# --- BDC digital ---
g_bdc = None
if df_citas is not None and len(df_citas):
    g_bdc = df_citas.groupby("BDC_L").agg(
        Registros=("Potencial", "count"), Potencial=("Potencial", "mean"),
        PDM=("PDM", "sum"), Ventas=("Venta", "sum")).reset_index()
    g_bdc["Potencial"] = g_bdc["Potencial"].round(1)
    g_bdc["% PDM"] = (g_bdc["PDM"] / g_bdc["Registros"] * 100).round(1)
    g_bdc["% Venta"] = (g_bdc["Ventas"] / g_bdc["Registros"] * 100).round(1)
    g_bdc = g_bdc.sort_values("% Venta", ascending=False)

# --- Marketing por canal ---
g_mkt = None
if df_mkt is not None and len(df_mkt):
    g_mkt = df_mkt.groupby("Canal").agg(
        Meses=("Fecha", "nunique"), Leads=("Leads", "sum"),
        Ventas=("Ventas", "sum"), Presupuesto=("Presupuesto", "sum")).reset_index()
    g_mkt["Costo / lead"] = (g_mkt["Presupuesto"] / g_mkt["Leads"]).round(0)
    g_mkt["Costo / venta"] = (g_mkt["Presupuesto"] / g_mkt["Ventas"]).round(0)
    g_mkt["Conv. %"] = (g_mkt["Ventas"] / g_mkt["Leads"] * 100).round(1)
    g_mkt["Leads / $1k"] = (g_mkt["Leads"] / g_mkt["Presupuesto"] * 1000).round(1)

# --- Productos BCS ---
if df_bcs is not None:
    df_bcs["Cumplimiento %"] = (df_bcs["Ventas_Reales"] / df_bcs["Objetivo"] * 100).round(0)

# --- Efectividad de leads: inicio vs actual ---
efec_inicio = float(df_leads["Efectividad_Leads"].head(6).mean())
efec_actual = float(df_leads["Efectividad_Leads"].tail(6).mean())

# --- KPIs de los ultimos 12 meses del resumen ---
r12 = df_resumen.tail(12)
k_ventas = int(r12["Ventas Real"].sum())
k_cumpl = float(r12["% CAL Ventas"].mean() * 100)
k_leads_mes = float(r12["Leads"].mean())
k_lead_venta = float((r12["Leads"].sum() / r12["Ventas Real"].sum()))
k_efec = efec_actual
n_meses_bajo = int((df_resumen["% CAL Ventas"] < 1).sum())

# =========================================================
# ENCABEZADO Y NAVEGACION
# =========================================================
st.markdown("<h1 style='color:" + GAC_AZUL + ";'>🚗 GAC · Insights de Negocio</h1>",
            unsafe_allow_html=True)
st.caption("Plaza Angelopolis · Analitica comercial: panorama, embudo, desempeno "
           "y evidencia del proceso estadistico")
st.divider()

if os.path.exists("logo_gac.png"):
    st.sidebar.image("logo_gac.png", width="stretch")
else:
    st.sidebar.markdown(
        "<h2 style='color:" + GAC_AZUL + "; text-align:center;'>GAC MOTOR</h2>",
        unsafe_allow_html=True)
st.sidebar.markdown("---")

VISTAS = ["📊 Panorama Ejecutivo", "🔻 Embudo de Ventas",
          "👥 Desempeno", "🧪 Laboratorio (evidencia)"]
vista = st.sidebar.radio("Vista:", VISTAS, key="nav_vista")

st.sidebar.markdown("---")
st.sidebar.write("📋 Bitacora de piso: **%d** registros" % len(df_bitacora))
st.sidebar.write("📈 Leads reales: **%d** meses" % len(df_leads))
st.sidebar.write("📊 Base resumen: **%d** meses" % len(df_resumen))
st.sidebar.write("🔻 Embudo: **%d** meses" % len(df_funnel))
if df_citas is not None:
    st.sidebar.write("📞 Citas digital: **%d** registros" % len(df_citas))
if df_mkt is not None:
    st.sidebar.write("📣 Marketing: **%d** obs." % len(df_mkt))
st.sidebar.markdown("---")
st.sidebar.caption("Evidencia del proceso estadistico:\n`archivo/app_etapas_original.py`")

st.caption("📍 Estas en: **" + vista + "**")

# =========================================================
# VISTA 1: PANORAMA EJECUTIVO
# =========================================================
if vista == VISTAS[0]:
    st.subheader("¿Como va la plaza?")

    alcance = st.radio("Periodo:", ["Ultimos 12 meses", "Todo el historico (30 meses)"],
                       horizontal=True)
    r = df_resumen.tail(12).copy() if alcance.startswith("Ultimos") else df_resumen.copy()
    n_meses_bajo_r = int((r["% CAL Ventas"] < 1).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Ventas (" + str(len(r)) + " meses)", f"{int(r['Ventas Real'].sum()):,}")
    k2.metric("Cumplimiento de objetivo", f"{float(r['% CAL Ventas'].mean())*100:.0f}%")
    k3.metric("Leads / mes", f"{r['Leads'].mean():,.0f}")
    k4.metric("Leads por venta", f"{r['Leads'].sum()/r['Ventas Real'].sum():.0f}")
    k5.metric("Efectividad de leads (ult. 6 meses)", f"{efec_actual:.0f}%")
    st.caption("Meses bajo objetivo en el periodo: **%d de %d**" % (n_meses_bajo_r, len(r)))

    # Tendencia: ventas vs objetivo + % de cumplimiento
    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["Periodo"], y=r["% CAL Ventas"] * 100, name="% cumplimiento",
                         marker_color=np.where(r["% CAL Ventas"] < 1, GAC_PLATA, GAC_AZUL),
                         yaxis="y2"))
    fig.add_trace(go.Scatter(x=r["Periodo"], y=r["Ventas Real"], name="Ventas reales",
                             mode="lines+markers", line=dict(color=GAC_OSCURO, width=3),
                             marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=r["Periodo"], y=r["Obj Ventas"], name="Objetivo",
                             mode="lines", line=dict(color=GAC_AZUL_CLARO, width=2, dash="dash")))
    fig.update_layout(font=ESTILO_FUENTE, height=440,
        title=dict(text="Ventas reales vs objetivo (linea) y % de cumplimiento (barras)",
                   font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        yaxis=dict(title="Ventas", gridcolor=REJILLA),
        yaxis2=dict(title="% cumplimiento", overlaying="y", side="right",
                    range=[0, 160], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch")

    anomalias = r[r["% CAL Ventas"] < 0.6][["Periodo", "Leads", "Ventas Real",
                                            "Obj Ventas", "% CAL Ventas"]].copy()
    anomalias["% CAL Ventas"] = (anomalias["% CAL Ventas"] * 100).round(0)
    if len(anomalias):
        st.markdown("**⚠️ Meses criticos (menos del 60% del objetivo):**")
        st.dataframe(anomalias.rename(columns={"% CAL Ventas": "% cumplimiento"}),
                     width="stretch", hide_index=True)

    st.markdown("### 🔎 Hallazgos clave (los datos ya los tienes; aqui estan redactados)")

    tasa_aprob = float(df_trans.loc[df_trans["Transicion"] == "Aprobacion de credito", "Tasa %"].iloc[0])
    tasa_agend = float(df_trans.loc[df_trans["Transicion"] == "Agendamiento", "Tasa %"].iloc[0])
    c1, c2, c3 = st.columns(3)
    with c1:
        tarjeta_insight("💳", "Aprobacion de credito", f" Solo {tasa_aprob:.0f}% de las solicitudes se aprueba",
            "El mayor cuello del embudo. 6 de cada 10 solicitudes se caen en "
            "aprobacion: revisar filtros de las financieras y documentacion "
            "que se pide al cliente.", ROJO_ALERTA)
    with c2:
        tarjeta_insight("📅", "Agendamiento de citas", f"Solo {tasa_agend:.0f}% de los leads efectivos agendan",
            "2 de cada 3 leads contactados no llegan a agendar cita. "
            "Oportunidad: reforzar el contacto BDC en las primeras 48 horas.",
            ROJO_ALERTA)
    with c3:
        tarjeta_insight("📉", "Calidad del lead", f"Efectividad: {efec_inicio:.0f}% → {efec_actual:.0f}%",
            "El volumen de leads casi se triplico desde 2024, pero la "
            "efectividad cayo ~20 puntos. Mas leads no significan mas ventas: "
            "vale mas la calidad de la captura que el volumen.", AMBAR)
    c4, c5, c6 = st.columns(3)
    with c4:
        if g_bdc is not None and len(g_bdc) >= 2:
            mejor = g_bdc.iloc[0]; peor = g_bdc.sort_values("Registros", ascending=False).iloc[0]
            tarjeta_insight("👥", "Brecha entre ejecutivos BDC",
                f"{peor['BDC_L']}: {peor['% Venta']:.0f}% vs {mejor['BDC_L']}: {mejor['% Venta']:.0f}%",
                f"{peor['BDC_L']} atiende el mayor volumen ({int(peor['Registros'])} registros) "
                f"con la menor conversion; {mejor['BDC_L']} convierte "
                f"{mejor['% Venta']/max(peor['% Venta'],0.1):.1f}x mas. Revisar asignacion y proceso.",
                AMBAR)
        else:
            tarjeta_insight("👥", "Brecha entre ejecutivos BDC", "Sin datos",
                            "La base de citas digitales no esta disponible.")
    with c5:
        if g_mkt is not None and len(g_mkt) >= 2 and "Piso" in set(g_mkt["Canal"]):
            cv = g_mkt.set_index("Canal")["Costo / venta"]
            ratio = cv["Piso"] / cv["Digital"]
            tarjeta_insight("📣", "Costo por venta: Piso vs Digital",
                f"Piso cuesta {ratio:.1f}x mas que Digital",
                f"${cv['Piso']:,.0f} por venta en piso vs ${cv['Digital']:,.0f} en digital. "
                "El piso convierte mejor (11.5% vs 1.4%), pero el costo por venta "
                "es desproporcionado: evaluar la mezcla de presupuesto.", AMBAR)
        else:
            tarjeta_insight("📣", "Marketing por canal", "Sin datos suficientes",
                            "La base de marketing no tiene los 3 canales completos.")
    with c6:
        if df_bcs is not None:
            bajo = df_bcs[df_bcs["Cumplimiento %"] < 60]
            alto = df_bcs[df_bcs["Cumplimiento %"] >= 150]
            txt_bajo = ", ".join(f"{m} ({c:.0f}%)" for m, c in
                                 zip(bajo["Modelo"], bajo["Cumplimiento %"])) or "ninguno"
            txt_alto = ", ".join(f"{m} ({c:.0f}%)" for m, c in
                                 zip(alto["Modelo"], alto["Cumplimiento %"])) or "ninguno"
            tarjeta_insight("🚗", "Objetivos por modelo mal calibrados",
                "Cumplimientos de " + f"{df_bcs['Cumplimiento %'].min():.0f}% a {df_bcs['Cumplimiento %'].max():.0f}%",
                f"Bajo objetivo: {txt_bajo}. Sobre-objetivo: {txt_alto}. "
                "La dispersion sugiere metas mal calibradas o un mix comercial "
                "desbalanceado; recalibrar objetivos por modelo.", AMBAR)
        else:
            tarjeta_insight("🚗", "Objetivos por modelo", "Sin datos",
                            "La base BCS no esta disponible.")

# =========================================================
# VISTA 2: EMBUDO DE VENTAS
# =========================================================
elif vista == VISTAS[1]:
    st.subheader("¿Donde se fuga la venta?")
    st.caption("Embudo mensual promedio (" + str(len(df_funnel)) +
               " meses) y tasas de transicion entre etapas. "
               "Barra roja = transicion bajo el benchmark interno.")

    col_f, col_t = st.columns([3, 4])
    with col_f:
        fig_fun = go.Figure(go.Funnel(
            y=[NOMBRE_ETAPA[e] for e in ETAPAS],
            x=prom_etapas[ETAPAS].round(0),
            textinfo="value+percent previous",
            marker_color=[GAC_AZUL_CLARO, GAC_PLATA, GAC_AZUL_MED, GAC_AZUL,
                          GAC_OSCURO, GAC_AZUL_MED, GAC_PLATA, GAC_AZUL]))
        fig_fun.update_layout(font=ESTILO_FUENTE, height=480,
            title=dict(text="Embudo promedio por mes", font=dict(size=15, color=GAC_AZUL)),
            paper_bgcolor=GAC_BLANCO,
            margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_fun, width="stretch")
        st.caption("Nota: 'Ventas' incluye compras de contado sin solicitud de "
                   "credito, por eso puede ser mayor que las SDC aprobadas.")

    with col_t:
        colores = [ROJO_ALERTA if t < b else GAC_AZUL
                   for t, b in zip(df_trans["Tasa %"], df_trans["Benchmark %"])]
        fig_tr = go.Figure(go.Bar(
            x=df_trans["Tasa %"], y=df_trans["Transicion"], orientation="h",
            marker_color=colores,
            text=df_trans["Tasa %"].map(lambda v: f"{v:.1f}%"),
            textposition="outside"))
        fig_tr.update_traces(
            hovertext=df_trans.apply(
                lambda r: f"{r['Desde']} → siguiente<br>Benchmark: {r['Benchmark %']:.0f}%",
                axis=1), hoverinfo="text")
        fig_tr.update_layout(font=ESTILO_FUENTE, height=480,
            title=dict(text="Tasa de transicion por etapa (benchmark en hover)",
                       font=dict(size=15, color=GAC_AZUL)),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            xaxis=dict(title="%", gridcolor=REJILLA, range=[0, 105]),
            yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_tr, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        fig_c = go.Figure(go.Scatter(
            x=df_funnel["Mes"], y=df_funnel["Conv_Lead_Venta"].round(2),
            mode="lines+markers", line=dict(color=GAC_AZUL, width=3),
            marker=dict(size=8), fill="tozeroy", fillcolor="#F3E4E7"))
        estilo_fig(fig_c, "Conversion lead → venta por mes (%)")
        fig_c.update_yaxes(gridcolor=REJILLA)
        st.plotly_chart(fig_c, width="stretch")
    with c2:
        fig_e = go.Figure(go.Scatter(
            x=df_leads["Periodo"], y=df_leads["Efectividad_Leads"],
            mode="lines+markers", line=dict(color=GAC_AZUL_MED, width=3),
            marker=dict(size=7)))
        fig_e.add_hline(y=df_leads["Efectividad_Leads"].mean(),
                        line_dash="dash", line_color=GAC_PLATA,
                        annotation_text="promedio")
        estilo_fig(fig_e, "Efectividad de leads por mes (%)")
        fig_e.update_yaxes(gridcolor=REJILLA)
        st.plotly_chart(fig_e, width="stretch")

    st.markdown("**Detalle mensual del embudo**")
    tabla_f = df_funnel[["Mes"] + ETAPAS + ["Conv_Lead_Venta"]].copy()
    tabla_f["Conv_Lead_Venta"] = tabla_f["Conv_Lead_Venta"].round(2)
    tabla_f.columns = ["Mes"] + [NOMBRE_ETAPA[e] for e in ETAPAS] + ["Conv. lead→venta %"]
    st.dataframe(tabla_f, width="stretch", hide_index=True)

# =========================================================
# VISTA 3: DESEMPENO
# =========================================================
elif vista == VISTAS[2]:
    st.subheader("¿Quien y que rinde?")

    tab_ases, tab_bdc, tab_mkt, tab_prod = st.tabs(
        ["Asesores de piso", "Ejecutivos BDC (digital)", "Marketing por canal",
         "Productos (BCS)"])

    with tab_ases:
        col1, col2 = st.columns([3, 4])
        with col1:
            st.markdown("**Ranking por ventas cerradas**")
            st.dataframe(g_asesor.rename(columns={
                "% PDM": "% prueba de manejo",
                "% Venta/visita": "% venta por visita"}),
                width="stretch", hide_index=True)
            sin_ventas = g_asesor[g_asesor["Ventas"] == 0]
            if len(sin_ventas):
                st.warning("Asesores sin ventas cerradas en el periodo: "
                           + ", ".join(sin_ventas["Asesor"])
                           + " (" + ", ".join(str(int(v)) + " visitas" for v in sin_ventas["Visitas"]) + "). "
                           "Revisar asignacion de leads, capacitacion de cierre o rotacion de piso.")
        with col2:
            fig_a = px.bar(g_asesor.sort_values("Ventas"), x="Ventas", y="Asesor",
                           orientation="h", text="Ventas",
                           color_discrete_sequence=[GAC_AZUL])
            estilo_fig(fig_a, "Ventas cerradas por asesor (bitacora de piso)")
            fig_a.update_traces(textposition="outside")
            fig_a.update_xaxes(gridcolor=REJILLA)
            st.plotly_chart(fig_a, width="stretch")

    with tab_bdc:
        if g_bdc is not None:
            col1, col2 = st.columns([3, 4])
            with col1:
                st.markdown("**Conversion por ejecutivo BDC (citas digitales)**")
                st.dataframe(g_bdc.rename(columns={
                    "Potencial": "Potencial medio (%)",
                    "% PDM": "% prueba de manejo", "% Venta": "% conversion a venta"}),
                    width="stretch", hide_index=True)
            with col2:
                fig_b = px.bar(g_bdc.sort_values("% Venta"), x="% Venta", y="BDC_L",
                               orientation="h", text="% Venta",
                               color_discrete_sequence=[GAC_AZUL_MED])
                estilo_fig(fig_b, "% de conversion a venta por ejecutivo BDC")
                fig_b.update_traces(textposition="outside")
                fig_b.update_xaxes(gridcolor=REJILLA)
                st.plotly_chart(fig_b, width="stretch")
            mejor_b = g_bdc.iloc[0]; peor_b = g_bdc.sort_values("Registros", ascending=False).iloc[0]
            if peor_b["% Venta"] < mejor_b["% Venta"] / 2:
                st.warning(f"**Brecha de desempeno:** {peor_b['BDC_L']} concentra el mayor volumen "
                           f"({int(peor_b['Registros'])} registros) pero converte solo "
                           f"{peor_b['% Venta']:.1f}%, contra {mejor_b['% Venta']:.1f}% de "
                           f"{mejor_b['BDC_L']}. Vale la pena auditar su proceso de seguimiento.")
        else:
            st.info("Base de citas digitales no disponible.")

    with tab_mkt:
        if g_mkt is not None:
            col1, col2 = st.columns([3, 4])
            with col1:
                st.markdown("**Eficiencia por canal (" +
                            str(int(g_mkt["Meses"].max())) + " meses completos)**")
                ver = g_mkt[["Canal", "Leads", "Ventas", "Presupuesto",
                             "Costo / lead", "Costo / venta", "Conv. %"]].copy()
                ver["Presupuesto"] = ver["Presupuesto"].map("${:,.0f}".format)
                ver["Costo / lead"] = ver["Costo / lead"].map("${:,.0f}".format)
                ver["Costo / venta"] = ver["Costo / venta"].map("${:,.0f}".format)
                st.dataframe(ver, width="stretch", hide_index=True)
                st.caption("El canal Piso genera pocos leads pero de altisima "
                           "conversion (trafico calido); Digital genera volumen "
                           "barato pero convierte poco. No son sustitutos: el "
                           "punto es la mezcla.")
            with col2:
                fig_m = px.bar(g_mkt, x="Canal", y="Costo / venta", text="Costo / venta",
                               color_discrete_sequence=[GAC_AZUL])
                estilo_fig(fig_m, "Costo de marketing por venta cerrada (menor = mejor)")
                fig_m.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
                fig_m.update_yaxes(gridcolor=REJILLA)
                st.plotly_chart(fig_m, width="stretch")
        else:
            st.info("Base de marketing sin meses completos de los 3 canales.")

    with tab_prod:
        if df_bcs is not None:
            col1, col2 = st.columns([3, 4])
            with col1:
                st.markdown("**Cumplimiento del objetivo por modelo**")
                st.dataframe(df_bcs.rename(columns={
                    "Objetivo": "Objetivo (unidades)",
                    "Ventas_Reales": "Ventas reales",
                    "Cumplimiento %": "Cumplimiento %"}),
                    width="stretch", hide_index=True)
            with col2:
                colores_m = [ROJO_ALERTA if v < 60 else (VERDE_OK if v >= 100 else AMBAR)
                             for v in df_bcs["Cumplimiento %"]]
                fig_p = px.bar(df_bcs.sort_values("Cumplimiento %"), x="Cumplimiento %",
                               y="Modelo", orientation="h", text="Cumplimiento %",
                               color_discrete_sequence=[GAC_AZUL])
                fig_p.update_traces(marker_color=colores_m, texttemplate="%{text:.0f}%",
                                    textposition="outside")
                fig_p.add_vline(x=100, line_dash="dash", line_color=GAC_OSCURO)
                estilo_fig(fig_p, "Cumplimiento % del objetivo por modelo (linea = 100%)")
                fig_p.update_xaxes(gridcolor=REJILLA, range=[0, max(260, float(df_bcs["Cumplimiento %"].max()) * 1.15)])
                st.plotly_chart(fig_p, width="stretch")
        else:
            st.info("Base BCS no disponible.")

# =========================================================
# VISTA 4: LABORATORIO (EVIDENCIA DEL PROCESO)
# =========================================================
else:
    st.subheader("🧪 Laboratorio — evidencia del proceso estadistico")
    st.markdown("""
    Las **13 vistas originales** del dashboard (Etapas I, II y III: analisis
    univariado, correlaciones, regresiones lineales y logisticas, y el
    despliegue ANOVA) se conservan intactas como evidencia del trabajo del
    equipo. No se perdio nada: la version anterior sigue versionada en este
    mismo repositorio.
    """)
    st.markdown("**Donde encontrar cada cosa:**")
    st.markdown("""
| Artefacto | Contenido |
|---|---|
| `archivo/app_etapas_original.py` | Dashboard completo con las 13 vistas originales (Etapas I–III) listo para correr con `streamlit run` |
| `DASHBOARD_GAC.ipynb` | Notebook que generaba y ejecutaba la version original |
| `base_resumen_limpio.csv` | Base mensual de 30 meses / 26 variables usada en las regresiones |
| `funnel_mensual.csv` | Embudo mensual de 18 meses usado en las regresiones del embudo |
| `bcs_modelos.csv` | 6 modelos GAC (Objetivo vs Ventas reales) del Balance Scorecard |
| Historial de git (`git log`) | Todos los commits del proceso, incluida la version previa a esta reestructuracion |
    """)
    st.info("Para consultar una vista estadistica puntual, corre la version "
            "archivada:  `streamlit run archivo/app_etapas_original.py`")
    st.caption("Nota de conciliacion: la bitacora de piso y el embudo mensual "
               "reportan conversiones distintas en etapas compartidas "
               "(p. ej. PDM → SDC: 5.3% vs 82.3%). Se deben a definiciones de "
               "captura distintas; antes de usar ambas fuentes en una misma "
               "decision, conviene alinearlas.")
