# Dashboard GAC - Insights de Negocio
# Reestructuracion NARRATIVA: 3 preguntas + anexo tecnico
#   1. Como vamos?   2. Donde se fuga?   3. Que hacemos?   4. Modelos y evidencia
# Modelos en vistas_modelos.py (SIN CAMBIOS). Evidencia: archivo/app_etapas_original.py
# Cambios: nav 5->4 vistas; Productos BCS->V1; Marketing->V2; Asesores+BDC->V3;
#          Vista4 por pregunta; fix tarjeta BDC; tarjetas KPI con HTML; ROI dinamico.

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import vistas_modelos as vm

st.set_page_config(page_title="GAC - Insights de Negocio", page_icon="🚗", layout="wide")

# =========================================================
# CARGA DE DATOS (identica a la version que ya funciona)
# =========================================================
@st.cache_data
def load_data():
    df_bitacora = pd.read_csv("Bitacora_Piso_Limpia.csv")
    df_leads = pd.read_csv("Leads_Reales_Limpia.csv")
    df_bitacora["Periodo"] = pd.to_datetime(df_bitacora["Fecha"]).dt.strftime("%Y-%m")
    df_leads["Fecha"] = pd.to_datetime(df_leads["Periodo"] + "-01")
    return df_bitacora, df_leads

@st.cache_data
def load_leads_detalle():
    try:
        d = pd.read_csv("base_leads_limpia.csv")
    except FileNotFoundError:
        return None
    d["Estado"] = d["Estado"].astype(str).str.strip().str.capitalize()
    d["Estado"] = d["Estado"].replace({"1 ventas": "Ventas", "1 venta": "Ventas",
                                       "Nan": "Sin dato"})
    d["Producto"] = d["Producto"].astype(str).str.strip().replace({"nan": "Sin especificar"})
    d = d[~d["Producto"].isin(["No envian producto valido"])]
    d["Temperatura"] = d["Temperatura"].astype(str).str.strip()
    d["Asesor"] = d["Asesor"].astype(str).str.strip().str.title()
    return d

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
    dfc["PDM_txt"] = dfc["PDM"].map({True: "Con PDM", False: "Sin PDM"})
    return dfc

@st.cache_data
def load_plazas_anova():
    try:
        dfp = pd.read_csv("Plazas.csv")
    except FileNotFoundError:
        return None
    dfp.columns = dfp.columns.str.strip()
    dfp["Fecha"] = pd.to_datetime(dfp["Fecha Origen"],
                                  format="%d/%m/%Y %H:%M", errors="coerce")
    dfp["Producto_L"] = dfp["Producto"].astype(str).str.strip().str.upper()
    dfp["Plaza_L"] = (dfp["Origen"].astype(str).str.strip().str.upper()
        .replace({"PLAZA ANGELOPOLIS": "Angelopolis",
                  "GRAN PATIO TLAXCALA": "Gran Patio Tlaxcala",
                  "GALERIAS TLAXCALA": "Galerias Tlaxcala"}))
    dfp = dfp[dfp["Producto_L"].isin(["EMZOOM", "GS8", "EMKOO"])
              & dfp["Fecha"].notna()
              & dfp["Plaza_L"].isin(["Angelopolis", "Gran Patio Tlaxcala"])]
    dfp["Semana"] = dfp["Fecha"].dt.to_period("W").astype(str)
    sem = (dfp.groupby(["Semana", "Producto_L", "Plaza_L"])
           .size().reset_index(name="Leads"))
    return sem

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
df_detalle = load_leads_detalle()
df_resumen, df_funnel, df_bcs = load_regresion_data()
df_citas = load_citas()
df_plazas_sem = load_plazas_anova()
df_mkt = load_mkt()
# =========================================================
# TEMA VINO TINTO OSCURO (presentacion GAC)
# =========================================================
GAC_AZUL       = "#D6335F"
GAC_AZUL_MED   = "#A8284C"
GAC_AZUL_CLARO = "#E89AAA"
GAC_PLATA      = "#C9747F"
GAC_OSCURO     = "#F3E6EA"
GAC_BLANCO     = "#251219"
PANEL          = "#221119"
BORDE_PANEL    = "#40222F"
TEXTO_SUAVE    = "#C9A2AE"
REJILLA        = "#3A2230"
VERDE_OK       = "#3FBF6B"
AMBAR          = "#E8A33D"
ROJO_ALERTA    = "#FF5468"
RAMPA_VINO     = ["#F0B9C7", "#E89AAA", "#C9747F", "#D6335F",
                  "#A8284C", "#7E1330", "#5C0A1E"]

ESTILO_FUENTE = dict(family="Arial, sans-serif", size=16, color=GAC_OSCURO)

# CSS global: tarjetas KPI e insight con gradientes vino (RESTAURADO:
# antes llegaba vacio y las tarjetas se veian como texto plano)
st.markdown("""
<style>
.kpi-card{
  border-radius:16px; padding:18px 20px; color:#fff;
  box-shadow:0 6px 18px rgba(0,0,0,.35); border:1px solid rgba(255,255,255,.08);
  min-height:118px; display:flex; flex-direction:column; justify-content:center;
}
.kpi-label{font-size:14px; opacity:.85; letter-spacing:.3px; margin-bottom:6px;}
.kpi-value{font-size:30px; font-weight:700; line-height:1.05;}
.kpi-sub{font-size:12px; opacity:.8; margin-top:6px;}
.insight-card{
  background:#221119; border:1px solid #40222F; border-left:6px solid #D6335F;
  border-radius:12px; padding:16px 18px; min-height:150px;
}
.insight-title{font-size:15px; font-weight:700; color:#F3E6EA; margin-bottom:6px;}
.insight-dato{font-size:22px; font-weight:700; margin-bottom:6px;}
.insight-detalle{font-size:13px; color:#C9A2AE; line-height:1.35;}
.badge-hallazgo{
  display:inline-block; background:#D6335F; color:#fff; font-size:11px;
  padding:2px 8px; border-radius:999px; margin-right:8px; vertical-align:middle;
}
</style>
""", unsafe_allow_html=True)

def estilo_fig(fig, titulo, alto=430):
    fig.update_layout(
        font=ESTILO_FUENTE, height=alto,
        title=dict(text=titulo, font=dict(size=18, color=GAC_AZUL),
                   x=0.02, xanchor="left", pad=dict(l=6)),
        margin=dict(l=70, r=40, t=70, b=60),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=GAC_AZUL,
                        font=dict(color=GAC_OSCURO, size=15, family="Arial")),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        yaxis=dict(gridcolor=REJILLA))
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    return fig

CONFIG_GRAFICA = dict(displaylogo=False, displayModeBar=True, scrollZoom=True,
                      modeBarButtonsToRemove=["select2d", "lasso2d"])
def grafica(fig, **kw):
    fig.update_layout(hoverlabel=dict(bgcolor=PANEL, bordercolor=GAC_AZUL,
                                      font=dict(color=GAC_OSCURO, size=15, family="Arial")))
    fig.update_xaxes(tickfont=dict(size=15))
    fig.update_yaxes(tickfont=dict(size=15))
    st.plotly_chart(fig, config=CONFIG_GRAFICA, **kw)

def tarjeta_insight(titulo, dato, detalle, color=GAC_AZUL):
    st.markdown(
        f'<div class="insight-card" style="border-left-color:{color}">'
        f'<div class="insight-title">{titulo}</div>'
        f'<div class="insight-dato" style="color:{color}">{dato}</div>'
        f'<div class="insight-detalle">{detalle}</div></div>',
        unsafe_allow_html=True)

DEGRADADOS_KPI = [
    "linear-gradient(135deg,#7A0E28 0%,#C42B54 55%,#E35A7C 100%)",
    "linear-gradient(135deg,#4A0A22 0%,#8F1F42 60%,#C42B54 100%)",
    "linear-gradient(135deg,#931232 0%,#D6335F 60%,#F0769A 100%)",
]

def tarjeta_kpi(etiqueta, valor, subtexto="", variante=0):
    g = DEGRADADOS_KPI[variante % len(DEGRADADOS_KPI)]
    sub = f'<div class="kpi-sub">{subtexto}</div>' if subtexto else ""
    st.markdown(
        f'<div class="kpi-card" style="background:{g}">'
        f'<div class="kpi-label">{etiqueta}</div>'
        f'<div class="kpi-value">{valor}</div>{sub}</div>',
        unsafe_allow_html=True)

# =========================================================
# PRE-CALCULOS DE NEGOCIO (identicos a la version que funciona)
# =========================================================
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

df_resumen["Periodo"] = (df_resumen["Año"].astype(str) + "-"
                         + df_resumen["Mes"].astype(int).astype(str).str.zfill(2))

g_asesor = df_bitacora.groupby("Asesor").agg(
    Visitas=("Cliente", "count"), PDM=("PDM", "sum"), SDC=("SDC", "sum"),
    Ventas=("Venta", "sum")).reset_index()
g_asesor["% PDM"] = (g_asesor["PDM"] / g_asesor["Visitas"] * 100).round(1)
g_asesor["% Venta/visita"] = (g_asesor["Ventas"] / g_asesor["Visitas"] * 100).round(1)
g_asesor = g_asesor.sort_values("Ventas", ascending=False)

visitas_mes = (df_bitacora.groupby(["Periodo", "Mes"], as_index=False)
               .size().rename(columns={"size": "Visitas"}))

g_bdc = None
if df_citas is not None and len(df_citas):
    g_bdc = df_citas.groupby("BDC_L").agg(
        Registros=("Potencial", "count"), Potencial=("Potencial", "mean"),
        PDM=("PDM", "sum"), Ventas=("Venta", "sum")).reset_index()
    g_bdc["Potencial"] = g_bdc["Potencial"].round(1)
    g_bdc["% PDM"] = (g_bdc["PDM"] / g_bdc["Registros"] * 100).round(1)
    g_bdc["% Venta"] = (g_bdc["Ventas"] / g_bdc["Registros"] * 100).round(1)
    g_bdc = g_bdc.sort_values("% Venta", ascending=False)

g_mkt = None
if df_mkt is not None and len(df_mkt):
    g_mkt = df_mkt.groupby("Canal").agg(
        Meses=("Fecha", "nunique"), Leads=("Leads", "sum"),
        Ventas=("Ventas", "sum"), Presupuesto=("Presupuesto", "sum")).reset_index()
    g_mkt["Costo / lead"] = (g_mkt["Presupuesto"] / g_mkt["Leads"]).round(0)
    g_mkt["Costo / venta"] = (g_mkt["Presupuesto"] / g_mkt["Ventas"]).round(0)
    g_mkt["Conv. %"] = (g_mkt["Ventas"] / g_mkt["Leads"] * 100).round(1)
    g_mkt["Leads / $1k"] = (g_mkt["Leads"] / g_mkt["Presupuesto"] * 1000).round(1)

if df_bcs is not None:
    df_bcs["Cumplimiento %"] = (df_bcs["Ventas_Reales"] / df_bcs["Objetivo"] * 100).round(0)

efec_inicio = float(df_leads["Efectividad_Leads"].head(6).mean())
efec_actual = float(df_leads["Efectividad_Leads"].tail(6).mean())

# --- ROI de la fuga de credito (DINAMICO: se calcula de los datos) ---
BENCH_APROBACION = 65.0
sdc_gen_mes = float(df_funnel["SDC_Generada"].mean())
tasa_aprob_actual = float((df_funnel["SDC_Aprobada"] / df_funnel["SDC_Generada"] * 100).mean())
roi_credito_mes = max(0.0, (BENCH_APROBACION - tasa_aprob_actual) / 100.0) * sdc_gen_mes
roi_credito_anio = roi_credito_mes * 12.0

# =========================================================
# ENCABEZADO Y NAVEGACION (3 preguntas + anexo tecnico)
# =========================================================
st.markdown('<h1 style="color:#D6335F;margin-bottom:0">GAC · Insights de Negocio</h1>',
            unsafe_allow_html=True)
st.caption("Plaza Angelopolis · Analitica comercial: como vamos, donde se fuga, "
           "que hacemos · y la evidencia estadistica que lo respalda")
st.divider()

if os.path.exists("logo_gac.png"):
    st.sidebar.image("logo_gac.png", width="stretch")
else:
    st.sidebar.markdown("GAC MOTOR", unsafe_allow_html=True)
st.sidebar.markdown("---")

VISTAS = ["1. ¿Cómo vamos?",
          "2. ¿Dónde se fuga el dinero?",
          "3. ¿Qué hacemos esta semana?",
          "4. Modelos y evidencia"]
vista = st.sidebar.radio("Vista:", VISTAS, key="nav_vista")

st.sidebar.markdown("---")
st.sidebar.write("Bitacora de piso: **%d** registros" % len(df_bitacora))
st.sidebar.write("Leads reales: **%d** meses" % len(df_leads))
if df_detalle is not None:
    st.sidebar.write("Leads plaza: **%d** registros" % len(df_detalle))
st.sidebar.write("Base resumen: **%d** meses" % len(df_resumen))
st.sidebar.write("Embudo: **%d** meses" % len(df_funnel))
if df_citas is not None:
    st.sidebar.write("Citas digital: **%d** registros" % len(df_citas))
if df_mkt is not None:
    st.sidebar.write("Marketing: **%d** obs." % len(df_mkt))
if df_plazas_sem is not None:
    st.sidebar.write("Plazas: **%d** semanas-prod." % len(df_plazas_sem))
st.sidebar.markdown("---")
st.sidebar.caption("Evidencia del proceso estadistico: `archivo/app_etapas_original.py`")

st.markdown(f'<span class="badge-hallazgo">Estas en</span> **{vista}**',
            unsafe_allow_html=True)
# =========================================================
# VISTA 1: COMO VAMOS?  (termometro: KPIs, ventas vs objetivo, productos)
# =========================================================
if vista == VISTAS[0]:
    st.subheader("¿Cómo va la plaza?")

    alcance = st.radio("Periodo:", ["Ultimos 12 meses", "Todo el historico (30 meses)"],
                       horizontal=True)
    r = df_resumen.tail(12).copy() if alcance.startswith("Ultimos") else df_resumen.copy()
    n_meses_bajo_r = int((r["% CAL Ventas"] < 1).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        tarjeta_kpi("Ventas (" + str(len(r)) + " meses)",
                    f"{int(r['Ventas Real'].sum()):,}", variante=0)
    with k2:
        tarjeta_kpi("Cumplimiento de objetivo",
                    f"{float(r['% CAL Ventas'].mean())*100:.0f}%", variante=1)
    with k3:
        tarjeta_kpi("Leads / mes", f"{r['Leads'].mean():,.0f}", variante=2)
    with k4:
        tarjeta_kpi("Leads por venta",
                    f"{r['Leads'].sum()/r['Ventas Real'].sum():.0f}", variante=1)
    with k5:
        tarjeta_kpi("Efectividad de leads", f"{efec_actual:.0f}%",
                    subtexto="ultimos 6 meses", variante=0)
    st.caption("Meses bajo objetivo en el periodo: **%d de %d**" % (n_meses_bajo_r, len(r)))

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
        title=dict(text="Ventas vs objetivo · % cumplimiento",
                   font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        yaxis=dict(title="Ventas", gridcolor=REJILLA),
        yaxis2=dict(title="% cumplimiento", overlaying="y", side="right",
                    range=[0, 160], showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    grafica(fig, width="stretch")

    anomalias = r[r["% CAL Ventas"] < 0.6][["Periodo", "Leads", "Ventas Real",
                                            "Obj Ventas", "% CAL Ventas"]].copy()
    anomalias["% CAL Ventas"] = (anomalias["% CAL Ventas"] * 100).round(0)
    if len(anomalias):
        st.markdown("**Meses criticos (menos del 60% del objetivo):**")
        st.dataframe(anomalias.rename(columns={"% CAL Ventas": "% cumplimiento"}),
                     width="stretch", hide_index=True)

    st.markdown("### Hallazgos clave")
    tasa_aprob = float(df_trans.loc[df_trans["Transicion"] == "Aprobacion de credito", "Tasa %"].iloc[0])
    tasa_agend = float(df_trans.loc[df_trans["Transicion"] == "Agendamiento", "Tasa %"].iloc[0])
    c1, c2, c3 = st.columns(3)
    with c1:
        tarjeta_insight("Aprobacion de credito", f"Solo {tasa_aprob:.0f}% de las solicitudes se aprueba",
            "El mayor cuello del embudo. Revisar filtros de las financieras y "
            "documentacion que se pide al cliente. Respaldo: ANOVA y reg. del embudo (vista 4).",
            ROJO_ALERTA)
    with c2:
        tarjeta_insight("Agendamiento de citas", f"Solo {tasa_agend:.0f}% de los leads efectivos agendan",
            "2 de cada 3 leads contactados no llegan a agendar cita. "
            "Oportunidad: reforzar el contacto BDC en las primeras 48 horas.",
            ROJO_ALERTA)
    with c3:
        tarjeta_insight("Calidad del lead", f"Efectividad: {efec_inicio:.0f}% → {efec_actual:.0f}%",
            "El volumen de leads casi se triplico desde 2024, pero la "
            "efectividad cayo ~20 puntos. Mas leads no significan mas ventas.",
            AMBAR)
    c4, c5, c6 = st.columns(3)
    with c4:
        # BUG CORREGIDO: "peor" ahora es el de MENOR conversion con volumen minimo,
        # no el de MAYOR volumen (que podia ser tambien el mejor).
        if g_bdc is not None and len(g_bdc) >= 2:
            mejor = g_bdc.iloc[0]
            elegibles = g_bdc[g_bdc["Registros"] >= 10]
            if len(elegibles) == 0:
                elegibles = g_bdc
            peor = elegibles.sort_values("% Venta").iloc[0]
            if peor["BDC_L"] != mejor["BDC_L"]:
                tarjeta_insight("Brecha entre ejecutivos BDC",
                    f"{mejor['BDC_L']}: {mejor['% Venta']:.0f}% vs {peor['BDC_L']}: {peor['% Venta']:.0f}%",
                    f"{mejor['BDC_L']} convierte "
                    f"{mejor['% Venta']/max(peor['% Venta'],0.1):.1f}x mas que {peor['BDC_L']}. "
                    "Revisar asignacion y proceso. Respaldo: ANOVA citas (vista 4).",
                    AMBAR)
            else:
                tarjeta_insight("Brecha entre ejecutivos BDC", "Sin brecha relevante",
                    "Las conversiones entre ejecutivos son homogeneas en el periodo.",
                    VERDE_OK)
        else:
            tarjeta_insight("Brecha entre ejecutivos BDC", "Sin datos",
                            "La base de citas digitales no esta disponible.")
    with c5:
        if g_mkt is not None and len(g_mkt) >= 2 and "Piso" in set(g_mkt["Canal"]) and "Digital" in set(g_mkt["Canal"]):
            cv = g_mkt.set_index("Canal")["Costo / venta"]
            ratio = cv["Piso"] / cv["Digital"]
            tarjeta_insight("Costo por venta: Piso vs Digital",
                f"Piso cuesta {ratio:.1f}x mas que Digital",
                f"${cv['Piso']:,.0f} por venta en piso vs ${cv['Digital']:,.0f} en digital. "
                "El piso convierte mejor, pero el costo es desproporcionado.",
                AMBAR)
        else:
            tarjeta_insight("Marketing por canal", "Sin datos suficientes",
                            "La base de marketing no tiene los 3 canales completos.")
    with c6:
        if df_bcs is not None:
            bajo = df_bcs[df_bcs["Cumplimiento %"] < 60]
            alto = df_bcs[df_bcs["Cumplimiento %"] >= 150]
            txt_bajo = ", ".join(f"{m} ({c:.0f}%)" for m, c in
                                 zip(bajo["Modelo"], bajo["Cumplimiento %"])) or "ninguno"
            txt_alto = ", ".join(f"{m} ({c:.0f}%)" for m, c in
                                 zip(alto["Modelo"], alto["Cumplimiento %"])) or "ninguno"
            tarjeta_insight("Objetivos por modelo mal calibrados",
                "Cumplimientos de " + f"{df_bcs['Cumplimiento %'].min():.0f}% a {df_bcs['Cumplimiento %'].max():.0f}%",
                f"Bajo objetivo: {txt_bajo}. Sobre-objetivo: {txt_alto}. "
                "La dispersion sugiere metas mal calibradas o un mix desbalanceado.",
                AMBAR)
        else:
            tarjeta_insight("Objetivos por modelo", "Sin datos",
                            "La base BCS no esta disponible.")

    # ---- Productos (BCS): movido aqui desde la antigua vista Desempeno ----
    st.divider()
    st.markdown("### ¿Qué productos cumplen su objetivo?")
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
            estilo_fig(fig_p, "Cumplimiento del objetivo por modelo")
            fig_p.update_xaxes(gridcolor=REJILLA,
                               range=[0, max(260, float(df_bcs["Cumplimiento %"].max()) * 1.15)])
            grafica(fig_p, width="stretch")
    else:
        st.info("Base BCS no disponible.")
# =========================================================
# VISTA 2: DONDE SE FUGA EL DINERO? (embudo + fugas + marketing)
# =========================================================
elif vista == VISTAS[1]:
    st.subheader("¿Dónde se fuga la venta?")
    st.caption("Embudo mensual promedio (" + str(len(df_funnel)) +
               " meses) y tasas de transicion entre etapas. "
               "Barra roja = transicion bajo el benchmark interno.")

    col_f, col_t = st.columns([3, 4])
    with col_f:
        fig_fun = go.Figure(go.Funnel(
            y=[NOMBRE_ETAPA[e] for e in ETAPAS],
            x=prom_etapas[ETAPAS].round(0),
            textinfo="value+percent previous",
            marker_color=RAMPA_VINO + ["#3E0717"]))
        fig_fun.update_layout(font=ESTILO_FUENTE, height=480,
            title=dict(text="Embudo promedio por mes", font=dict(size=18, color=GAC_AZUL)),
            paper_bgcolor=GAC_BLANCO,
            margin=dict(l=10, r=10, t=60, b=10))
        grafica(fig_fun, width="stretch")
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
                lambda r: f"{r['Desde']} → siguiente. Benchmark: {r['Benchmark %']:.0f}%",
                axis=1), hoverinfo="text")
        fig_tr.update_layout(font=ESTILO_FUENTE, height=480,
            title=dict(text="Transicion por etapa vs benchmark",
                       font=dict(size=15, color=GAC_AZUL)),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            xaxis=dict(title="%", gridcolor=REJILLA, range=[0, 105]),
            yaxis=dict(autorange="reversed"), showlegend=False)
        grafica(fig_tr, width="stretch")

    # ---- ROI de la fuga de credito (el punchline, calculado dinamico) ----
    st.markdown("### ¿Cuánto vale cerrar la fuga de crédito?")
    r1, r2, r3 = st.columns(3)
    with r1:
        tarjeta_kpi("Tasa actual de aprobacion", f"{tasa_aprob_actual:.0f}%",
                    subtexto=f"benchmark: {BENCH_APROBACION:.0f}%", variante=1)
    with r2:
        tarjeta_kpi("Ventas extra / mes", f"+{roi_credito_mes:.1f}",
                    subtexto="si se alcanza el benchmark", variante=2)
    with r3:
        tarjeta_kpi("Ventas extra / año", f"+{roi_credito_anio:.0f}",
                    subtexto="sin gastar mas en marketing", variante=0)
    st.caption(f"Calculo: con {sdc_gen_mes:.0f} solicitudes de credito/mes promedio, "
               f"subir la aprobacion de {tasa_aprob_actual:.0f}% al benchmark de "
               f"{BENCH_APROBACION:.0f}% son +{roi_credito_mes:.1f} ventas/mes "
               f"({(BENCH_APROBACION - tasa_aprob_actual):.0f} pts × SDC), sin un peso mas en marketing.")

    c1, c2, c3 = st.columns(3)
    with c1:
        fig_c = go.Figure(go.Scatter(
            x=df_funnel["Mes"], y=df_funnel["Conv_Lead_Venta"].round(2),
            mode="lines+markers", line=dict(color=GAC_AZUL, width=3),
            marker=dict(size=8), fill="tozeroy",
            fillcolor="rgba(214,51,95,0.15)"))
        estilo_fig(fig_c, "Conversion lead → venta (%)")
        fig_c.update_yaxes(gridcolor=REJILLA)
        grafica(fig_c, width="stretch")
    with c2:
        fig_e = go.Figure(go.Scatter(
            x=df_leads["Periodo"], y=df_leads["Efectividad_Leads"],
            mode="lines+markers", line=dict(color=GAC_AZUL_MED, width=3),
            marker=dict(size=7)))
        fig_e.add_hline(y=df_leads["Efectividad_Leads"].mean(),
                        line_dash="dash", line_color=GAC_PLATA,
                        annotation_text="promedio")
        estilo_fig(fig_e, "Efectividad de leads (%)")
        fig_e.update_yaxes(gridcolor=REJILLA)
        grafica(fig_e, width="stretch")
    with c3:
        fig_v = go.Figure(go.Bar(
            x=visitas_mes["Periodo"], y=visitas_mes["Visitas"],
            marker_color=GAC_AZUL_MED))
        estilo_fig(fig_v, "Visitas registradas por mes")
        fig_v.update_xaxes(tickangle=-45)
        grafica(fig_v, width="stretch")

    st.markdown("**Detalle mensual del embudo**")
    tabla_f = df_funnel[["Mes"] + ETAPAS + ["Conv_Lead_Venta"]].copy()
    tabla_f["Conv_Lead_Venta"] = tabla_f["Conv_Lead_Venta"].round(2)
    tabla_f.columns = ["Mes"] + [NOMBRE_ETAPA[e] for e in ETAPAS] + ["Conv. lead→venta %"]
    st.dataframe(tabla_f, width="stretch", hide_index=True)

    # ---- Marketing por canal (movido aqui: misma conversacion que el embudo) ----
    st.divider()
    st.markdown("### El presupuesto al revés: costo por venta por canal")
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
                       "barato pero convierte poco. No son sustitutos.")
        with col2:
            fig_m = px.bar(g_mkt, x="Canal", y="Costo / venta", text="Costo / venta",
                           color_discrete_sequence=[GAC_AZUL])
            estilo_fig(fig_m, "Costo por venta por canal")
            fig_m.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_m.update_yaxes(gridcolor=REJILLA)
            grafica(fig_m, width="stretch")
        st.caption("Respaldo estadistico del efecto del canal: ANOVA marketing (vista 4).")
    else:
        st.info("Base de marketing sin meses completos de los 3 canales.")
# =========================================================
# VISTA 3: QUE HACEMOS ESTA SEMANA? (proyeccion + alarma + a quien replicar)
# =========================================================
elif vista == VISTAS[2]:
    st.subheader("¿Qué hacemos esta semana?")
    st.caption("Tres palancas accionables: proyectar el cierre del mes, anticipar "
               "un mes flojo, y replicar a quienes mejor ejecutan el proceso.")

    st.markdown("### 1 · ¿Llegamos al cierre? Proyeccion del mes")
    vm.vista_reg_funnel(df_funnel)

    st.divider()
    st.markdown("### 2 · ¿Se viene un mes flojo? Alarma temprana")
    vm.vista_logistica_ventas(df_resumen)

    st.divider()
    st.markdown("### 3 · ¿A quién replicamos? Asesores de piso y BDC")
    tab_ases, tab_bdc = st.tabs(["Asesores de piso", "Ejecutivos BDC (digital)"])

    with tab_ases:
        col1, col2 = st.columns([3, 4])
        with col1:
            st.markdown("**Ranking por ventas cerradas**")
            st.dataframe(g_asesor.rename(columns={
                "% PDM": "% prueba de manejo",
                "% Venta/visita": "% venta por visita"}),
                width="stretch", hide_index=True)
        with col2:
            fig_a = px.bar(g_asesor.sort_values("Ventas"), x="Ventas", y="Asesor",
                           orientation="h", text="Ventas",
                           color_discrete_sequence=[GAC_AZUL])
            estilo_fig(fig_a, "Ventas cerradas por asesor")
            fig_a.update_traces(textposition="outside")
            fig_a.update_xaxes(gridcolor=REJILLA)
            grafica(fig_a, width="stretch")

        sin_ventas = g_asesor[g_asesor["Ventas"] == 0]
        top_a = g_asesor.iloc[0]
        share_top = top_a["Ventas"] / max(g_asesor["Ventas"].sum(), 1) * 100
        ha1, ha2 = st.columns(2)
        with ha1:
            tarjeta_insight("Lider de piso (a replicar)",
                f"{top_a['Asesor']}: {int(top_a['Ventas'])} ventas",
                f"Concentra el {share_top:.0f}% de las ventas de piso y convierte "
                f"{top_a['% Venta/visita']:.0f}% de sus visitas. Su proceso es la "
                "referencia para capacitar al resto del equipo.", VERDE_OK)
        with ha2:
            if len(sin_ventas):
                tarjeta_insight("Asesores sin ventas cerradas",
                    f"{len(sin_ventas)} de {len(g_asesor)}",
                    "Sin cierres en el periodo: " + ", ".join(sin_ventas["Asesor"]) +
                    " (" + ", ".join(str(int(v)) + " visitas" for v in sin_ventas["Visitas"]) +
                    "). Revisar asignacion de leads, capacitacion de cierre o "
                    "rotacion de piso.", ROJO_ALERTA)
            else:
                tarjeta_insight("Asesores sin ventas cerradas", "Ninguno",
                    "Todos los asesores cerraron al menos una venta en el periodo.",
                    VERDE_OK)

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
                estilo_fig(fig_b, "Conversion a venta por ejecutivo BDC")
                fig_b.update_traces(textposition="outside")
                fig_b.update_xaxes(gridcolor=REJILLA)
                grafica(fig_b, width="stretch")

            mejor_b = g_bdc.iloc[0]
            peor_b = g_bdc.sort_values("Registros", ascending=False).iloc[0]
            hb1, hb2 = st.columns(2)
            with hb1:
                tarjeta_insight("Mejor conversion BDC (a replicar)",
                    f"{mejor_b['BDC_L']}: {mejor_b['% Venta']:.1f}%",
                    f"Con {int(mejor_b['Registros'])} registros atendidos. "
                    "Documentar su guion de seguimiento y replicarlo en el equipo. "
                    "Respaldo: ANOVA citas (vista 4).", VERDE_OK)
            with hb2:
                if peor_b["% Venta"] < mejor_b["% Venta"] / 2:
                    tarjeta_insight("Brecha de desempeno",
                        f"{peor_b['BDC_L']}: {peor_b['% Venta']:.1f}%",
                        f"Concentra el mayor volumen ({int(peor_b['Registros'])} registros) "
                        f"pero convierte {peor_b['% Venta']:.1f}% contra {mejor_b['% Venta']:.1f}% "
                        f"de {mejor_b['BDC_L']}. Auditar su proceso de seguimiento.",
                        ROJO_ALERTA)
                else:
                    tarjeta_insight("Brecha de desempeno", "Sin brecha critica",
                        "Ningun ejecutivo de alto volumen convierte a menos de la "
                        "mitad del mejor del equipo.", VERDE_OK)
        else:
            st.info("Base de citas digitales no disponible.")

# =========================================================
# VISTA 4: MODELOS Y EVIDENCIA (demostracion, organizada por pregunta)
# =========================================================
else:
    st.subheader("Modelos y evidencia estadistica")
    st.caption("Todo el modelado del proceso (Etapas II y III), organizado por la "
               "pregunta de negocio que responde. Trazable a los notebooks del equipo.")

    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f = st.tabs([
        "A · ¿Qué mueve las ventas?",
        "B · Proyeccion (reg. embudo)",
        "C · Alarma (log. ventas)",
        "D · Clasificar visita (log. PDM)",
        "E · ANOVA (quién y dónde)",
        "F · Laboratorio"])

    with tab_a:
        st.markdown("#### ¿Qué variables se mueven con las ventas?")
        vm.vista_correlaciones(df_resumen)
        st.divider()
        st.markdown("#### Regresion lineal simple")
        vm.vista_reg_simple(df_resumen, df_bcs)
        st.divider()
        st.markdown("#### Regresion lineal multiple")
        vm.vista_reg_multiple(df_resumen)

    with tab_b:
        vm.vista_reg_funnel(df_funnel)

    with tab_c:
        vm.vista_logistica_ventas(df_resumen)

    with tab_d:
        vm.vista_logistica_pdm(df_bitacora)
        st.warning("Nota metodologica: la variable `Venta` aparece como predictora pero "
                   "ocurre DESPUES de la prueba de manejo (fuga temporal). Se conserva "
                   "para replicar el notebook original; en produccion deberia excluirse.")

    with tab_e:
        vm.vista_anova(df_citas, df_mkt, df_plazas_sem)

    with tab_f:
        st.markdown("#### Laboratorio — evidencia del proceso estadistico")
        st.markdown("""
        Las **13 vistas originales** del dashboard (Etapas I, II y III) se conservan
        intactas como evidencia del trabajo del equipo. Los modelos ya estan integrados
        en vivo en las pestanas A-E de esta vista.
        """)
        st.markdown("**Donde encontrar cada cosa:**")
        st.markdown("""
| Artefacto | Contenido |
|---|---|
| `vistas_modelos.py` | Modelos predictivos y ANOVA integrados (esta vista) |
| `archivo/app_etapas_original.py` | Dashboard completo con las 13 vistas originales |
| `DASHBOARD_GAC.ipynb` | Notebook que generaba la version original |
| `notebooks/etapa1_limpieza/` | Limpieza y pipeline (Entregable 1) |
| `notebooks/etapa2_modelado/` | Regresiones simple, multiple y logistica |
| `notebooks/etapa3_anova/` | Despliegue ANOVA (marketing y plazas) |
| `base_resumen_limpio.csv` | 30 meses / 26 variables (regresiones) |
| `funnel_mensual.csv` | Embudo de 18 meses (reg. del embudo) |
| `bcs_modelos.csv` | 6 modelos GAC (Balance Scorecard) |
        """)
        st.info("Para consultar la version original completa:  "
                "`streamlit run archivo/app_etapas_original.py`")
        st.caption("Nota de conciliacion: la bitacora de piso y el embudo mensual "
                   "reportan conversiones distintas en etapas compartidas "
                   "(p. ej. PDM → SDC: 5.3% vs 82.3%). Se deben a definiciones de "
                   "captura distintas; conviene alinearlas antes de usar ambas en una "
                   "misma decision.")