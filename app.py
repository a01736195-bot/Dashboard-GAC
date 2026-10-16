# Dashboard GAC - Analisis Univariado + Modelado Predictivo + Despliegue ANOVA
# Tema vino tinto / borgona (presentacion GAC) - diseno plano
# Etapa III: ANOVA one-way y multifactorial (scipy + statsmodels, plantilla 7_ANOVA.ipynb)

import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# =========================================================
# CONFIGURACION DE PAGINA
# =========================================================
st.set_page_config(
    page_title="Analisis GAC",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# CARGA DE DATOS
# =========================================================
@st.cache_data
def load_data():
    df_bitacora = pd.read_csv("Bitacora_Piso_Limpia.csv")
    df_leads = pd.read_csv("Leads_Reales_Limpia.csv")

    # Periodo para el filtro (YYYY-MM)
    df_bitacora["Periodo"] = (
        pd.to_datetime(df_bitacora["Fecha"]).dt.strftime("%Y-%m")
    )
    df_leads["Fecha"] = pd.to_datetime(df_leads["Periodo"] + "-01")

    return df_bitacora, df_leads

@st.cache_data
def load_leads_detalle():
    # Base de leads de plaza (detalle). Si no existe el archivo, devolvemos None
    try:
        d = pd.read_csv("base_leads_limpia.csv")
    except FileNotFoundError:
        return None

    # Limpieza defensiva (por si el CSV cambia)
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
    # Bases para la Etapa II (modelado predictivo)
    df_resumen = pd.read_csv("base_resumen_limpio.csv").bfill().ffill()
    df_funnel = pd.read_csv("funnel_mensual.csv")
    # Base BCS: 6 modelos (Objetivo vs Ventas reales)
    try:
        df_bcs = pd.read_csv("bcs_modelos.csv")
    except FileNotFoundError:
        df_bcs = None
    return df_resumen, df_funnel, df_bcs

df_bitacora, df_leads = load_data()
df_detalle = load_leads_detalle()
df_resumen, df_funnel, df_bcs = load_regresion_data()

# =========================================================
# CARGA DE DATOS - ETAPA III (DESPLIEGUE ANOVA)
# =========================================================
@st.cache_data
def load_citas_anova():
    # Citas Digital: seguimiento de leads del canal digital (668 registros)
    # Preprocesamiento identico al notebook Actividad_4_ANOVA.ipynb
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
        if x.startswith("antonio/saul"): return None
        if x.startswith("antonio"): return "Antonio"
        if x.startswith(("ignacio", "igancio")): return "Ignacio"
        if x.startswith("saul"): return "Saul"
        return None

    def limpiar_estatus(x):
        x = str(x).strip().lower()
        if x.startswith(("activo", "acitvo", "acrivo", "ativo", "activio")):
            return "Activo"
        if x.startswith("finalizado"): return "Finalizado"
        if x.startswith("venta"): return "Venta"
        return None

    dfc["BDC_L"] = dfc["BDC"].apply(limpiar_bdc)
    dfc["Estatus_L"] = dfc["Estatus de Lead"].apply(limpiar_estatus)
    dfc = dfc[dfc["Potencial"].notna() & dfc["BDC_L"].notna()
              & dfc["Estatus_L"].notna()].copy()
    # Outliers por IQR en la variable de respuesta
    q1, q3 = dfc["Potencial"].quantile([0.25, 0.75])
    iqr = q3 - q1
    dfc = dfc[(dfc["Potencial"] >= q1 - 1.5 * iqr)
              & (dfc["Potencial"] <= q3 + 1.5 * iqr)].copy()
    dfc["PDM_txt"] = dfc["PDM"].map({True: "Con PDM", False: "Sin PDM"})
    return dfc

@st.cache_data
def load_plazas_anova():
    # Plazas: leads de piso -> variable de respuesta = leads por semana
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
def load_mkt_anova():
    # Marketing: bloque "Presupuesto mensual de MKT por canal" (parser del
    # notebook ANOVA_Marketing_GAC.ipynb) -> Canal x Mes (Leads, Ventas, Presupuesto)
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
    datos["Anio"] = datos["Fecha"].dt.year.astype(str)
    completos = datos.dropna(subset=["Leads", "Ventas", "Presupuesto"])
    fc = completos.groupby("Fecha")["Canal"].nunique()
    fc = fc[fc.eq(3)].index
    dfm = (completos[completos["Fecha"].isin(fc)]
           .sort_values(["Fecha", "Canal"]).reset_index(drop=True))
    return dfm

df_citas = load_citas_anova()
df_plazas_sem = load_plazas_anova()
df_mkt = load_mkt_anova()

# =========================================================
# PALETA VINO TINTO / BORGONA (presentacion GAC)
# =========================================================
GAC_AZUL      = "#800020"   # tinto principal
GAC_AZUL_MED  = "#9B1B30"   # bordo
GAC_AZUL_CLARO= "#DDA0A8"   # rosa palido
GAC_PLATA     = "#C9747F"   # rosa medio (lineas/secundarios)
GAC_OSCURO    = "#4A0414"   # vino oscuro (texto)
GAC_BLANCO    = "#FFFFFF"
REJILLA       = "#F3E4E7"   # rejilla suave rosada
GRIS_DATOS    = "#9AA3B0"   # datos/contexto en gris (teoria de color)

ESTILO_FUENTE = dict(family="Arial, sans-serif", size=13, color=GAC_OSCURO)
ESCALA_VINO   = [[0, "#FBF2F3"], [0.5, "#DDA0A8"], [1, "#800020"]]

# Variables numericas de la base resumen (para correlaciones y multiple)
LISTA_NUM = ["Leads", "Efectivos", "Ventas Reportadas", "Obj Ventas",
             "Ventas Real", "Conversion", "SDC", "Autorizadas",
             "Formalizado", "Contado", "Credito", "PDM", "Obj PDM",
             "Variacion", "Lead / Venta"]

# =========================================================
# HELPERS - ETAPA III (DESPLIEGUE ANOVA)
# =========================================================
PALETA_ANOVA = [GAC_AZUL, GAC_PLATA, GAC_AZUL_MED, GAC_AZUL_CLARO]

def formato_p(p):
    if pd.isna(p):
        return "—"
    return f"{p:.1e}" if p < 0.001 else f"{p:.4f}"

def tabla_anova_bonita(tabla, nombres=None):
    nombres = nombres or {}
    fuentes = [nombres.get(str(i), str(i)) for i in tabla.index]
    return pd.DataFrame({
        "Fuente": fuentes,
        "Suma de cuadrados": tabla["sum_sq"].map(lambda v: f"{v:,.2f}"),
        "gl": tabla["df"].map(lambda v: f"{v:.0f}"),
        "F": tabla["F"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}"),
        "p-valor": tabla["PR(>F)"].map(formato_p)})

def fig_box(df, x, y, orden, titulo, ey):
    fig = px.box(df, x=x, y=y, category_orders={x: orden}, color=x,
                 color_discrete_sequence=PALETA_ANOVA, points="outliers")
    fig.update_layout(font=ESTILO_FUENTE, height=430, showlegend=False,
        title=dict(text=titulo, font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        xaxis=dict(title=None), yaxis=dict(title=ey, gridcolor=REJILLA))
    return fig

def fig_bar_ic95(df, x, y, orden, titulo, ey):
    res = df.groupby(x)[y].agg(["mean", "sem"]).reindex(orden).reset_index()
    fig = go.Figure(go.Bar(
        x=res[x], y=res["mean"],
        error_y=dict(type="data", array=1.96 * res["sem"], color=GAC_OSCURO),
        marker_color=PALETA_ANOVA[:len(res)],
        text=np.round(res["mean"], 1), textposition="outside"))
    fig.update_layout(font=ESTILO_FUENTE, height=430, showlegend=False,
        title=dict(text=titulo, font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        xaxis=dict(title=None),
        yaxis=dict(title=ey, gridcolor=REJILLA,
                   range=[0, res["mean"].max() * 1.25]))
    return fig

def fig_violin(df, x, y, orden, titulo, ey):
    fig = px.violin(df, x=x, y=y, category_orders={x: orden}, color=x,
                    box=True, color_discrete_sequence=PALETA_ANOVA)
    fig.update_layout(font=ESTILO_FUENTE, height=400, showlegend=False,
        title=dict(text=titulo, font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        xaxis=dict(title=None), yaxis=dict(title=ey, gridcolor=REJILLA))
    return fig

def fig_interaccion(df, x, linea, y, orden_x, titulo, ey):
    piv = (df.pivot_table(index=x, columns=linea, values=y,
                          aggfunc="mean", observed=False)
           .reindex(orden_x))
    fig = go.Figure()
    for i, col in enumerate(piv.columns):
        fig.add_trace(go.Scatter(
            x=piv.index.astype(str), y=piv[col],
            mode="lines+markers", name=str(col),
            line=dict(color=PALETA_ANOVA[i % len(PALETA_ANOVA)], width=3),
            marker=dict(size=10)))
    fig.update_layout(font=ESTILO_FUENTE, height=420,
        title=dict(text=titulo, font=dict(size=15, color=GAC_AZUL)),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        xaxis=dict(title=None), yaxis=dict(title=ey, gridcolor=REJILLA),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    return fig

# =========================================================
# ENCABEZADO
# =========================================================
st.markdown(
    "<h1 style='color:" + GAC_AZUL + ";'>🚗 Analisis GAC</h1>",
    unsafe_allow_html=True
)
st.caption("Etapa I · Analisis univariado | Etapa II · Modelado predictivo | "
           "Etapa III · Despliegue ANOVA | Plaza Angelopolis")
st.divider()

# =========================================================
# BARRA LATERAL
# =========================================================
if os.path.exists("logo_gac.png"):
    st.sidebar.image("logo_gac.png", use_container_width=True)
else:
    st.sidebar.markdown(
        "<h2 style='color:" + GAC_AZUL + "; text-align:center;'>GAC MOTOR</h2>",
        unsafe_allow_html=True
    )
st.sidebar.markdown("---")

vista = st.sidebar.selectbox(
    "Selecciona una vista:",
    ["Extraccion de Caracteristicas",
     "Solicitudes de Credito por Asesor",
     "Visitas por Mes",
     "Explorador de Variables",
     "Leads Detalle (Plaza)",
     "Analisis de Correlaciones",
     "Regresion Lineal Simple",
     "Regresion Lineal Multiple (Resumen)",
     "Regresion del Embudo (FUNNEL)",
     "Despliegue ANOVA"]
)

st.sidebar.markdown("---")
st.sidebar.write("📋 Bitacora de Piso: **%d** registros" % len(df_bitacora))
st.sidebar.write("📈 Leads Reales: **%d** meses" % len(df_leads))
if df_detalle is not None:
    st.sidebar.write("🏙️ Leads Plaza: **%d** registros" % len(df_detalle))
st.sidebar.write("📊 Base resumen: **%d** meses" % len(df_resumen))
st.sidebar.write("🔻 Embudo: **%d** meses" % len(df_funnel))
if df_bcs is not None:
    st.sidebar.write("🎯 Modelos BCS: **%d**" % len(df_bcs))
st.sidebar.markdown("---")
st.sidebar.caption("Etapa III · ANOVA")
if df_citas is not None:
    st.sidebar.write("📞 Citas Digital: **%d** registros" % len(df_citas))
if df_mkt is not None:
    st.sidebar.write("📣 Marketing: **%d** obs." % len(df_mkt))
if df_plazas_sem is not None:
    st.sidebar.write("🏙️ Plazas: **%d** semanas-prod." % len(df_plazas_sem))

# =========================================================
# VISTA 1: EXTRACCION DE CARACTERISTICAS (ETAPA I)
# =========================================================
if vista == "Extraccion de Caracteristicas":

    col1, col2 = st.columns(2)

    # ---------- GRAFICA 1: PARETO DE ASESOR ----------
    with col1:
        st.subheader("1. Visitas por Asesor (Pareto)")

        tabla_asesor = df_bitacora["Asesor"].value_counts().reset_index()
        tabla_asesor.columns = ["categorias", "frecuencia"]
        tabla_asesor["porcentaje"] = (
            tabla_asesor["frecuencia"] / tabla_asesor["frecuencia"].sum() * 100
        ).round(1)
        tabla_asesor["porcentaje_acumulado"] = tabla_asesor["porcentaje"].cumsum().round(1)

        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            x=tabla_asesor["categorias"],
            y=tabla_asesor["frecuencia"],
            name="Visitas",
            marker_color=GAC_AZUL,
            text=tabla_asesor["frecuencia"],
            textposition="outside"
        ))
        fig1.add_trace(go.Scatter(
            x=tabla_asesor["categorias"],
            y=tabla_asesor["porcentaje_acumulado"],
            name="% acumulado",
            yaxis="y2",
            mode="lines+markers+text",
            text=tabla_asesor["porcentaje_acumulado"].astype(str) + "%",
            textposition="top center",
            line=dict(color=GAC_PLATA, width=3),
            marker=dict(size=9, color=GAC_PLATA)
        ))
        fig1.update_layout(
            font=ESTILO_FUENTE,
            title=dict(text="Pareto de visitas atendidas por asesor",
                       font=dict(size=16, color=GAC_AZUL)),
            xaxis=dict(title="Asesor", tickangle=-30),
            yaxis=dict(title="Numero de visitas", gridcolor=REJILLA),
            yaxis2=dict(overlaying="y", side="right",
                        range=[0, 110], title="% acumulado",
                        gridcolor=GAC_BLANCO),
            height=480,
            plot_bgcolor=GAC_BLANCO,
            paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1)
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("Aurelio, Pedro y Mauricio concentran la mayor carga de atencion.")

    # ---------- GRAFICA 2: DONA DE PRUEBA DE MANEJO ----------
    with col2:
        st.subheader("2. Visitas con y sin Prueba de Manejo")

        tabla_pdm = df_bitacora["Prueba_Manejo"].value_counts().reset_index()
        tabla_pdm.columns = ["categorias", "frecuencia"]

        fig2 = px.pie(
            data_frame=tabla_pdm,
            names="categorias",
            values="frecuencia",
            hole=0.5,
            color="categorias",
            color_discrete_map={"Sí": GAC_AZUL, "No": GAC_AZUL_CLARO}
        )
        fig2.update_traces(
            textinfo="percent+label+value",
            textfont=dict(size=14),
            marker=dict(line=dict(color=GAC_BLANCO, width=3))
        )
        fig2.update_layout(
            font=ESTILO_FUENTE,
            title=dict(text="Proporcion de pruebas de manejo (PDM)",
                       font=dict(size=16, color=GAC_AZUL)),
            height=480,
            paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5)
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("La mayoria de las visitas se va sin realizar prueba de manejo.")

    # ---------- GRAFICA 3: EFECTIVIDAD DE LEADS POR MES ----------
    st.subheader("3. Efectividad de Leads por Mes")

    fig3 = px.bar(
        data_frame=df_leads,
        x="Periodo",
        y="Efectividad_Leads",
        color="Nivel_Efectividad",
        color_discrete_map={"Alta": GAC_OSCURO, "Media": GAC_AZUL, "Baja": GAC_AZUL_CLARO},
        category_orders={"Nivel_Efectividad": ["Baja", "Media", "Alta"]},
        text="Efectividad_Leads"
    )
    fig3.update_traces(textposition="outside", textfont=dict(size=11))
    fig3.add_hline(y=55, line_dash="dash", line_color=GAC_PLATA,
                   annotation_text="55%", annotation_position="top left")
    fig3.add_hline(y=70, line_dash="dash", line_color=GAC_PLATA,
                   annotation_text="70%", annotation_position="top left")
    fig3.update_layout(
        font=ESTILO_FUENTE,
        title=dict(text="Efectividad de leads por periodo (corte: 55% y 70%)",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title="Periodo"),
        yaxis=dict(title="Efectividad (%)", range=[0, 95], gridcolor=REJILLA),
        height=520,
        plot_bgcolor=GAC_BLANCO,
        paper_bgcolor=GAC_BLANCO,
        bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("14 de 28 meses se ubican en nivel Media; feb-2025 fue el mejor mes (83%).")

    # ---------- TABLA RESUMEN ----------
    st.subheader("Tabla de frecuencias - Asesor")
    st.dataframe(
        tabla_asesor.style.format({"porcentaje": "{:.1f}%",
                                   "porcentaje_acumulado": "{:.1f}%"}),
        use_container_width=True
    )

# =========================================================
# VISTA 2: SOLICITUDES DE CREDITO POR ASESOR (SDC)
# =========================================================
elif vista == "Solicitudes de Credito por Asesor":

    st.subheader("1. Solicitudes de Credito (SDC) - Top 5 asesores")

    top_asesores = (
        df_bitacora.loc[df_bitacora["Asesor"] != "Otros", "Asesor"]
        .value_counts()
        .head(5)
        .index
    )

    sdc_asesor = df_bitacora[df_bitacora["Asesor"].isin(top_asesores)].copy()
    sdc_asesor["SDC"] = sdc_asesor["SDC"].map({1: "Si solicito", 0: "No solicito"})

    sdc_por_asesor = (
        sdc_asesor.groupby(["Asesor", "SDC"])
        .size()
        .reset_index(name="Clientes")
    )

    fig_sdc = px.bar(
        sdc_por_asesor,
        x="Asesor",
        y="Clientes",
        color="SDC",
        barmode="group",
        text="Clientes",
        color_discrete_map={"Si solicito": GAC_AZUL, "No solicito": GAC_AZUL_CLARO}
    )
    fig_sdc.update_traces(textposition="outside", textfont=dict(size=13))
    fig_sdc.update_layout(
        font=ESTILO_FUENTE,
        title=dict(text="Solicitudes de credito de los 5 asesores principales",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title="Asesor"),
        yaxis=dict(title="Numero de clientes", gridcolor=REJILLA),
        height=480,
        plot_bgcolor=GAC_BLANCO,
        paper_bgcolor=GAC_BLANCO,
        bargap=0.3,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig_sdc, use_container_width=True)

    st.subheader("Detalle")
    st.dataframe(sdc_por_asesor, use_container_width=True)

# =========================================================
# VISTA 3: VISITAS POR MES
# =========================================================
elif vista == "Visitas por Mes":

    st.subheader("1. Visitas Registradas por Mes")

    visitas_mes = (
        df_bitacora.groupby(["Mes_Num", "Mes"], as_index=False)
        .size()
        .rename(columns={"size": "Visitas"})
        .sort_values("Mes_Num")
    )

    fig_mes = px.bar(
        visitas_mes,
        x="Mes",
        y="Visitas",
        text="Visitas",
        color_discrete_sequence=[GAC_AZUL]
    )
    fig_mes.update_traces(textposition="outside", textfont=dict(size=13),
                          marker_line_color=GAC_AZUL, marker_line_width=1)
    fig_mes.update_layout(
        font=ESTILO_FUENTE,
        title=dict(text="Visitas registradas en la bitacora de piso por mes",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title="Mes", categoryorder="array",
                   categoryarray=visitas_mes["Mes"]),
        yaxis=dict(title="Numero de visitas", gridcolor=REJILLA),
        height=480,
        plot_bgcolor=GAC_BLANCO,
        paper_bgcolor=GAC_BLANCO,
        bargap=0.35,
        showlegend=False
    )
    st.plotly_chart(fig_mes, use_container_width=True)

    st.subheader("Detalle")
    st.dataframe(visitas_mes[["Mes", "Visitas"]], use_container_width=True)

# =========================================================
# VISTA 4: EXPLORADOR DE VARIABLES (ETAPA I)
# =========================================================
elif vista == "Explorador de Variables":

    st.subheader("Explorador de variables por frecuencia")

    # ---------- SELECTORES ----------
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        Base = st.selectbox("Base de datos",
                            ["Bitacora de piso", "Base de leads"])
    with col_sel2:
        if Base == "Bitacora de piso":
            df = df_bitacora.copy()
            Lista = ["Asesor", "Estatus_Lead", "Prueba_Manejo", "PDM",
                     "SDC", "Venta", "Inter_Gerente", "Mes"]
        else:
            df = df_leads.copy()
            Lista = ["Nivel_Efectividad", "Año", "Mes", "Mes_Num"]

        Variable_Cat = st.selectbox("Variable categorica", Lista)

    # ---------- FILTRO DE PERIODO ----------
    Periodos = sorted(df["Periodo"].dropna().unique())
    Periodos_Seleccionados = st.multiselect("Periodo", Periodos,
                                            default=Periodos)
    df = df[df["Periodo"].isin(Periodos_Seleccionados)]

    if df.empty:
        st.warning("Selecciona al menos un periodo para mostrar las graficas.")
        st.stop()

    # ---------- METRICAS ----------
    Indicador_A, Indicador_B, Indicador_C = st.columns(3)

    if Base == "Bitacora de piso":
        Indicador_A.metric("Registros", len(df))
        Indicador_B.metric("Ventas", int(df["Venta"].sum()))
        Indicador_C.metric("Tasa de venta", f"{df['Venta'].mean():.1%}")

        Tabla_Tiempo = (df.groupby("Fecha").size()
                        .reset_index(name="frecuencia"))
        Titulo_Area = "Registros por dia"
        Etiqueta_Y = "Registros"
    else:
        Indicador_A.metric("Meses", len(df))
        Indicador_B.metric("Ventas reportadas", f"{df['Ventas'].sum():,.0f}")
        Indicador_C.metric("Leads totales", f"{df['Total'].sum():,.2f}")

        Tabla_Tiempo = (df.groupby("Fecha", as_index=False)["Ventas"]
                        .sum().rename(columns={"Ventas": "frecuencia"}))
        Titulo_Area = "Ventas por mes"
        Etiqueta_Y = "Ventas"

    # ---------- TABLA DE FRECUENCIAS ----------
    Tabla_frecuencias = (
        df[Variable_Cat].fillna("Sin dato").astype(str)
        .value_counts().reset_index()
    )
    Tabla_frecuencias.columns = ["categorias", "frecuencia"]

    # ---------- FILA 1: BARRAS Y PASTEL ----------
    Contenedor_A, Contenedor_B = st.columns(2)

    with Contenedor_A:
        st.write("Grafico de Barras")
        figure1 = px.bar(
            data_frame=Tabla_frecuencias,
            x="categorias", y="frecuencia",
            title=f"Frecuencia de {Variable_Cat}",
            text="frecuencia",
            color_discrete_sequence=[GAC_AZUL]
        )
        figure1.update_traces(textposition="outside")
        figure1.update_layout(
            font=ESTILO_FUENTE, height=350,
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            xaxis=dict(tickangle=-30), showlegend=False,
            yaxis=dict(gridcolor=REJILLA)
        )
        st.plotly_chart(figure1, use_container_width=True)

    with Contenedor_B:
        st.write("Grafico de Pastel")
        figure2 = px.pie(
            data_frame=Tabla_frecuencias,
            names="categorias", values="frecuencia",
            title=f"Distribucion de {Variable_Cat}",
            color_discrete_sequence=[GAC_AZUL, GAC_AZUL_MED,
                                     GAC_AZUL_CLARO, GAC_PLATA,
                                     GAC_OSCURO]
        )
        figure2.update_traces(textinfo="percent+label")
        figure2.update_layout(font=ESTILO_FUENTE, height=350,
                              paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(figure2, use_container_width=True)

    # ---------- FILA 2: DONA Y AREA ----------
    Contenedor_C, Contenedor_D = st.columns(2)

    with Contenedor_C:
        st.write("Grafico de Anillo o Dona")
        figure3 = px.pie(
            data_frame=Tabla_frecuencias,
            names="categorias", values="frecuencia",
            hole=0.4,
            title=f"Participacion de {Variable_Cat}",
            color_discrete_sequence=[GAC_AZUL, GAC_PLATA,
                                     GAC_AZUL_MED, GAC_AZUL_CLARO,
                                     GAC_OSCURO]
        )
        figure3.update_traces(textinfo="percent+label")
        figure3.update_layout(font=ESTILO_FUENTE, height=350,
                              paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(figure3, use_container_width=True)

    with Contenedor_D:
        st.write("Grafico de Area")
        figure4 = px.area(
            data_frame=Tabla_Tiempo,
            x="Fecha", y="frecuencia",
            title=Titulo_Area,
            color_discrete_sequence=[GAC_AZUL]
        )
        figure4.update_layout(
            font=ESTILO_FUENTE, height=350,
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            xaxis_title="Fecha", yaxis_title=Etiqueta_Y,
            yaxis=dict(gridcolor=REJILLA)
        )
        st.plotly_chart(figure4, use_container_width=True)

    # ---------- TABLAS EN EXPANDER ----------
    with st.expander("Ver tabla de frecuencias y datos filtrados"):
        st.write("Tabla de frecuencias")
        st.dataframe(Tabla_frecuencias, hide_index=True,
                     use_container_width=True)
        st.write("Datos filtrados")
        st.dataframe(df, hide_index=True, use_container_width=True)

# =========================================================
# VISTA 5: LEADS DETALLE (PLAZA) (ETAPA I)
# =========================================================
elif vista == "Leads Detalle (Plaza)":

    if df_detalle is None:
        st.error("No se encontro el archivo 'base_leads_limpia.csv' "
                 "en la carpeta del proyecto.")
        st.stop()

    st.subheader("Leads captados en Plaza Angelopolis")

    # ---------- KPIs ----------
    total = len(df_detalle)
    n_ventas = int((df_detalle["Estado"] == "Ventas").sum())
    n_contacto = int((df_detalle["Estado"] == "Contactado").sum())
    tasa_cierre = n_ventas / total if total else 0

    K1, K2, K3, K4 = st.columns(4)
    K1.metric("Leads captados", total)
    K2.metric("Contactados", n_contacto)
    K3.metric("Ventas concretadas", n_ventas)
    K4.metric("Tasa de cierre", f"{tasa_cierre:.1%}")

    # ---------- FILA 1: HEATMAP DE PRODUCTO + PASTEL DE ESTADO ----------
    Cont_A, Cont_B = st.columns(2)

    with Cont_A:
        st.write("Heatmap")
        Tabla_producto = df_detalle["Producto"].value_counts().reset_index()
        Tabla_producto.columns = ["producto", "frecuencia"]
        Matriz = Tabla_producto.set_index("producto")[["frecuencia"]].T
        fig_hm = px.imshow(
            Matriz, text_auto=True, aspect="auto",
            title="Frecuencia por producto",
            color_continuous_scale=ESCALA_VINO
        )
        fig_hm.update_layout(
            font=ESTILO_FUENTE, height=350,
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    with Cont_B:
        st.write("Grafico de Pastel")
        Tabla_estatus = df_detalle["Estado"].value_counts().reset_index()
        Tabla_estatus.columns = ["estatus", "frecuencia"]
        fig_pie = px.pie(
            data_frame=Tabla_estatus, names="estatus", values="frecuencia",
            title="Estatus de los leads", color="estatus",
            color_discrete_map={"Ventas": GAC_AZUL, "Contactado": GAC_AZUL_MED,
                                "Finalizado": GAC_AZUL_CLARO, "Sin dato": GAC_PLATA}
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(font=ESTILO_FUENTE, height=350,
                              paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---------- FILA 2: AREA DE ESTATUS LEAD (bitacora) ----------
    st.write("Grafico de Area")
    Tabla_area = df_bitacora["Estatus_Lead"].value_counts().reset_index()
    Tabla_area.columns = ["estatus", "frecuencia"]
    fig_area = px.area(
        data_frame=Tabla_area, x="estatus", y="frecuencia",
        title="Frecuencia por estatus de lead (bitacora de piso)",
        color_discrete_sequence=[GAC_AZUL]
    )
    fig_area.update_traces(mode="lines")
    fig_area.update_layout(
        font=ESTILO_FUENTE, height=380,
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        xaxis_title="Estatus", yaxis_title="Frecuencia",
        yaxis=dict(gridcolor=REJILLA)
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # ---------- SELECTOR CONECTADO ----------
    st.subheader("Frecuencia por variable")
    Variable_Det = st.selectbox("Variable",
                                ["Producto", "Temperatura", "Estado", "Asesor"])
    Tabla_det = (df_detalle[Variable_Det].fillna("Sin dato").astype(str)
                 .value_counts().reset_index())
    Tabla_det.columns = ["categorias", "frecuencia"]

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_bar = px.bar(
            data_frame=Tabla_det, x="categorias", y="frecuencia",
            title=f"Frecuencia de {Variable_Det}",
            text="frecuencia",
            color_discrete_sequence=[GAC_AZUL]
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            font=ESTILO_FUENTE, height=380,
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            xaxis=dict(tickangle=-30), showlegend=False,
            yaxis=dict(gridcolor=REJILLA)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_b2:
        fig_dona = px.pie(
            data_frame=Tabla_det, names="categorias", values="frecuencia",
            hole=0.4,
            title=f"Participacion de {Variable_Det}",
            color_discrete_sequence=[GAC_AZUL, GAC_AZUL_MED, GAC_AZUL_CLARO,
                                     GAC_PLATA, GAC_OSCURO]
        )
        fig_dona.update_traces(textinfo="percent+label")
        fig_dona.update_layout(font=ESTILO_FUENTE, height=380,
                               paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(fig_dona, use_container_width=True)

    # ---------- TABLAS ----------
    with st.expander("Ver tablas de frecuencias"):
        st.write("Producto")
        st.dataframe(Tabla_producto, hide_index=True, use_container_width=True)
        st.write("Estatus de leads")
        st.dataframe(Tabla_estatus, hide_index=True, use_container_width=True)
        st.write(f"Frecuencia de {Variable_Det}")
        st.dataframe(Tabla_det, hide_index=True, use_container_width=True)

# =========================================================
# VISTA 6: ANALISIS DE CORRELACIONES (ETAPA II)
# =========================================================
elif vista == "Analisis de Correlaciones":

    st.subheader("Matriz de correlaciones entre variables numericas")

    sel_corr = st.multiselect("Variables a incluir en el heatmap",
                              LISTA_NUM,
                              default=["Leads", "Efectivos", "SDC",
                                       "Autorizadas", "PDM", "Ventas Real"])
    if len(sel_corr) < 2:
        st.warning("Selecciona al menos 2 variables.")
        st.stop()

    matriz = df_resumen[sel_corr].corr().round(2)

    fig_corr = px.imshow(matriz, text_auto=True, aspect="auto",
                         title="Matriz de correlacion (Pearson)",
                         color_continuous_scale=ESCALA_VINO,
                         zmin=-1, zmax=1)
    fig_corr.update_layout(font=ESTILO_FUENTE, height=560,
                           plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Correlacion de cada variable con Ventas Real")
    tabla_corr = (df_resumen[LISTA_NUM].corr()["Ventas Real"]
                  .drop("Ventas Real").sort_values(ascending=False)
                  .reset_index())
    tabla_corr.columns = ["Variable", "Correlacion con Ventas Real"]
    fig_bar = px.bar(tabla_corr, x="Correlacion con Ventas Real", y="Variable",
                     orientation="h", text="Correlacion con Ventas Real",
                     color_discrete_sequence=[GAC_AZUL])
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(font=ESTILO_FUENTE, height=520,
                          plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
                          xaxis=dict(range=[-1, 1], gridcolor=REJILLA))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info("Las variables con mayor correlacion positiva con las ventas "
            "(Autorizadas, Formalizado, SDC) son las mejores candidatas "
            "para el modelo de regresion.")

# =========================================================
# VISTA 7: REGRESION LINEAL SIMPLE (UNIFICADA)
# Base "BCS - 6 modelos"      -> regresion_lineal_simple.ipynb
# Base "Base resumen mensual" -> Regresion_Simple_Resumen.ipynb
# =========================================================
elif vista == "Regresion Lineal Simple":

    st.subheader("Regresion lineal simple")
    st.caption("Solo las bases de las regresiones simples del equipo · "
               "defaults = variables originales de cada notebook")

    # ---------- SELECTOR DE BASE ----------
    Base_sel = st.selectbox(
        "Base de datos",
        ["BCS - 6 modelos (Balance Scorecard)",
         "Base resumen mensual (30 meses)"])

    # Variables disponibles segun la base elegida
    if Base_sel.startswith("BCS"):
        if df_bcs is None:
            st.error("No se encontro el archivo 'bcs_modelos.csv' "
                     "en la carpeta del proyecto.")
            st.stop()
        df_reg = df_bcs
        opts_x = ["Objetivo"]
        opts_y = ["Ventas_Reales"]
        default_y = "Ventas_Reales"
    else:
        df_reg = df_resumen
        opts_x = LISTA_NUM
        opts_y = LISTA_NUM
        default_y = "Ventas Real"

    es_bcs = Base_sel.startswith("BCS")

    # ---------- SELECTORES X / Y (formato separado) ----------
    col_x, col_y = st.columns(2)
    with col_x:
        Variable_x = st.selectbox("Variable X (independiente)", opts_x,
                                  index=0, key="x_simple")
    with col_y:
        Variable_y = st.selectbox("Variable Y (dependiente)", opts_y,
                                  index=opts_y.index(default_y),
                                  key="y_simple")

    if Variable_x == Variable_y:
        st.warning("Las variables X e Y deben ser diferentes.")
        st.stop()

    # ---------- MODELO ----------
    model = LinearRegression()
    model.fit(X=df_reg[[Variable_x]], y=df_reg[Variable_y])
    y_pred = model.predict(df_reg[[Variable_x]])

    intercepto = model.intercept_
    pendiente = model.coef_[0]
    r2 = model.score(df_reg[[Variable_x]], df_reg[Variable_y])
    r = df_reg[Variable_x].corr(df_reg[Variable_y])

    M1, M2, M3, M4 = st.columns(4)
    M1.metric("Intercepto (b0)", f"{intercepto:.4f}")
    M2.metric("Pendiente (b1)", f"{pendiente:.4f}")
    M3.metric("R²", f"{r2:.4f}")
    M4.metric("Correlacion (r)", f"{r:.4f}")
    st.markdown(
        f"**Ecuacion:** `{Variable_y} = {intercepto:.4f} "
        f"+ ({pendiente:.4f}) × {Variable_x}`"
    )

    # ---------- GRAFICA ----------
    fig = go.Figure()
    if es_bcs:
        fig.add_trace(go.Scatter(
            x=df_reg[Variable_x], y=df_reg[Variable_y],
            mode="markers+text", text=df_reg["Modelo"],
            textposition="top center",
            name="Ventas reales",
            marker=dict(color=GRIS_DATOS, size=13)
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df_reg[Variable_x], y=df_reg[Variable_y],
            mode="markers", name="Datos reales",
            marker=dict(color=GRIS_DATOS, size=11)
        ))
    orden = df_reg[Variable_x].argsort()
    fig.add_trace(go.Scatter(
        x=df_reg[Variable_x].iloc[orden],
        y=pd.Series(y_pred).iloc[orden],
        mode="lines", name="Recta de regresion",
        line=dict(color=GAC_AZUL, width=3)
    ))
    titulo_graf = ("Regresion lineal del objetivo y las ventas reales por modelo"
                   if es_bcs else f"{Variable_y} ~ {Variable_x}")
    fig.update_layout(
        font=ESTILO_FUENTE, height=480,
        title=dict(text=titulo_graf, font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title=Variable_x),
        yaxis=dict(title=Variable_y, gridcolor=REJILLA),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---------- TABLA ----------
    st.subheader("Valores reales, predichos y residuales")
    if es_bcs:
        tabla = pd.DataFrame({
            "Modelo": df_reg["Modelo"],
            Variable_x: df_reg[Variable_x],
            "Real": df_reg[Variable_y].round(2),
            "Predicho": np.round(y_pred, 2)})
    else:
        tabla = pd.DataFrame({
            "Año": df_reg["Año"], "Mes": df_reg["Mes"],
            "Real": df_reg[Variable_y].round(2),
            "Predicho": np.round(y_pred, 2)})
    tabla["Residual"] = (tabla["Real"] - tabla["Predicho"]).round(2)
    st.dataframe(tabla, hide_index=True, use_container_width=True)

# =========================================================
# VISTA 9: REGRESION LINEAL MULTIPLE (RESUMEN)
# Replicado de: codigos (1).ipynb (vista "Regresión Lineal")
# =========================================================
elif vista == "Regresion Lineal Multiple (Resumen)":

    st.subheader("Regresion lineal multiple")
    st.caption("Fuente: base_resumen_limpio.csv · notebook: codigos (1).ipynb")

    col_y2, _ = st.columns(2)
    with col_y2:
        Variable_y = st.selectbox("Variable objetivo (Y)", LISTA_NUM,
                                  index=LISTA_NUM.index("Ventas Real"),
                                  key="y_multi")
    Variables_x = st.multiselect(
        "Variables independientes del modelo multiple (X)",
        [v for v in LISTA_NUM if v != Variable_y],
        default=["Efectivos", "PDM", "SDC"])

    if len(Variables_x) == 0:
        st.warning("Selecciona al menos una variable independiente.")
        st.stop()

    model = LinearRegression()
    model.fit(X=df_resumen[Variables_x], y=df_resumen[Variable_y])
    y_pred = model.predict(df_resumen[Variables_x])

    r2 = model.score(df_resumen[Variables_x], df_resumen[Variable_y])
    r = np.sqrt(r2)
    r2_adj = 1 - (1 - r2) * (len(df_resumen) - 1) / \
        (len(df_resumen) - len(Variables_x) - 1)

    ecuacion = f"{Variable_y} = {model.intercept_:.4f}"
    for nombre, coef in zip(Variables_x, model.coef_):
        signo = "+" if coef >= 0 else "-"
        ecuacion += f" {signo} {abs(coef):.4f}·({nombre})"

    M1, M2, M3 = st.columns(3)
    M1.metric("R² (determinacion)", f"{r2:.4f}")
    M2.metric("R (correlacion multiple)", f"{r:.4f}")
    M3.metric("R² ajustado", f"{r2_adj:.4f}")
    st.markdown(f"**Ecuacion del modelo:** `{ecuacion}`")

    C1, C2 = st.columns(2)

    with C1:
        st.write("Analisis de Correlaciones")
        matriz = df_resumen[Variables_x + [Variable_y]].corr().round(2)
        fig_hm = px.imshow(matriz, text_auto=True, aspect="auto",
                           title="Matriz de Correlacion",
                           color_continuous_scale=ESCALA_VINO,
                           zmin=-1, zmax=1)
        fig_hm.update_layout(font=ESTILO_FUENTE, height=420,
                             plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(fig_hm, use_container_width=True)

    with C2:
        st.write("Correlacion Lineal Multiple")
        minimo = min(df_resumen[Variable_y].min(), y_pred.min())
        maximo = max(df_resumen[Variable_y].max(), y_pred.max())
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_resumen[Variable_y], y=y_pred, mode="markers",
            name="Meses", marker=dict(color=GRIS_DATOS, size=11)
        ))
        fig_sc.add_trace(go.Scatter(
            x=[minimo, maximo], y=[minimo, maximo], mode="lines",
            name="Linea ideal (y = x)",
            line=dict(color=GAC_AZUL, width=2, dash="dash")
        ))
        fig_sc.update_layout(
            font=ESTILO_FUENTE, height=420,
            title=dict(text="Modelo Lineal Multiple",
                       font=dict(size=16, color=GAC_AZUL)),
            xaxis=dict(title="Reales"),
            yaxis=dict(title="Predichos", gridcolor=REJILLA),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                        xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.subheader("Coeficientes del modelo")
    tabla_coef = pd.DataFrame({
        "Variable": Variables_x,
        "Coeficiente": np.round(model.coef_, 4)})
    tabla_coef.loc[len(tabla_coef)] = ["Intercepto", round(model.intercept_, 4)]
    st.dataframe(tabla_coef, hide_index=True, use_container_width=True)

    with st.expander("Ver tabla de valores reales vs predichos"):
        tabla = pd.DataFrame({
            "Año": df_resumen["Año"], "Mes": df_resumen["Mes"],
            "Real": df_resumen[Variable_y].round(2),
            "Predicho": np.round(y_pred, 2)})
        tabla["Residual"] = (tabla["Real"] - tabla["Predicho"]).round(2)
        st.dataframe(tabla, hide_index=True, use_container_width=True)

# =========================================================
# VISTA 10: REGRESION DEL EMBUDO (FUNNEL)
# Replicado de: FUNNEL_REGRESION.IPYNB + selectores interactivos
# =========================================================
elif vista == "Regresion del Embudo (FUNNEL)":

    st.subheader("Regresiones sobre el embudo de ventas")
    st.caption("Fuente: FUNNEL.csv (18 meses, GAC Angelopolis) · "
               "notebook: FUNNEL_REGRESION.IPYNB · defaults = variables originales")

    # ---------- SELECTORES (misma funcion que la vista multiple) ----------
    LISTA_FUNNEL = ["Ventas", "Leads_Generados", "Leads_Efectivos",
                    "Cita_Programada", "Cita_Efectiva", "Prueba_Manejo",
                    "SDC_Generada", "SDC_Aprobada", "Formalizado", "Mes_Num"]
    col_yf, _ = st.columns(2)
    with col_yf:
        Variable_y = st.selectbox("Variable objetivo (Y)", LISTA_FUNNEL,
                                  index=LISTA_FUNNEL.index("Ventas"),
                                  key="y_funnel")
    opciones_x = [v for v in LISTA_FUNNEL if v != Variable_y]
    vars_m1 = st.multiselect(
        "Regresion 1 · Variables independientes (X)",
        opciones_x,
        default=[v for v in ["Cita_Efectiva", "Prueba_Manejo", "SDC_Aprobada",
                             "Formalizado"] if v in opciones_x],
        key="x1_funnel")
    vars_m2 = st.multiselect(
        "Regresion 2 · Variables independientes (X)",
        opciones_x,
        default=[v for v in ["Mes_Num", "SDC_Aprobada"] if v in opciones_x],
        key="x2_funnel")

    if len(vars_m1) == 0 or len(vars_m2) == 0:
        st.warning("Cada modelo necesita al menos una variable X.")
        st.stop()

    y_real = df_funnel[Variable_y]

    # ---------- REGRESION 1 ----------
    st.markdown(f"**Regresion 1 · {Variable_y} ~ variables seleccionadas**")
    m1 = LinearRegression().fit(df_funnel[vars_m1], y_real)
    pred1 = m1.predict(df_funnel[vars_m1])
    r2_m1 = m1.score(df_funnel[vars_m1], y_real)

    c1, c2 = st.columns(2)
    c1.metric("R²", f"{r2_m1:.4f}")
    c2.metric("Correlacion multiple", f"{np.sqrt(r2_m1):.4f}")

    tabla_m1 = pd.DataFrame({"Variable": vars_m1,
                             "Coeficiente": np.round(m1.coef_, 4)})
    tabla_m1.loc[len(tabla_m1)] = ["Intercepto", round(m1.intercept_, 4)]
    st.dataframe(tabla_m1, hide_index=True, use_container_width=True)
    mayor = tabla_m1.iloc[np.argmax(np.abs(m1.coef_))]["Variable"]
    st.caption(f"Variable con mayor peso en el Modelo 1: {mayor}")

    # ---------- REGRESION 2 ----------
    st.markdown(f"**Regresion 2 · {Variable_y} ~ variables seleccionadas**")
    m2 = LinearRegression().fit(df_funnel[vars_m2], y_real)
    pred2 = m2.predict(df_funnel[vars_m2])
    r2_m2 = m2.score(df_funnel[vars_m2], y_real)

    c3, c4 = st.columns(2)
    c3.metric("R²", f"{r2_m2:.4f}")
    c4.metric("Correlacion multiple", f"{np.sqrt(r2_m2):.4f}")

    # ---------- DISPERSION: REALES VS PREDICHAS (ambos modelos) ----------
    st.subheader(f"{Variable_y}: reales vs predichas")
    minimo = min(y_real.min(), pred1.min(), pred2.min())
    maximo = max(y_real.max(), pred1.max(), pred2.max())

    D1, D2 = st.columns(2)
    with D1:
        fig_d1 = go.Figure()
        fig_d1.add_trace(go.Scatter(
            x=df_funnel["Ventas"], y=np.round(pred1, 1), mode="markers",
            name="Meses", marker=dict(color=GRIS_DATOS, size=11)
        ))
        fig_d1.add_trace(go.Scatter(
            x=[minimo, maximo], y=[minimo, maximo], mode="lines",
            name="Linea ideal (y = x)",
            line=dict(color=GAC_AZUL, width=2, dash="dash")
        ))
        fig_d1.update_layout(
            font=ESTILO_FUENTE, height=400,
            title=dict(text=f"Regresion 1 (R² = {r2_m1:.2f})",
                       font=dict(size=15, color=GAC_AZUL)),
            xaxis=dict(title=f"{Variable_y} reales"),
            yaxis=dict(title=f"{Variable_y} predichas", gridcolor=REJILLA),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                        xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_d1, use_container_width=True)
    with D2:
        fig_d2 = go.Figure()
        fig_d2.add_trace(go.Scatter(
            x=df_funnel["Ventas"], y=np.round(pred2, 1), mode="markers",
            name="Meses", marker=dict(color=GRIS_DATOS, size=11)
        ))
        fig_d2.add_trace(go.Scatter(
            x=[minimo, maximo], y=[minimo, maximo], mode="lines",
            name="Linea ideal (y = x)",
            line=dict(color=GAC_AZUL, width=2, dash="dash")
        ))
        fig_d2.update_layout(
            font=ESTILO_FUENTE, height=400,
            title=dict(text=f"Regresion 2 (R² = {r2_m2:.2f})",
                       font=dict(size=15, color=GAC_AZUL)),
            xaxis=dict(title=f"{Variable_y} reales"),
            yaxis=dict(title=f"{Variable_y} predichas", gridcolor=REJILLA),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                        xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_d2, use_container_width=True)

    # ---------- SERIE TEMPORAL ----------
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=y_real, mode="lines+markers",
        name=f"{Variable_y} reales", line=dict(color=GRIS_DATOS, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=np.round(pred1, 1), mode="lines+markers",
        name=f"Regresion 1 (R²={r2_m1:.2f})",
        line=dict(color=GAC_AZUL, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=np.round(pred2, 1), mode="lines+markers",
        name=f"Regresion 2 (R²={r2_m2:.2f})",
        line=dict(color=GAC_AZUL_CLARO, width=2.5, dash="dash")
    ))
    fig.update_layout(
        font=ESTILO_FUENTE, height=480,
        title=dict(text="Ventas reales vs predichas por mes",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title="Mes", tickangle=-45),
        yaxis=dict(title="Ventas", gridcolor=REJILLA),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabla real vs predicho (Regresion 2)"):
        tabla = pd.DataFrame({
            "Mes": df_funnel["Mes"],
            f"{Variable_y} reales": y_real,
            f"{Variable_y} predichas": np.round(pred2, 2)})
        tabla["Residual"] = (tabla[f"{Variable_y} reales"]
                             - tabla[f"{Variable_y} predichas"]).round(2)
        st.dataframe(tabla, hide_index=True, use_container_width=True)

# =========================================================
# VISTA 10: DESPLIEGUE ANOVA (ETAPA III)
# Metodologia de la plantilla 7_ANOVA.ipynb:
# scipy.stats.f_oneway + statsmodels anova_lm · alfa = 0.05
# =========================================================
elif vista == "Despliegue ANOVA":

    st.subheader("Despliegue ANOVA - analisis de varianza")
    st.caption("Un ANOVA one-way y uno multifactorial por base · "
               "H0: las medias de los grupos son iguales · α = 0.05")

    tab_citas, tab_mkt, tab_plazas = st.tabs(
        ["📞 Citas Digital", "📣 Marketing por canal", "🏙️ Plazas"])

    # -----------------------------------------------------
    # TAB 1: CITAS DIGITAL (Potencial de compra)
    # -----------------------------------------------------
    with tab_citas:
        if df_citas is None:
            st.error("No se encontro el archivo 'Citas_Digital.csv' "
                     "en la carpeta del proyecto.")
        else:
            ORDEN_BDC = ["Ana", "Antonio", "Ignacio", "Saul"]
            st.markdown("#### Potencial de compra (%) segun agente BDC")
            st.caption("668 registros crudos → 531 tras la limpieza: 57 nulos "
                       "en la respuesta, unificacion de typos (estatus y BDC) "
                       "y recorte de 67 outliers por IQR (rango 55–95%).")

            # --- ANOVA one-way (scipy + statsmodels, como en clase) ---
            grupos_c = [g["Potencial"].values
                        for _, g in df_citas.groupby("BDC_L")]
            anova_c = stats.f_oneway(*grupos_c)
            mod_c = ols("Potencial ~ BDC_L", data=df_citas).fit()
            tab_c = sm.stats.anova_lm(mod_c, typ=1)
            eta2_c = tab_c["sum_sq"]["BDC_L"] / tab_c["sum_sq"].sum()
            lev_c = stats.levene(*grupos_c)

            st.markdown("**H₀:** el potencial promedio es igual entre los 4 "
                        "agentes · **H₁:** al menos un promedio difiere")
            K1, K2, K3, K4 = st.columns(4)
            K1.metric("F (one-way)", f"{anova_c.statistic:.2f}")
            K2.metric("p-valor", f"{anova_c.pvalue:.1e}")
            K3.metric("η² (tamano del efecto)", f"{eta2_c:.3f}")
            K4.metric("Levene (p)", f"{lev_c.pvalue:.2f}")

            C1, C2 = st.columns(2)
            with C1:
                st.plotly_chart(fig_box(df_citas, "BDC_L", "Potencial",
                    ORDEN_BDC, "Boxplot del potencial por agente",
                    "Potencial (%)"), use_container_width=True)
            with C2:
                st.plotly_chart(fig_bar_ic95(df_citas, "BDC_L", "Potencial",
                    ORDEN_BDC, "Media del potencial con IC 95%",
                    "Potencial (%)"), use_container_width=True)
            st.plotly_chart(fig_violin(df_citas, "BDC_L", "Potencial",
                ORDEN_BDC, "Distribucion del potencial (violin)",
                "Potencial (%)"), use_container_width=True)

            st.write("**Tabla ANOVA one-way** (scipy f_oneway + anova_lm typ 1)")
            st.dataframe(tabla_anova_bonita(tab_c,
                {"BDC_L": "Agente BDC", "Residual": "Residual"}),
                hide_index=True, use_container_width=True)
            medias_bdc = df_citas.groupby("BDC_L")["Potencial"].mean()
            st.success(
                f"**Interpretacion:** p = {anova_c.pvalue:.1e} < 0.05 → **se "
                f"rechaza H₀**: el agente BDC si influye en el potencial de "
                f"compra. Saul promedia {medias_bdc['Saul']:.1f}% contra "
                f"{medias_bdc.drop('Saul').mean():.1f}% del resto. "
                f"η² = {eta2_c:.3f}: el agente explica ~{eta2_c*100:.1f}% de "
                f"la variabilidad (efecto chico pero real). Levene p = "
                f"{lev_c.pvalue:.2f} > 0.05 → varianzas homogeneas, el ANOVA "
                f"es valido.")

            st.divider()
            st.markdown("#### ANOVA multifactorial: agente BDC × prueba de manejo")
            mod_c2 = ols("Potencial ~ BDC_L + PDM_txt + BDC_L:PDM_txt",
                         data=df_citas).fit()
            tab_c2 = sm.stats.anova_lm(mod_c2, typ=1)
            st.dataframe(tabla_anova_bonita(tab_c2,
                {"BDC_L": "Agente BDC", "PDM_txt": "Prueba de manejo",
                 "BDC_L:PDM_txt": "BDC × PDM", "Residual": "Residual"}),
                hide_index=True, use_container_width=True)

            C3, C4 = st.columns([3, 2])
            with C3:
                st.plotly_chart(fig_interaccion(df_citas, "BDC_L", "PDM_txt",
                    "Potencial", ORDEN_BDC, "Grafica de interaccion BDC × PDM",
                    "Potencial promedio"), use_container_width=True)
            with C4:
                st.write("Potencial promedio por celda")
                st.dataframe(df_citas.pivot_table(
                    index="BDC_L", columns="PDM_txt", values="Potencial",
                    aggfunc="mean").reindex(ORDEN_BDC).round(1))
            p_bdc = tab_c2.loc["BDC_L", "PR(>F)"]
            p_pdm = tab_c2.loc["PDM_txt", "PR(>F)"]
            p_int = tab_c2.loc["BDC_L:PDM_txt", "PR(>F)"]
            st.info(
                f"**Interpretacion:** agente BDC (p = {formato_p(p_bdc)}) y "
                f"prueba de manejo (p = {formato_p(p_pdm)}) son efectos "
                f"principales significativos; la **interaccion no es "
                f"significativa** (p = {formato_p(p_int)}): las lineas son "
                f"casi paralelas → la prueba de manejo suma potencial por "
                f"igual sin importar el agente (efectos aditivos).")
            st.caption("Robustez: con los 67 outliers incluidos el one-way da "
                       "F = 16.18 (p < 0.001) — la conclusion no depende del "
                       "recorte. Limitacion declarada: Ignacio queda con "
                       "n = 15 tras el recorte.")

    # -----------------------------------------------------
    # TAB 2: MARKETING POR CANAL (Leads y Presupuesto)
    # -----------------------------------------------------
    with tab_mkt:
        if df_mkt is None:
            st.error("No se encontro el archivo 'MKT_DIGITAL.csv' "
                     "en la carpeta del proyecto.")
        else:
            from statsmodels.stats.oneway import anova_oneway
            ORDEN_CANAL = ["Digital", "Plaza", "Piso"]
            st.markdown("#### Leads mensuales segun canal de marketing")
            st.caption(f"Bloque 'Presupuesto mensual de MKT por canal' del "
                       f"archivo maestro → {len(df_mkt)} observaciones "
                       f"(3 canales × {df_mkt['Fecha'].nunique()} meses, "
                       f"{df_mkt['Fecha'].min():%b-%Y} a "
                       f"{df_mkt['Fecha'].max():%b-%Y}).")

            # --- ANOVA one-way: Leads ~ Canal ---
            grupos_m = [g["Leads"].values
                        for _, g in df_mkt.groupby("Canal")]
            anova_m = stats.f_oneway(*grupos_m)
            mod_m = ols("Leads ~ C(Canal)", data=df_mkt).fit()
            tab_m = sm.stats.anova_lm(mod_m, typ=2)
            eta2_m = tab_m["sum_sq"]["C(Canal)"] / tab_m["sum_sq"].sum()
            lev_m = stats.levene(*grupos_m)
            welch_m = anova_oneway(grupos_m, use_var="unequal")

            st.markdown("**H₀:** los leads mensuales promedio son iguales en "
                        "los 3 canales · **H₁:** al menos un canal difiere")
            K1, K2, K3, K4 = st.columns(4)
            K1.metric("F (one-way)", f"{anova_m.statistic:.2f}")
            K2.metric("p-valor", f"{anova_m.pvalue:.1e}")
            K3.metric("η² (tamano del efecto)", f"{eta2_m:.3f}")
            K4.metric("Levene (p)", f"{lev_m.pvalue:.4f}")

            C1, C2 = st.columns(2)
            with C1:
                st.plotly_chart(fig_box(df_mkt, "Canal", "Leads",
                    ORDEN_CANAL, "Boxplot de leads por canal",
                    "Leads mensuales"), use_container_width=True)
            with C2:
                st.plotly_chart(fig_bar_ic95(df_mkt, "Canal", "Leads",
                    ORDEN_CANAL, "Media de leads con IC 95%",
                    "Leads mensuales"), use_container_width=True)

            st.write("**Tabla ANOVA one-way** (anova_lm typ 2)")
            st.dataframe(tabla_anova_bonita(tab_m,
                {"C(Canal)": "Canal", "Residual": "Residual"}),
                hide_index=True, use_container_width=True)
            medias_canal = df_mkt.groupby("Canal")["Leads"].mean()
            st.success(
                f"**Interpretacion:** p = {anova_m.pvalue:.1e} < 0.05 → **se "
                f"rechaza H₀**: el canal determina los leads mensuales. "
                f"Digital genera {medias_canal['Digital']:.0f} leads/mes "
                f"contra {medias_canal['Plaza']:.0f} de Plaza y "
                f"{medias_canal['Piso']:.0f} de Piso. η² = {eta2_m:.2f}: el "
                f"canal explica ~{eta2_m*100:.0f}% de la variabilidad — es "
                f"el efecto mas grande de los tres analisis.")
            st.warning(
                f"**Supuestos:** Levene p = {lev_m.pvalue:.4f} < 0.05 → las "
                f"varianzas no son homogeneas (a mayor media, mayor "
                f"dispersion). Como robustez se corre el ANOVA de Welch: "
                f"F = {welch_m.statistic:.1f}, p = {welch_m.pvalue:.1e} → "
                f"la conclusion se mantiene.")

            st.divider()
            st.markdown("#### ANOVA multifactorial: presupuesto ~ canal × año")
            mod_p = ols("Presupuesto ~ C(Canal, Sum) * C(Anio, Sum)",
                        data=df_mkt).fit()
            tab_p = sm.stats.anova_lm(mod_p, typ=3)
            st.dataframe(tabla_anova_bonita(tab_p,
                {"Intercept": "Intercept", "C(Canal, Sum)": "Canal",
                 "C(Anio, Sum)": "Año",
                 "C(Canal, Sum):C(Anio, Sum)": "Canal × Año",
                 "Residual": "Residual"}),
                hide_index=True, use_container_width=True)
            st.plotly_chart(fig_interaccion(df_mkt, "Anio", "Canal",
                "Presupuesto", ["2025", "2026"],
                "Grafica de interaccion canal × año (presupuesto)",
                "Presupuesto promedio ($)"), use_container_width=True)
            p_can = tab_p.loc["C(Canal, Sum)", "PR(>F)"]
            p_anio = tab_p.loc["C(Anio, Sum)", "PR(>F)"]
            p_intm = tab_p.loc["C(Canal, Sum):C(Anio, Sum)", "PR(>F)"]
            pres_canal = df_mkt.groupby("Canal")["Presupuesto"].mean()
            st.info(
                f"**Interpretacion:** el presupuesto difiere por canal "
                f"(F = {tab_p.loc['C(Canal, Sum)', 'F']:.2f}, p = "
                f"{formato_p(p_can)}): Piso invierte "
                f"${pres_canal['Piso']:,.0f}/mes, Plaza "
                f"${pres_canal['Plaza']:,.0f} y Digital "
                f"${pres_canal['Digital']:,.0f}. El año (p = "
                f"{formato_p(p_anio)}) y la interaccion (p = "
                f"{formato_p(p_intm)}) **no son significativos**: la "
                f"asignacion por canal fue estable entre 2025 y 2026.")

    # -----------------------------------------------------
    # TAB 3: PLAZAS (Leads captados por semana)
    # -----------------------------------------------------
    with tab_plazas:
        if df_plazas_sem is None:
            st.error("No se encontro el archivo 'Plazas.csv' "
                     "en la carpeta del proyecto.")
        else:
            ORDEN_PROD = ["EMZOOM", "GS8", "EMKOO"]
            st.markdown("#### Leads captados por semana segun producto y plaza")
            st.caption("303 leads de piso → variable de respuesta construida: "
                       f"leads por semana ({len(df_plazas_sem)} celdas "
                       "semana × producto × plaza).")

            # --- ANOVA one-way: Leads ~ Producto ---
            grupos_p = [g["Leads"].values
                        for _, g in df_plazas_sem.groupby("Producto_L")]
            anova_p = stats.f_oneway(*grupos_p)
            mod_pl = ols("Leads ~ Producto_L", data=df_plazas_sem).fit()
            tab_pl = sm.stats.anova_lm(mod_pl, typ=1)

            st.markdown("**H₀:** los leads semanales promedio son iguales "
                        "entre productos · **H₁:** al menos un producto difiere")
            K1, K2, K3 = st.columns(3)
            K1.metric("F (one-way)", f"{anova_p.statistic:.2f}")
            K2.metric("p-valor", formato_p(anova_p.pvalue))
            K3.metric("gl", f"{int(tab_pl['df']['Producto_L'])} / "
                             f"{int(tab_pl['df']['Residual'])}")

            C1, C2 = st.columns(2)
            with C1:
                st.plotly_chart(fig_box(df_plazas_sem, "Producto_L", "Leads",
                    ORDEN_PROD, "Boxplot de leads semanales por producto",
                    "Leads por semana"), use_container_width=True)
            with C2:
                st.plotly_chart(fig_bar_ic95(df_plazas_sem, "Producto_L",
                    "Leads", ORDEN_PROD, "Media de leads con IC 95%",
                    "Leads por semana"), use_container_width=True)

            st.write("**Tabla ANOVA one-way**")
            st.dataframe(tabla_anova_bonita(tab_pl,
                {"Producto_L": "Producto", "Residual": "Residual"}),
                hide_index=True, use_container_width=True)
            medias_prod = df_plazas_sem.groupby("Producto_L")["Leads"].mean()
            st.success(
                f"**Interpretacion:** p = {formato_p(anova_p.pvalue)} < 0.05 "
                f"→ **se rechaza H₀**: el producto influye en los leads "
                f"semanales. GS8 capta {medias_prod['GS8']:.1f} leads/semana "
                f"contra {medias_prod['EMKOO']:.1f} de EMKOO y "
                f"{medias_prod['EMZOOM']:.1f} de EMZOOM.")

            st.divider()
            st.markdown("#### ANOVA multifactorial: leads ~ producto × plaza")
            mod_pl2 = ols("Leads ~ Producto_L + Plaza_L + Producto_L:Plaza_L",
                          data=df_plazas_sem).fit()
            tab_pl2 = sm.stats.anova_lm(mod_pl2, typ=1)
            st.dataframe(tabla_anova_bonita(tab_pl2,
                {"Producto_L": "Producto", "Plaza_L": "Plaza",
                 "Producto_L:Plaza_L": "Producto × Plaza",
                 "Residual": "Residual"}),
                hide_index=True, use_container_width=True)
            st.plotly_chart(fig_interaccion(df_plazas_sem, "Producto_L",
                "Plaza_L", "Leads", ORDEN_PROD,
                "Grafica de interaccion producto × plaza",
                "Leads semanales promedio"), use_container_width=True)
            p_prod = tab_pl2.loc["Producto_L", "PR(>F)"]
            p_plaza = tab_pl2.loc["Plaza_L", "PR(>F)"]
            p_intp = tab_pl2.loc["Producto_L:Plaza_L", "PR(>F)"]
            st.info(
                f"**Interpretacion:** producto significativo (p = "
                f"{formato_p(p_prod)}); plaza sola no (p = "
                f"{formato_p(p_plaza)}), pero la **interaccion producto × "
                f"plaza si es significativa** (p = {formato_p(p_intp)}): la "
                f"mezcla de productos que jala leads cambia segun la "
                f"ubicacion — lo que se vende en Angelopolis no se comporta "
                f"igual en Tlaxcala.")

st.divider()
st.caption("Dashboard GAC | Etapa I: Analisis univariado · "
           "Etapa II: Modelado predictivo · Etapa III: Despliegue ANOVA | "
           "Plaza Angelopolis")
