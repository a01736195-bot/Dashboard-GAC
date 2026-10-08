# Dashboard GAC - Analisis Univariado + Modelado Predictivo
# Tema vino tinto / borgona (presentacion GAC) - diseno plano

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

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
    return df_resumen, df_funnel

df_bitacora, df_leads = load_data()
df_detalle = load_leads_detalle()
df_resumen, df_funnel = load_regresion_data()

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

# Variables numericas disponibles para los modelos (Etapa II)
LISTA_NUM = ["Leads", "Efectivos", "Ventas Reportadas", "Obj Ventas",
             "Ventas Real", "Conversion", "SDC", "Autorizadas",
             "Formalizado", "Contado", "Credito", "PDM", "Obj PDM",
             "Variacion", "Lead / Venta"]

# =========================================================
# ENCABEZADO
# =========================================================
st.markdown(
    "<h1 style='color:" + GAC_AZUL + ";'>🚗 Analisis GAC</h1>",
    unsafe_allow_html=True
)
st.caption("Etapa I · Analisis univariado | Etapa II · Modelado predictivo | Plaza Angelopolis")
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
     "Regresion Lineal Multiple",
     "Regresion del Embudo (bonus)"]
)

st.sidebar.markdown("---")
st.sidebar.write("📋 Bitacora de Piso: **%d** registros" % len(df_bitacora))
st.sidebar.write("📈 Leads Reales: **%d** meses" % len(df_leads))
if df_detalle is not None:
    st.sidebar.write("🏙️ Leads Plaza: **%d** registros" % len(df_detalle))
st.sidebar.write("📊 Base resumen: **%d** meses" % len(df_resumen))
st.sidebar.write("🔻 Embudo: **%d** meses" % len(df_funnel))

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
# VISTA 7: REGRESION LINEAL SIMPLE (ETAPA II)
# =========================================================
elif vista == "Regresion Lineal Simple":

    st.subheader("Regresion lineal simple (una variable X)")

    col_x, col_y = st.columns(2)
    with col_x:
        Variable_x = st.selectbox("Variable independiente (X)", LISTA_NUM,
                                  index=LISTA_NUM.index("Leads"))
    with col_y:
        Variable_y = st.selectbox("Variable objetivo (Y)", LISTA_NUM,
                                  index=LISTA_NUM.index("Ventas Real"))

    model = LinearRegression()
    model.fit(X=df_resumen[[Variable_x]], y=df_resumen[Variable_y])
    y_pred = model.predict(df_resumen[[Variable_x]])

    intercepto = model.intercept_
    pendiente = model.coef_[0]
    r2 = model.score(df_resumen[[Variable_x]], df_resumen[Variable_y])
    r = np.sqrt(r2)

    M1, M2, M3 = st.columns(3)
    M1.metric("R² (determinacion)", f"{r2:.4f}")
    M2.metric("r (correlacion)", f"{r:.4f}")
    M3.metric("Pendiente (b1)", f"{pendiente:.4f}")
    st.markdown(
        f"**Ecuacion del modelo:** `{Variable_y} = {intercepto:.4f} "
        f"+ ({pendiente:.4f}) × {Variable_x}`"
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_resumen[Variable_x], y=df_resumen[Variable_y], mode="markers",
        name="Datos reales", marker=dict(color=GRIS_DATOS, size=11)
    ))
    orden = df_resumen[Variable_x].argsort()
    fig.add_trace(go.Scatter(
        x=df_resumen[Variable_x].iloc[orden], y=pd.Series(y_pred).iloc[orden],
        mode="lines", name="Recta de regresion",
        line=dict(color=GAC_AZUL, width=3)
    ))
    fig.update_layout(
        font=ESTILO_FUENTE, height=480,
        title=dict(text=f"{Variable_y} ~ {Variable_x}",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title=Variable_x),
        yaxis=dict(title=Variable_y, gridcolor=REJILLA),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabla de valores reales, predichos y residuales"):
        tabla = pd.DataFrame({
            "Año": df_resumen["Año"], "Mes": df_resumen["Mes"],
            "Real": df_resumen[Variable_y].round(2),
            "Predicho": np.round(y_pred, 2)})
        tabla["Residual"] = (tabla["Real"] - tabla["Predicho"]).round(2)
        st.dataframe(tabla, hide_index=True, use_container_width=True)

# =========================================================
# VISTA 8: REGRESION LINEAL MULTIPLE (ETAPA II)
# =========================================================
elif vista == "Regresion Lineal Multiple":

    st.subheader("Regresion lineal multiple (varias variables X)")

    col_y2, _ = st.columns(2)
    with col_y2:
        Variable_y = st.selectbox("Variable objetivo (Y)", LISTA_NUM,
                                  index=LISTA_NUM.index("Ventas Real"),
                                  key="y_multi")
    Variables_x = st.multiselect(
        "Variables independientes (X)",
        [v for v in LISTA_NUM if v != Variable_y],
        default=["Efectivos", "SDC", "Autorizadas", "PDM"])

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
        st.write("Matriz de correlacion (variables del modelo)")
        matriz = df_resumen[Variables_x + [Variable_y]].corr().round(2)
        fig_hm = px.imshow(matriz, text_auto=True, aspect="auto",
                           color_continuous_scale=ESCALA_VINO,
                           zmin=-1, zmax=1)
        fig_hm.update_layout(font=ESTILO_FUENTE, height=420,
                             plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(fig_hm, use_container_width=True)

    with C2:
        st.write("Ventas reales vs predichas")
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
# VISTA 9: REGRESION DEL EMBUDO (BONUS)
# =========================================================
elif vista == "Regresion del Embudo (bonus)":

    st.subheader("Regresion sobre el embudo de ventas (FUNNEL)")
    st.caption("Modelos ajustados con los 18 meses del embudo GAC Angelopolis")

    st.markdown("**Modelo 1 · Ventas ~ etapas del embudo**")
    vars_m1 = ["Cita_Efectiva", "Prueba_Manejo", "SDC_Aprobada", "Formalizado"]
    m1 = LinearRegression().fit(df_funnel[vars_m1], df_funnel["Ventas"])
    pred1 = m1.predict(df_funnel[vars_m1])
    r2_m1 = m1.score(df_funnel[vars_m1], df_funnel["Ventas"])

    c1, c2 = st.columns(2)
    c1.metric("R²", f"{r2_m1:.4f}")
    c2.metric("R (correlacion multiple)", f"{np.sqrt(r2_m1):.4f}")

    tabla_m1 = pd.DataFrame({"Variable": vars_m1,
                             "Coeficiente": np.round(m1.coef_, 4)})
    tabla_m1.loc[len(tabla_m1)] = ["Intercepto", round(m1.intercept_, 4)]
    st.dataframe(tabla_m1, hide_index=True, use_container_width=True)
    st.caption("La variable con mayor peso es SDC_Aprobada: cada solicitud de "
               "credito aprobada suma ~0.61 ventas al mes.")

    st.markdown("**Modelo 2 · Ventas ~ tendencia temporal + solicitudes aprobadas**")
    vars_m2 = ["Mes_Num", "SDC_Aprobada"]
    m2 = LinearRegression().fit(df_funnel[vars_m2], df_funnel["Ventas"])
    pred2 = m2.predict(df_funnel[vars_m2])
    r2_m2 = m2.score(df_funnel[vars_m2], df_funnel["Ventas"])

    c3, c4 = st.columns(2)
    c3.metric("R²", f"{r2_m2:.4f}")
    c4.metric("R (correlacion multiple)", f"{np.sqrt(r2_m2):.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=df_funnel["Ventas"], mode="lines+markers",
        name="Ventas reales", line=dict(color=GRIS_DATOS, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=np.round(pred1, 1), mode="lines+markers",
        name=f"Modelo 1 (R²={r2_m1:.2f})", line=dict(color=GAC_AZUL, width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=np.round(pred2, 1), mode="lines+markers",
        name=f"Modelo 2 (R²={r2_m2:.2f})",
        line=dict(color=GAC_AZUL_CLARO, width=2.5, dash="dash")
    ))
    fig.update_layout(
        font=ESTILO_FUENTE, height=480,
        title=dict(text="Ventas reales vs predichas por los modelos del embudo",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title="Mes", tickangle=-45),
        yaxis=dict(title="Ventas", gridcolor=REJILLA),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver tabla real vs predicho (Modelo 2)"):
        tabla = pd.DataFrame({
            "Mes": df_funnel["Mes"],
            "Ventas reales": df_funnel["Ventas"],
            "Ventas predichas": np.round(pred2, 2)})
        tabla["Residual"] = (tabla["Ventas reales"]
                             - tabla["Ventas predichas"]).round(2)
        st.dataframe(tabla, hide_index=True, use_container_width=True)

st.divider()
st.caption("Dashboard GAC | Etapa I: Analisis univariado · "
           "Etapa II: Modelado predictivo | Plaza Angelopolis")
