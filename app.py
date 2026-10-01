# Dashboard GAC - Analisis Univariado
# Tema vino tinto / borgona (presentacion GAC) - diseno plano

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

df_bitacora, df_leads = load_data()

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

ESTILO_FUENTE = dict(family="Arial, sans-serif", size=13, color=GAC_OSCURO)

# =========================================================
# ENCABEZADO
# =========================================================
st.markdown(
    "<h1 style='color:" + GAC_AZUL + ";'>🚗 Analisis Univariado GAC</h1>",
    unsafe_allow_html=True
)
st.caption("Bitacora de Piso y Leads Reales | Plaza Angelopolis")
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
     "Explorador de Variables"]
)

st.sidebar.markdown("---")
st.sidebar.write("📋 Bitacora de Piso: **%d** registros" % len(df_bitacora))
st.sidebar.write("📈 Leads Reales: **%d** meses" % len(df_leads))

# =========================================================
# VISTA 1: EXTRACCION DE CARACTERISTICAS
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
# VISTA 4: EXPLORADOR DE VARIABLES (plantilla del curso)
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

st.divider()
st.caption("Dashboard GAC | Analisis Univariado | Datos: Bitacora de Piso y Leads Reales")
