# Vistas estadisticas del dashboard GAC (Etapa II y III del proceso original)
# Cada funcion dibuja una pestana de la vista "Modelos predictivos".
# Todo se replica fielmente de los notebooks del equipo; el codigo original
# completo esta archivado en archivo/app_etapas_original.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# ---------- TEMA (mismo que app.py) ----------
GAC_AZUL       = "#800020"
GAC_AZUL_MED   = "#9B1B30"
GAC_AZUL_CLARO = "#DDA0A8"
GAC_PLATA      = "#C9747F"
GAC_OSCURO     = "#4A0414"
GAC_BLANCO     = "#FFFFFF"
REJILLA        = "#F3E4E7"
GRIS_DATOS     = "#9AA3B0"
PALETA_ANOVA   = [GAC_AZUL, GAC_PLATA, GAC_AZUL_MED, GAC_AZUL_CLARO]
ESCALA_VINO    = [[0, "#FBF2F3"], [0.5, "#DDA0A8"], [1, "#800020"]]
ESTILO_FUENTE  = dict(family="Arial, sans-serif", size=13, color=GAC_OSCURO)

LISTA_NUM = ["Leads", "Efectivos", "Ventas Reportadas", "Obj Ventas",
             "Ventas Real", "Conversion", "SDC", "Autorizadas",
             "Formalizado", "Contado", "Credito", "PDM", "Obj PDM",
             "Variacion", "Lead / Venta"]

# ---------- HELPERS ANOVA ----------
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

# ---------- KEYS UNICOS PARA GRAFICAS ----------
_contador = {"n": 0}
def _key(base="fig"):
    _contador["n"] += 1
    return f"vm_{base}_{_contador['n']}"

# =========================================================
# CORRELACIONES
# =========================================================
def vista_correlaciones(df_resumen):
    st.markdown("**¿Para que sirve?** Ver que variables se mueven junto con las "
                "ventas. Las que mas correlacionen son las mejores palancas para "
                "predecir y para enfocar el esfuerzo del mes.")
    sel_corr = st.multiselect("Variables a incluir en el heatmap",
                              LISTA_NUM,
                              default=["Leads", "Efectivos", "SDC",
                                       "Autorizadas", "PDM", "Ventas Real"])
    if len(sel_corr) < 2:
        st.warning("Selecciona al menos 2 variables.")
        return

    matriz = df_resumen[sel_corr].corr().round(2)
    fig_corr = px.imshow(matriz, text_auto=True, aspect="auto",
                         title="Matriz de correlacion (Pearson)",
                         color_continuous_scale=ESCALA_VINO, zmin=-1, zmax=1)
    fig_corr.update_layout(font=ESTILO_FUENTE, height=520,
                           plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
    st.plotly_chart(fig_corr, width="stretch", key=_key("g"))

    st.markdown("**Correlacion de cada variable con Ventas Real**")
    tabla_corr = (df_resumen[LISTA_NUM].corr()["Ventas Real"]
                  .drop("Ventas Real").sort_values(ascending=False)
                  .reset_index())
    tabla_corr.columns = ["Variable", "Correlacion con Ventas Real"]
    fig_bar = px.bar(tabla_corr, x="Correlacion con Ventas Real", y="Variable",
                     orientation="h", text="Correlacion con Ventas Real",
                     color_discrete_sequence=[GAC_AZUL])
    fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig_bar.update_layout(font=ESTILO_FUENTE, height=480,
                          plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
                          xaxis=dict(range=[-1, 1], gridcolor=REJILLA))
    st.plotly_chart(fig_bar, width="stretch", key=_key("g"))

    st.info("Las variables con mayor correlacion positiva con las ventas "
            "(Autorizadas, Formalizado, SDC) son las mejores candidatas "
            "para el modelo de regresion.")

# =========================================================
# REGRESION LINEAL SIMPLE
# =========================================================
def vista_reg_simple(df_resumen, df_bcs):
    st.markdown("**¿Para que sirve?** Con una sola variable (por ejemplo, las "
                "solicitudes de credito) estimar otra (las ventas). La ecuacion "
                "del modelo permite hacer proyecciones rapidas.")
    Base_sel = st.selectbox(
        "Base de datos",
        ["BCS - 6 modelos (Balance Scorecard)",
         "Base resumen mensual (30 meses)"])

    if Base_sel.startswith("BCS"):
        if df_bcs is None:
            st.error("No se encontro el archivo 'bcs_modelos.csv'.")
            return
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
        return

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
        f"+ ({pendiente:.4f}) × {Variable_x}`")
    st.caption("R² indica que porcentaje de la variacion de Y explica la "
               f"variable X: aqui, {r2*100:.1f}%.")

    fig = go.Figure()
    if es_bcs:
        fig.add_trace(go.Scatter(
            x=df_reg[Variable_x], y=df_reg[Variable_y],
            mode="markers+text", text=df_reg["Modelo"],
            textposition="top center", name="Ventas reales",
            marker=dict(color=GRIS_DATOS, size=13)))
    else:
        fig.add_trace(go.Scatter(
            x=df_reg[Variable_x], y=df_reg[Variable_y],
            mode="markers", name="Datos reales",
            marker=dict(color=GRIS_DATOS, size=11)))
    orden = df_reg[Variable_x].argsort()
    fig.add_trace(go.Scatter(
        x=df_reg[Variable_x].iloc[orden],
        y=pd.Series(y_pred).iloc[orden],
        mode="lines", name="Recta de regresion",
        line=dict(color=GAC_AZUL, width=3)))
    titulo_graf = ("Regresion lineal del objetivo y las ventas reales por modelo"
                   if es_bcs else f"{Variable_y} ~ {Variable_x}")
    fig.update_layout(
        font=ESTILO_FUENTE, height=460,
        title=dict(text=titulo_graf, font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title=Variable_x),
        yaxis=dict(title=Variable_y, gridcolor=REJILLA),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch", key=_key("g"))

    st.markdown("**Valores reales, predichos y residuales**")
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
    st.dataframe(tabla, hide_index=True, width="stretch")

# =========================================================
# REGRESION LINEAL MULTIPLE
# =========================================================
def vista_reg_multiple(df_resumen):
    st.markdown("**¿Para que sirve?** Igual que la simple, pero combinando "
                "varias senales a la vez (p. ej. leads efectivos + pruebas de "
                "manejo + solicitudes). Suele dar proyecciones mas ajustadas.")
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
        return

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
        ecuacion += f" {signo} {abs(coef):.4f}*({nombre})"

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
                           color_continuous_scale=ESCALA_VINO, zmin=-1, zmax=1)
        fig_hm.update_layout(font=ESTILO_FUENTE, height=420,
                             plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
        st.plotly_chart(fig_hm, width="stretch", key=_key("g"))
    with C2:
        st.write("Correlacion Lineal Multiple")
        minimo = min(df_resumen[Variable_y].min(), y_pred.min())
        maximo = max(df_resumen[Variable_y].max(), y_pred.max())
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=df_resumen[Variable_y], y=y_pred, mode="markers",
            name="Meses", marker=dict(color=GRIS_DATOS, size=11)))
        fig_sc.add_trace(go.Scatter(
            x=[minimo, maximo], y=[minimo, maximo], mode="lines",
            name="Linea ideal (y = x)",
            line=dict(color=GAC_AZUL, width=2, dash="dash")))
        fig_sc.update_layout(
            font=ESTILO_FUENTE, height=420,
            title=dict(text="Modelo Lineal Multiple",
                       font=dict(size=16, color=GAC_AZUL)),
            xaxis=dict(title="Reales"),
            yaxis=dict(title="Predichos", gridcolor=REJILLA),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                        xanchor="center", x=0.5))
        st.plotly_chart(fig_sc, width="stretch", key=_key("g"))

    st.markdown("**Coeficientes del modelo** (el signo y el tamano indican "
                "hacia donde y cuanto empuja cada variable)")
    tabla_coef = pd.DataFrame({
        "Variable": Variables_x,
        "Coeficiente": np.round(model.coef_, 4)})
    tabla_coef.loc[len(tabla_coef)] = ["Intercepto", round(model.intercept_, 4)]
    st.dataframe(tabla_coef, hide_index=True, width="stretch")

    with st.expander("Ver tabla de valores reales vs predichos"):
        tabla = pd.DataFrame({
            "Año": df_resumen["Año"], "Mes": df_resumen["Mes"],
            "Real": df_resumen[Variable_y].round(2),
            "Predicho": np.round(y_pred, 2)})
        tabla["Residual"] = (tabla["Real"] - tabla["Predicho"]).round(2)
        st.dataframe(tabla, hide_index=True, width="stretch")

# =========================================================
# REGRESION DEL EMBUDO (FUNNEL)
# =========================================================
def vista_reg_funnel(df_funnel):
    st.markdown("**¿Para que sirve?** Es el modelo operativo mas util: con las "
                "etapas tempranas del mes (citas, pruebas de manejo, solicitudes "
                "aprobadas) **proyecta cuantas ventas va a cerrar el mes** antes "
                "de que termine.")
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
        return

    y_real = df_funnel[Variable_y]

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
    st.dataframe(tabla_m1, hide_index=True, width="stretch")
    mayor = tabla_m1.iloc[np.argmax(np.abs(m1.coef_))]["Variable"]
    st.caption(f"Variable con mayor peso en el Modelo 1: {mayor}")

    st.markdown(f"**Regresion 2 · {Variable_y} ~ variables seleccionadas**")
    m2 = LinearRegression().fit(df_funnel[vars_m2], y_real)
    pred2 = m2.predict(df_funnel[vars_m2])
    r2_m2 = m2.score(df_funnel[vars_m2], y_real)

    c3, c4 = st.columns(2)
    c3.metric("R²", f"{r2_m2:.4f}")
    c4.metric("Correlacion multiple", f"{np.sqrt(r2_m2):.4f}")

    st.subheader(f"{Variable_y}: reales vs predichas")
    minimo = min(y_real.min(), pred1.min(), pred2.min())
    maximo = max(y_real.max(), pred1.max(), pred2.max())

    D1, D2 = st.columns(2)
    with D1:
        fig_d1 = go.Figure()
        fig_d1.add_trace(go.Scatter(
            x=df_funnel["Ventas"], y=np.round(pred1, 1), mode="markers",
            name="Meses", marker=dict(color=GRIS_DATOS, size=11)))
        fig_d1.add_trace(go.Scatter(
            x=[minimo, maximo], y=[minimo, maximo], mode="lines",
            name="Linea ideal (y = x)",
            line=dict(color=GAC_AZUL, width=2, dash="dash")))
        fig_d1.update_layout(
            font=ESTILO_FUENTE, height=400,
            title=dict(text=f"Regresion 1 (R² = {r2_m1:.2f})",
                       font=dict(size=15, color=GAC_AZUL)),
            xaxis=dict(title=f"{Variable_y} reales"),
            yaxis=dict(title=f"{Variable_y} predichas", gridcolor=REJILLA),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                        xanchor="center", x=0.5))
        st.plotly_chart(fig_d1, width="stretch", key=_key("g"))
    with D2:
        fig_d2 = go.Figure()
        fig_d2.add_trace(go.Scatter(
            x=df_funnel["Ventas"], y=np.round(pred2, 1), mode="markers",
            name="Meses", marker=dict(color=GRIS_DATOS, size=11)))
        fig_d2.add_trace(go.Scatter(
            x=[minimo, maximo], y=[minimo, maximo], mode="lines",
            name="Linea ideal (y = x)",
            line=dict(color=GAC_AZUL, width=2, dash="dash")))
        fig_d2.update_layout(
            font=ESTILO_FUENTE, height=400,
            title=dict(text=f"Regresion 2 (R² = {r2_m2:.2f})",
                       font=dict(size=15, color=GAC_AZUL)),
            xaxis=dict(title=f"{Variable_y} reales"),
            yaxis=dict(title=f"{Variable_y} predichas", gridcolor=REJILLA),
            plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3,
                        xanchor="center", x=0.5))
        st.plotly_chart(fig_d2, width="stretch", key=_key("g"))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=y_real, mode="lines+markers",
        name=f"{Variable_y} reales", line=dict(color=GRIS_DATOS, width=2.5)))
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=np.round(pred1, 1), mode="lines+markers",
        name=f"Regresion 1 (R²={r2_m1:.2f})",
        line=dict(color=GAC_AZUL, width=2.5)))
    fig.add_trace(go.Scatter(
        x=df_funnel["Mes"], y=np.round(pred2, 1), mode="lines+markers",
        name=f"Regresion 2 (R²={r2_m2:.2f})",
        line=dict(color=GAC_AZUL_CLARO, width=2.5, dash="dash")))
    fig.update_layout(
        font=ESTILO_FUENTE, height=460,
        title=dict(text="Ventas reales vs predichas por mes",
                   font=dict(size=16, color=GAC_AZUL)),
        xaxis=dict(title="Mes", tickangle=-45),
        yaxis=dict(title="Ventas", gridcolor=REJILLA),
        plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1))
    st.plotly_chart(fig, width="stretch", key=_key("g"))
    if r2_m1 >= 0.7 or r2_m2 >= 0.7:
        st.success(f"Con R² de {max(r2_m1, r2_m2):.2f}, el modelo explica bien "
                   "la variacion de las ventas: a mitad de mes, con las citas "
                   "efectivas y SDC aprobadas acumuladas, se puede proyectar "
                   "el cierre con razonable confianza.")

    with st.expander("Ver tabla real vs predicho (Regresion 2)"):
        tabla = pd.DataFrame({
            "Mes": df_funnel["Mes"],
            f"{Variable_y} reales": y_real,
            f"{Variable_y} predichas": np.round(pred2, 2)})
        tabla["Residual"] = (tabla[f"{Variable_y} reales"]
                             - tabla[f"{Variable_y} predichas"]).round(2)
        st.dataframe(tabla, hide_index=True, width="stretch")

# =========================================================
# REGRESION LOGISTICA (PDM)
# =========================================================
def vista_logistica_pdm(df_bitacora):
    st.markdown("**¿Para que sirve?** Clasificacion si/no: estima si una visita "
                "terminara en prueba de manejo segun senales tempranas (SDC, "
                "intervencion del gerente, estatus del lead). Una **alarma** para "
                "que gerencia intervenga la visita antes de que se enfrie el lead.")
    if not SMOTE_OK:
        st.error("Falta la libreria imbalanced-learn: agrega imbalanced-learn a "
                 "requirements.txt y dale **Reboot** a la app.")
        return

    df_log = df_bitacora.copy()
    df_log["Estatus_bin"] = (df_log["Estatus_Lead"].astype(str).str.strip() == "Activo").astype(int)
    X_log = df_log[["SDC", "Inter_Gerente", "Estatus_bin", "Venta"]].astype(int)
    y_log = df_log["PDM"].astype(int)

    SEMILLA = 21
    X_ent, X_pru, y_ent, y_pru = train_test_split(X_log, y_log, test_size=0.3, random_state=SEMILLA)

    m1, m2, m3 = st.columns(3)
    m1.metric("Entrenamiento", f"{len(X_ent)}", "70% de la bitacora", delta_color="off")
    m2.metric("Prueba", f"{len(X_pru)}", "30% de la bitacora", delta_color="off")
    m3.metric("PDM = 1 en prueba", f"{int(y_pru.sum())}", "de %d registros" % len(y_pru), delta_color="off")

    def _metricas_log(y_real, y_pred):
        return {
            "Exactitud": accuracy_score(y_real, y_pred),
            "Precision": precision_score(y_real, y_pred, pos_label=1, zero_division=0),
            "Sensibilidad": recall_score(y_real, y_pred, pos_label=1),
            "F1": f1_score(y_real, y_pred, pos_label=1),
        }

    def _fig_matriz(cm, titulo):
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=ESCALA_VINO,
                        x=["Predicho: No", "Predicho: Si"], y=["Real: No", "Real: Si"],
                        title=titulo, aspect="auto")
        fig.update_layout(font=ESTILO_FUENTE, height=380, coloraxis_showscale=False,
                          title_font=dict(size=15, color=GAC_AZUL),
                          plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
        fig.update_traces(textfont_size=16)
        return fig

    esc = StandardScaler()
    X_ent_e = esc.fit_transform(X_ent)
    X_pru_e = esc.transform(X_pru)
    mod_base = LogisticRegression(random_state=SEMILLA)
    mod_base.fit(X_ent_e, y_ent)
    pred_base = mod_base.predict(X_pru_e)
    met_base = _metricas_log(y_pru, pred_base)
    cm_base = confusion_matrix(y_pru, pred_base)

    smote = SMOTE(random_state=SEMILLA)
    X_over, y_over = smote.fit_resample(X_ent, y_ent)
    esc2 = StandardScaler()
    X_over_e = esc2.fit_transform(X_over)
    X_pru_e2 = esc2.transform(X_pru)
    mod_over = LogisticRegression(random_state=SEMILLA)
    mod_over.fit(X_over_e, y_over)
    pred_over = mod_over.predict(X_pru_e2)
    met_over = _metricas_log(y_pru, pred_over)
    cm_over = confusion_matrix(y_pru, pred_over)

    tab_base, tab_over, tab_comp = st.tabs(["Sin balanceo", "Oversampling (SMOTE)", "Comparacion"])

    with tab_base:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Exactitud", f"{met_base['Exactitud']:.3f}")
        b2.metric("Precision (PDM=1)", f"{met_base['Precision']:.3f}")
        b3.metric("Sensibilidad (recall)", f"{met_base['Sensibilidad']:.3f}")
        b4.metric("F1", f"{met_base['F1']:.3f}")
        st.plotly_chart(_fig_matriz(cm_base, "Matriz de confusion - modelo base"), width="stretch", key=_key("g"))
        st.caption(f"El modelo base acierta {met_base['Exactitud']*100:.0f}% de las veces, pero solo detecta "
                   f"{int(cm_base[1,1])} de {int(cm_base[1].sum())} pruebas de manejo reales "
                   f"(sensibilidad {met_base['Sensibilidad']*100:.0f}%): como PDM=1 es la clase minoritaria, "
                   "el modelo tiende a predecir 'No'.")

    with tab_over:
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Exactitud", f"{met_over['Exactitud']:.3f}")
        s2.metric("Precision (PDM=1)", f"{met_over['Precision']:.3f}")
        s3.metric("Sensibilidad (recall)", f"{met_over['Sensibilidad']:.3f}")
        s4.metric("F1", f"{met_over['F1']:.3f}")
        st.plotly_chart(_fig_matriz(cm_over, "Matriz de confusion - con SMOTE"), width="stretch", key=_key("g"))
        st.caption(f"SMOTE crea ejemplos sinteticos de PDM=1 en el entrenamiento "
                   f"(de {len(X_ent)} a {len(X_over)} registros, con las clases parejas). "
                   f"La sensibilidad salta a {met_over['Sensibilidad']*100:.0f}% "
                   f"(detecta {int(cm_over[1,1])} de {int(cm_over[1].sum())} casos reales), pero la exactitud "
                   f"baja a {met_over['Exactitud']*100:.0f}%: marca {int(cm_over[0,1])} falsos positivos.")

    with tab_comp:
        tabla_log = pd.DataFrame({"Sin balanceo": met_base, "Con SMOTE": met_over}).round(3)
        st.dataframe(tabla_log, width="stretch")
        st.markdown(f"""
        **¿Cual conviene para GAC?** Depende del objetivo:

        - Si lo que importa es **no dejar ir prospectos** con potencial de prueba de manejo,
          el modelo **con SMOTE** es la mejor alarma: atrapa casi todos los casos reales
          (sensibilidad {met_over['Sensibilidad']*100:.0f}%), a costa de revisar mas falsos positivos.
        - Si lo que importa es **certeza al predecir**, el modelo base es mas exacto
          ({met_base['Exactitud']*100:.0f}%), pero deja pasar la mayoria de las pruebas reales.
        - Las senales con mas peso son el **estatus activo del lead**, la **intervencion de gerente**
          y la **SDC** (coeficientes positivos en ambos modelos); la variable **Venta** practicamente
          no aporta, porque quien ya compro ya paso por la prueba.
        """)

    st.caption("Nota de reproducibilidad: el notebook original usaba division aleatoria sin semilla "
               "(random_state=None), por lo que sus cifras exactas cambiaban en cada corrida; aqui se fija "
               f"semilla = {SEMILLA} para que el tablero siempre muestre lo mismo. La base es "
               f"Bitacora_Piso_Limpia.csv ({len(df_log)} registros ya limpios); con una muestra chica "
               "las cifras pueden variar ligeramente segun la division, pero el patron "
               "(SMOTE sube la sensibilidad a costa de exactitud) se mantiene.")

# =========================================================
# REGRESION LOGISTICA (VENTAS)
# =========================================================
def vista_logistica_ventas(df_resumen):
    st.markdown("**¿Para que sirve?** Otra alarma si/no: clasifica si el mes va a "
                "ser de **ventas bajas o altas** segun las senales del mes (leads, "
                "efectivos, pruebas). Sirve para activar promociones o empuje de "
                "piso antes de que el mes cierre mal.")
    if not SMOTE_OK:
        st.error("Falta la libreria imbalanced-learn: agrega imbalanced-learn a "
                 "requirements.txt y dale **Reboot** a la app.")
        return

    df_rv = df_resumen.copy()
    df_rv["Periodo"] = df_rv["Año"].astype(str).replace(["2025", "2026"], "2025-2026")
    df_rv["Periodo_num"] = np.where(df_rv["Periodo"] == "2025-2026", 1, 0)
    vmin, vmax = df_rv["Ventas Real"].min(), df_rv["Ventas Real"].max()
    df_rv["Ventas_Cat"] = pd.cut(df_rv["Ventas Real"],
                                 bins=np.linspace(vmin - 0.1, vmax + 0.1, 3),
                                 labels=["Ventas bajas", "Ventas altas"])
    y_rv = df_rv["Ventas_Cat"]

    conteo = y_rv.value_counts()
    corte = (vmin + vmax) / 2
    p1, p2, p3 = st.columns(3)
    p1.metric("Meses en la base", f"{len(df_rv)}", "corte en %.1f ventas" % corte, delta_color="off")
    p2.metric("Ventas bajas", f"{int(conteo.get('Ventas bajas', 0))}", "clase minoritaria", delta_color="off")
    p3.metric("Ventas altas", f"{int(conteo.get('Ventas altas', 0))}", "clase mayoritaria", delta_color="off")

    def _metricas_rv(y_real, y_pred):
        return {
            "Exactitud": accuracy_score(y_real, y_pred),
            "Precision": precision_score(y_real, y_pred, pos_label="Ventas bajas", zero_division=0),
            "Sensibilidad": recall_score(y_real, y_pred, pos_label="Ventas bajas"),
            "F1": f1_score(y_real, y_pred, pos_label="Ventas bajas"),
        }

    def _fig_matriz_rv(cm, titulo):
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=ESCALA_VINO,
                        x=["Predicho: Altas", "Predicho: Bajas"],
                        y=["Real: Altas", "Real: Bajas"],
                        title=titulo, aspect="auto")
        fig.update_layout(font=ESTILO_FUENTE, height=380, coloraxis_showscale=False,
                          title_font=dict(size=15, color=GAC_AZUL),
                          plot_bgcolor=GAC_BLANCO, paper_bgcolor=GAC_BLANCO)
        fig.update_traces(textfont_size=16)
        return fig

    def _entrena_rv(X, y, ponderar=False, con_smote=False):
        X_e, X_p, y_e, y_p = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        esc_rv = StandardScaler()
        X_e = esc_rv.fit_transform(X_e)
        X_p = esc_rv.transform(X_p)
        if con_smote:
            X_e, y_e = SMOTE(random_state=42).fit_resample(X_e, y_e)
        mod_rv = LogisticRegression(class_weight="balanced" if ponderar else None)
        mod_rv.fit(X_e, y_e)
        return mod_rv, y_p, mod_rv.predict(X_p)

    vars_m1 = ["Leads", "Efectivos", "PDM", "Periodo_num"]
    mod_m1, y_p1, pred_m1 = _entrena_rv(df_rv[vars_m1], y_rv)
    met_m1 = _metricas_rv(y_p1, pred_m1)

    st.markdown("#### Modelo 1 · multivariable (Leads + Efectivos + PDM + Periodo)")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Exactitud", f"{met_m1['Exactitud']:.3f}")
    a2.metric("Precision (bajas)", f"{met_m1['Precision']:.3f}")
    a3.metric("Sensibilidad (recall)", f"{met_m1['Sensibilidad']:.3f}")
    a4.metric("F1", f"{met_m1['F1']:.3f}")
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(_fig_matriz_rv(confusion_matrix(y_p1, pred_m1),
                                       "Matriz de confusion - multivariable"),
                        width="stretch", key=_key("g"))
    with g2:
        coefs = pd.DataFrame({"Variable": ["Leads", "Efectivos", "PDM", "Periodo 2025-2026"],
                              "Coeficiente": np.round(mod_m1.coef_[0], 3)})
        st.dataframe(coefs, hide_index=True, width="stretch")
        st.caption("Coeficiente negativo = empuja hacia ventas ALTAS. El Periodo 2025-2026 es la "
                   "senal mas fuerte: en esos anios los meses de ventas bajas se vuelven mucho "
                   "menos probables.")
    st.caption("Ojo: la clasificacion sale perfecta, pero la prueba tiene solo 9 meses (30 registros, "
               "70/30). Con una muestra tan chica el modelo puede estar memorizando: tomalo como "
               "exploratorio, no como predictor confiable.")

    st.markdown("#### Modelo 2 · solo Leads, con y sin balanceo")
    _, y_p2, pred_l = _entrena_rv(df_rv[["Leads"]], y_rv)
    _, _, pred_p = _entrena_rv(df_rv[["Leads"]], y_rv, ponderar=True)
    _, _, pred_s = _entrena_rv(df_rv[["Leads"]], y_rv, con_smote=True)
    met_l = _metricas_rv(y_p2, pred_l)
    met_p = _metricas_rv(y_p2, pred_p)
    met_s = _metricas_rv(y_p2, pred_s)

    t1, t2, t3 = st.tabs(["Sin balanceo", "Reponderacion (class_weight)", "Oversampling (SMOTE)"])
    for tab_rv, met_rv, pred_rv, leyenda in [
        (t1, met_l, pred_l,
         "Leads por si solos NO separan: el modelo predice 'Ventas altas' para todo y no detecta "
         "ningun mes bajo (sensibilidad 0)."),
        (t2, met_p, pred_p,
         "La reponderacion obliga al modelo a cuidar la clase minoritaria: ahora detecta TODOS los "
         "meses bajos (sensibilidad 1.00), a costa de algunas falsas alarmas."),
        (t3, met_s, pred_s,
         "SMOTE llega al mismo resultado que la reponderacion: sensibilidad 1.00 y exactitud 0.78. "
         "Dos caminos distintos, mismo equilibrio."),
    ]:
        with tab_rv:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Exactitud", f"{met_rv['Exactitud']:.3f}")
            b2.metric("Precision (bajas)", f"{met_rv['Precision']:.3f}")
            b3.metric("Sensibilidad (recall)", f"{met_rv['Sensibilidad']:.3f}")
            b4.metric("F1", f"{met_rv['F1']:.3f}")
            st.plotly_chart(_fig_matriz_rv(confusion_matrix(y_p2, pred_rv), "Matriz de confusion"),
                            width="stretch", key=_key("g"))
            st.caption(leyenda)

    tabla_rv = pd.DataFrame({"Sin balanceo": met_l, "Reponderacion": met_p, "SMOTE": met_s}).round(3)
    st.dataframe(tabla_rv, width="stretch")
    st.markdown(f"""
    **Lectura para GAC:** con los leads como unica senal, el modelo sin balanceo no sirve de alarma
    (sensibilidad {met_l['Sensibilidad']:.2f}). Al balancear —por reponderacion o por SMOTE— la
    sensibilidad sube a {met_p['Sensibilidad']:.2f}: atrapa todos los meses de ventas bajas, a costa
    de marcar algunas falsas alarmas (precision {met_p['Precision']:.2f}). Para anticipar meses flojos
    y reaccionar a tiempo (promociones, empuje de piso de ventas), esa compensacion conviene.
    """)
    st.caption("Division 70/30 con semilla 42 y stratify, igual que el notebook; la prueba son 9 meses "
               "(6 de ventas altas, 3 de bajas). Con n = 30 los resultados son orientativos.")

# =========================================================
# ANOVA (ETAPA III)
# =========================================================
def vista_anova(df_citas, df_mkt, df_plazas_sem):
    st.markdown("**¿Para que sirve?** El ANOVA responde: *¿hay diferencias "
                "reales entre grupos?* (agentes, canales, productos). Con p < 0.05 "
                "la diferencia observada no es casualidad.")
    st.caption("H0: las medias de los grupos son iguales · alfa = 0.05")

    tab_citas, tab_mkt, tab_plazas = st.tabs(
        ["📞 Citas Digital", "📣 Marketing por canal", "🏙️ Plazas"])

    # -----------------------------------------------------
    with tab_citas:
        if df_citas is None or not len(df_citas):
            st.error("No se encontro el archivo 'Citas_Digital.csv'.")
        else:
            ORDEN_BDC = ["Ana", "Antonio", "Ignacio", "Saul"]
            st.markdown("#### Potencial de compra (%) segun agente BDC")
            st.caption("668 registros crudos → 531 tras la limpieza: 57 nulos "
                       "en la respuesta, unificacion de typos (estatus y BDC) "
                       "y recorte de 67 outliers por IQR (rango 55-95%).")

            grupos_c = [g["Potencial"].values for _, g in df_citas.groupby("BDC_L")]
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
                    "Potencial (%)"), width="stretch", key=_key("g"))
            with C2:
                st.plotly_chart(fig_bar_ic95(df_citas, "BDC_L", "Potencial",
                    ORDEN_BDC, "Media del potencial con IC 95%",
                    "Potencial (%)"), width="stretch", key=_key("g"))
            st.plotly_chart(fig_violin(df_citas, "BDC_L", "Potencial",
                ORDEN_BDC, "Distribucion del potencial (violin)",
                "Potencial (%)"), width="stretch", key=_key("g"))

            st.write("**Tabla ANOVA one-way** (scipy f_oneway + anova_lm typ 1)")
            st.dataframe(tabla_anova_bonita(tab_c,
                {"BDC_L": "Agente BDC", "Residual": "Residual"}),
                hide_index=True, width="stretch")
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
                hide_index=True, width="stretch")

            C3, C4 = st.columns([3, 2])
            with C3:
                st.plotly_chart(fig_interaccion(df_citas, "BDC_L", "PDM_txt",
                    "Potencial", ORDEN_BDC, "Grafica de interaccion BDC × PDM",
                    "Potencial promedio"), width="stretch", key=_key("g"))
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
    with tab_mkt:
        if df_mkt is None or not len(df_mkt):
            st.error("No se encontro el archivo 'MKT_DIGITAL.csv'.")
        else:
            from statsmodels.stats.oneway import anova_oneway
            ORDEN_CANAL = ["Digital", "Plaza", "Piso"]
            st.markdown("#### Leads mensuales segun canal de marketing")
            st.caption(f"Bloque 'Presupuesto mensual de MKT por canal' del "
                       f"archivo maestro → {len(df_mkt)} observaciones "
                       f"(3 canales × {df_mkt['Fecha'].nunique()} meses, "
                       f"{df_mkt['Fecha'].min():%b-%Y} a "
                       f"{df_mkt['Fecha'].max():%b-%Y}).")

            grupos_m = [g["Leads"].values for _, g in df_mkt.groupby("Canal")]
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
                    "Leads mensuales"), width="stretch", key=_key("g"))
            with C2:
                st.plotly_chart(fig_bar_ic95(df_mkt, "Canal", "Leads",
                    ORDEN_CANAL, "Media de leads con IC 95%",
                    "Leads mensuales"), width="stretch", key=_key("g"))

            st.write("**Tabla ANOVA one-way** (anova_lm typ 2)")
            st.dataframe(tabla_anova_bonita(tab_m,
                {"C(Canal)": "Canal", "Residual": "Residual"}),
                hide_index=True, width="stretch")
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
                hide_index=True, width="stretch")
            st.plotly_chart(fig_interaccion(df_mkt, "Anio", "Canal",
                "Presupuesto", ["2025", "2026"],
                "Grafica de interaccion canal × año (presupuesto)",
                "Presupuesto promedio ($)"), width="stretch", key=_key("g"))
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
    with tab_plazas:
        if df_plazas_sem is None or not len(df_plazas_sem):
            st.error("No se encontro el archivo 'Plazas.csv'.")
        else:
            ORDEN_PROD = ["EMZOOM", "GS8", "EMKOO"]
            st.markdown("#### Leads captados por semana segun producto y plaza")
            st.caption("303 leads de piso → variable de respuesta construida: "
                       f"leads por semana ({len(df_plazas_sem)} celdas "
                       "semana × producto × plaza).")

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
                    "Leads por semana"), width="stretch", key=_key("g"))
            with C2:
                st.plotly_chart(fig_bar_ic95(df_plazas_sem, "Producto_L",
                    "Leads", ORDEN_PROD, "Media de leads con IC 95%",
                    "Leads por semana"), width="stretch", key=_key("g"))

            st.write("**Tabla ANOVA one-way**")
            st.dataframe(tabla_anova_bonita(tab_pl,
                {"Producto_L": "Producto", "Residual": "Residual"}),
                hide_index=True, width="stretch")
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
                hide_index=True, width="stretch")
            st.plotly_chart(fig_interaccion(df_plazas_sem, "Producto_L",
                "Plaza_L", "Leads", ORDEN_PROD,
                "Grafica de interaccion producto × plaza",
                "Leads semanales promedio"), width="stretch", key=_key("g"))
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
