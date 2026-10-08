# Dashboard GAC - Analisis Univariado + Regresion Lineal

Dashboard unificado de las operaciones de la concesionaria GAC (Plaza
Angelopolis), con tema visual vino tinto / borgona y diseno plano.

## Contenido

**Etapa I. Modelado explicativo (Analisis comparativo)**: extraccion de
caracteristicas, analisis univariado de las variables categoricas mas
significativas.

**Etapa II. Modelado predictivo**: analisis de correlaciones y regresion
lineal simple y multiple, superponiendo en cada grafica la dispersion de
los datos predichos.

Cada regresion del equipo tiene su vista DEDICADA, con las mismas
variables, indicadores, ecuacion, tabla y grafica de su notebook de
origen (sin mezclar).

## Archivos del repositorio

| Archivo | Descripcion |
|---|---|
| `app.py` | Aplicacion principal del dashboard (Streamlit, 10 vistas) |
| `DASHBOARD_GAC.ipynb` | Notebook que genera `app.py` y lo corre |
| `Bitacora_Piso_Limpia.csv` | Base limpia de la bitacora de piso (nulos, outliers y transformaciones ya aplicadas) |
| `Leads_Reales_Limpia.csv` | Base limpia de leads reales mensual |
| `base_leads_limpia.csv` | Base de leads de plaza (detalle) con preprocesamiento aplicado |
| `base_resumen_limpio.csv` | Base resumen mensual (30 meses, 26 variables) para la Etapa II |
| `funnel_mensual.csv` | Embudo mensual limpio (18 meses) extraido del FUNNEL.csv original |
| `bcs_modelos.csv` | 6 modelos GAC (Objetivo vs Ventas reales) del Balance Scorecard |
| `logo_gac.png` | Logo de GAC Motor (menu lateral) |
| `.streamlit/config.toml` | Tema vino tinto / borgona |
| `requirements.txt` | Dependencias (incluye scikit-learn) |

## Vistas del dashboard y trazabilidad con los notebooks del equipo

| Vista | Que muestra | Notebook de origen |
|---|---|---|
| 1. Extraccion de Caracteristicas | Pareto de asesores, dona PDM, efectividad de leads | `Analisis_GAC_Limpieza_Graficos.ipynb` |
| 2. Solicitudes de Credito por Asesor | Top 5 asesores con SDC si/no | `DASHBOARD.IPYNB` |
| 3. Visitas por Mes | Conteo mensual de visitas | `DASHBOARD.IPYNB` |
| 4. Explorador de Variables | Barras/pastel/dona/area + KPIs + filtros | `Act1.ipynb` |
| 5. Leads Detalle (Plaza) | Heatmap de producto, pastel de estatus, KPIs | `codigos.ipynb` |
| 6. Analisis de Correlaciones | Heatmap de la matriz de correlacion | Etapa II (todos) |
| 7. Regresion Objetivo vs Ventas (BCS) | 6 modelos: intercepto 3.0546, pendiente 0.5153, R² 0.4746, r 0.6889 | `regresion_lineal_simple.ipynb` |
| 8. Regresion Ventas ~ Leads (Resumen) | 30 meses: X=Leads, Y=Ventas Real + tabla real/predicho/residual | `Regresion_Simple_Resumen.ipynb` |
| 9. Regresion Lineal Multiple (Resumen) | Selector Y + multiselect X (default: Efectivos, PDM, SDC) | `codigos (1).ipynb` |
| 10. Regresion del Embudo (FUNNEL) | Regresion 1 (R² 0.7053) y Regresion 2 (R² 0.7834) | `FUNNEL_REGRESION.IPYNB` |

## Como correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O bien abrir `DASHBOARD_GAC.ipynb` y ejecutar las celdas en orden.

## Despliegue

Listo para Streamlit Community Cloud: conecta este repositorio y usa
`app.py` como archivo principal.
