# Dashboard GAC - Insights de Negocio

Dashboard de la concesionaria GAC (Plaza Angelopolis) con tema vino tinto /
borgona y diseno plano. Reestructurado en octubre 2026 para enfocarse en
**insights de negocio**: cada vista responde una pregunta concreta en lugar de
replicar las tecnicas estadisticas de los notebooks de origen.

## Las 4 vistas de negocio

| Vista | Pregunta que responde | Contenido |
|---|---|---|
| 1. Panorama Ejecutivo | ¿Como va la plaza? | KPIs (ventas, cumplimiento, leads/venta, efectividad), tendencia vs objetivo a 30 meses, meses criticos y 6 tarjetas de insight redactadas en lenguaje de negocio |
| 2. Embudo de Ventas | ¿Donde se fuga la venta? | Embudo mensual promedio, tasa de transicion por etapa vs benchmark (cuellos en rojo), conversion lead→venta y efectividad de leads en el tiempo |
| 3. Desempeno | ¿Quien y que rinde? | Asesores de piso, ejecutivos BDC (digital), marketing por canal (costo/venta) y productos (cumplimiento BCS) |
| 4. Laboratorio (evidencia) | ¿Donde quedo el proceso? | Indice de la evidencia estadistica archivada y nota de conciliacion entre fuentes |

## Evidencia del proceso (no se perdio nada)

Las 13 vistas originales (Etapas I-III: analisis univariado, correlaciones,
regresiones lineales y logisticas, despliegue ANOVA) se conservan intactas:

- `archivo/app_etapas_original.py` - dashboard completo original, listo para
  correr con `streamlit run archivo/app_etapas_original.py`
- `DASHBOARD_GAC.ipynb` - notebook que generaba y ejecutaba esa version
- Historial de git (`git log`) - todos los commits del proceso, incluido el
  commit previo a esta reestructuracion

## Archivos de datos

| Archivo | Descripcion |
|---|---|
| `Bitacora_Piso_Limpia.csv` | Base limpia de la bitacora de piso |
| `Leads_Reales_Limpia.csv` | Base limpia de leads reales mensual |
| `base_leads_limpia.csv` | Base de leads de plaza (detalle) |
| `base_resumen_limpio.csv` | Base resumen mensual (30 meses, 26 variables) |
| `funnel_mensual.csv` | Embudo mensual limpio (18 meses) |
| `MKT_DIGITAL.csv` | Marketing mensual por canal (presupuesto, leads, ventas) |
| `Citas_Digital.csv` | Seguimiento de leads del canal digital |
| `Plazas.csv` | Leads de piso por plaza y producto |
| `bcs_modelos.csv` | 6 modelos GAC (Objetivo vs Ventas reales) del Balance Scorecard |
| `logo_gac.png` | Logo de GAC Motor (menu lateral) |
| `.streamlit/config.toml` | Tema vino tinto / borgona |

## Como correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

Listo para Streamlit Community Cloud: conecta este repositorio y usa
`app.py` como archivo principal.
