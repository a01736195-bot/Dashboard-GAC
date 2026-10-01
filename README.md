# Actividad 1 - Despliegue de Analisis Univariado (GAC)

Dashboard en Streamlit con el analisis univariado de las bases de la
concesionaria GAC (Bitacora de Piso y Leads Reales), con tema visual
corporativo GAC (azul y plata) y diseno plano.

## Contenido del repositorio

| Archivo | Descripcion |
|---|---|
| `app.py` | Aplicacion principal del dashboard (Streamlit) |
| `DASHBOARD_GAC.ipynb` | Notebook que genera `app.py` y lo corre |
| `Bitacora_Piso_Limpia.csv` | Base limpia de la bitacora de piso (nulos, outliers y transformaciones ya aplicadas) |
| `Leads_Reales_Limpia.csv` | Base limpia de leads reales mensual (nulos, outliers y transformaciones ya aplicadas) |
| `.streamlit/config.toml` | Tema vino tinto / borgona del dashboard |
| `logo_gac.png` | Logo de GAC Motor (se muestra en el menu lateral) |
| `requirements.txt` | Dependencias para el despliegue |

## Vistas del dashboard

1. **Extraccion de caracteristicas**: Pareto de visitas por asesor,
   dona de visitas con/sin prueba de manejo (PDM) y barras de
   efectividad de leads por mes (con cortes al 55% y 70%).
2. **Solicitudes de credito por asesor (SDC)**: Top 5 asesores con
   solicitudes de credito si/no.
3. **Visitas por mes**: conteo mensual de visitas registradas.
4. **Explorador de variables**: plantilla del curso con selector de base
   y variable categorica, filtro por periodo, metricas KPI, graficos de
   barras, pastel, dona y area, y tablas de frecuencias/datos filtrados.

## Como correrlo localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O bien abrir `DASHBOARD_GAC.ipynb` y ejecutar las celdas en orden.

## Graficas

Las graficas son interactivas (Plotly). Para descargar cualquier grafica
como **PNG**: pasa el cursor sobre la grafica y usa el icono de camara
📷 que aparece en la esquina superior derecha ("Download plot as png").

## Despliegue

El dashboard esta listo para Streamlit Community Cloud: conecta este
repositorio y usa `app.py` como archivo principal.
