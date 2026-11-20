# ¿Se puede anticipar el impacto de un artículo? · Entrega 4

Aplicación web del Proyecto Integrador de Ciencia de Datos (UTN FRM, 2026) · Grupo 10, comisión 5K09:
Bustelo, Cuello, Lemos, Masuet y Vazquez.

**Aplicación publicada:** https://jeanpaulmst.github.io/entrega4-impacto-citas/

Es una sola página HTML (`index.html`) con formato de artículo científico, que cuenta el proyecto como una copa de martini.
El índice lateral separa el tallo, que guía el autor, de la copa, que maneja el usuario:

| Parte | Sección | Qué muestra |
|---|---|---|
| Portada | El proyecto | La pregunta, la oración central, tres números (error ×2,0 en el número de citas; 65 % de aciertos de tercio; 49 % de precisión en candidatos a top 10 %) y el recorrido de las 4 entregas con el problema de cada una |
| Tallo | §1 Los datos | Distribución de citas por tramos y terciles, y cuánto se lleva cada tercil (Figura 1, apéndice A) |
| Tallo | §2 Lo que encontramos | Prestigio de la revista × referencias → % de top 10 % (Figura 2, apéndice B: H2 se sostuvo, H4 no) |
| Tallo | §3 El modelo | Tres preguntas, tres modelos (cuántas citas, en qué tercio, si es top 10 %) y por qué: comparación de familias, error por tramo, tercio predicho contra real y triage por deciles (Figuras 3–6, apéndice C) |
| Tallo | §4 Lo que logramos | Qué se pudo responder, qué no y por qué |
| Copa | S1 Explorá | Los 10.326 artículos con ejes elegibles (citas reales, esperadas, probabilidad, prestigio, referencias, autores), color, tendencia, búsqueda, 4 filtros, recuadro de selección, correlación entre ejes, ficha de cada artículo y ranking de los 15 más citados / más sorprendentes |
| Copa | S2 Probá el modelo | Deslizadores con las 12 columnas → probabilidad de top 10 % (logística elegida) y citas esperadas (Ridge) |

Todos los gráficos están hechos con **Altair** y se dibujan con Vega-Lite en el navegador.
La población es **un solo campo y un solo año**: artículos de Ciencia de Materiales publicados en revistas en 2019 (OpenAlex).

## Cómo se regenera

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn altair
python build_app.py --datos /ruta/a/ultimo_ok.csv.gz
```

- `--datos`: el CSV que produce el pipeline de Airflow del grupo (repositorio `airflow-openalex`, archivo `include/frozen/ultimo_ok.csv.gz`). No se incluye acá por su tamaño.
- `--notebook`: el notebook de la Entrega 3 (por defecto, `Entrega_3_Grupo_10.ipynb` de esta carpeta).
- `--salida`: dónde escribir la página (por defecto `index.html`).

`build_app.py` **ejecuta el notebook de la Entrega 3 celda por celda** (semilla 42) y toma de ahí la partición, los modelos y las tablas.
Por eso lo que muestra la aplicación coincide con el notebook, y si el pipeline produce datos nuevos, volver a correr el script actualiza la página.
Tarda unos 3 minutos.

## Cómo llega el modelo al navegador

- **Explorá:** las predicciones viajan con los datos. Los artículos de entrenamiento se predicen con validación cruzada (5 folds) y los del test con el modelo final, así que ningún artículo lo predice un modelo que lo vio.
- **Probá el modelo:** el clasificador elegido es una regresión logística (`TunedThresholdClassifierCV`, umbral 0,31). Su fórmula (ecuación 2 de la página) se escribe dentro del gráfico: `log(1 + x)`, imputación por mediana, estandarización y coeficientes, todo exportado del pipeline (`modelo_navegador.json`). El regresor elegido (Gradient Boosting) no se puede escribir en el navegador. Por eso en esa sección las citas esperadas salen de Ridge (MAE CV 0,709 contra 0,685 del Boosting), y la página lo rotula como aproximación.
- Los deslizadores van del mínimo del entrenamiento al percentil 99,5: no se pueden cargar valores que el modelo nunca vio.

## Archivos

| Archivo | Qué es |
|---|---|
| `index.html` | La aplicación generada (la que publica GitHub Pages) |
| `build_app.py` | Script de construcción: ejecuta el notebook, calcula predicciones, arma los gráficos Altair y la página |
| `plantilla.html` | Estructura, textos y estilos de la página, con marcadores `{{…}}` que completa el script |
| `Entrega_3_Grupo_10.ipynb` | Notebook de modelado de la Entrega 3 (con el Objetivo 3 en terciles) |
| `modelo_navegador.json` | Coeficientes y transformaciones exportados para «Probá el modelo» |

La página carga Vega, Vega-Lite y Vega-Embed desde jsDelivr con versiones fijas, y las fuentes desde Google Fonts, así que necesita conexión a internet.
