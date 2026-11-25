"""
build_app.py · Entrega 4 · Grupo 10 (5K09)

Genera la aplicación web (index.html) del proyecto «¿Se puede anticipar el impacto de un artículo?».

Cómo funciona:
  1. Ejecuta, celda por celda, el notebook de la Entrega 3 (Entrega_3_Grupo_10.ipynb) sobre el dataset
     que produce el pipeline de Airflow. Así los modelos, la partición y todos los números de la app
     son exactamente los del notebook (misma semilla, mismo código).
  2. Calcula las predicciones que viajan con los datos: cada artículo de entrenamiento lo predice un
     modelo que no lo vio (validación cruzada) y cada artículo del test lo predice el modelo final.
  3. Exporta los coeficientes de la Regresión Logística elegida (y de Ridge) para que la sección
     «Probá el modelo» calcule la predicción dentro del navegador.
  4. Arma los gráficos con Altair y escribe index.html.

Uso:
  python build_app.py --datos ../ruta/ultimo_ok.csv.gz
  (por defecto busca include/frozen/ultimo_ok.csv.gz del repositorio del pipeline)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import altair as alt
import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
NOTEBOOK_DEF = next((r for r in [AQUI / "Entrega_3_Grupo_10.ipynb", AQUI.parent / "Entrega_3_Grupo_10.ipynb"] if r.exists()),
                    AQUI / "Entrega_3_Grupo_10.ipynb")
DATOS_DEF = Path.home() / "Documents/Proyectos/airflow-openalex/include/frozen/ultimo_ok.csv.gz"

# Versiones exactas de las librerías de Vega que usa la página (compatibles con Altair 6 / Vega-Lite 6)
VEGA, VEGA_LITE, VEGA_EMBED = "6.4.0", "6.4.1", "7.1.0"

# ------------------------------------------------------------------
# Paleta: cada color significa lo mismo en toda la aplicación
# ------------------------------------------------------------------
COL_TERCIL = {"T1 · bajo": "#b9c7d6", "T2 · medio": "#5b86b0", "T3 · alto": "#1d3a5f"}
TERCILES = list(COL_TERCIL)
COL_TOP = "#c9552a"       # top 10 % (real o candidato)
COL_RESTO = "#a8a39b"     # no top 10 %
COL_PRED = "#c9552a"      # lo que dice el modelo
COL_REAL = "#1d3a5f"      # lo que pasó
TINTA, TENUE, FONDO = "#1c1b19", "#4a4740", "#fdfcf8"


def fmt(x, nd=2):
    """Número con coma decimal y punto de miles (es-AR)."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def pct(x, nd=1):
    return f"{fmt(100 * x, nd)}\u00a0%"


# ------------------------------------------------------------------
# 1. Ejecutar el notebook de la Entrega 3
# ------------------------------------------------------------------
def ejecutar_notebook(ruta_nb: Path, ruta_datos: Path) -> dict:
    nb = json.loads(ruta_nb.read_text(encoding="utf-8"))
    celdas = [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "__main__", "display": lambda *a, **k: None}
    trabajo = Path(tempfile.mkdtemp(prefix="entrega4_"))
    (trabajo / "ultimo_ok.csv.gz").symlink_to(ruta_datos.resolve())
    previo = os.getcwd()
    os.chdir(trabajo)
    try:
        import matplotlib.pyplot as plt
        plt.show = lambda *a, **k: plt.close("all")
        t0 = time.time()
        for i, codigo in enumerate(celdas, 1):
            exec(compile(codigo, f"<celda {i}>", "exec"), g)
            plt.close("all")
            print(f"  celda {i:2d}/{len(celdas)} ejecutada ({time.time() - t0:5.0f} s)", flush=True)
    finally:
        os.chdir(previo)
    return g


# ------------------------------------------------------------------
# 2. Datos para la aplicación
# ------------------------------------------------------------------
def a_tercil_txt(codigos):
    return pd.Series(codigos).map(dict(enumerate(TERCILES))).values


def tramo_citas(c):
    bordes = [-1, 0, 4, 9, 19, 49, 99, np.inf]
    etiq = ["0", "1–4", "5–9", "10–19", "20–49", "50–99", "100+"]
    return pd.cut(c, bordes, labels=etiq)


def preparar_datos(g):
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    df, X_train, X_test = g["df"], g["X_train"], g["X_test"]
    y_reg_train, y_clf_train = g["y_reg_train"], g["y_clf_train"]

    # Predicción de log(1 + citas): train fuera de muestra (CV), test con el modelo final
    pred_log = pd.Series(index=df.index, dtype=float)
    pred_log.loc[X_train.index] = cross_val_predict(clone(g["modelo_reg_final"]), X_train, y_reg_train, cv=g["cv_reg"], n_jobs=-1)
    pred_log.loc[X_test.index] = g["modelo_reg_final"].predict(X_test)

    # Probabilidad de top 10 % con la logística elegida (mismo criterio)
    logistica = g["candidatos"][g["NOMBRE_FINAL"]]
    proba = pd.Series(index=df.index, dtype=float)
    proba.loc[X_train.index] = cross_val_predict(clone(logistica), X_train, y_clf_train, cv=g["cv_clf"], method="predict_proba", n_jobs=-1)[:, 1]
    proba.loc[X_test.index] = g["modelo_clf_final"].predict_proba(X_test)[:, 1]

    cortes_tercil = g["modelo_q_final"][-1].cortes_          # cortes aprendidos en la Sección 4.4
    tercil_pred = np.digitize(pred_log.values, cortes_tercil, right=True)
    tercil_real = g["a_tercil"](df["cited_by_count"]).values

    datos = pd.DataFrame({
        "titulo": df["title"].fillna("(sin título)").str.replace("\ufffdC", "–", regex=False).str.replace("\ufffd", "", regex=False).str.slice(0, 110),
        "revista": df["primary_location_source_display_name"].fillna("(sin dato)").str.slice(0, 60),
        "subcampo": df["primary_topic_subfield_display_name"].fillna("(sin dato)"),
        "oa": df["oa_status"].astype(str),
        "core": df["source_is_core"].astype(bool),
        "citas": df["cited_by_count"].astype(int),
        "citas_pred": np.round(np.expm1(np.clip(pred_log, 0, None)), 1),
        "prob": proba.round(3),
        "candidato": proba >= g["UMBRAL"],
        "top10": df["is_top10_percent"].astype(bool),
        "tercil_real": a_tercil_txt(tercil_real),
        "tercil_pred": a_tercil_txt(tercil_pred),
        "prestigio": df["source_pre_cites_per_work"].round(1),
        "refs": df["referenced_works_count"].astype(int),
        "autores": df["authors_count"].astype(int),
        "particion": np.where(df.index.isin(X_test.index), "test", "entrenamiento"),
    }, index=df.index)
    return datos, cortes_tercil


def exportar_modelos(g):
    """Coeficientes y transformaciones del pipeline, para recalcular la predicción en el navegador."""
    X_train = g["X_train"]
    logistica = g["modelo_clf_final"].estimator_          # Pipeline ajustado dentro de TunedThresholdClassifierCV
    ridge = g["modelo_ridge_ref"]

    def leer(pipe):
        prep = pipe.named_steps["preparar"]
        num = prep.named_transformers_["numericas"]
        cat = prep.named_transformers_["categoricas"]
        nombres_num = list(prep.transformers_[0][2])
        coef = np.ravel(pipe.named_steps["modelo"].coef_)
        inter = float(np.ravel(pipe.named_steps["modelo"].intercept_)[0])
        feats = []
        for j, n in enumerate(nombres_num):
            feats.append({
                "nombre": n,
                "mediana": float(num.named_steps["imputar"].statistics_[j]),
                "media": float(num.named_steps["escalar"].mean_[j]),
                "escala": float(num.named_steps["escalar"].scale_[j]),
                "coef": float(coef[j]),
            })
        cats = list(cat.named_steps["codificar"].categories_[0])
        coef_cat = {c: float(coef[len(nombres_num) + k]) for k, c in enumerate(cats)}
        return {"num": feats, "oa": coef_cat, "intercepto": inter}

    pred_ridge_train = ridge.predict(X_train)
    rangos = {}
    # pasos finos: un artículo real del test se tiene que poder cargar exacto en los deslizadores
    for c, paso in [("source_pre_cites_per_work", 0.1), ("referenced_works_count", 1), ("funders_count", 1),
                    ("last_author_pre_cites", 1), ("first_author_pre_cites", 1), ("institution_max_h_index", 1),
                    ("institutions_count", 1), ("authors_count", 1)]:
        s = X_train[c].dropna()
        rangos[c] = {"min": float(s.min()), "max": float(np.ceil(s.quantile(0.995) / paso) * paso),
                     "max_visto": float(s.max()), "paso": paso, "mediana": float(s.median())}
    return {
        "logistica": leer(logistica),
        "ridge": leer(ridge),
        "umbral": float(g["UMBRAL"]),
        "cortes_tercil_ridge": [float(x) for x in np.quantile(pred_ridge_train, [1 / 3, 2 / 3])],
        "rangos": rangos,
        "oa_categorias": sorted(X_train["oa_status"].astype(str).unique().tolist()),
        "tasa_base": float(g["y_clf_train"].mean()),
    }


# ------------------------------------------------------------------
# 3. Gráficos (Altair)
# ------------------------------------------------------------------
def tema():
    return {
        "config": {
            "view": {"stroke": None},
            "font": "STIX Two Text, Times New Roman, Georgia, serif",
            "axis": {"labelColor": TINTA, "titleColor": TINTA, "gridColor": "#ebe6da", "domainColor": "#8a857a",
                     "tickColor": "#8a857a", "labelFontSize": 12.5, "titleFontSize": 13, "titleFontWeight": 500},
            "legend": {"labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": 600, "labelColor": TINTA},
            "header": {"labelFontSize": 12, "titleFontSize": 12},
            "title": {"fontSize": 14, "fontWeight": 600, "fontStyle": "normal", "anchor": "start", "color": TINTA, "subtitleColor": TENUE},
            "background": "transparent",
        }
    }


alt.themes.register("grupo10", tema)
alt.themes.enable("grupo10")
alt.data_transformers.disable_max_rows()

ESC_TERCIL = alt.Scale(domain=TERCILES, range=list(COL_TERCIL.values()))


def g_datos(datos):
    """Histograma de citas por tramos (alineados a los cortes de tercil) + cuánto se lleva cada tercil."""
    bordes = [-1, 0, 1, 3, 6, 10, 15, 23, 35, 55, 99, 179, 319, 599, np.inf]
    etiq = ["0", "1", "2–3", "4–6", "7–10", "11–15", "16–23", "24–35", "36–55", "56–99", "100–179", "180–319", "320–599", "600+"]
    d = datos.assign(tramo=pd.cut(datos["citas"], bordes, labels=etiq).astype(str))
    hist = d.groupby(["tramo", "tercil_real"], observed=True).size().reset_index(name="articulos")
    hist = hist[hist["articulos"] > 0]
    hist["pct"] = hist["articulos"] / len(d)

    barras = alt.Chart(hist).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("tramo:O", sort=etiq, title="citas recibidas (2019 – oct. 2026)", axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("articulos:Q", title="artículos"),
        color=alt.Color("tercil_real:N", scale=ESC_TERCIL, title="tercil de citas",
                        legend=alt.Legend(orient="top", direction="horizontal")),
        tooltip=[alt.Tooltip("tramo:O", title="citas"), alt.Tooltip("articulos:Q", title="artículos", format=",d"),
                 alt.Tooltip("pct:Q", title="% de la población", format=".1%"), alt.Tooltip("tercil_real:N", title="tercil")],
    ).properties(width=470, height=260, title=alt.Title("(a) Artículos por tramo de citas"))

    reparto = []
    for t in TERCILES:
        m = d["tercil_real"] == t
        reparto.append({"medida": "artículos", "tercil": t, "parte": m.mean()})
        reparto.append({"medida": "citas", "tercil": t, "parte": d.loc[m, "citas"].sum() / d["citas"].sum()})
    rep = pd.DataFrame(reparto)
    rep["orden"] = rep["tercil"].map({t: i for i, t in enumerate(TERCILES)})
    base = alt.Chart(rep).encode(
        x=alt.X("medida:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=13), sort=["artículos", "citas"]),
        y=alt.Y("parte:Q", stack="normalize", title="parte del total", axis=alt.Axis(format="%")),
        order=alt.Order("orden:Q"),
    )
    pila = base.mark_bar(width=56).encode(
        color=alt.Color("tercil:N", scale=ESC_TERCIL, legend=None),
        tooltip=[alt.Tooltip("medida:N"), alt.Tooltip("tercil:N"), alt.Tooltip("parte:Q", format=".1%", title="parte")],
    )
    texto = base.mark_text(color="white", fontWeight=600, fontSize=12).encode(
        text=alt.Text("parte:Q", format=".0%"),
        y=alt.Y("parte:Q", stack="normalize", bandPosition=0.5),
        opacity=alt.condition("datum.parte > 0.06", alt.value(1), alt.value(0)),
    )
    lado = (pila + texto).properties(width=170, height=260, title=alt.Title("(b) Parte de cada tercio"))
    return alt.hconcat(barras, lado, spacing=36).resolve_scale(color="shared")


def g_hallazgo(datos):
    """Figura 2: % de artículos en el tercio alto de citas según dos pistas del día de publicación.

    La revista (prestigio previo) y la bibliografía (cantidad de referencias) se parten en tercios de igual tamaño.
    El gráfico compara los extremos: ninguna pista alta, una sola o las dos. La grilla completa de 3 × 3 va al apéndice B.
    """
    d = datos.dropna(subset=["prestigio"])
    nivel_rev = pd.qcut(d["prestigio"].rank(method="first"), 3, labels=False)
    nivel_bib = pd.qcut(d["refs"].rank(method="first"), 3, labels=False)
    alto = d["tercil_real"] == "T3 · alto"
    grilla = pd.DataFrame({"rev": nivel_rev, "bib": nivel_bib, "alto": alto}).groupby(["rev", "bib"])["alto"].agg(["mean", "size"])
    base = alto.mean()

    lim_rev = d["prestigio"].quantile([1 / 3, 2 / 3]).values
    lim_bib = d["refs"].quantile([1 / 3, 2 / 3]).values
    r_lo, r_hi = fmt(lim_rev[0], 0), fmt(lim_rev[1], 0)
    b_lo, b_hi = fmt(lim_bib[0], 0), fmt(lim_bib[1], 0)
    grupos = [("Revista poco citada y bibliografía corta", f"revista: menos de {r_lo} citas por trabajo · bibliografía: menos de {b_lo} referencias", 0, 0),
              ("Bibliografía larga, revista poco citada", f"revista: menos de {r_lo} citas por trabajo · bibliografía: más de {b_hi} referencias", 0, 2),
              ("Revista muy citada, bibliografía corta", f"revista: más de {r_hi} citas por trabajo · bibliografía: menos de {b_lo} referencias", 2, 0),
              ("Revista muy citada y bibliografía larga", f"revista: más de {r_hi} citas por trabajo · bibliografía: más de {b_hi} referencias", 2, 2)]
    barras = pd.DataFrame([{"grupo": nom, "detalle": det, "t3": grilla.loc[(r, b), "mean"], "n": int(grilla.loc[(r, b), "size"])}
                           for nom, det, r, b in grupos])

    # Un solo lienzo en unidades 0–100: a la izquierda los tercios de la Figura 1, a la derecha los grupos,
    # y una flecha de cada grupo al tercio alto, más gruesa cuanto más artículos del grupo llegan.
    X0, LARGO = 44, 46                      # dónde empiezan las barras de los grupos y cuánto mide el 100 %
                                            # (sobra lugar a la derecha para el porcentaje, que va después de la pista)
    filas = [86, 62, 38, 14]                # centro vertical de cada grupo (de arriba abajo)
    destino = [93, 86, 79, 72]              # dónde llega cada flecha dentro del bloque del tercio alto
    barras["y"], barras["yt"] = filas, destino
    barras["x_ini"], barras["x_fin"], barras["x_tope"], barras["x_punta"] = X0, X0 + LARGO * barras["t3"], X0 + LARGO, 23.2
    barras["pct_txt"] = barras["t3"].map(lambda v: pct(v, 0))
    Xu = alt.X("x:Q", scale=alt.Scale(domain=[0, 100]), axis=None)
    Yu = alt.Y("y:Q", scale=alt.Scale(domain=[0, 108]), axis=None)
    tt = [alt.Tooltip("grupo:N"), alt.Tooltip("detalle:N", title="quiénes son"), alt.Tooltip("n:Q", title="artículos", format=",d"),
          alt.Tooltip("t3:Q", title="llegan al tercio alto", format=".1%")]

    # columna de los tercios (la misma partición y los mismos colores de la Figura 1)
    bloques = []
    for i, t in enumerate(TERCILES):
        c = datos.loc[datos["tercil_real"] == t, "citas"]
        rango = f"{fmt(c.min(), 0)} o más citas" if i == 2 else f"{fmt(c.min(), 0)} a {fmt(c.max(), 0)} citas"
        bloques.append({"tercil": t, "nombre": ["tercio bajo", "tercio medio", "tercio alto"][i], "rango": rango,
                        "x": 0, "x2": 22, "xm": 11, "y": 32 * i, "y2": 32 * i + 31, "yc": 32 * i + 15.5})
    bl = pd.DataFrame(bloques)
    col = alt.Chart(bl).mark_rect(cornerRadius=2).encode(
        x=Xu, x2="x2:Q", y=Yu, y2="y2:Q",
        color=alt.Color("tercil:N", scale=ESC_TERCIL, legend=None),
        tooltip=[alt.Tooltip("nombre:N", title="tercio"), alt.Tooltip("rango:N", title="citas")])
    col_nom = alt.Chart(bl).mark_text(fontSize=13, fontWeight=600, dy=-8).encode(
        x=alt.X("xm:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("yc:Q", scale=alt.Scale(domain=[0, 108]), axis=None), text="nombre:N",
        color=alt.condition("datum.tercil == 'T1 · bajo'", alt.value(TINTA), alt.value("white")))
    col_rango = alt.Chart(bl).mark_text(fontSize=11.5, dy=9).encode(
        x=alt.X("xm:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("yc:Q", scale=alt.Scale(domain=[0, 108]), axis=None), text="rango:N",
        color=alt.condition("datum.tercil == 'T1 · bajo'", alt.value(TINTA), alt.value("white")))

    # flechas: curva de la barra del grupo al bloque del tercio alto, con la punta horizontal
    tramos = []
    for _, r in barras.iterrows():
        for k, (x, y) in enumerate([(X0 - 1, r.y), (X0 - 9, r.y), (30, r.yt), (23.6, r.yt)]):
            tramos.append({"grupo": r.grupo, "k": k, "x": x, "y": y, "t3": r.t3})
    tr = pd.DataFrame(tramos)
    grosor = alt.Scale(domain=[0, 1], range=[1, 9])
    flechas = alt.Chart(tr).mark_line(interpolate="basis", color=COL_TERCIL["T3 · alto"], opacity=0.75).encode(
        x=Xu, y=Yu, detail="grupo:N", order="k:Q", strokeWidth=alt.StrokeWidth("t3:Q", scale=grosor, legend=None))
    puntas = alt.Chart(barras).mark_point(shape="triangle-left", filled=True, opacity=1, color=COL_TERCIL["T3 · alto"]).encode(
        x=alt.X("x_punta:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("yt:Q", scale=alt.Scale(domain=[0, 108]), axis=None),
        size=alt.Size("t3:Q", scale=alt.Scale(domain=[0, 1], range=[60, 260]), legend=None))

    # cada grupo: pista gris = los 100 artículos del grupo; parte pintada = los que llegan al tercio alto
    base_g = alt.Chart(barras)
    pista = base_g.mark_rect(color="#e9e5dc").encode(
        x=alt.X("x_ini:Q", scale=alt.Scale(domain=[0, 100]), axis=None), x2="x_tope:Q",
        y=alt.Y("y_lo:Q", scale=alt.Scale(domain=[0, 108]), axis=None), y2="y_hi:Q", tooltip=tt
    ).transform_calculate(y_lo="datum.y - 3", y_hi="datum.y + 3")
    lleno = base_g.mark_rect(color=COL_TERCIL["T3 · alto"]).encode(
        x=alt.X("x_ini:Q", scale=alt.Scale(domain=[0, 100]), axis=None), x2="x_fin:Q",
        y=alt.Y("y_lo:Q", scale=alt.Scale(domain=[0, 108]), axis=None), y2="y_hi:Q", tooltip=tt
    ).transform_calculate(y_lo="datum.y - 3", y_hi="datum.y + 3")
    # el porcentaje va al final de la pista gris, no de la parte pintada, para que la marca del promedio no lo tape
    valor = base_g.mark_text(align="left", dx=6, fontSize=15, fontWeight=700, color=TINTA).encode(
        x=alt.X("x_tope:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=Yu, text="pct_txt:N")
    nombre = base_g.mark_text(align="left", baseline="bottom", fontSize=13.5, fontWeight=600, color=TINTA).encode(
        x=alt.X("x_ini:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("y_nom:Q", scale=alt.Scale(domain=[0, 108]), axis=None), text="grupo:N"
    ).transform_calculate(y_nom="datum.y + 9.2")
    detalle = base_g.mark_text(align="left", baseline="bottom", fontSize=11.5, color="#6b665c").encode(
        x=alt.X("x_ini:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("y_det:Q", scale=alt.Scale(domain=[0, 108]), axis=None), text="detalle:N"
    ).transform_calculate(y_det="datum.y + 4.4")

    # promedio: por definición, un tercio de todos los artículos está en el tercio alto
    ref = pd.DataFrame({"x": [X0 + LARGO * base], "y": [0], "y2": [97], "yr": [5], "t": [f"promedio de todos: {pct(base, 0)}"]})
    marcas = pd.DataFrame({"x": X0 + LARGO * base, "y": [y - 4.5 for y in filas], "y2": [y + 4.5 for y in filas]})
    linea = alt.Chart(marcas).mark_rule(strokeDash=[3, 2], strokeWidth=1.5, color=TINTA).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("y:Q", scale=alt.Scale(domain=[0, 108]), axis=None), y2="y2:Q")
    rotulo = alt.Chart(ref).mark_text(align="left", dx=4, fontSize=11.5, fontStyle="italic", color=TENUE).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=alt.Y("yr:Q", scale=alt.Scale(domain=[0, 108]), axis=None), text="t:N")

    titulos = pd.DataFrame([{"x": 0, "y": 104, "t": "Los tercios de citas de la Figura 1"},
                            {"x": X0, "y": 104, "t": "De cada 100 artículos del grupo, cuántos llegan al tercio alto"}])
    tit = alt.Chart(titulos).mark_text(align="left", fontSize=12.5, fontStyle="italic", color=TENUE).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[0, 100]), axis=None), y=Yu, text="t:N")

    graf = (col + col_nom + col_rango + flechas + puntas + pista + lleno + valor + nombre + detalle + linea + rotulo + tit
            ).properties(width=660, height=330)

    # Grilla completa (apéndice B)
    rev_nom = ["poco citada", "media", "muy citada"]
    bib_nom = ["corta", "media", "larga"]
    tabla = pd.DataFrame({f"bibliografía {bib_nom[b]}": [f"{pct(grilla.loc[(r, b), 'mean'], 1)} ({fmt(grilla.loc[(r, b), 'size'], 0)})"
                                                           for r in range(3)] for b in range(3)},
                         index=pd.Index([f"revista {n}" for n in rev_nom], name="% en el tercio alto (artículos)"))
    textos = {
        "h_ninguna": pct(barras.t3[0], 0), "h_bib": pct(barras.t3[1], 0), "h_rev": pct(barras.t3[2], 0), "h_dos": pct(barras.t3[3], 0),
        "h_dos_no": pct(1 - barras.t3[3], 0), "h_base": pct(base, 0),
        "h_cortes_rev": " y ".join(fmt(v, 0) for v in lim_rev), "h_cortes_bib": " y ".join(fmt(v, 0) for v in lim_bib),
        "h_sin_rev": pct(1 - len(d) / len(datos), 0), "h_n_chico": fmt(min(barras.n[1], barras.n[2]), 0),
        "tabla_hallazgo": tabla_html(tabla),
    }
    return graf, textos


def g_tramos(g):
    tab = g["tabla_tramos_err"].reset_index()
    tab = tab.rename(columns={tab.columns[0]: "tramo"})
    tab["tramo"] = tab["tramo"].astype(str)
    largo = pd.concat([
        tab[["tramo", "articulos", "mediana_citas_real"]].rename(columns={"mediana_citas_real": "citas"}).assign(serie="real"),
        tab[["tramo", "articulos", "mediana_citas_pred"]].rename(columns={"mediana_citas_pred": "citas"}).assign(serie="predicha por el modelo"),
    ])
    orden = ["0", "1–4", "5–9", "10–19", "20–49", "50–99", "100+"]
    esc = alt.Scale(domain=["real", "predicha por el modelo"], range=[COL_REAL, COL_PRED])
    y = alt.Y("tramo:O", sort=orden, title="tramo de citas reales")
    x = alt.X("citas:Q", scale=alt.Scale(type="symlog"), title="mediana de citas (escala logarítmica)",
              axis=alt.Axis(values=[0, 1, 2, 5, 10, 20, 50, 100, 200]))
    linea = alt.Chart(tab).mark_rule(color="#c9c2b6", strokeWidth=3).encode(
        y=y, x=alt.X("mediana_citas_real:Q", scale=alt.Scale(type="symlog")), x2="mediana_citas_pred:Q")
    puntos = alt.Chart(largo).mark_circle(size=150, opacity=1).encode(
        y=y, x=x, color=alt.Color("serie:N", scale=esc, title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("tramo:O", title="tramo"), alt.Tooltip("serie:N", title=""),
                 alt.Tooltip("citas:Q", title="mediana de citas", format=".1f"), alt.Tooltip("articulos:Q", title="artículos del test")],
    )
    return (linea + puntos).properties(width=560, height=260)


def g_triage(g):
    p = g["proba_test_final"]
    y = g["y_clf_test"].values
    dec = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    tab = pd.DataFrame({"decil": dec + 1, "top": y, "p": p}).groupby("decil").agg(
        tasa=("top", "mean"), pmin=("p", "min"), pmax=("p", "max"), n=("top", "size")).reset_index()
    tab["sobre_umbral"] = np.where(tab["pmin"] >= g["UMBRAL"], "el modelo lo marca como candidato",
                                   np.where(tab["pmax"] >= g["UMBRAL"], "parte sobre el umbral", "debajo del umbral"))
    tab["decil_txt"] = tab["decil"].map(lambda d: f"D{d}")
    esc = alt.Scale(domain=["debajo del umbral", "parte sobre el umbral", "el modelo lo marca como candidato"],
                    range=[COL_RESTO, "#e3a586", COL_TOP])
    barras = alt.Chart(tab).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
        x=alt.X("decil_txt:O", sort=[f"D{i}" for i in range(1, 11)],
                title="artículos del test ordenados por probabilidad predicha (deciles, de menor a mayor)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("tasa:Q", title="% que realmente fue top 10 %", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 0.7])),
        color=alt.Color("sobre_umbral:N", scale=esc, title=None, legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("decil_txt:O", title="decil"), alt.Tooltip("tasa:Q", title="% top 10 % real", format=".1%"),
                 alt.Tooltip("pmin:Q", title="prob. desde", format=".2f"), alt.Tooltip("pmax:Q", title="prob. hasta", format=".2f"),
                 alt.Tooltip("n:Q", title="artículos")],
    )
    texto = alt.Chart(tab).mark_text(dy=-8, fontSize=11, color=TINTA).encode(
        x=alt.X("decil_txt:O", sort=[f"D{i}" for i in range(1, 11)]), y="tasa:Q", text=alt.Text("tasa:Q", format=".0%"))
    base = float(y.mean())
    regla = alt.Chart(pd.DataFrame({"y": [base], "txt": [f"al azar: {pct(base, 0)}"]})).mark_rule(
        strokeDash=[5, 4], color=TINTA).encode(y="y:Q")
    regla_txt = alt.Chart(pd.DataFrame({"y": [base], "txt": [f"al azar: {pct(base, 0)}"]})).mark_text(
        align="left", x=4, dy=-7, fontSize=11, color=TINTA).encode(y="y:Q", text="txt:N")
    return (barras + texto + regla + regla_txt).properties(width=620, height=270)


def g_comparacion(g):
    """Comparación de familias en CV para los dos objetivos (media ± desvío)."""
    r = g["tabla_reg_cv"].reset_index().rename(columns={"index": "modelo"})
    r = r[r["modelo"] != "piso (media)"]
    r = r.assign(lo=r["MAE cv"] - r["desvío MAE"], hi=r["MAE cv"] + r["desvío MAE"],
                 elegido=r["modelo"].eq("gradient boosting").map({True: "elegido", False: "otro"}))
    esc = alt.Scale(domain=["elegido", "otro", "piso"], range=[COL_PRED, COL_REAL, COL_RESTO])
    r.loc[r["modelo"].str.startswith("piso"), "elegido"] = "piso"
    orden = r.sort_values("MAE cv")["modelo"].tolist()
    base = alt.Chart(r).encode(y=alt.Y("modelo:N", sort=orden, title=None))
    reg = (base.mark_rule(strokeWidth=2, color=TENUE).encode(x=alt.X("lo:Q", title="MAE en log(1 + citas), CV 5 folds (menos es mejor)",
                                                                    scale=alt.Scale(zero=False)), x2="hi:Q")
           + base.mark_circle(size=110, opacity=1).encode(x="MAE cv:Q", color=alt.Color("elegido:N", scale=esc, legend=None),
                                                          tooltip=[alt.Tooltip("modelo:N"), alt.Tooltip("MAE cv:Q", format=".3f"),
                                                                   alt.Tooltip("desvío MAE:Q", format=".3f"), alt.Tooltip("MAE train:Q", format=".3f")])
           ).properties(width=300, height=170, title=alt.Title("(a) Regresión · MAE ± desvío"))
    td = g["tabla_desbalance"].reset_index().rename(columns={"index": "modelo"})
    col = [c for c in td.columns if "anidada" in c and "f1" in c][0]
    dcol = [c for c in td.columns if "desvío" in c and "anidada" in c][0]
    td = td.assign(f1=td[col], de=td[dcol])
    td["lo"], td["hi"] = td["f1"] - td["de"], td["f1"] + td["de"]
    td["elegido"] = np.where(td["modelo"] == g["NOMBRE_FINAL"], "elegido", "otro")
    piso = pd.DataFrame([{"modelo": "piso (siempre «no»)", "f1": 0.468, "de": 0, "lo": 0.468, "hi": 0.468, "elegido": "piso"}])
    piso["f1"] = g["tabla_clf_cv"].iloc[0]["f1_macro cv"]
    piso["lo"] = piso["hi"] = piso["f1"]
    td = pd.concat([td[["modelo", "f1", "de", "lo", "hi", "elegido"]], piso])
    orden2 = td.sort_values("f1", ascending=False)["modelo"].tolist()
    base2 = alt.Chart(td).encode(y=alt.Y("modelo:N", sort=orden2, title=None))
    clf = (base2.mark_rule(strokeWidth=2, color=TENUE).encode(x=alt.X("lo:Q", title="f1 macro con umbral ajustado, CV anidada (más es mejor)",
                                                                     scale=alt.Scale(zero=False)), x2="hi:Q")
           + base2.mark_circle(size=110, opacity=1).encode(x="f1:Q", color=alt.Color("elegido:N", scale=esc, legend=None),
                                                           tooltip=[alt.Tooltip("modelo:N"), alt.Tooltip("f1:Q", format=".3f", title="f1 macro"),
                                                                    alt.Tooltip("de:Q", format=".3f", title="desvío")])
           ).properties(width=300, height=170, title=alt.Title("(b) Clasificación top 10 % · f1 macro ± desvío"))
    return alt.hconcat(reg, clf, spacing=60)


def g_terciles(datos):
    """Objetivo 3 en el test: de los artículos a los que el modelo asignó cada tercio, en qué tercio terminaron."""
    t = datos[datos["particion"] == "test"]
    tab = t.groupby(["tercil_pred", "tercil_real"]).size().reset_index(name="articulos")
    tab["parte"] = tab["articulos"] / tab.groupby("tercil_pred")["articulos"].transform("sum")
    tab["orden"] = tab["tercil_real"].map({x: i for i, x in enumerate(TERCILES)})
    tab["dijo"] = "el modelo dijo " + tab["tercil_pred"]
    base = alt.Chart(tab).encode(
        y=alt.Y("dijo:N", sort=["el modelo dijo " + x for x in TERCILES], title=None, axis=alt.Axis(labelFontSize=13, labelLimit=320)),
        x=alt.X("parte:Q", stack="normalize", title="en qué tercio de citas terminaron (artículos del test)", axis=alt.Axis(format="%")),
        order=alt.Order("orden:Q"),
    )
    barras = base.mark_bar(height=34).encode(
        color=alt.Color("tercil_real:N", scale=ESC_TERCIL, title="tercil real", legend=alt.Legend(orient="top")),
        tooltip=[alt.Tooltip("tercil_pred:N", title="el modelo dijo"), alt.Tooltip("tercil_real:N", title="terminó en"),
                 alt.Tooltip("articulos:Q", title="artículos", format=",d"), alt.Tooltip("parte:Q", title="parte", format=".1%")],
    )
    texto = base.mark_text(fontSize=12.5, fontWeight=600).encode(
        x=alt.X("parte:Q", stack="normalize", bandPosition=0.5),
        text=alt.Text("parte:Q", format=".0%"),
        color=alt.condition("datum.tercil_real == 'T1 · bajo'", alt.value(TINTA), alt.value("white")),
        opacity=alt.condition("datum.parte > 0.05", alt.value(1), alt.value(0)),
    )
    return (barras + texto).properties(width=560, height=170)


# Variables que el usuario puede poner en cada eje de «Explorá»: campo -> (nombre, marcas del eje)
EJES = {
    "citas_pred": ("citas que esperaba el modelo", [0, 1, 2, 5, 10, 20, 50, 100, 200]),
    "citas": ("citas reales (2019 – oct. 2026)", [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]),
    "prob": ("probabilidad de top 10 % que dio el modelo", [0, 0.05, 0.1, 0.2, 0.31, 0.5, 0.75, 1]),
    "prestigio": ("prestigio de la revista (citas por trabajo 2012–2018)", [0, 1, 2, 5, 10, 20, 50, 100, 200]),
    "refs": ("referencias del artículo", [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]),
    "autores": ("autores del artículo", [1, 2, 3, 5, 10, 20, 30]),
}
ORDEN_RANKING = {
    "citas": "más citas reales",
    "citas_pred": "más citas esperadas por el modelo",
    "prob": "mayor probabilidad de top 10 %",
    "sorpresa": "más citas de las que esperaba el modelo",
    "decepcion": "menos citas de las que esperaba el modelo",
}


def g_explora(datos):
    subcampos = ["Todos"] + datos["subcampo"].value_counts().index.tolist()
    oas = ["Todos"] + sorted(datos["oa"].unique().tolist())
    nombres_ejes = [v[0] for v in EJES.values()]
    # qué se mira: ejes, color, tendencia (tocan el mapeo de x, y y color)
    p_x = alt.param(name="eje_x", value="citas_pred", bind=alt.binding_select(options=list(EJES), labels=nombres_ejes, name="Eje horizontal ", element="#c-x"))
    p_y = alt.param(name="eje_y", value="citas", bind=alt.binding_select(options=list(EJES), labels=nombres_ejes, name="Eje vertical ", element="#c-y"))
    p_col = alt.param(name="color_por", value="tercil_pred", bind=alt.binding_select(
        options=["tercil_pred", "tercil_real"], labels=["tercio que predijo el modelo", "tercio real"], name="Color ", element="#c-color"))
    p_tend = alt.param(name="ver_tend", value=False, bind=alt.binding_checkbox(name="Mostrar la tendencia (mediana) ", element="#c-tend"))
    p_orden = alt.param(name="orden_rk", value="citas", bind=alt.binding_select(
        options=list(ORDEN_RANKING), labels=list(ORDEN_RANKING.values()), name="Ranking: los 15 con ", element="#c-orden"))
    # qué artículos: filtros (achican el universo, tocan los datos de todos los paneles)
    p_busca = alt.param(name="f_busca", value="", bind=alt.binding(input="search", placeholder="palabra del título o de la revista",
                                                                   name="Buscar ", element="#c-busca"))
    p_sub = alt.param(name="f_subcampo", value="Todos", bind=alt.binding_select(options=subcampos, name="Subcampo ", element="#c-subcampo"))
    p_oa = alt.param(name="f_oa", value="Todos", bind=alt.binding_select(options=oas, name="Acceso abierto ", element="#c-oa"))
    p_part = alt.param(name="f_part", value="todos", bind=alt.binding_radio(options=["todos", "entrenamiento", "test"],
                                                                         name="Partición ", element="#c-part"))
    p_prob = alt.param(name="f_prob", value=0, bind=alt.binding_range(min=0, max=0.9, step=0.05,
                                                                       name="Probabilidad mínima de top 10 % ", element="#c-prob"))
    filtro = ("datum._s == 1 || ((f_subcampo == 'Todos' || datum.subcampo == f_subcampo) && (f_oa == 'Todos' || datum.oa == f_oa)"
              " && (f_part == 'todos' || datum.particion == f_part) && datum.prob >= f_prob"
              " && (f_busca == '' || indexof(lower(datum.titulo + ' ' + datum.revista), lower(f_busca)) >= 0)"
              # los artículos sin historial de revista no se pueden ubicar en un eje de prestigio: tampoco se cuentan
              " && ((eje_x != 'prestigio' && eje_y != 'prestigio') || isValid(datum.prestigio)))")
    brush = alt.selection_interval(name="zona", encodings=["x", "y"])
    elegido = alt.selection_point(name="elegido", fields=["titulo"], on="click", empty=False)

    # Una fila «centinela» (_s = 1) que pasa todos los filtros: garantiza que «En números» y el aviso de
    # «ningún artículo» tengan siempre una fila que agregar, aunque los filtros dejen vacío el universo.
    d = datos.reset_index(drop=True).assign(_s=0)
    centinela = {c: (False if d[c].dtype == bool else ("" if d[c].dtype == object else None)) for c in d.columns}
    d = pd.concat([d, pd.DataFrame([{**centinela, "_s": 1}])], ignore_index=True)
    tope_citas = float(max(datos["citas"].max(), datos["citas_pred"].max()))
    tooltip = [alt.Tooltip("titulo:N", title="artículo"), alt.Tooltip("revista:N"), alt.Tooltip("subcampo:N"),
               alt.Tooltip("citas:Q", title="citas reales"), alt.Tooltip("citas_pred:Q", title="citas que esperaba el modelo", format=".0f"),
               alt.Tooltip("prob:Q", title="probabilidad de top 10 %", format=".0%"),
               alt.Tooltip("top10:N", title="¿fue top 10 %?"), alt.Tooltip("tercil_pred:N", title="tercil predicho"),
               alt.Tooltip("tercil_real:N", title="tercil real"), alt.Tooltip("prestigio:Q", title="prestigio de la revista"),
               alt.Tooltip("refs:Q", title="referencias"), alt.Tooltip("autores:Q", title="autores"),
               alt.Tooltip("particion:N", title="partición")]
    # el valor de cada eje sale de la variable que eligió el usuario
    base_s = alt.Chart(d).transform_filter(filtro).transform_calculate(vx="datum[eje_x]", vy="datum[eje_y]", vc="datum[color_por]")
    base = base_s.transform_filter("datum._s == 0")          # lo que se dibuja: sin la centinela

    def marcas(eje):
        return alt.ExprRef("{" + ", ".join(f"{k}: {v[1]}" for k, v in EJES.items()) + "}[" + eje + "]")

    etiqueta = "eje_{0} == 'prob' ? format(datum.value, '.0%') : format(datum.value, ',~r')"
    X = alt.X("vx:Q", scale=alt.Scale(type="symlog"), title=None, axis=alt.Axis(values=marcas("eje_x"), labelExpr=etiqueta.format("x")))
    Y = alt.Y("vy:Q", scale=alt.Scale(type="symlog"), title=None, axis=alt.Axis(values=marcas("eje_y"), labelExpr=etiqueta.format("y")))
    W, H = 500, 420
    disp = base.mark_circle(size=16).encode(
        x=X, y=Y,
        color=alt.condition(brush, alt.Color("vc:N", scale=ESC_TERCIL, title="tercil de citas", legend=alt.Legend(orient="top")),
                            alt.value("#ddd8cf")),
        opacity=alt.condition(elegido, alt.value(1), alt.value(0.55)),
        size=alt.condition(elegido, alt.value(160), alt.value(16)),
        stroke=alt.condition("datum.top10", alt.value(COL_TOP), alt.value(None)),
        strokeWidth=alt.condition("datum.top10", alt.value(0.7), alt.value(0)),
        tooltip=tooltip,
    ).add_params(p_x, p_y, p_col, p_tend, p_orden, p_busca, p_sub, p_oa, p_part, p_prob, brush, elegido).properties(width=W, height=H)
    # diagonal «acertó»: sólo tiene sentido cuando los dos ejes son citas
    diag = alt.Chart(pd.DataFrame({"vx": [0, 2000], "vy": [0, 2000]})).transform_filter(
        "(eje_x == 'citas_pred' && eje_y == 'citas') || (eje_x == 'citas' && eje_y == 'citas_pred')"
    ).mark_line(color=TENUE, strokeDash=[4, 4]).encode(x=X, y=Y)
    # tendencia: mediana del eje vertical por tramo (en log) del horizontal
    tend = base.transform_filter("ver_tend && isValid(datum.vx) && isValid(datum.vy)").transform_calculate(
        lx="log(1 + datum.vx)").transform_bin("bx", "lx", bin=alt.Bin(maxbins=24)).transform_aggregate(
        mx="median(vx)", my="median(vy)", n="count()", groupby=["bx"]).transform_filter("datum.n >= 15").mark_line(
        color=TINTA, strokeWidth=2.5, point=alt.OverlayMarkDef(color=TINTA, size=28)).encode(
        x=alt.X("mx:Q", scale=alt.Scale(type="symlog")), y=alt.Y("my:Q", scale=alt.Scale(type="symlog")),
        tooltip=[alt.Tooltip("mx:Q", title="mediana del eje horizontal", format=",.2~f"),
                 alt.Tooltip("my:Q", title="mediana del eje vertical", format=",.2~f"), alt.Tooltip("n:Q", title="artículos en el tramo")])
    nombres = "{" + ", ".join(f"{k}: '{v[0]}'" for k, v in EJES.items()) + "}"
    rotulo = alt.Chart(pd.DataFrame({"a": [0]}))
    tit_x = rotulo.mark_text(fontSize=13, fontWeight=500, color=TINTA, text=alt.ExprRef(f"{nombres}[eje_x] + (eje_x == 'prob' ? '' : ' (escala log.)')")).encode(
        x=alt.value(W / 2), y=alt.value(H + 34))
    tit_y = rotulo.mark_text(fontSize=13, fontWeight=500, color=TINTA, angle=270, text=alt.ExprRef(f"{nombres}[eje_y] + (eje_y == 'prob' ? '' : ' (escala log.)')")).encode(
        x=alt.value(-46), y=alt.value(H / 2))

    resumen = base.transform_filter(brush).mark_bar().encode(
        y=alt.Y("tercil_pred:N", sort=TERCILES, title="tercil predicho"),
        x=alt.X("count():Q", stack="normalize", title="tercil real de esos artículos", axis=alt.Axis(format="%")),
        color=alt.Color("tercil_real:N", scale=ESC_TERCIL, title="tercil real", legend=None),
        order=alt.Order("tercil_real:N"),
        tooltip=[alt.Tooltip("tercil_pred:N", title="predicho"), alt.Tooltip("tercil_real:N", title="real"),
                 alt.Tooltip("count():Q", title="artículos")],
    ).properties(width=300, height=150, title=alt.Title("Lo predicho contra lo real", subtitle="artículos visibles (o dentro del recuadro)"))

    # contadores y correlación (en log) entre los dos ejes elegidos, sobre lo visible
    tabla = base_s.transform_filter({"or": [{"param": "zona"}, "datum._s == 1"]}).transform_calculate(
        real="1 - datum._s", top10_n="datum.top10 ? 1 : 0", cand_n="datum.candidato ? 1 : 0", tp_n="(datum.candidato && datum.top10) ? 1 : 0",
        ok="datum._s == 0 && isValid(datum.vx) && isValid(datum.vy) ? 1 : 0",
    ).transform_calculate(
        lx="datum.ok ? log(1 + datum.vx) : 0", ly="datum.ok ? log(1 + datum.vy) : 0",
    ).transform_calculate(lxx="datum.lx * datum.lx", lyy="datum.ly * datum.ly", lxy="datum.lx * datum.ly").transform_aggregate(
        n="sum(real)", top="sum(top10_n)", cand="sum(cand_n)", tp="sum(tp_n)",
        m="sum(ok)", sx="sum(lx)", sy="sum(ly)", sxx="sum(lxx)", syy="sum(lyy)", sxy="sum(lxy)",
    ).transform_calculate(
        r="(datum.m * datum.sxy - datum.sx * datum.sy) / sqrt((datum.m * datum.sxx - datum.sx * datum.sx) * (datum.m * datum.syy - datum.sy * datum.sy))",
    ).transform_calculate(
        t_n="format(datum.n, ',d')", t_top="format(datum.top, ',d')", t_cand="format(datum.cand, ',d')", t_tp="format(datum.tp, ',d')",
        t_r="isFinite(datum.r) ? format(datum.r, '.2f') : '—'",
    ).transform_fold(["t_n", "t_top", "t_cand", "t_tp", "t_r"], as_=["k", "v"]).transform_calculate(
        etiqueta="{'t_n':'artículos visibles','t_top':'top 10 % reales','t_cand':'candidatos del modelo',"
                 "'t_tp':'candidatos que acertaron','t_r':'correlación entre los ejes (en log)'}[datum.k]",
        orden="indexof(['t_n','t_top','t_cand','t_tp','t_r'], datum.k)",
    )
    txt_k = tabla.mark_text(align="left", fontSize=12, color=TENUE).encode(
        x=alt.value(0), y=alt.Y("orden:O", axis=None), text="etiqueta:N")
    txt_v = tabla.mark_text(align="right", fontSize=13, fontWeight=600, color=TINTA).encode(
        x=alt.value(300), y=alt.Y("orden:O", axis=None), text="v:N")
    cuadro = (txt_k + txt_v).properties(width=300, height=125, title=alt.Title("En números"))

    detalle = base.transform_filter(elegido).transform_calculate(
        filas="[{k:'artículo', v: datum.titulo}, {k:'revista', v: datum.revista}, {k:'citas reales', v: format(datum.citas, ',d')},"
              " {k:'citas esperadas', v: format(datum.citas_pred, ',.0f')}, {k:'prob. top 10 %', v: format(datum.prob, '.0%') + (datum.candidato ? ' → candidato' : ' → no')},"
              " {k:'¿fue top 10 %?', v: datum.top10 ? 'sí' : 'no'}, {k:'tercil', v: 'predicho ' + datum.tercil_pred + ' · real ' + datum.tercil_real},"
              " {k:'revista · refs.', v: (isValid(datum.prestigio) ? format(datum.prestigio, ',.1f') + ' citas/trabajo' : 'sin historial') + ' · ' + datum.refs + ' referencias'}]"
    ).transform_flatten(["filas"]).transform_calculate(k="datum.filas.k", v="datum.filas.v").transform_window(fila="row_number()")
    det_k = detalle.mark_text(align="left", fontSize=11, color=TENUE).encode(x=alt.value(0), y=alt.Y("fila:O", axis=None), text="k:N")
    det_v = detalle.mark_text(align="left", fontSize=11, color=TINTA, limit=200).encode(x=alt.value(100), y=alt.Y("fila:O", axis=None), text="v:N")
    ficha = (det_k + det_v).properties(width=300, height=150, title=alt.Title("Artículo elegido", subtitle="hacé clic en un punto"))

    # ranking: los 15 artículos visibles que más se destacan según el criterio elegido
    rk = base.transform_filter(brush).transform_calculate(
        sorpresa="log((1 + datum.citas) / (1 + datum.citas_pred))").transform_calculate(
        vo="orden_rk == 'decepcion' ? -datum.sorpresa : datum[orden_rk]").transform_filter("isValid(datum.vo)").transform_window(
        puesto="row_number()", sort=[alt.SortField("vo", order="descending")]).transform_filter("datum.puesto <= 15")
    # los títulos van en una columna de texto aparte (sin eje), para no correr el origen del panorama de arriba
    Yrk = alt.Y("titulo:N", sort=alt.EncodingSortField("puesto", order="ascending"), title=None, axis=None)
    Xrk = alt.X("citas:Q", scale=alt.Scale(type="symlog", domain=[0, tope_citas]), title="citas (escala log.): barra = reales, punto naranja = las que esperaba el modelo",
                axis=alt.Axis(values=[0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]))
    tt_rk = [alt.Tooltip("puesto:Q"), alt.Tooltip("titulo:N", title="artículo"), alt.Tooltip("revista:N"),
             alt.Tooltip("citas:Q", title="citas reales"), alt.Tooltip("citas_pred:Q", title="citas que esperaba el modelo", format=".0f"),
             alt.Tooltip("prob:Q", title="probabilidad de top 10 %", format=".0%"), alt.Tooltip("top10:N", title="¿fue top 10 %?")]
    rk_titulos = rk.transform_calculate(rotulo="datum.puesto + '. ' + datum.titulo").mark_text(
        align="left", fontSize=11.5, color=TINTA, limit=380).encode(
        y=Yrk, x=alt.value(0), text="rotulo:N", tooltip=tt_rk,
        color=alt.condition(elegido, alt.value("#1f4e8c"), alt.value(TINTA))).properties(
        width=380, height=alt.Step(17),
        title=alt.Title(alt.ExprRef("'Los 15 artículos visibles con ' + " + "{" + ", ".join(f"{k}: '{v}'" for k, v in ORDEN_RANKING.items()) + "}[orden_rk]"),
                        subtitle="cambiá el criterio arriba; respeta los filtros y el recuadro"))
    rk_barras = rk.mark_bar(height=13, color=COL_REAL).encode(
        y=Yrk, x=Xrk, opacity=alt.condition(elegido, alt.value(1), alt.value(0.75)), tooltip=tt_rk)
    rk_pred = rk.mark_point(shape="circle", filled=True, size=70, color=COL_PRED, opacity=1).encode(
        y=Yrk, x=alt.X("citas_pred:Q", scale=alt.Scale(type="symlog", domain=[0, tope_citas])), tooltip=tt_rk)
    ranking = alt.hconcat(rk_titulos, (rk_barras + rk_pred).properties(width=440, height=alt.Step(17)), spacing=10)

    vacio = base_s.transform_calculate(real="1 - datum._s").transform_aggregate(n="sum(real)").transform_filter("datum.n == 0").mark_text(
        fontSize=15, fontWeight=600, color=TENUE, text="Ningún artículo cumple estos filtros: probá aflojar alguno").encode(
        x=alt.value(W / 2), y=alt.value(H / 2))
    izquierda = disp + diag + tend + tit_x + tit_y + vacio
    derecha = alt.vconcat(resumen, cuadro, ficha, spacing=22)
    return alt.vconcat(alt.hconcat(izquierda, derecha, spacing=36).resolve_scale(color="independent"), ranking,
                       spacing=36).resolve_scale(color="independent")


def g_proba(m):
    """Probá el modelo: la predicción se calcula dentro del gráfico con los coeficientes exportados."""
    R = m["rangos"]
    nombres = {
        "source_pre_cites_per_work": "Citas por trabajo de la revista (2012–2018)",
        "referenced_works_count": "Referencias del artículo",
        "authors_count": "Autores",
        "institutions_count": "Instituciones",
        "institution_max_h_index": "Índice h de la mejor institución",
        "last_author_pre_cites": "Citas previas del último autor",
        "first_author_pre_cites": "Citas previas del primer autor",
        "funders_count": "Financiadores",
    }
    params = {}
    for c, etiqueta in nombres.items():
        r = R[c]
        params[c] = alt.param(name=f"p_{c}", value=round(round(r["mediana"] / r["paso"]) * r["paso"], 6),
                              bind=alt.binding_range(min=r["min"], max=r["max"], step=r["paso"], name=etiqueta + " ",
                                                     element=f"#d-{c}"))
    params["oa"] = alt.param(name="p_oa", value="closed", bind=alt.binding_select(options=m["oa_categorias"], name="Acceso abierto ", element="#d-oa"))
    params["core"] = alt.param(name="p_core", value=True, bind=alt.binding_checkbox(name="Revista core de OpenAlex ", element="#d-core"))
    params["pubmed"] = alt.param(name="p_pubmed", value=False, bind=alt.binding_checkbox(name="Indexado en PubMed ", element="#d-pubmed"))
    params["sinrev"] = alt.param(name="p_sinrev", value=False, bind=alt.binding_checkbox(name="La revista no tiene historial 2012–2018 ", element="#d-sinrev"))

    def valor(nombre_feat, md):
        """Expresión Vega del valor crudo (antes de estandarizar) de cada columna del modelo."""
        med = md["mediana"]
        if nombre_feat == "log_source_pre_cites_per_work":
            return f"(p_sinrev ? {med} : log(1 + p_source_pre_cites_per_work))"
        if nombre_feat == "log_institution_max_h_index":
            return f"(p_institutions_count == 0 ? {med} : log(1 + p_institution_max_h_index))"
        if nombre_feat.startswith("log_"):
            return f"log(1 + p_{nombre_feat[4:]})"
        return {
            "sin_dato_revista": "(p_sinrev ? 1 : 0)",
            "sin_dato_ultimo_autor": "0",
            "sin_dato_primer_autor": "0",
            "sin_instituciones": "(p_institutions_count == 0 ? 1 : 0)",
            "sin_referencias": "(p_referenced_works_count == 0 ? 1 : 0)",
            "source_is_core": "(p_core ? 1 : 0)",
            "in_pubmed": "(p_pubmed ? 1 : 0)",
        }[nombre_feat]

    def lineal(mod):
        partes = [f"{mod['intercepto']:.6f}"]
        for f in mod["num"]:
            partes.append(f"({f['coef']:.6f}) * (({valor(f['nombre'], f)}) - ({f['media']:.6f})) / ({f['escala']:.6f})")
        partes.append("(" + " + ".join(f"(p_oa == '{c}' ? {v:.6f} : 0)" for c, v in mod["oa"].items()) + ")")
        return " + ".join(partes)

    z_log = lineal(m["logistica"])
    z_ridge = lineal(m["ridge"])
    c1, c2 = m["cortes_tercil_ridge"]
    U = m["umbral"]
    base = alt.Chart(pd.DataFrame({"x": [0]})).transform_calculate(
        z=z_log).transform_calculate(prob="1 / (1 + exp(-datum.z))").transform_calculate(
        logc=z_ridge).transform_calculate(
        citas="max(0, exp(datum.logc) - 1)",
        tercil=f"datum.logc <= {c1} ? 'T1 · bajo' : (datum.logc <= {c2} ? 'T2 · medio' : 'T3 · alto')",
        veredicto=f"datum.prob >= {U} ? 'Candidato a top 10 %' : 'No llega al umbral de candidato'",
        t_prob=f"abs(datum.prob - {U}) < 0.005 ? format(datum.prob, '.2%') : format(datum.prob, '.0%')",
        fuera="(p_source_pre_cites_per_work > " + str(R["source_pre_cites_per_work"]["max_visto"]) + ") ? 1 : 0",
    )
    fondo = alt.Chart(pd.DataFrame({"a": [0], "b": [1]})).mark_bar(color="#ece8e0", height=34).encode(
        x=alt.X("a:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="probabilidad de ser top 10 %", values=[0, 0.12, U, 0.5, 0.75, 1])),
        x2="b:Q")
    barra = base.mark_bar(height=34).encode(
        x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1])), x2=alt.datum(0),
        color=alt.condition(f"datum.prob >= {U}", alt.value(COL_TOP), alt.value(COL_RESTO)),
    )
    regla_u = alt.Chart(pd.DataFrame({"u": [U]})).mark_rule(color=TINTA, strokeWidth=2).encode(x="u:Q")
    txt_u = alt.Chart(pd.DataFrame({"u": [U], "t": [f"umbral {fmt(U, 2)}"]})).mark_text(dy=-26, fontSize=11, color=TINTA).encode(x="u:Q", text="t:N")
    regla_b = alt.Chart(pd.DataFrame({"u": [m["tasa_base"]]})).mark_rule(color=TENUE, strokeDash=[3, 3]).encode(x="u:Q")
    txt_b = alt.Chart(pd.DataFrame({"u": [m["tasa_base"]], "t": ["al azar"]})).mark_text(dy=26, fontSize=11, color=TENUE).encode(x="u:Q", text="t:N")
    panel_prob = (fondo + barra + regla_u + txt_u + regla_b + txt_b).properties(width=440, height=90)

    num = base.mark_text(align="left", baseline="middle", fontSize=44, fontWeight=600).encode(
        x=alt.value(0), y=alt.value(30), text="t_prob:N",
        color=alt.condition(f"datum.prob >= {U}", alt.value(COL_TOP), alt.value(TINTA)))
    verd = base.mark_text(align="left", baseline="middle", fontSize=19, fontWeight=600).encode(
        x=alt.value(118), y=alt.value(18), text="veredicto:N",
        color=alt.condition(f"datum.prob >= {U}", alt.value(COL_TOP), alt.value(TENUE)))
    citas_l = base.transform_calculate(t="'Citas esperadas a ~7 años: ' + format(datum.citas, '.0f') + ' · tercil ' + datum.tercil").mark_text(
        align="left", baseline="middle", fontSize=14.5, color=TINTA).encode(x=alt.value(118), y=alt.value(44), text="t:N")
    nota_l = base.mark_text(align="left", baseline="middle", fontSize=12, fontStyle="italic", color=TENUE).encode(
        x=alt.value(118), y=alt.value(62), text=alt.value("(citas: aproximación lineal con Ridge)"))
    panel_txt = (num + verd + citas_l + nota_l).properties(width=440, height=72)

    graf = alt.vconcat(panel_txt, panel_prob, spacing=10)
    return graf.add_params(*params.values())


# ------------------------------------------------------------------
# 4. Textos con números (salen del notebook ejecutado)
# ------------------------------------------------------------------
def tabla_html(df, formatos=None, indice=True):
    formatos = formatos or {}
    cab = ([df.index.name or ""] if indice else []) + list(df.columns)
    filas = []
    for idx, fila in df.iterrows():
        celdas = [str(idx)] if indice else []
        for c in df.columns:
            v = fila[c]
            f = formatos.get(c)
            if f is not None and isinstance(v, (int, float, np.floating, np.integer)) and not pd.isna(v):
                celdas.append(f(v))
            else:
                celdas.append("" if pd.isna(v) else str(v))
        filas.append("<tr>" + "".join(f"<td>{x}</td>" for x in celdas) + "</tr>")
    return ('<div class="tabla"><table><thead><tr>' + "".join(f"<th>{c}</th>" for c in cab) + "</tr></thead><tbody>"
            + "".join(filas) + "</tbody></table></div>")


def calcular_textos(g, datos, m):
    from scipy.stats import spearmanr
    from sklearn.metrics import confusion_matrix

    df, dfc = g["df"], g["df_crudo"]
    T = {}
    # Embudo de población
    paso1 = dfc["source_type"] == "journal"
    paso2 = paso1 & (dfc["authors_count"] > 0)
    paso3 = paso2 & dfc["is_top10_percent"].notna()
    ed = dfc["doi"].str.contains("/ange.", regex=False, na=False)
    paso4 = paso3 & ~ed
    embudo = pd.DataFrame([
        ("Capa Plata del pipeline", len(dfc), "Artículos de Ciencia de Materiales de 2019 traídos de la API de OpenAlex y enriquecidos."),
        ("Publicados en revistas", int(paso1.sum()), "Se sacan repositorios, conferencias y libros: la pregunta es sobre artículos de revista."),
        ("Con al menos un autor", int(paso2.sum()), "Sin autores no hay trayectoria previa que medir."),
        ("Con is_top10_percent definido", int(paso3.sum()), "Hace falta la etiqueta del Objetivo 2."),
        ("Sin la edición alemana de Angewandte Chemie", int(paso4.sum()), "Es una traducción del mismo artículo («/ange.»): duplicaría casos."),
        ("Sin títulos repetidos (población final)", len(df), "Se queda la versión más citada de cada título."),
        ("Entrenamiento (80 %)", len(g["X_train"]), "Acá se elige todo, con validación cruzada de 5 folds."),
        ("Test (20 %, estratificado)", len(g["X_test"]), "Se separa antes de mirar nada y se usa una sola vez."),
    ], columns=["Paso", "Artículos", "Por qué"]).set_index("Paso")
    T["tabla_embudo"] = tabla_html(embudo, {"Artículos": lambda v: fmt(v, 0)})

    c = df["cited_by_count"]
    T["n"] = fmt(len(df), 0)
    T["n_revistas"] = fmt(df["primary_location_source_id"].nunique(), 0)
    T["mediana"] = fmt(c.median(), 0)
    T["max"] = fmt(c.max(), 0)
    T["cero"] = pct((c == 0).mean())
    k = int(np.ceil(0.01 * len(c)))
    T["top1"] = pct(c.sort_values(ascending=False).iloc[:k].sum() / c.sum(), 0)
    t3 = datos["tercil_real"] == "T3 · alto"
    T["t3_citas"] = pct(datos.loc[t3, "citas"].sum() / c.sum(), 0)
    T["t1_citas"] = pct(datos.loc[datos["tercil_real"] == "T1 · bajo", "citas"].sum() / c.sum(), 1)
    T["cortes"] = " y ".join(fmt(x, 0) for x in g["CORTES_Q"])
    T["top10"] = pct(df["is_top10_percent"].mean())
    # Por qué la tasa de top 10 % no es 10 %: la etiqueta es de OpenAlex (todo el subcampo y año)
    top_crudo = dfc["is_top10_percent"].astype(float)
    T["top10_crudo"] = pct(top_crudo.mean())
    T["n_fuera"] = fmt(int((~paso1).sum()), 0)
    T["top10_fuera"] = pct(top_crudo[~paso1].mean())

    # Hallazgos
    T["rho_prest"] = fmt(spearmanr(df["source_pre_cites_per_work"], c, nan_policy="omit").statistic, 2)
    T["rho_refs"] = fmt(spearmanr(df["referenced_works_count"], c).statistic, 2)
    T["rho_mes"] = fmt(spearmanr(df["publication_month"], c, nan_policy="omit").statistic, 3)
    y = np.log1p(c)

    def eta2(grupo):
        m_ = y.groupby(grupo).transform("mean")
        return ((m_ - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    T["eta_oa"] = fmt(eta2(df["oa_status"].astype(str)), 3)
    T["eta_oa_core"] = fmt(eta2(df["oa_status"].astype(str) + "_" + df["source_is_core"].astype(str)), 3)
    tasa = df.groupby(["source_is_core", df["oa_status"].astype(str)])["is_top10_percent"].mean()
    T["core_top"] = pct(df.loc[df["source_is_core"], "is_top10_percent"].mean())
    T["nocore_top"] = pct(df.loc[~df["source_is_core"], "is_top10_percent"].mean())
    mes = df.groupby("publication_month")["cited_by_count"].median()
    T["mes_ene"], T["mes_dic"] = fmt(mes.get(1, np.nan), 0), fmt(mes.get(12, np.nan), 0)

    # Modelo · regresión
    rt = g["resumen_test_reg"]
    T["r2"] = fmt(rt.loc["Gradient Boosting (Elegido)", "R² (Test)"], 2)
    T["r2_pct"] = pct(rt.loc["Gradient Boosting (Elegido)", "R² (Test)"], 0)
    T["mae_test"] = fmt(rt.loc["Gradient Boosting (Elegido)", "MAE log (Test)"], 3)
    T["mae_piso"] = fmt(rt.loc[rt.index[0], "MAE log (Test)"], 3)
    T["mae_cv"] = fmt(rt.loc["Gradient Boosting (Elegido)", "MAE log (CV)"], 3)
    T["mae_ridge_cv"] = fmt(rt.loc["Regresión Lineal (Ridge)", "MAE log (CV)"], 3)
    T["mae_ridge_test"] = fmt(rt.loc["Regresión Lineal (Ridge)", "MAE log (Test)"], 3)
    T["err_med"] = fmt(rt.loc["Gradient Boosting (Elegido)", "Error mediano en citas reales"], 1)
    T["err_med_piso"] = fmt(rt.loc[rt.index[0], "Error mediano en citas reales"], 1)
    tt = g["tabla_tramos_err"]
    T["hit_real"] = fmt(tt.loc["100+", "mediana_citas_real"], 0)
    T["hit_pred"] = fmt(tt.loc["100+", "mediana_citas_pred"], 0)
    T["t1019_real"] = fmt(tt.loc["10–19", "mediana_citas_real"], 0)
    T["t1019_pred"] = fmt(tt.loc["10–19", "mediana_citas_pred"], 1)
    T["cero_pred"] = fmt(tt.loc["0", "mediana_citas_pred"], 1)
    grid = g["busquedas_reg"]["gradient boosting"]
    T["gb_params"] = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in grid.best_params_.items())
    T["desv_gb"] = fmt(grid.cv_results_["std_test_score"][grid.best_index_], 3)

    # Modelo · clasificación
    tf = g["tabla_final_clf"]
    fila_el = [i for i in tf.index if "Elegido" in i][0]
    fila_otro = [i for i in tf.index if "referencia" in i][0]
    T["f1_test"] = fmt(tf.loc[fila_el, "f1_macro (Test)"], 3)
    T["f1_piso"] = fmt(tf.loc["Piso (siempre False)", "f1_macro (Test)"], 3)
    T["prec"] = pct(tf.loc[fila_el, "precision top 10%"], 0)
    T["rec"] = pct(tf.loc[fila_el, "recall top 10%"], 0)
    T["lift"] = fmt(tf.loc[fila_el, "precision top 10%"] / g["y_clf_test"].mean(), 1)
    T["f1_otro"] = fmt(tf.loc[fila_otro, "f1_macro (Test)"], 3)
    T["umbral"] = fmt(g["UMBRAL"], 2)
    td = g["tabla_desbalance"]
    col = [c_ for c_ in td.columns if "anidada" in c_ and "f1" in c_][0]
    dcol = [c_ for c_ in td.columns if "desvío" in c_ and "anidada" in c_][0]
    T["f1_log_cv"] = fmt(td.loc["Logística sin balancear", col], 3)
    T["f1_log_de"] = fmt(td.loc["Logística sin balancear", dcol], 3)
    T["f1_gb_cv"] = fmt(td.loc["Boosting sin balancear", col], 3)
    T["f1_gb_de"] = fmt(td.loc["Boosting sin balancear", dcol], 3)
    yt = g["y_clf_test"].values.astype(bool)
    pt = np.asarray(g["pred_clf_test"]).astype(bool)
    T["tp"], T["fp"], T["fn"] = fmt((pt & yt).sum(), 0), fmt((pt & ~yt).sum(), 0), fmt((~pt & yt).sum(), 0)
    T["pos_test"] = fmt(yt.sum(), 0)
    dec = pd.qcut(pd.Series(g["proba_test_final"]).rank(method="first"), 10, labels=False)
    tasa_dec = pd.Series(yt).groupby(dec.values).mean()
    T["dec10"] = pct(tasa_dec.iloc[-1], 0)
    T["dec_bajos"] = pct(pd.Series(yt)[dec.values < 5].mean(), 1)
    T["base"] = pct(yt.mean(), 0)

    # Terciles (Objetivo 3)
    rq = g["resumen_test_q"]
    T["q_f1"] = fmt(rq.loc["Regresión discretizada (elegido)", "f1_macro"], 3)
    T["q_acc"] = pct(rq.loc["Regresión discretizada (elegido)", "accuracy"], 0)
    T["q_pm1"] = pct(rq.loc["Regresión discretizada (elegido)", "acierto ±1 tercil"], 0)
    T["q_kappa"] = fmt(rq.loc["Regresión discretizada (elegido)", "kappa cuadrático"], 3)
    T["q_piso_acc"] = pct(rq.loc[rq.index[0], "accuracy"], 0)
    # Qué dice cada tercio predicho en el test (los mismos datos de la Figura 5 y de «Explorá»)
    dt = datos[datos["particion"] == "test"]
    T["ter_t1_ok"] = pct((dt.loc[dt.tercil_pred == "T1 · bajo", "tercil_real"] == "T1 · bajo").mean(), 0)
    T["ter_t3_ok"] = pct((dt.loc[dt.tercil_pred == "T3 · alto", "tercil_real"] == "T3 · alto").mean(), 0)
    T["ter_ext"] = pct((((dt.tercil_pred == "T1 · bajo") & (dt.tercil_real == "T3 · alto"))
                        | ((dt.tercil_pred == "T3 · alto") & (dt.tercil_real == "T1 · bajo"))).mean(), 0)
    # Error típico del regresor como factor multiplicativo sobre (1 + citas)
    T["factor"] = fmt(np.exp(rt.loc["Gradient Boosting (Elegido)", "MAE log (Test)"]), 1)

    # Tablas técnicas
    tr = g["tabla_reg_cv"][["MAE train", "MAE cv", "desvío MAE", "R² cv"]].copy()
    T["tabla_reg"] = tabla_html(tr.rename_axis("Modelo"), {c_: (lambda v: fmt(v, 3)) for c_ in tr.columns})
    tg = g["resumen_grillas_reg"].copy()
    tg = tg[[c_ for c_ in tg.columns if c_ in ("MAE mejor de la grilla", "desvío cv", "MAE train", "mejores hiperparámetros")]]
    T["tabla_grilla"] = tabla_html(tg.rename_axis("Modelo"), {c_: (lambda v: fmt(v, 3)) for c_ in tg.columns if c_ != "mejores hiperparámetros"})
    T["tabla_test_reg"] = tabla_html(rt.rename_axis("Modelo"), {c_: (lambda v: fmt(v, 3)) for c_ in rt.columns})
    T["tabla_tramos"] = tabla_html(tt[["articulos", "MAE_log", "sesgo_log", "mediana_citas_real", "mediana_citas_pred"]].rename(columns={
        "articulos": "Artículos", "MAE_log": "MAE (log)", "sesgo_log": "Sesgo (log)", "mediana_citas_real": "Mediana real",
        "mediana_citas_pred": "Mediana predicha"}).rename_axis("Tramo de citas"),
        {"Artículos": lambda v: fmt(v, 0), "MAE (log)": lambda v: fmt(v, 2), "Sesgo (log)": lambda v: fmt(v, 2),
         "Mediana real": lambda v: fmt(v, 0), "Mediana predicha": lambda v: fmt(v, 1)})
    tc = g["tabla_clf_cv"][["f1_macro train", "f1_macro cv", "desvío f1_macro", "f1 top 10% (cv)", "accuracy (cv)"]]
    T["tabla_clf"] = tabla_html(tc.rename_axis("Modelo (umbral 0,50)"), {c_: (lambda v: fmt(v, 3)) for c_ in tc.columns})
    tdb = td[[c_ for c_ in td.columns if c_ != "hiperparámetros (grilla 3.2)"]]
    T["tabla_desb"] = tabla_html(tdb.rename_axis("Variante"), {c_: (lambda v: fmt(v, 3)) for c_ in tdb.columns})
    T["tabla_test_clf"] = tabla_html(tf.rename_axis("Estrategia (test)"), {c_: (lambda v: fmt(v, 3)) for c_ in tf.columns})
    cm = confusion_matrix(yt, pt)
    cmdf = pd.DataFrame(cm, index=["Real: resto", "Real: top 10 %"], columns=["Predicho: resto", "Predicho: candidato"]).rename_axis("")
    T["tabla_cm"] = tabla_html(cmdf, {c_: (lambda v: fmt(v, 0)) for c_ in cmdf.columns})
    pr = g["imp_perm_reg"].sort_values(ascending=False).head(6) if isinstance(g["imp_perm_reg"], pd.Series) else None
    pc = g["imp_perm_clf"].sort_values(ascending=False).head(6) if isinstance(g["imp_perm_clf"], pd.Series) else None
    if pr is not None and pc is not None:
        imp = pd.DataFrame({"Columna (regresión)": pr.index, "Sube el MAE al barajarla": [fmt(v, 3) for v in pr.values],
                            "Columna (clasificación)": pc.index, "Baja el f1 macro al barajarla": [fmt(v, 3) for v in pc.values]})
        T["tabla_imp"] = tabla_html(imp, indice=False)
    else:
        T["tabla_imp"] = ""
    coefs = pd.DataFrame({"Columna": [f["nombre"] for f in m["logistica"]["num"]] + [f"oa_status = {k}" for k in m["logistica"]["oa"]],
                          "Logística (top 10 %)": [f["coef"] for f in m["logistica"]["num"]] + list(m["logistica"]["oa"].values()),
                          "Ridge (log citas)": [f["coef"] for f in m["ridge"]["num"]] + list(m["ridge"]["oa"].values())})
    coefs = coefs.reindex(coefs["Logística (top 10 %)"].abs().sort_values(ascending=False).index)
    T["tabla_coef"] = tabla_html(coefs, {"Logística (top 10 %)": lambda v: fmt(v, 3), "Ridge (log citas)": lambda v: fmt(v, 3)}, indice=False)
    tq = g["resumen_test_q"]
    T["tabla_q"] = tabla_html(tq.rename_axis("Modelo (test)"), {c_: (lambda v: fmt(v, 3)) for c_ in tq.columns})
    return T


# ------------------------------------------------------------------
# 5. Página
# ------------------------------------------------------------------
def ejemplos_probar(g, datos, m):
    """Artículos reales del test para cargar en «Probá el modelo».

    Sólo entran los que caben en los deslizadores (del mínimo al percentil 99,5 de cada columna) y no están
    pegados al umbral, para que lo que muestran los controles sea exactamente lo que se calcula.
    """
    R, U = m["rangos"], float(g["UMBRAL"])
    X_test = g["X_test"]
    d = datos.loc[X_test.index].join(X_test[["source_pre_cites_per_work", "funders_count", "last_author_pre_cites", "first_author_pre_cites",
                                             "institution_max_h_index", "institutions_count", "in_pubmed"]])
    d = d.dropna(subset=["source_pre_cites_per_work", "last_author_pre_cites", "first_author_pre_cites"])
    d["source_pre_cites_per_work"] = d["source_pre_cites_per_work"].round(1)
    col_slider = {"source_pre_cites_per_work": "source_pre_cites_per_work", "referenced_works_count": "refs", "authors_count": "autores",
                  "institutions_count": "institutions_count", "institution_max_h_index": "institution_max_h_index",
                  "last_author_pre_cites": "last_author_pre_cites", "first_author_pre_cites": "first_author_pre_cites", "funders_count": "funders_count"}
    entra = pd.Series(True, index=d.index)
    for c, col in col_slider.items():
        v = d[col] if c != "institution_max_h_index" else d[col].fillna(R[c]["mediana"])
        entra &= v.between(R[c]["min"], R[c]["max"])
    d = d[entra]
    # la probabilidad del modelo, recalculada con los valores exactos que se cargan
    X = X_test.loc[d.index].copy()
    X["source_pre_cites_per_work"] = d["source_pre_cites_per_work"]
    d["prob_exacta"] = g["modelo_clf_final"].predict_proba(X)[:, 1]
    d = d[(d["prob_exacta"] - U).abs() >= 0.03]
    cand = d["prob_exacta"] >= U
    elegidos = []
    for cond, n in [((d.top10) & cand, 3), ((~d.top10) & (~cand), 2), ((d.top10) & (~cand), 2), ((~d.top10) & cand, 1)]:
        sub = d[cond].sort_values("prob_exacta", ascending=False)
        idx = np.linspace(0, len(sub) - 1, n).astype(int) if len(sub) else []
        elegidos += [sub.iloc[i] for i in idx]
    res = []
    for r in elegidos:
        es_cand = r.prob_exacta >= U
        tipo = ("acierto · fue top 10 %" if r.top10 and es_cand else "acierto · no fue top 10 %" if not r.top10 and not es_cand
                else "se le escapó · fue top 10 %" if r.top10 else "falsa alarma · no fue top 10 %")
        h = R["institution_max_h_index"]["mediana"] if pd.isna(r.institution_max_h_index) else r.institution_max_h_index
        res.append({
            "etiqueta": f"{r.titulo[:60]}… · {r.revista[:32]} · {r.citas} citas ({tipo})",
            "valores": {
                "p_source_pre_cites_per_work": float(r.source_pre_cites_per_work), "p_referenced_works_count": int(r.refs),
                "p_authors_count": int(r.autores), "p_institutions_count": int(r.institutions_count),
                "p_institution_max_h_index": float(round(h)),
                "p_last_author_pre_cites": float(r.last_author_pre_cites), "p_first_author_pre_cites": float(r.first_author_pre_cites),
                "p_funders_count": int(r.funders_count), "p_oa": r.oa, "p_core": bool(r.core), "p_pubmed": bool(r.in_pubmed),
                "p_sinrev": False,
            },
            "prob_modelo": float(r.prob_exacta),
        })
    return res


def construir(ruta_nb: Path, ruta_datos: Path, salida: Path):
    print(f"Ejecutando el notebook {ruta_nb.name} sobre {ruta_datos} …")
    g = ejecutar_notebook(ruta_nb, ruta_datos)
    print("Calculando predicciones fuera de muestra …")
    datos, _ = preparar_datos(g)
    modelos = exportar_modelos(g)
    graf_hall, textos_hall = g_hallazgo(datos)
    specs = {
        "g-datos": g_datos(datos),
        "g-hallazgo": graf_hall,
        "g-comparacion": g_comparacion(g),
        "g-tramos": g_tramos(g),
        "g-terciles": g_terciles(datos),
        "g-triage": g_triage(g),
        "g-explora": g_explora(datos),
        "g-proba": g_proba(modelos),
    }
    specs = {k: v.to_dict() for k, v in specs.items()}
    T = calcular_textos(g, datos, modelos)
    T.update(textos_hall)
    T["fecha_datos"] = time.strftime("%d/%m/%Y", time.localtime(ruta_datos.resolve().stat().st_mtime))
    T["fecha_build"] = time.strftime("%d/%m/%Y")
    T["slider_tope"] = fmt(modelos["rangos"]["referenced_works_count"]["max"], 0)
    ejemplos = ejemplos_probar(g, datos, modelos)
    opciones = "".join(f'<option value="{i}">{e["etiqueta"]}</option>' for i, e in enumerate(ejemplos))

    html = (AQUI / "plantilla.html").read_text(encoding="utf-8")
    for k, v in T.items():
        html = html.replace("{{" + k + "}}", str(v))
    html = (html.replace("{{opciones_ejemplos}}", opciones)
                .replace("{{SPECS}}", json.dumps(specs, ensure_ascii=False, separators=(",", ":")))
                .replace("{{EJEMPLOS}}", json.dumps(ejemplos, ensure_ascii=False))
                .replace("{{MEDIANAS}}", json.dumps({f"p_{c}": round(round(r["mediana"] / r["paso"]) * r["paso"], 6) for c, r in modelos["rangos"].items()}))
                .replace("{{VEGA}}", VEGA).replace("{{VEGA_LITE}}", VEGA_LITE).replace("{{VEGA_EMBED}}", VEGA_EMBED))
    faltan = sorted(set(re.findall(r"\{\{(\w+)\}\}", html)))
    if faltan:
        raise SystemExit(f"Faltan valores en la plantilla: {faltan}")
    salida.write_text(html, encoding="utf-8")
    (AQUI / "modelo_navegador.json").write_text(json.dumps(modelos, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo: {salida} ({salida.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notebook", type=Path, default=NOTEBOOK_DEF)
    ap.add_argument("--datos", type=Path, default=DATOS_DEF)
    ap.add_argument("--salida", type=Path, default=AQUI / "index.html")
    a = ap.parse_args()
    construir(a.notebook, a.datos, a.salida)
