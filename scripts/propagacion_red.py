#!/usr/bin/env python3
"""
Propagación en redes: Implementaciones ligeras de GUILD (RWR) y DIAMOnD

Este script ofrece una implementación de referencia, documentada y sencilla, de dos
estrategias clásicas de priorización de genes basadas en redes:

1) Difusión tipo GUILD mediante Random Walk with Restart (RWR)
   - Produce un ranking continuo de todos los nodos según su proximidad difusa a las semillas.
   - Iteración: p <- (1 - alpha) * W @ p + alpha * p0, donde W es la matriz de transición
     (adyacencia normalizada por columnas) y p0 es el vector inicial sobre las semillas.

2) Expansión tipo DIAMOnD con prueba hipergeométrica
   - Añade iterativamente el nodo más significativamente conectado al módulo de semillas actual,
     midiendo la probabilidad (cola) de observar al menos c enlaces al módulo dado el grado del candidato.

Novedades (robustez de entrada y gráficos automáticos)
-----------------------------------------------------
- Si el archivo pasado en `--seeds` **no existe** o está **vacío**, se usan por defecto
  las semillas **ENO1, PGK1, HK2** (se avisa por logging).
- Si **ninguna semilla** está en la red, el script **elige automáticamente** como semillas
  los **3 nodos de mayor grado** en la red (comportamiento fijo, no configurable).
- **Máximo 3 semillas**: si se proporcionan más de 3, se **recortan a las 3 primeras** (se avisa).
- Se puede pasar una lista de semillas **inline** con `--seeds-inline ENO1,PGK1,HK2`.
- **Gráficos automáticos**:
  - Si ejecutas **solo GUILD** o **solo DIAMOnD**, se generan 2 figuras informativas del algoritmo.
  - Si al terminar hay **resultados de ambos métodos** disponibles en `results/`, se genera además
    un gráfico comparativo de **Jaccard Top-N vs N** automáticamente (sin flags extra).

Entradas
--------
- Archivo de aristas (tab/coma/espacios). Las dos primeras columnas son nodos (HUGO).
  La tercera (opcional) es peso. Grafo no dirigido. Si hay pesos, RWR puede usarlos.
- Archivo de semillas: un gen por línea (opcional). Por defecto, si falta, se usan ENO1, PGK1, HK2.

Salidas
-------
- Archivo TSV en `results/` con cabeceras descriptivas. Las columnas dependen del algoritmo.
- Figuras PNG en `results/plots/`.

Ejemplos rápidos (CLI)
----------------------
RWR (GUILD):
    python scripts/propagacion_red.py --algo guild --input data/network_guild.txt --output results/guild_results.tsv --alpha 0.5 --tol 1e-9 --max-iters 10000

DIAMOnD:
    python scripts/propagacion_red.py --algo diamond --input data/network_diamond.txt --output results/diamond_results_k200.tsv --k 200

Dependencias
------------
- pandas, networkx, numpy, matplotlib (para las figuras). No se requiere SciPy (hipergeométrica exacta con `math.comb`).
"""
from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import sys
from typing import Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# Utilidades de E/S y preprocesado
# ----------------------------

def _detectar_separador(ruta: str) -> str:
    """Detecta heurísticamente el delimitador entre tab, coma o espacio.

    Devuelve un separador válido para `pandas.read_csv` ("\t", "," o "\s+").
    """
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                if "	" in linea:
                    return "	"
                if "," in linea:
                    return ","
                return "\s+"  # uno o más espacios
    return "\s+"


def cargar_red(path: str) -> nx.Graph:
    """Carga un grafo no dirigido a partir de una lista de aristas.

    Asume que las dos primeras columnas son IDs de nodos (str). La tercera columna,
    si existe, se interpreta como peso. Elimina auto-bucles y colapsa múltiples aristas.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de red: {path}")

    sep = _detectar_separador(path)
    try:
        df = pd.read_csv(path, sep=sep, header=None, comment="#", engine="python")
    except Exception as e:
        raise RuntimeError(f"Error al leer la red '{path}': {e}")

    if df.shape[1] < 2:
        raise ValueError("La red debe tener al menos dos columnas (u v [w]).")

    df = df.iloc[:, :3]
    df.columns = ["u", "v", "w"][: df.shape[1]]

    # Normaliza etiquetas a mayúsculas
    df["u"] = df["u"].astype(str).str.strip().str.upper()
    df["v"] = df["v"].astype(str).str.strip().str.upper()

    # Elimina auto-bucles
    df = df[df["u"] != df["v"]].copy()

    G = nx.Graph()
    if "w" in df.columns:
        for _, row in df.iterrows():
            u, v = row["u"], row["v"]
            w = row["w"] if not pd.isna(row["w"]) else 1.0
            try:
                w = float(w)
            except Exception:
                w = 1.0
            G.add_edge(u, v, weight=w)
    else:
        aristas = df[["u", "v"]].itertuples(index=False, name=None)
        G.add_edges_from(aristas)

    logging.info(
        "Red cargada: %d nodos, %d aristas desde '%s'",
        G.number_of_nodes(), G.number_of_edges(), path,
    )
    return G


def _parsear_seeds_inline(cadena: str | None) -> List[str]:
    if not cadena:
        return []
    return [s.strip().upper() for s in str(cadena).split(",") if s.strip()]


def cargar_semillas(path: str | None, seeds_inline: str | None = None) -> List[str]:
    """Carga semillas desde archivo o desde `--seeds-inline`.

    Prioridad: 1) `seeds_inline` (cadena "A,B,C"), 2) archivo `path`, 3) defecto [ENO1, PGK1, HK2].
    Normaliza a mayúsculas y deduplica. Máximo 3 semillas.
    """
    semillas_defecto = ["ENO1", "PGK1", "HK2"]

    semillas = _parsear_seeds_inline(seeds_inline)
    if semillas:
        logging.info("Semillas recibidas por CLI: %s", ", ".join(semillas))
    else:
        if path is None or not os.path.exists(path):
            logging.warning(
                "Archivo de semillas no encontrado (%s). Se usarán las semillas por defecto: %s",
                path, ", ".join(semillas_defecto),
            )
            semillas = semillas_defecto
        else:
            with open(path, "r", encoding="utf-8") as f:
                semillas = [line.strip().upper() for line in f if line.strip()]
            if not semillas:
                logging.warning(
                    "El archivo de semillas está vacío (%s). Se usarán las semillas por defecto: %s",
                    path, ", ".join(semillas_defecto),
                )
                semillas = semillas_defecto

    # Deduplicado preservando orden y límite a 3
    vistas = set()
    res = []
    for g in semillas:
        if g not in vistas:
            vistas.add(g)
            res.append(g)
        if len(res) == 3:
            if len(semillas) > 3:
                logging.warning("Se proporcionaron %d semillas; se usarán solo las 3 primeras: %s", len(semillas), ", ".join(res))
            break
    return res


# ----------------------------
# GUILD-like: Random Walk with Restart
# ----------------------------

def rwr(
    G: nx.Graph,
    semillas: Iterable[str],
    alpha: float = 0.5,
    tol: float = 1e-9,
    max_iters: int = 10000,
    usar_pesos: bool = False,
    auto_seeds_si_vacias: bool = True,
) -> pd.DataFrame:
    """Random Walk with Restart (RWR) sobre un grafo no dirigido.

    Si ninguna semilla está presente y `auto_seeds_si_vacias` es True, selecciona
    automáticamente los **3 nodos de mayor grado** como semillas.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("La red está vacía.")

    nodos = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodos)}

    semillas = [s.upper() for s in semillas]
    semillas_in = [s for s in semillas if s in idx]
    faltantes = [s for s in semillas if s not in idx]
    if faltantes:
        logging.warning("%d semillas no están en la red y se ignorarán: %s", len(faltantes), ", ".join(faltantes))

    if not semillas_in:
        if auto_seeds_si_vacias:
            top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:3]
            semillas_in = [n for n, _ in top]
            logging.warning(
                "Ninguna semilla está en la red. Se usarán automáticamente como semillas los 3 nodos de mayor grado: %s",
                ", ".join(semillas_in),
            )
        else:
            raise ValueError("Ninguna semilla está presente en la red.")

    n = len(nodos)

    # Construcción de la matriz de transición W (columnas suman 1)
    A = np.zeros((n, n), dtype=float)

    if usar_pesos:
        for u, v, data in G.edges(data=True):
            w = float(data.get("weight", 1.0))
            A[idx[u], idx[v]] += w
            A[idx[v], idx[u]] += w
    else:
        for u, v in G.edges():
            A[idx[u], idx[v]] = 1.0
            A[idx[v], idx[u]] = 1.0

    sum_col = A.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.divide(A, sum_col, where=sum_col != 0)
        W[:, sum_col == 0] = 0.0

    # Vector semilla p0 (masa uniforme sobre semillas presentes)
    p0 = np.zeros(n, dtype=float)
    masa = 1.0 / len(semillas_in)
    for s in semillas_in:
        p0[idx[s]] = masa

    p = p0.copy()
    for it in range(1, max_iters + 1):
        p_next = (1.0 - alpha) * (W @ p) + alpha * p0
        diff = np.abs(p_next - p).sum()
        if diff < tol:
            logging.info("RWR convergió en %d iteraciones (L1 diff=%.3e)", it, diff)
            p = p_next
            break
        p = p_next
    else:
        logging.warning(
            "RWR alcanzó max_iters=%d sin cumplir tol=%.3e (diff final=%.3e)", max_iters, tol, diff
        )

    df = pd.DataFrame({"gene": nodos, "score_guild": p})
    df.sort_values("score_guild", ascending=False, inplace=True, ignore_index=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# ----------------------------
# DIAMOnD-like: expansión con p-valor hipergeométrico
# ----------------------------

def _hipergeom_sf(x: int, K: int, N: int, n: int) -> float:
    """Función de cola P[X >= x] para la hipergeométrica(N, K, n) usando `math.comb`.

    X = número de éxitos en n extracciones sin reemplazo de una población de tamaño N
    con K éxitos. Devuelve la probabilidad de observar al menos x éxitos.
    """
    x = max(0, x)
    sup = min(n, K)
    denom = math.comb(N, n)
    if denom == 0:
        return 1.0
    cola = 0.0
    for i in range(x, sup + 1):
        cola += math.comb(K, i) * math.comb(N - K, n - i) / denom
    return float(min(max(cola, 0.0), 1.0))


def diamond(
    G: nx.Graph,
    semillas: Iterable[str],
    k: int = 200,
    auto_seeds_si_vacias: bool = True,
) -> pd.DataFrame:
    """Expansión tipo DIAMOnD basada en p-valor hipergeométrico.

    Si ninguna semilla está presente y `auto_seeds_si_vacias` es True, selecciona
    automáticamente los **3 nodos de mayor grado** como semillas iniciales.
    """
    if G.number_of_nodes() == 0:
        raise ValueError("La red está vacía.")

    todos = set(G.nodes())
    semillas = [s.upper() for s in semillas]
    S_in = [s for s in semillas if s in todos]
    faltantes = [s for s in semillas if s not in todos]
    if faltantes:
        logging.warning("%d semillas no están en la red y se ignorarán: %s", len(faltantes), ", ".join(faltantes))

    if not S_in:
        if auto_seeds_si_vacias:
            top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:3]
            S_in = [n for n, _ in top]
            logging.warning(
                "Ninguna semilla está en la red. Se usarán automáticamente como semillas los 3 nodos de mayor grado: %s",
                ", ".join(S_in),
            )
        else:
            raise ValueError("Ninguna semilla está presente en la red.")

    S: Set[str] = set(S_in)
    N = G.number_of_nodes()

    resultados: List[Tuple[str, float, int]] = []  # (gen, pval, paso)

    for paso in range(1, k + 1):
        mejor_gen = None
        mejor_p = None
        mejores_enlaces = -1

        for v in todos - S:
            dv = G.degree(v)
            if dv == 0:
                continue
            enlaces_S = sum(1 for u in G.neighbors(v) if u in S)
            if enlaces_S == 0:
                continue

            pval = _hipergeom_sf(enlaces_S, K=len(S), N=N - 1, n=dv)

            # Selección por menor p-valor; desempates: más enlaces a S, luego mayor grado
            if (mejor_p is None) or (pval < mejor_p) or (
                math.isclose(pval, mejor_p) and (enlaces_S > mejores_enlaces)
            ) or (
                math.isclose(pval, mejor_p) and (enlaces_S == mejores_enlaces) and (dv > G.degree(mejor_gen) if mejor_gen else True)
            ):
                mejor_gen = v
                mejor_p = pval
                mejores_enlaces = enlaces_S

        if mejor_gen is None:
            logging.info(
                "DIAMOnD se detuvo en el paso %d: no hay candidatos conectados al módulo (|S|=%d).",
                paso, len(S)
            )
            break

        S.add(mejor_gen)
        resultados.append((mejor_gen, float(mejor_p), paso))

    df = pd.DataFrame(resultados, columns=["gene", "score_diamond", "step_added"])
    return df


# ----------------------------
# Gráficos automáticos
# ----------------------------

def _asegurar_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _guardar_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info("Figura guardada en '%s'", path)


def _plots_guild(df_g: pd.DataFrame, plots_dir: str) -> None:
    _asegurar_dir(plots_dir)
    # 1) Histograma/densidad de score_guild
    vals = df_g["score_guild"].values
    plt.figure()
    counts, bins, _ = plt.hist(vals, bins=50, density=True, alpha=0.6)
    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(centers, counts)
    plt.xlabel("score_guild")
    plt.ylabel("Densidad")
    plt.title("Distribución de score (GUILD)")
    _guardar_fig(os.path.join(plots_dir, "guild_dist.png"))

    # 2) CDF de score_guild
    plt.figure()
    s = np.sort(vals)
    y = np.arange(1, len(s) + 1) / len(s)
    plt.plot(s, y)
    plt.xlabel("score_guild")
    plt.ylabel("CDF")
    plt.title("CDF de score (GUILD)")
    _guardar_fig(os.path.join(plots_dir, "guild_cdf.png"))


def _plots_diamond(df_d: pd.DataFrame, plots_dir: str) -> None:
    _asegurar_dir(plots_dir)
    # Transformación -log10(p)
    p = df_d["score_diamond"].astype(float).values
    eps = 1e-300
    neglogp = -np.log10(np.clip(p, eps, 1.0))

    # 1) Histograma/densidad
    plt.figure()
    counts, bins, _ = plt.hist(neglogp, bins=50, density=True, alpha=0.6)
    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.plot(centers, counts)
    plt.xlabel("-log10(p-valor)")
    plt.ylabel("Densidad")
    plt.title("Distribución de significancia (DIAMOnD)")
    _guardar_fig(os.path.join(plots_dir, "diamond_dist.png"))

    # 2) CDF
    plt.figure()
    s = np.sort(neglogp)
    y = np.arange(1, len(s) + 1) / len(s)
    plt.plot(s, y)
    plt.xlabel("-log10(p-valor)")
    plt.ylabel("CDF")
    plt.title("CDF de significancia (DIAMOnD)")
    _guardar_fig(os.path.join(plots_dir, "diamond_cdf.png"))


def _plots_comparativos(df_g: pd.DataFrame, df_d: pd.DataFrame, plots_dir: str) -> None:
    _asegurar_dir(plots_dir)
    # Jaccard Top-N vs N
    # Ranking GUILD: por score_guild descendente
    g_ranked = df_g.sort_values("score_guild", ascending=False, ignore_index=True)["gene"].tolist()
    # Ranking DIAMOnD: por step_added ascendente
    d_ranked = df_d.sort_values("step_added", ascending=True, ignore_index=True)["gene"].tolist()

    maxN = min(200, len(g_ranked), len(d_ranked))
    if maxN < 10:
        logging.warning("Muy pocos elementos para Jaccard (N<10). Se omitirá el comparativo.")
        return

    Ns = list(range(10, maxN + 1, 10))
    jaccs = []
    g_set = set()
    d_set = set()
    for N in Ns:
        g_set.update(g_ranked[:N])
        d_set.update(d_ranked[:N])
        inter = len(g_set & d_set)
        union = len(g_set | d_set)
        j = inter / union if union else 0.0
        jaccs.append(j)

    plt.figure()
    plt.plot(Ns, jaccs, marker="o")
    plt.xlabel("N (Top-N)")
    plt.ylabel("Jaccard(Top-N)")
    plt.title("Solapamiento Top-N: GUILD vs DIAMOnD")
    _guardar_fig(os.path.join(plots_dir, "jaccard_topN.png"))


# ----------------------------
# Orquestación y CLI
# ----------------------------

def guardar_tsv(df: pd.DataFrame, ruta_salida: str) -> None:
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, sep="	", index=False)
    logging.info("Resultados guardados en '%s' (%d filas)", ruta_salida, len(df))


def parsear_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Propagación en redes: GUILD (RWR) y DIAMOnD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--algo", required=True, choices=["guild", "diamond"], help="Algoritmo a ejecutar")
    p.add_argument("--input", required=True, help="Ruta al archivo de aristas (u v [w])")
    p.add_argument("--seeds", required=False, default="data/genes_seed.txt", help="Ruta al archivo de semillas (opcional)")
    p.add_argument("--seeds-inline", required=False, default=None, help="Lista de semillas separadas por coma, p. ej., ENO1,PGK1,HK2")
    p.add_argument("--no-auto-seeds", action="store_true", help="Desactiva la selección automática de semillas por grado si ninguna semilla aparece en la red")
    p.add_argument("--output", required=True, help="Ruta del TSV de salida")

    # Parámetros RWR
    p.add_argument("--alpha", type=float, default=0.5, help="Probabilidad de reinicio (guild)")
    p.add_argument("--tol", type=float, default=1e-9, help="Tolerancia L1 de convergencia (guild)")
    p.add_argument("--max-iters", type=int, default=10000, help="Máximo de iteraciones (guild)")
    p.add_argument("--usar-pesos", action="store_true", help="Usar pesos de arista si existen (guild)")

    # Parámetros DIAMOnD
    p.add_argument("--k", type=int, default=200, help="Número de nodos a añadir (diamond)")

    # Miscelánea
    p.add_argument("--log", default="INFO", help="Nivel de logging: DEBUG, INFO, WARNING, ERROR")

    return p.parse_args(argv)


def _plots_auto_postrun(args, df, out_path):
    # Directorio de plots al lado de la carpeta de salida
    base_dir = os.path.dirname(os.path.abspath(out_path))
    plots_dir = os.path.join(base_dir, "plots")

    # Plots del algoritmo ejecutado
    if args.algo == "guild":
        _plots_guild(df, plots_dir)
    elif args.algo == "diamond":
        _plots_diamond(df, plots_dir)

    # Intento de comparativa si existen resultados del otro algoritmo
    try:
        if args.algo == "guild":
            # Buscar algún TSV de DIAMOnD en la carpeta base de resultados
            candidatos = glob.glob(os.path.join(base_dir, "diamond*.tsv")) + glob.glob(os.path.join(base_dir, "*diamond*.tsv"))
            if candidatos:
                df_d = pd.read_csv(sorted(candidatos)[0], sep="	")
                _plots_comparativos(df, df_d, plots_dir)
        else:
            candidatos = glob.glob(os.path.join(base_dir, "guild*.tsv")) + glob.glob(os.path.join(base_dir, "*guild*.tsv"))
            if candidatos:
                df_g = pd.read_csv(sorted(candidatos)[0], sep="	")
                _plots_comparativos(df_g, df, plots_dir)
    except Exception as e:
        logging.warning("No se pudieron generar los plots comparativos automáticamente: %s", e)


def main(argv: List[str] | None = None) -> int:
    args = parsear_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, str(args.log).upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    try:
        G = cargar_red(args.input)
        semillas = cargar_semillas(args.seeds, seeds_inline=args.seeds_inline)
        auto_ok = not args.no_auto_seeds

        if args.algo == "guild":
            df = rwr(
                G,
                semillas=semillas,
                alpha=args.alpha,
                tol=args.tol,
                max_iters=args.max_iters,
                usar_pesos=args.usar_pesos,
                auto_seeds_si_vacias=auto_ok,
            )
        elif args.algo == "diamond":
            df = diamond(
                G,
                semillas=semillas,
                k=args.k,
                auto_seeds_si_vacias=auto_ok,
            )
        else:
            raise ValueError(f"Algoritmo desconocido: {args.algo}")

        guardar_tsv(df, args.output)
        _plots_auto_postrun(args, df, args.output)
        return 0

    except Exception as e:
        logging.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
