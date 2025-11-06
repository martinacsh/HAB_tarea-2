
# Tarea 2: Network Propagation

# Propagación de los genes COX4I2, ND1 y ATP6

Este proyecto implementa un ejemplo reproducible de propagación en redes biológicas usando dos enfoques complementarios:

- **GUILD / Random Walk with Restart (RWR)**: difusión con reinicio sobre una red de interacción; prioriza nodos cercanos a las semillas en la topología.
- **DIAMOnD**: expansión del módulo de enfermedad mediante prueba hipergeométrica; añade genes con exceso de conexiones al módulo actual.

**Semillas empleadas**: `ENO1`, `PGK1`, `HK2` (glucólisis).

**Qué haremos**: ejecutar GUILD y/o DIAMOnD sobre redes de ejemplo, generar _rankings_ de genes y visualizar resultados y comparativas (p. ej., **Jaccard Top-N**).

---

## Bases teóricas (resumen)

### Random Walk with Restart (RWR)

Partiendo de un vector semilla \( \mathbf{p}_0 \), se itera

\[
\mathbf{p}_{t+1} = (1-\alpha)\, W\, \mathbf{p}_t + \alpha\, \mathbf{p}_0
\]

donde \( W \) es la **matriz de transición** (columnas normalizadas) del grafo. El fijo \( \mathbf{p}^\* \) es un ranking de proximidad a las semillas.

### DIAMOnD (DIseAse MOdule Detection)

Dado un conjunto \( S \) (inicialmente las semillas), para cada candidato \( v \notin S \) se calcula el **p-valor hipergeométrico** de observar \(\ge x\) enlaces de \( v \) hacia \( S \) dado su **grado** \( n \) en una red de \( N \) nodos con \( |S| = K \).  
En cada paso se añade el gen con **menor p-valor** y se actualiza \( S \).

---

## Comparativa Top-N

El **Jaccard(Top-N)** evalúa la estabilidad/consenso entre listas de GUILD y DIAMOnD al crecer \( N \).

---

## Metodología

### 1) Entrada

- **Red** en `data/` con formato de aristas: `u v [w]` (peso opcional).  
  Delimitador **autodetectado**: tab, coma o espacios.
- **Semillas**: archivo `data/genes_seed.txt` (una por línea) o _inline_ `--seeds-inline ENO1,PGK1,HK2`.
- **Normalización** automática a **MAYÚSCULAS** y uso de **máximo 3 semillas** (avisa si hay más).
- Si **ninguna semilla** está en la red (y no desactivas el comportamiento), se usan automáticamente los **3 nodos de mayor grado**.

### 2) GUILD (RWR)

- **Construcción** de \( W \) (usa pesos si existen con `--usar-pesos`).
- **Iteración** hasta convergencia con `--alpha`, `--tol` y `--max-iters`.
- **Salida**: `gene, score_guild, rank`.

### 3) DIAMOnD

- **Expansión** paso a paso con p-valor hipergeométrico; desempates por **nº de enlaces a \( S \)** y **grado**.
- **Salida**: `gene, score_diamond (p-valor), step_added`.

### 4) Exportación y visualización

- **TSV** en `results/` (si no se especifica `-o`, se generan rutas por defecto).
- **Figuras** en `results/plots/`:
  - **GUILD**: distribución/CDF (y variantes en log para mayor legibilidad).
  - **DIAMOnD**: histograma y CDF de \(-\log_{10}(p)\); opcionalmente, \(-\log_{10}(p)\) vs paso.
  - **Comparativa**: **Jaccard(Top-N)** GUILD vs DIAMOnD.

---

## Justificación de métodos

### Dualidad local vs. significancia

- **GUILD** capta **proximidad topológica**.  
- **DIAMOnD** detecta **exceso de conexión** al módulo con soporte **estadístico**.  
Usarlos en paralelo permite **intersecar y comparar** listas, priorizando **candidatos robustos**.

### Criterios de selección prácticos

- **GUILD**: Top-N o percentil (1–5%).
- **DIAMOnD**: Top-k (parámetro del método) y/o umbral por p-valor (opcionalmente **FDR** si se calcula).
- **Intersección** de listas y **Jaccard** para evaluar estabilidad.

---

## Librerías y trazabilidad

- `networkx` para grafos, `numpy`/`pandas` para cálculo y E/S, `matplotlib` para figuras.  
- **Logging** informativo y rutas de salida creadas automáticamente.

---

## Robustez y reproducibilidad

- **CLI** con subcomandos: `guild`, `diamond`, `both`.
- **Compatibilidad** con _flags_ legacy `--algoritmo`/`--algo` (mapa a los subcomandos).
- **Headless-safe**: _backend_ de Matplotlib preparado para entornos sin servidor gráfico.
- **Advertencias útiles**: semillas fuera de la red, convergencia de RWR, paradas tempranas de DIAMOnD.

---

## Resultados esperados (semillas ENO1, PGK1, HK2)

- **GUILD** priorizará genes **próximos a glucólisis**.
- **DIAMOnD** tenderá a destacar nodos con **enriquecimiento de enlaces** al módulo de glucólisis (dependiendo de la red de entrada).

---

## Guía de ejecución

instalación dependencias: 
pip install -r requirements.txt


Ejecutar ambos métodos (usa los defaults del repo)
python scripts/propagacion_red.py both --seeds data/genes_seed.txt -k 200 -a 0.5 --log INFO


Solo GUILD (RWR) con el input por defecto

python scripts/propagacion_red.py guild -i data/network_guild.txt --seeds-inline ENO1,PGK1,HK2 -a 0.5 --tol 1e-9 --max-iters 10000 --usar-pesos -o results/network_guild.tsv --log DEBUG


Solo DIAMOnD con el input por defecto
python scripts/propagacion_red.py diamond -i data/network_diamond.txt --seeds data/genes_seed.txt -k 200 -o results/network_diamond_k200.tsv --log INFO


---

## Preparación del entorno


