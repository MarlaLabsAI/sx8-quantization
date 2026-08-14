"""
PIPELINE ADAPTADO A RESOLUCION Y TONO
=====================================
Los scripts del estudio se calibraron para imagen3 (1080x1920, sepia):
  sigma=15, threshold=10, REGION_SIZE=100, radios 20-120px, gap=10,
  CROSS_CENTER=(416,416), GRID_LINES fijas.

Jeshua2-izq es 1185x2321 en B/N: factor de escala = 2321/1080 = 2.149.

Este script ESCALA todos los parametros espaciales por el factor y
normaliza el tono (cv2.normalize 0-255, como hace el estudio), y
re-ejecuta las metricas clave del estudio sobre Jeshua2-izq.

Parametros escalados:
  sigma        = 15 * f
  REGION_SIZE  = 100 * f
  radios       = [20, 40, 60, 80, 100, 120] * f
  gap          = 10 * f
  radius_brazos= 20 * f
  threshold    = 10.0 (relativo a 0-255 tras normalizar, igual que el estudio)
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N_CONTROLS = 100
N_WORKERS = 12

# ============================================================================
# 1. CARGAR Y NORMALIZAR (mismo pipeline que el estudio)
# ============================================================================
def load_profile(img_path, normalize=True):
    """Perfil central normalizado a 0-255 (como analisis_chip_profundo.py)."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if normalize:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    else:
        img = img.astype(np.float64)
    h, w = img.shape
    return img[:, w // 2].astype(np.float64)

def build_recurrence(profile, sigma, threshold=10.0):
    """Matriz de recurrencia con sigma escalado."""
    p = ndimage.gaussian_filter1d(profile, sigma=sigma)
    n = len(p)
    diff = np.abs(p[:, None] - p[None, :])
    return (diff < threshold).astype(np.float32)

# ============================================================================
# 2. METRICAS DEL ESTUDIO (con parametros escalados)
# ============================================================================
def study_grid_and_cells(R, gap):
    """CHIP-4 + CHIP-5 exactos del estudio, con gap escalado."""
    n = R.shape[0]
    quadrant = R[:n//2, :n//2]
    row_density = quadrant.mean(axis=1)
    col_density = quadrant.mean(axis=0)
    thr_row = row_density.mean() + row_density.std()
    thr_col = col_density.mean() + col_density.std()
    strong_rows = np.where(row_density > thr_row)[0]
    strong_cols = np.where(col_density > thr_col)[0]
    def group_lines(lines, gap):
        if len(lines) == 0:
            return []
        groups = []
        cur = [lines[0]]
        for line in lines[1:]:
            if line - cur[-1] <= gap:
                cur.append(line)
            else:
                groups.append(int(np.mean(cur)))
                cur = [line]
        groups.append(int(np.mean(cur)))
        return groups
    grid_rows = group_lines(strong_rows, gap)
    grid_cols = group_lines(strong_cols, gap)
    sim = float("nan")
    n_cells = 0
    if len(grid_rows) > 1 and len(grid_cols) > 1:
        cells = []
        for i in range(min(5, len(grid_rows)-1)):
            for j in range(min(5, len(grid_cols)-1)):
                r1, r2 = grid_rows[i], grid_rows[i+1]
                c1, c2 = grid_cols[j], grid_cols[j+1]
                if r2-r1 > 10 and c2-c1 > 10:
                    cells.append(quadrant[r1:r2, c1:c2])
        if len(cells) > 1:
            cell_size = min(c.shape[0] for c in cells)
            cells_resized = np.array([cv2.resize(c, (cell_size, cell_size)) for c in cells])
            n_cells = len(cells_resized)
            sm = np.zeros((n_cells, n_cells))
            for i in range(n_cells):
                for j in range(n_cells):
                    sm[i, j] = np.mean(cells_resized[i] == cells_resized[j])
            sim = float(np.mean(sm[np.triu_indices(n_cells, 1)]))
    return len(grid_rows), len(grid_cols), sim, n_cells

def box_counting_dimension(R, min_size=2, max_size=None):
    h, w = R.shape
    if max_size is None:
        max_size = min(h, w) // 2
    sizes = []
    s = min_size
    while s <= max_size:
        sizes.append(s)
        s = int(s * 1.5) + 1
    sizes = np.array(sizes)
    counts = []
    for s in sizes:
        n_h, n_w = h // s, w // s
        if n_h == 0 or n_w == 0:
            continue
        blocks = R[: n_h * s, : n_w * s].reshape(n_h, s, n_w, s)
        counts.append((blocks.sum(axis=(1, 3)) > 0).sum())
    counts = np.array(counts)
    valid = counts > 0
    if valid.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0 / sizes[valid]), np.log(counts[valid]), 1)
    return float(slope)

def multifractal_multiscale(R, box_sizes=(4, 8, 16, 32), q_values=None):
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    h, w = R.shape
    tau_q = np.zeros(len(q_values))
    for qi, q in enumerate(q_values):
        if q == 0:
            tau_q[qi] = 0.0
            continue
        log_eps, log_sum = [], []
        for s in box_sizes:
            n_h, n_w = h // s, w // s
            if n_h == 0 or n_w == 0:
                continue
            blocks = R[: n_h * s, : n_w * s].reshape(n_h, s, n_w, s)
            measure = blocks.sum(axis=(1, 3)) / (s * s)
            measure = measure[measure > 0]
            if len(measure) == 0:
                continue
            measure = measure / measure.sum()
            log_eps.append(np.log(1.0 / s))
            log_sum.append(np.log(np.sum(measure ** q)))
        if len(log_eps) >= 3:
            slope, _ = np.polyfit(log_eps, log_sum, 1)
            tau_q[qi] = slope
        else:
            tau_q[qi] = float("nan")
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q
    valid = ~np.isnan(alpha)
    if valid.sum() < 3:
        return {"delta_alpha": float("nan"), "alpha_peak": float("nan"),
                "skewness": float("nan"), "alpha_min": float("nan"), "alpha_max": float("nan")}
    alpha_v, f_v = alpha[valid], f_alpha[valid]
    a_min, a_max = alpha_v.min(), alpha_v.max()
    a_peak = alpha_v[np.argmax(f_v)]
    return {"delta_alpha": float(a_max - a_min), "alpha_peak": float(a_peak),
            "skewness": float((a_max - a_peak) - (a_peak - a_min)),
            "alpha_min": float(a_min), "alpha_max": float(a_max)}

def central_peak_ratio(R, radius):
    n = R.shape[0]
    cy, cx = n // 2, n // 2
    center = R[cy-radius:cy+radius, cx-radius:cx+radius]
    return float(center.mean() / R.mean()) if R.mean() > 0 else float("nan")

def directional_quadrants(profile):
    n = len(profile)
    D = profile[:, None] - profile[None, :]
    h = n // 2
    return {"Q1": float(D[:h,:h].mean()), "Q2": float(D[:h,h:].mean()),
            "Q3": float(D[h:,:h].mean()), "Q4": float(D[h:,h:].mean())}

def mi_center_cells(R, rows, cols, radius):
    n = R.shape[0]
    cy, cx = n // 2, n // 2
    center = R[cy-radius:cy+radius, cx-radius:cx+radius]
    if center.size == 0 or len(rows) < 2 or len(cols) < 2:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    def mi(a, b):
        a = np.clip(a, 0, 1).astype(np.uint8)
        b = np.clip(b, 0, 1).astype(np.uint8)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                c[i, j] = np.mean((a == i) & (b == j))
        c /= c.sum()
        pa, pb = c.sum(axis=1), c.sum(axis=0)
        m = 0.0
        for i in range(2):
            for j in range(2):
                if c[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                    m += c[i, j] * np.log2(c[i, j] / (pa[i] * pb[j]))
        return m
    mis, dists = [], []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            cell = R[rows[i]:rows[i+1], cols[j]:cols[j+1]]
            if cell.size == 0:
                continue
            ccx = (cols[j] + cols[j+1]) // 2
            ccy = (rows[i] + rows[i+1]) // 2
            dists.append(np.sqrt((ccx - cx)**2 + (ccy - cy)**2))
            mis.append(mi(center, cell))
    if len(mis) < 3:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    corr, _ = pearsonr(dists, mis)
    return {"mean_MI": float(np.mean(mis)), "correlation_MI_distance": float(corr)}

def all_metrics(R, profile, gap, radius):
    gr, gc, sim, nc = study_grid_and_cells(R, gap)
    rows, cols = [], []
    # Re-detectar posiciones para MI (necesita posiciones reales)
    n = R.shape[0]
    quadrant = R[:n//2, :n//2]
    row_density = quadrant.mean(axis=1)
    col_density = quadrant.mean(axis=0)
    thr_row = row_density.mean() + row_density.std()
    thr_col = col_density.mean() + col_density.std()
    strong_rows = np.where(row_density > thr_row)[0]
    strong_cols = np.where(col_density > thr_col)[0]
    def group_lines(lines, gap):
        if len(lines) == 0:
            return []
        groups = []
        cur = [lines[0]]
        for line in lines[1:]:
            if line - cur[-1] <= gap:
                cur.append(line)
            else:
                groups.append(int(np.mean(cur)))
                cur = [line]
        groups.append(int(np.mean(cur)))
        return groups
    rows = group_lines(strong_rows, gap)
    cols = group_lines(strong_cols, gap)
    return {
        "density": float(R.mean()),
        "fractal_dimension": box_counting_dimension(R),
        "multifractal": multifractal_multiscale(R),
        "grid_rows": int(gr), "grid_cols": int(gc),
        "cell_similarity": sim,
        "central_peak_ratio": central_peak_ratio(R, radius),
        "directional": directional_quadrants(profile),
        "mi_center_cells": mi_center_cells(R, rows, cols, radius),
    }

# ============================================================================
# 3. CONTROLES
# ============================================================================
def control_permutation(p): return rng.permutation(p)
def control_gaussian(p): return rng.normal(p.mean(), p.std(), size=len(p))
def control_ar1(p):
    phi = 0.99
    n = len(p)
    noise = rng.normal(0, p.std() * np.sqrt(1 - phi**2), size=n)
    x = np.zeros(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x - x.mean() + p.mean()

def worker_metrics(args):
    R, profile, gap, radius = args
    return all_metrics(R, profile, gap, radius)

# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"config": {}, "imagen3": {}, "jeshua2": {}, "controles_jeshua2": {},
              "comparacion": {}}

    # Parametros originales (imagen3) y escalados (Jeshua2)
    f = 2321 / 1080  # factor de escala del perfil
    params_orig = {"sigma": 15.0, "gap": 10, "radius": 20, "region": 100}
    params_j2 = {"sigma": 15.0 * f, "gap": int(10 * f), "radius": int(20 * f), "region": int(100 * f)}
    report["config"] = {"factor_escala": f, "params_imagen3": params_orig, "params_jeshua2": params_j2}
    print("=" * 70, flush=True)
    print(f"FACTOR DE ESCALA: {f:.3f}", flush=True)
    print(f"  sigma: {params_orig['sigma']} -> {params_j2['sigma']:.1f}", flush=True)
    print(f"  gap:   {params_orig['gap']} -> {params_j2['gap']}", flush=True)
    print(f"  radius:{params_orig['radius']} -> {params_j2['radius']}", flush=True)
    print("=" * 70, flush=True)

    # --- Imagen3 con parametros originales (referencia) ---
    print("\nIMAGEN3 (parametros originales):", flush=True)
    p3 = load_profile(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"))
    R3 = build_recurrence(p3, params_orig["sigma"])
    m3 = all_metrics(R3, p3, params_orig["gap"], params_orig["radius"])
    report["imagen3"] = m3
    print(f"  D={m3['fractal_dimension']:.4f} | grid={m3['grid_rows']}x{m3['grid_cols']} | sim={m3['cell_similarity']:.4f} | pico={m3['central_peak_ratio']:.2f} | MI-dist={m3['mi_center_cells']['correlation_MI_distance']:.4f} | Q2={m3['directional']['Q2']:.3f}", flush=True)

    # --- Jeshua2-izq con parametros ESCALADOS ---
    print("\nJESHUA2-IZQ (parametros escalados):", flush=True)
    j2 = cv2.imread(os.path.join(BASE, "Re_verificacion", "Jeshua2.jpg"), cv2.IMREAD_GRAYSCALE)
    half = j2[:, :j2.shape[1]//2]
    tmp = "/tmp/opencode/jeshua2_izq.png"
    cv2.imwrite(tmp, half)
    pj = load_profile(tmp)
    Rj = build_recurrence(pj, params_j2["sigma"])
    mj = all_metrics(Rj, pj, params_j2["gap"], params_j2["radius"])
    report["jeshua2"] = mj
    print(f"  D={mj['fractal_dimension']:.4f} | grid={mj['grid_rows']}x{mj['grid_cols']} | sim={mj['cell_similarity']:.4f} | pico={mj['central_peak_ratio']:.2f} | MI-dist={mj['mi_center_cells']['correlation_MI_distance']:.4f} | Q2={mj['directional']['Q2']:.3f}", flush=True)

    # --- Controles sobre Jeshua2 con parametros escalados ---
    print(f"\nCONTROLES JESHUA2 ({N_CONTROLS} x 3, parametros escalados):", flush=True)
    kinds = ["permutation", "gaussian", "ar1"]
    control_metrics = {k: [] for k in kinds}
    for kind in kinds:
        tasks = []
        for i in range(N_CONTROLS):
            if kind == "permutation":
                pc = control_permutation(pj)
            elif kind == "gaussian":
                pc = control_gaussian(pj)
            else:
                pc = control_ar1(pj)
            Rc = build_recurrence(pc, params_j2["sigma"])
            tasks.append((Rc, pc, params_j2["gap"], params_j2["radius"]))
        with Pool(N_WORKERS) as pool:
            res = list(pool.imap_unordered(worker_metrics, tasks))
        control_metrics[kind] = res
        print(f"  [{kind}] {N_CONTROLS} controles listos", flush=True)

    # --- Significancia ---
    print("\nSIGNIFICANCIA (Jeshua2, parametros escalados):", flush=True)
    sig = {}
    for kind in kinds:
        sig[kind] = {}
        for metric, real_val in [("fractal_dimension", mj["fractal_dimension"]),
                                 ("cell_similarity", mj["cell_similarity"]),
                                 ("central_peak_ratio", mj["central_peak_ratio"]),
                                 ("grid_rows", mj["grid_rows"]),
                                 ("grid_cols", mj["grid_cols"])]:
            vals = [mc[metric] for mc in control_metrics[kind]
                    if not (isinstance(mc[metric], float) and np.isnan(mc[metric]))]
            if not vals:
                continue
            vals = np.array(vals)
            mean, std = vals.mean(), vals.std()
            z = (real_val - mean) / std if std > 0 else float("nan")
            pct = float((vals < real_val).mean() * 100)
            pval = float((np.abs(vals) >= abs(real_val)).mean()) if real_val != 0 else 1.0
            sig[kind][metric] = {"real": real_val, "control_mean": float(mean),
                                 "control_std": float(std), "z_score": float(z),
                                 "percentile": pct, "p_value_empirical": pval}
            print(f"  [{kind}] {metric}: real={real_val:.4f} | control={mean:.4f}±{std:.4f} | z={z:+.2f} | pct={pct:.1f}% | p={pval:.4f}", flush=True)
        vals = [mc["multifractal"]["delta_alpha"] for mc in control_metrics[kind]
                if not np.isnan(mc["multifractal"]["delta_alpha"])]
        if vals:
            vals = np.array(vals)
            real_da = mj["multifractal"]["delta_alpha"]
            z = (real_da - vals.mean()) / vals.std() if vals.std() > 0 else float("nan")
            pct = float((vals < real_da).mean() * 100)
            sig[kind]["delta_alpha"] = {"real": real_da, "control_mean": float(vals.mean()),
                                        "control_std": float(vals.std()), "z_score": float(z),
                                        "percentile": pct}
            print(f"  [{kind}] delta_alpha: real={real_da:.4f} | control={vals.mean():.4f}±{vals.std():.4f} | z={z:+.2f} | pct={pct:.1f}%", flush=True)
        vals = [mc["mi_center_cells"]["correlation_MI_distance"] for mc in control_metrics[kind]
                if not np.isnan(mc["mi_center_cells"]["correlation_MI_distance"])]
        if vals:
            vals = np.array(vals)
            real_mi = mj["mi_center_cells"]["correlation_MI_distance"]
            z = (real_mi - vals.mean()) / vals.std() if vals.std() > 0 else float("nan")
            pct = float((vals < real_mi).mean() * 100)
            sig[kind]["mi_distance_corr"] = {"real": real_mi, "control_mean": float(vals.mean()),
                                             "control_std": float(vals.std()), "z_score": float(z),
                                             "percentile": pct}
            print(f"  [{kind}] MI-distancia: real={real_mi:.4f} | control={vals.mean():.4f}±{vals.std():.4f} | z={z:+.2f} | pct={pct:.1f}%", flush=True)
    report["controles_jeshua2"] = sig

    # --- Comparacion ---
    print("\nCOMPARACION imagen3 (orig) vs Jeshua2 (escalado):", flush=True)
    comp = {}
    for metric in ["fractal_dimension", "cell_similarity", "central_peak_ratio", "grid_rows", "grid_cols"]:
        comp[metric] = {"imagen3": m3[metric], "jeshua2": mj[metric]}
        print(f"  {metric}: imagen3={m3[metric]:.4f} | jeshua2={mj[metric]:.4f}", flush=True)
    comp["delta_alpha"] = {"imagen3": m3["multifractal"]["delta_alpha"], "jeshua2": mj["multifractal"]["delta_alpha"]}
    comp["mi_distance"] = {"imagen3": m3["mi_center_cells"]["correlation_MI_distance"],
                           "jeshua2": mj["mi_center_cells"]["correlation_MI_distance"]}
    comp["directional_Q2"] = {"imagen3": m3["directional"]["Q2"], "jeshua2": mj["directional"]["Q2"]}
    report["comparacion"] = comp
    print(f"  delta_alpha: imagen3={comp['delta_alpha']['imagen3']:.4f} | jeshua2={comp['delta_alpha']['jeshua2']:.4f}", flush=True)
    print(f"  MI-dist: imagen3={comp['mi_distance']['imagen3']:.4f} | jeshua2={comp['mi_distance']['jeshua2']:.4f}", flush=True)
    print(f"  Q2: imagen3={comp['directional_Q2']['imagen3']:.4f} | jeshua2={comp['directional_Q2']['jeshua2']:.4f}", flush=True)

    out_json = os.path.join(OUT, "pipeline_adaptado_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
