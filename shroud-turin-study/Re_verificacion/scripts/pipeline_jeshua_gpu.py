"""
RE-VERIFICACION CON IMAGENES JESHUA (ALTA RESOLUCION) - VERSION GPU
===================================================================
Optimizado: GPU (torch CUDA) para matrices + multiprocessing (12 cores)
para controles + progreso visible en tiempo real.

Las imagenes Jeshua1/Jeshua2 contienen DOS fotos (anverso/reverso)
partidas por la mitad en vertical. Se parte cada JPG, se selecciona
la mejor mitad y se ejecutan las metricas del estudio + controles.
"""

import os
import sys
import json
import time
import numpy as np
import cv2
import torch
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
REV = os.path.join(BASE, "Re_verificacion")
OUT = os.path.join(REV, "resultados")
os.makedirs(OUT, exist_ok=True)

N_CONTROLS = 100
N_WORKERS = 12
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# ============================================================================
# METRICAS (vectorizadas numpy, sin bucles python pesados)
# ============================================================================
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

def multifractal_spectrum_multiscale(R, box_sizes=(4, 8, 16, 32), q_values=None):
    """Espectro multifractal multi-escala VECTORIZADO (21 q a la vez por escala)."""
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    h, w = R.shape
    tau_q = np.zeros(len(q_values))
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
        # Vectorizado: todos los q a la vez
        log_eps = np.log(1.0 / s)
        for qi, q in enumerate(q_values):
            if q == 0:
                continue
            tau_q[qi] += np.log(np.sum(measure ** q)) / np.log(1.0 / s) if False else 0
        # Mejor: regresion log-log acumulada
        for qi, q in enumerate(q_values):
            if q == 0:
                continue
            tau_q[qi] += np.log(np.sum(measure ** q))
    # tau_q ahora es suma de log(sum) sobre escalas; dividir por n_escalas y usar
    # regresion implicita: tau(q) ~ slope * log(1/s) -> slope = tau_q / sum(log(1/s))
    n_scales = len(box_sizes)
    if n_scales == 0:
        return {"delta_alpha": float("nan"), "alpha_peak": float("nan"),
                "skewness": float("nan"), "alpha_min": float("nan"), "alpha_max": float("nan")}
    # Recalcular con regresion real por q (correcto)
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

def detect_grid(R, min_distance=10, prominence=0.05):
    row_proj = ndimage.gaussian_filter1d(R.mean(axis=1), sigma=3)
    col_proj = ndimage.gaussian_filter1d(R.mean(axis=0), sigma=3)
    rows, _ = find_peaks(row_proj, distance=min_distance, prominence=prominence * row_proj.std())
    cols, _ = find_peaks(col_proj, distance=min_distance, prominence=prominence * col_proj.std())
    return rows, cols

def cell_similarity(R, rows, cols):
    if len(rows) < 2 or len(cols) < 2:
        return float("nan")
    n = min(5, len(rows) - 1)
    if n < 2:
        return float("nan")
    start = len(rows) // 2 - n // 2
    r0 = rows[start:start + n + 1]
    c0 = cols[start:start + n + 1]
    if len(r0) < n + 1 or len(c0) < n + 1:
        return float("nan")
    cells = []
    for i in range(n):
        for j in range(n):
            cell = R[r0[i]:r0[i + 1], c0[j]:c0[j + 1]]
            if cell.size > 0:
                cells.append(cell)
    if len(cells) < 2:
        return float("nan")
    sims = []
    for a in range(len(cells)):
        for b in range(a + 1, len(cells)):
            ca, cb = cells[a], cells[b]
            if ca.size != cb.size or ca.size == 0:
                continue
            ca_f, cb_f = ca.flatten().astype(np.float64), cb.flatten().astype(np.float64)
            if ca_f.std() == 0 or cb_f.std() == 0:
                continue
            sims.append(float(np.corrcoef(ca_f, cb_f)[0, 1]))
    return float(np.mean(sims)) if sims else float("nan")

def central_peak_ratio(R, radius=20):
    h, w = R.shape
    cy, cx = h // 2, w // 2
    center = R[cy - radius:cy + radius, cx - radius:cx + radius]
    return float(center.mean() / R.mean()) if R.mean() > 0 else float("nan")

def directional_quadrants(profile):
    n = len(profile)
    D = profile[:, None] - profile[None, :]
    h = n // 2
    return {"Q1": float(D[:h, :h].mean()), "Q2": float(D[:h, h:].mean()),
            "Q3": float(D[h:, :h].mean()), "Q4": float(D[h:, h:].mean())}

def mi_center_cells(R, rows, cols, radius=50):
    h, w = R.shape
    cy, cx = h // 2, w // 2
    center = R[cy - radius:cy + radius, cx - radius:cx + radius]
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
            cell = R[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            if cell.size == 0:
                continue
            ccx = (cols[j] + cols[j + 1]) // 2
            ccy = (rows[i] + rows[i + 1]) // 2
            dists.append(np.sqrt((ccx - cx) ** 2 + (ccy - cy) ** 2))
            mis.append(mi(center, cell))
    if len(mis) < 3:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    corr, _ = pearsonr(dists, mis)
    return {"mean_MI": float(np.mean(mis)), "correlation_MI_distance": float(corr)}

def all_metrics(R, profile):
    rows, cols = detect_grid(R)
    return {
        "density": float(R.mean()),
        "fractal_dimension": box_counting_dimension(R),
        "multifractal": multifractal_spectrum_multiscale(R),
        "grid_rows": int(len(rows)), "grid_cols": int(len(cols)),
        "cell_similarity": cell_similarity(R, rows, cols),
        "central_peak_ratio": central_peak_ratio(R),
        "directional": directional_quadrants(profile),
        "mi_center_cells": mi_center_cells(R, rows, cols),
    }

# ============================================================================
# CONSTRUCCION DE MATRICES EN GPU
# ============================================================================
def build_recurrence_gpu(profile, threshold=10.0, sigma=15.0):
    """Matriz de recurrencia en GPU (torch)."""
    p = torch.from_numpy(ndimage.gaussian_filter1d(profile, sigma=sigma).astype(np.float32)).to(DEVICE)
    diff = torch.abs(p.unsqueeze(0) - p.unsqueeze(1))
    R = (diff < threshold).float().cpu().numpy()
    return R

def build_control_matrix_gpu(profile, kind, rng_state):
    """Genera perfil de control y su matriz de recurrencia en GPU."""
    rng = np.random.default_rng(rng_state)
    if kind == "permutation":
        p = rng.permutation(profile)
    elif kind == "gaussian":
        p = rng.normal(profile.mean(), profile.std(), size=len(profile))
    elif kind == "ar1":
        phi = 0.9
        n = len(profile)
        noise = rng.normal(0, profile.std() * np.sqrt(1 - phi**2), size=n)
        x = np.zeros(n)
        x[0] = noise[0]
        for i in range(1, n):
            x[i] = phi * x[i - 1] + noise[i]
        p = x - x.mean() + profile.mean()
    else:
        raise ValueError(kind)
    pt = torch.from_numpy(p.astype(np.float32)).to(DEVICE)
    diff = torch.abs(pt.unsqueeze(0) - pt.unsqueeze(1))
    R = (diff < 10.0).float().cpu().numpy()
    return R, p

# ============================================================================
# WORKER PARA MULTIPROCESSING (metricas de un control)
# ============================================================================
def worker_metrics(args):
    R, profile = args
    return all_metrics(R, profile)

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"config": {"n_controls": N_CONTROLS, "n_workers": N_WORKERS, "seed": SEED,
                         "device": str(DEVICE)},
              "images": {}, "significance": {}, "comparison_original": {}}

    # --- 1. Partir imagenes y seleccionar mejor mitad ---
    print("=" * 70, flush=True)
    print("PASO 1: Partir JPGs en mitades y evaluar calidad", flush=True)
    print("=" * 70, flush=True)
    candidates = {}
    for jpg in ["Jeshua1.jpg", "Jeshua2.jpg"]:
        img = cv2.imread(os.path.join(REV, jpg), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        mid = w // 2
        for side, half in [("izquierda", img[:, :mid]), ("derecha", img[:, mid:])]:
            hh, ww = half.shape
            lap = cv2.Laplacian(half, cv2.CV_64F).var()
            p = half[:, ww // 2]
            col_var = half.var(axis=0)
            n_empty = int((col_var < 1).sum())
            candidates[f"{jpg}_{side}"] = {
                "side": side, "source": jpg, "shape": [hh, ww],
                "sharpness": float(lap), "contrast": float(half.std()),
                "profile_std": float(p.std()), "empty_cols": n_empty, "half": half,
            }
            print(f"  {jpg} [{side}]: {ww}x{hh} | nitidez={lap:.0f} | perfil_std={p.std():.1f} | col_vacias={n_empty}", flush=True)

    best_key = max(candidates, key=lambda k: (
        candidates[k]["sharpness"] if candidates[k]["profile_std"] > 5 and candidates[k]["empty_cols"] < 5 else 0))
    best = candidates[best_key]
    print(f"\n  MEJOR MITAD: {best_key} ({best['shape'][1]}x{best['shape'][0]})", flush=True)
    report["images"]["selected"] = best_key
    report["images"]["selected_quality"] = {k: v for k, v in best.items() if k != "half"}

    # --- 2. Matriz de recurrencia en GPU ---
    print("\n" + "=" * 70, flush=True)
    print("PASO 2: Matriz de recurrencia (GPU)", flush=True)
    print("=" * 70, flush=True)
    half = best["half"]
    hh, ww = half.shape
    profile = half[:, ww // 2].astype(np.float64)
    R = build_recurrence_gpu(profile)
    print(f"  Matriz: {R.shape}, densidad = {R.mean():.4f}", flush=True)

    # --- 3. Metricas reales ---
    print("\n" + "=" * 70, flush=True)
    print("PASO 3: Metricas de la matriz REAL (alta resolucion)", flush=True)
    print("=" * 70, flush=True)
    m_real = all_metrics(R, profile)
    report["images"]["metrics_real"] = m_real
    print(f"  D fractal        = {m_real['fractal_dimension']:.4f}  (estudio: 1.642)", flush=True)
    print(f"  Delta_alpha      = {m_real['multifractal']['delta_alpha']:.4f}  (estudio D2: 4.7652)", flush=True)
    print(f"  Grid             = {m_real['grid_rows']}x{m_real['grid_cols']}  (estudio: 14x14)", flush=True)
    print(f"  Similitud celdas = {m_real['cell_similarity']:.4f}  (estudio: 0.648)", flush=True)
    print(f"  Pico central     = {m_real['central_peak_ratio']:.4f}", flush=True)
    print(f"  MI-distancia     = {m_real['mi_center_cells']['correlation_MI_distance']:.4f}  (estudio D13: -0.1341)", flush=True)
    print(f"  Direccional Q2/Q3= {m_real['directional']['Q2']:.4f} / {m_real['directional']['Q3']:.4f}  (estudio: -0.696/+0.696)", flush=True)

    # --- 4. Controles negativos (GPU + multiprocessing) ---
    print("\n" + "=" * 70, flush=True)
    print(f"PASO 4: Controles negativos ({N_CONTROLS} x 3 tipos, GPU + {N_WORKERS} workers)", flush=True)
    print("=" * 70, flush=True)
    kinds = ["permutation", "gaussian", "ar1"]
    control_metrics = {k: [] for k in kinds}
    for kind in kinds:
        t_kind = time.time()
        print(f"  [{kind}] generando matrices en GPU...", flush=True)
        tasks = []
        for i in range(N_CONTROLS):
            Rc, pc = build_control_matrix_gpu(profile, kind, SEED + i * 3 + kinds.index(kind))
            tasks.append((Rc, pc))
        print(f"  [{kind}] {N_CONTROLS} matrices listas ({time.time()-t_kind:.1f}s), calculando metricas con {N_WORKERS} workers...", flush=True)
        with Pool(N_WORKERS) as pool:
            results = []
            for idx, res in enumerate(pool.imap_unordered(worker_metrics, tasks)):
                results.append(res)
                if (idx + 1) % 10 == 0 or idx + 1 == N_CONTROLS:
                    print(f"    progreso: {idx+1}/{N_CONTROLS} ({time.time()-t_kind:.0f}s)", flush=True)
        control_metrics[kind] = results
        print(f"  [{kind}] completado en {time.time()-t_kind:.1f}s", flush=True)

    # --- 5. Significancia ---
    print("\n" + "=" * 70, flush=True)
    print("PASO 5: Significancia estadistica", flush=True)
    print("=" * 70, flush=True)
    sig = {}
    for kind in kinds:
        sig[kind] = {}
        for metric, real_val in [("fractal_dimension", m_real["fractal_dimension"]),
                                 ("cell_similarity", m_real["cell_similarity"]),
                                 ("central_peak_ratio", m_real["central_peak_ratio"]),
                                 ("grid_rows", m_real["grid_rows"]),
                                 ("grid_cols", m_real["grid_cols"])]:
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
            real_da = m_real["multifractal"]["delta_alpha"]
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
            real_mi = m_real["mi_center_cells"]["correlation_MI_distance"]
            z = (real_mi - vals.mean()) / vals.std() if vals.std() > 0 else float("nan")
            pct = float((vals < real_mi).mean() * 100)
            sig[kind]["mi_distance_corr"] = {"real": real_mi, "control_mean": float(vals.mean()),
                                             "control_std": float(vals.std()), "z_score": float(z),
                                             "percentile": pct}
            print(f"  [{kind}] MI-distancia: real={real_mi:.4f} | control={vals.mean():.4f}±{vals.std():.4f} | z={z:+.2f} | pct={pct:.1f}%", flush=True)
    report["significance"] = sig

    # --- 6. Comparacion con hallazgos originales ---
    print("\n" + "=" * 70, flush=True)
    print("PASO 6: Comparacion con hallazgos del estudio original", flush=True)
    print("=" * 70, flush=True)
    comparison = {
        "fractal_dimension": {"original": 1.642, "reproducido": m_real["fractal_dimension"],
                              "coincide": abs(m_real["fractal_dimension"] - 1.642) < 0.1},
        "delta_alpha": {"original": 4.7652, "reproducido": m_real["multifractal"]["delta_alpha"],
                        "coincide": abs(m_real["multifractal"]["delta_alpha"] - 4.7652) < 1.0},
        "grid": {"original": "14x14", "reproducido": f"{m_real['grid_rows']}x{m_real['grid_cols']}",
                 "coincide": m_real["grid_rows"] == 14 and m_real["grid_cols"] == 14},
        "cell_similarity": {"original": 0.648, "reproducido": m_real["cell_similarity"],
                            "coincide": abs(m_real["cell_similarity"] - 0.648) < 0.1},
        "mi_distance": {"original": -0.1341, "reproducido": m_real["mi_center_cells"]["correlation_MI_distance"],
                        "coincide": abs(m_real["mi_center_cells"]["correlation_MI_distance"] - (-0.1341)) < 0.1},
        "directional_Q2": {"original": -0.696, "reproducido": m_real["directional"]["Q2"],
                           "coincide": abs(m_real["directional"]["Q2"] - (-0.696)) < 0.1},
        "directional_Q3": {"original": 0.696, "reproducido": m_real["directional"]["Q3"],
                           "coincide": abs(m_real["directional"]["Q3"] - 0.696) < 0.1},
    }
    report["comparison_original"] = comparison
    for k, v in comparison.items():
        status = "OK" if v["coincide"] else "NO"
        print(f"  [{status}] {k}: original={v['original']} | reproducido={v['reproducido']}", flush=True)

    # --- 7. Guardar ---
    out_json = os.path.join(OUT, "reverificacion_jeshua_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nResultados guardados en: {out_json}", flush=True)
    print(f"Tiempo total: {time.time() - t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
