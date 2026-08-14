"""
PIPELINE DE RE-VERIFICACION ESTADISTICA
========================================
Estudio: Sabana Santa de Turin - Analisis de Matrices de Recurrencia
Objetivo: Someter los hallazgos clave a controles negativos (hipotesis nula)
          y verificar su significancia estadistica.

Metodologia:
  1. Reproducir exactamente la matriz de recurrencia del estudio original
     (perfil columna central, gaussiano sigma=15, umbral 10.0)
  2. Generar 100 controles x 3 tipos (permutacion, gaussiano, AR1)
  3. Re-ejecutar las metricas clave sobre la matriz real y los controles
  4. Calcular z-scores, percentiles y p-valores empiricos
  5. Barrido de umbrales de recurrencia (5/10/15/20)
  6. Diagnostico de imagen2 (fallo silencioso del estudio original)

No modifica ningun archivo original. Todo se guarda en Re_verificacion/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.stats import pearsonr

# ============================================================================
# CONFIGURACION
# ============================================================================
BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
IMG3 = os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg")
IMG2 = os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen2_dos_caras.jpeg")
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

N_CONTROLS = 100
SEED = 42
rng = np.random.default_rng(SEED)

# ============================================================================
# 1. REPRODUCCION DE LA MATRIZ DE RECURRENCIA (metodo original)
# ============================================================================
def build_recurrence(img_path, threshold=10.0, sigma=15.0, normalize=False):
    """Replica el metodo del estudio: perfil columna central + gaussiano + umbral."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    if normalize:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    else:
        img = img.astype(np.float64)
    h, w = img.shape
    profile = img[:, w // 2]
    profile_smooth = ndimage.gaussian_filter1d(profile, sigma=sigma)
    n = len(profile_smooth)
    diff = np.abs(profile_smooth[:, None] - profile_smooth[None, :])
    R = (diff < threshold).astype(np.float32)
    return R, profile_smooth

# ============================================================================
# 2. GENERACION DE CONTROLES (hipotesis nula)
# ============================================================================
def control_permutation(profile):
    """Permuta el perfil real: destruye estructura, conserva distribucion."""
    return rng.permutation(profile)

def control_gaussian(profile):
    """Ruido gaussiano con misma media y desviacion que el perfil real."""
    return rng.normal(profile.mean(), profile.std(), size=len(profile))

def control_ar1(profile):
    """Ruido coloreado AR(1): autocorrelacion como senal real, sin estructura."""
    phi = 0.9
    n = len(profile)
    noise = rng.normal(0, profile.std() * np.sqrt(1 - phi**2), size=n)
    x = np.zeros(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    x = x - x.mean() + profile.mean()
    return x

def build_control_matrix(profile, kind):
    if kind == "permutation":
        p = control_permutation(profile)
    elif kind == "gaussian":
        p = control_gaussian(profile)
    elif kind == "ar1":
        p = control_ar1(profile)
    else:
        raise ValueError(kind)
    diff = np.abs(p[:, None] - p[None, :])
    return (diff < 10.0).astype(np.float32), p

# ============================================================================
# 3. METRICAS CLAVE (replicando los metodos del estudio)
# ============================================================================
def box_counting_dimension(R, min_size=2, max_size=None):
    """Dimension fractal por box-counting (CHIP-6)."""
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
        occupied = (blocks.sum(axis=(1, 3)) > 0).sum()
        counts.append(occupied)
    counts = np.array(counts)
    valid = counts > 0
    if valid.sum() < 3:
        return float("nan")
    log_s = np.log(1.0 / sizes[valid])
    log_n = np.log(counts[valid])
    slope, _ = np.polyfit(log_s, log_n, 1)
    return float(slope)

def multifractal_spectrum_multiscale(R, box_sizes=(4, 8, 16, 32), q_values=None):
    """Espectro multifractal MULTI-ESCALA (fix del bug del estudio que usaba
    una sola escala de caja). Regresion log-log de tau(q) sobre escalas."""
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    h, w = R.shape
    tau_q = np.zeros(len(q_values))
    for qi, q in enumerate(q_values):
        if q == 0:
            tau_q[qi] = 0.0
            continue
        log_eps = []
        log_sum = []
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
                "skewness": float("nan"), "alpha_min": float("nan"),
                "alpha_max": float("nan")}
    alpha_v = alpha[valid]
    f_v = f_alpha[valid]
    a_min, a_max = alpha_v.min(), alpha_v.max()
    a_peak = alpha_v[np.argmax(f_v)]
    return {
        "delta_alpha": float(a_max - a_min),
        "alpha_peak": float(a_peak),
        "skewness": float((a_max - a_peak) - (a_peak - a_min)),
        "alpha_min": float(a_min),
        "alpha_max": float(a_max),
    }

def detect_grid(R, min_distance=10, prominence=0.05):
    """Deteccion de grid por proyecciones fila/columna (CHIP-4)."""
    h, w = R.shape
    row_proj = R.mean(axis=1)
    col_proj = R.mean(axis=0)
    row_proj_s = ndimage.gaussian_filter1d(row_proj, sigma=3)
    col_proj_s = ndimage.gaussian_filter1d(col_proj, sigma=3)
    thr = row_proj_s.mean() + 0.5 * row_proj_s.std()
    rows, _ = find_peaks(row_proj_s, distance=min_distance, prominence=prominence * row_proj_s.std())
    cols, _ = find_peaks(col_proj_s, distance=min_distance, prominence=prominence * col_proj_s.std())
    return rows, cols

def cell_similarity(R, rows, cols, n_cells=25):
    """Similitud media entre celdas del grid (CHIP-5)."""
    if len(rows) < 2 or len(cols) < 2:
        return float("nan")
    # Tomar hasta 5x5 celdas centrales
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
            ca_f = ca.flatten().astype(np.float64)
            cb_f = cb.flatten().astype(np.float64)
            if ca_f.std() == 0 or cb_f.std() == 0:
                continue
            sims.append(float(np.corrcoef(ca_f, cb_f)[0, 1]))
    if not sims:
        return float("nan")
    return float(np.mean(sims))

def central_peak_ratio(R, radius=20):
    """Densidad del centro vs densidad global (CHIP-8 / P6)."""
    h, w = R.shape
    cy, cx = h // 2, w // 2
    center = R[cy - radius:cy + radius, cx - radius:cx + radius]
    c_dens = center.mean()
    g_dens = R.mean()
    return float(c_dens / g_dens) if g_dens > 0 else float("nan")

def directional_quadrants(profile):
    """Matriz direccional D(i,j)=profile[i]-profile[j], medias por cuadrante (A2)."""
    n = len(profile)
    D = profile[:, None] - profile[None, :]
    h = n // 2
    Q1 = D[:h, :h]
    Q2 = D[:h, h:]
    Q3 = D[h:, :h]
    Q4 = D[h:, h:]
    return {
        "Q1": float(Q1.mean()), "Q2": float(Q2.mean()),
        "Q3": float(Q3.mean()), "Q4": float(Q4.mean()),
    }

def mi_center_cells(R, rows, cols, radius=50):
    """MI entre centro y celdas + correlacion con distancia (D13)."""
    h, w = R.shape
    cy, cx = h // 2, w // 2
    center = R[cy - radius:cy + radius, cx - radius:cx + radius]
    if center.size == 0 or len(rows) < 2 or len(cols) < 2:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    # MI discreta con bins
    def mi(a, b, bins=16):
        a = np.clip(a, 0, 1).astype(np.uint8)
        b = np.clip(b, 0, 1).astype(np.uint8)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                c[i, j] = np.mean((a == i) & (b == j))
        c /= c.sum()
        pa = c.sum(axis=1)
        pb = c.sum(axis=0)
        m = 0.0
        for i in range(2):
            for j in range(2):
                if c[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                    m += c[i, j] * np.log2(c[i, j] / (pa[i] * pb[j]))
        return m
    mis = []
    dists = []
    for i in range(len(rows) - 1):
        for j in range(len(cols) - 1):
            cell = R[rows[i]:rows[i + 1], cols[j]:cols[j + 1]]
            if cell.size == 0:
                continue
            ccx = (cols[j] + cols[j + 1]) // 2
            ccy = (rows[i] + rows[i + 1]) // 2
            dist = np.sqrt((ccx - cx) ** 2 + (ccy - cy) ** 2)
            m = mi(center, cell)
            mis.append(m)
            dists.append(dist)
    if len(mis) < 3:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    corr, _ = pearsonr(dists, mis)
    return {"mean_MI": float(np.mean(mis)), "correlation_MI_distance": float(corr)}

def all_metrics(R, profile):
    """Ejecuta todas las metricas clave sobre una matriz."""
    rows, cols = detect_grid(R)
    m = {}
    m["density"] = float(R.mean())
    m["fractal_dimension"] = box_counting_dimension(R)
    m["multifractal"] = multifractal_spectrum_multiscale(R)
    m["grid_rows"] = int(len(rows))
    m["grid_cols"] = int(len(cols))
    m["cell_similarity"] = cell_similarity(R, rows, cols)
    m["central_peak_ratio"] = central_peak_ratio(R)
    m["directional"] = directional_quadrants(profile)
    m["mi_center_cells"] = mi_center_cells(R, rows, cols)
    return m

# ============================================================================
# 4. EJECUCION PRINCIPAL
# ============================================================================
def main():
    t0 = time.time()
    report = {"config": {"n_controls": N_CONTROLS, "seed": SEED,
                         "threshold": 10.0, "sigma": 15.0},
              "reproduction": {}, "controls": {}, "threshold_sweep": {},
              "imagen2_diagnosis": {}}

    # --- 4.1 Reproduccion de la matriz real (imagen3) ---
    print("=" * 70)
    print("PASO 1: Reproduccion de la matriz de recurrencia (imagen3)")
    print("=" * 70)
    R_real, profile_real = build_recurrence(IMG3)
    print(f"  Matriz: {R_real.shape}, densidad = {R_real.mean():.4f}")
    print(f"  (estudio original: 1080x1080, densidad 0.1338)")
    report["reproduction"]["density"] = float(R_real.mean())
    report["reproduction"]["shape"] = list(R_real.shape)
    report["reproduction"]["matches_original"] = abs(R_real.mean() - 0.1338) < 0.01

    # --- 4.2 Metricas de la matriz real ---
    print("\nPASO 2: Metricas de la matriz REAL")
    m_real = all_metrics(R_real, profile_real)
    report["reproduction"]["metrics_real"] = m_real
    print(f"  D fractal        = {m_real['fractal_dimension']:.4f}  (estudio: 1.642)")
    print(f"  Delta_alpha      = {m_real['multifractal']['delta_alpha']:.4f}  (estudio D2: 4.7652)")
    print(f"  Grid             = {m_real['grid_rows']}x{m_real['grid_cols']}  (estudio: 14x14)")
    print(f"  Similitud celdas = {m_real['cell_similarity']:.4f}  (estudio: 0.648)")
    print(f"  Pico central     = {m_real['central_peak_ratio']:.4f}")
    print(f"  MI-distancia     = {m_real['mi_center_cells']['correlation_MI_distance']:.4f}  (estudio D13: -0.1341)")
    print(f"  Direccional Q2/Q3= {m_real['directional']['Q2']:.4f} / {m_real['directional']['Q3']:.4f}  (estudio: -0.696/+0.696)")

    # --- 4.3 Controles negativos ---
    print("\n" + "=" * 70)
    print(f"PASO 3: Controles negativos ({N_CONTROLS} x 3 tipos)")
    print("=" * 70)
    kinds = ["permutation", "gaussian", "ar1"]
    control_metrics = {k: [] for k in kinds}
    for kind in kinds:
        for i in range(N_CONTROLS):
            Rc, pc = build_control_matrix(profile_real, kind)
            mc = all_metrics(Rc, pc)
            control_metrics[kind].append(mc)
        print(f"  {kind}: {N_CONTROLS} controles generados")

    # --- 4.4 Significancia estadistica ---
    print("\n" + "=" * 70)
    print("PASO 4: Significancia estadistica (z-scores, percentiles, p-valores)")
    print("=" * 70)
    sig = {}
    for kind in kinds:
        sig[kind] = {}
        for metric, real_val in [("fractal_dimension", m_real["fractal_dimension"]),
                                 ("cell_similarity", m_real["cell_similarity"]),
                                 ("central_peak_ratio", m_real["central_peak_ratio"]),
                                 ("grid_rows", m_real["grid_rows"]),
                                 ("grid_cols", m_real["grid_cols"])]:
            vals = [mc[metric] for mc in control_metrics[kind] if not (isinstance(mc[metric], float) and np.isnan(mc[metric]))]
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
            print(f"  [{kind}] {metric}: real={real_val:.4f} | control mean={mean:.4f}±{std:.4f} | z={z:+.2f} | pct={pct:.1f}% | p={pval:.4f}")
        # Multifractal
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
            print(f"  [{kind}] delta_alpha: real={real_da:.4f} | control mean={vals.mean():.4f}±{vals.std():.4f} | z={z:+.2f} | pct={pct:.1f}%")
        # MI-distancia
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
            print(f"  [{kind}] MI-distancia: real={real_mi:.4f} | control mean={vals.mean():.4f}±{vals.std():.4f} | z={z:+.2f} | pct={pct:.1f}%")
    report["significance"] = sig

    # --- 4.5 Barrido de umbrales ---
    print("\n" + "=" * 70)
    print("PASO 5: Barrido de umbrales de recurrencia (imagen3)")
    print("=" * 70)
    sweep = {}
    for thr in [5.0, 10.0, 15.0, 20.0]:
        R_t, _ = build_recurrence(IMG3, threshold=thr)
        rows, cols = detect_grid(R_t)
        m_t = all_metrics(R_t, profile_real)
        sweep[str(thr)] = {"density": float(R_t.mean()), "grid": f"{len(rows)}x{len(cols)}",
                           "fractal_dimension": m_t["fractal_dimension"],
                           "central_peak_ratio": m_t["central_peak_ratio"]}
        print(f"  umbral={thr:5.1f}: densidad={R_t.mean():.4f} | grid={len(rows)}x{len(cols)} | D={m_t['fractal_dimension']:.4f} | pico={m_t['central_peak_ratio']:.4f}")
    report["threshold_sweep"] = sweep

    # --- 4.6 Diagnostico imagen2 ---
    print("\n" + "=" * 70)
    print("PASO 6: Diagnostico imagen2 (fallo silencioso del estudio)")
    print("=" * 70)
    img2 = cv2.imread(IMG2, cv2.IMREAD_GRAYSCALE)
    diag2 = {}
    if img2 is None:
        diag2["error"] = "no se pudo leer"
    else:
        h2, w2 = img2.shape
        diag2["shape"] = [h2, w2]
        diag2["dtype"] = str(img2.dtype)
        diag2["min_max"] = [int(img2.min()), int(img2.max())]
        diag2["mean_std"] = [float(img2.mean()), float(img2.std())]
        diag2["contrast"] = float(img2.std() / (img2.mean() + 1e-9))
        R2, p2 = build_recurrence(IMG2)
        if R2 is not None:
            diag2["recurrence_density"] = float(R2.mean())
            diag2["recurrence_shape"] = list(R2.shape)
            diag2["profile_std"] = float(p2.std())
            diag2["profile_range"] = [float(p2.min()), float(p2.max())]
            # % de pares dentro del umbral
            diag2["pairs_within_threshold_pct"] = float((np.abs(p2[:, None] - p2[None, :]) < 10.0).mean() * 100)
        print(f"  shape={diag2.get('shape')} | contraste={diag2.get('contrast', 'N/A'):.3f} | densidad recurrencia={diag2.get('recurrence_density', 'N/A')}")
        print(f"  rango perfil=[{diag2.get('profile_range', ['?','?'])[0]:.1f}, {diag2.get('profile_range', ['?','?'])[1]:.1f}] | std={diag2.get('profile_std', '?'):.1f}")
        print(f"  pares dentro umbral: {diag2.get('pairs_within_threshold_pct', '?'):.2f}%")
    report["imagen2_diagnosis"] = diag2

    # --- 4.7 Guardar ---
    out_json = os.path.join(OUT, "reverificacion_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=lambda o: bool(o) if isinstance(o, (np.bool_, bool)) else float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nResultados guardados en: {out_json}")
    print(f"Tiempo total: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
