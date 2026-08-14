"""
ANALISIS FINAL: METODOS EXACTOS DEL ESTUDIO SOBRE JESHUA2 CORREGIDA
===================================================================
Paso 1 (hecho): correccion de iluminacion verificada.
  - perfil_highpass: gradiente 14% del std (igual que imagen3 15%),
    estructura preservada 99.7%. MEJOR METODO.

Este script:
  1. Corrige TODA la imagen (cada columna con su tendencia de baja
     frecuencia, mismo sigma_hp) - coherente para cualquier metrica
  2. Aplica los metodos EXACTOS del estudio (CHIP y D) sobre:
     - imagen3 corregida (parametros originales) [referencia]
     - Jeshua2-izq corregida (parametros escalados x2.149)
  3. Ejecuta controles negativos (100 x 3 tipos) sobre el perfil
     corregido de Jeshua2 con parametros escalados
  4. Compara todo y guarda resultados

NO modifica ningun archivo original.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import pearsonr
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
TMP = "/tmp/opencode"
rng = np.random.default_rng(42)
N_CONTROLS = 100
N_WORKERS = 12
FACTOR = 2321 / 1080  # 2.149

# ============================================================================
# 1. CORRECCION DE ILUMINACION (highpass por columna)
# ============================================================================
def corregir_iluminacion_highpass(img, sigma_hp=None):
    """Resta a cada columna su tendencia de baja frecuencia (highpass).
    Preserva la estructura local (99.7%) y elimina el gradiente global."""
    img = img.astype(np.float32)
    if sigma_hp is None:
        sigma_hp = img.shape[0] / 8.0
    tendencia = ndimage.gaussian_filter1d(img, sigma=sigma_hp, axis=0)
    corr = img - tendencia + img.mean()
    return np.clip(corr, 0, 255).astype(np.uint8)

# ============================================================================
# 2. METODOS EXACTOS DEL ESTUDIO
# ============================================================================
def metodo_chip(img, ksize=15, threshold=10.0):
    """analisis_chip_profundo.py: normaliza + GaussianBlur((ksize,1),0)."""
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (ksize,1), 0).flatten()
    R = (np.abs(perfil[:, None] - perfil[None, :]) < threshold).astype(float)
    return R, perfil

def metodo_d(img, sigma=15.0, threshold=10.0):
    """tests_D1_D10: imagen cruda + gaussian_filter1d(sigma)."""
    h, w = img.shape
    perfil = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=sigma)
    R = (np.abs(perfil[:, None] - perfil[None, :]) < threshold).astype(float)
    return R, perfil

# ============================================================================
# 3. METRICAS DEL ESTUDIO
# ============================================================================
def box_counting_2d(binary_img):
    """CHIP-6 exacto."""
    p = max(binary_img.shape)
    n = 2**int(np.ceil(np.log2(p)))
    padded = np.pad(binary_img, ((0, n-binary_img.shape[0]), (0, n-binary_img.shape[1])), 'constant')
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = []
    for s in sizes:
        reshaped = padded.reshape(n//s, s, n//s, s)
        counts.append(np.sum(np.any(reshaped, axis=(1,3))))
    counts = np.array(counts)
    valid = counts > 0
    if np.sum(valid) < 2:
        return 0.0
    return -np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)[0]

def grid_y_cruz(R, gap=10):
    """CHIP-4 + CHIP-8 exactos."""
    n = R.shape[0]
    qsize = n // 2
    q = R[:qsize, :qsize]
    rd = np.mean(q, axis=1); cd = np.mean(q, axis=0)
    thr_r = np.mean(rd) + np.std(rd); thr_c = np.mean(cd) + np.std(cd)
    sr = np.where(rd > thr_r)[0]; sc = np.where(cd > thr_c)[0]
    def gl(lines, gap):
        if len(lines) == 0:
            return []
        groups = []; cur = [lines[0]]
        for line in lines[1:]:
            if line - cur[-1] <= gap:
                cur.append(line)
            else:
                groups.append(int(np.mean(cur))); cur = [line]
        groups.append(int(np.mean(cur)))
        return groups
    gr = gl(sr, gap); gc = gl(sc, gap)
    cx = int(np.argmax(np.mean(q, axis=0)))
    cy = int(np.argmax(np.mean(q, axis=1)))
    return len(gr), len(gc), cx, cy, qsize, gr, gc

def similitud_celdas(R, gr, gc):
    """CHIP-5 exacto: fraccion de pixeles iguales."""
    n = R.shape[0]
    q = R[:n//2, :n//2]
    if len(gr) < 2 or len(gc) < 2:
        return float("nan"), 0
    cells = []
    for i in range(min(5, len(gr)-1)):
        for j in range(min(5, len(gc)-1)):
            r1, r2 = gr[i], gr[i+1]
            c1, c2 = gc[j], gc[j+1]
            if r2-r1 > 10 and c2-c1 > 10:
                cells.append(q[r1:r2, c1:c2])
    if len(cells) < 2:
        return float("nan"), 0
    cs = min(c.shape[0] for c in cells)
    cells_r = np.array([cv2.resize(c, (cs, cs)) for c in cells])
    nc = len(cells_r)
    sm = np.zeros((nc, nc))
    for i in range(nc):
        for j in range(nc):
            sm[i, j] = np.mean(cells_r[i] == cells_r[j])
    return float(np.mean(sm[np.triu_indices(nc, 1)])), nc

def multifractal_multiscale(R, box_sizes=(4, 8, 16, 32), q_values=None):
    """Espectro multifractal MULTI-ESCALA (fix del bug de una escala)."""
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
            nh, nw = h // s, w // s
            if nh == 0 or nw == 0:
                continue
            blocks = R[:nh*s, :nw*s].reshape(nh, s, nw, s)
            measure = blocks.sum(axis=(1,3)) / (s*s)
            measure = measure[measure > 0]
            if len(measure) == 0:
                continue
            measure = measure / measure.sum()
            log_eps.append(np.log(1.0/s))
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
        return {"delta_alpha": float("nan")}
    av, fv = alpha[valid], f_alpha[valid]
    return {"delta_alpha": float(av.max() - av.min())}

def central_peak_ratio(R, radius=20):
    n = R.shape[0]
    cy, cx = n//2, n//2
    c = R[cy-radius:cy+radius, cx-radius:cx+radius]
    return float(c.mean() / R.mean()) if R.mean() > 0 else float("nan")

def directional_quadrants(profile):
    n = len(profile)
    D = profile[:, None] - profile[None, :]
    h = n // 2
    return {"Q1": float(D[:h,:h].mean()), "Q2": float(D[:h,h:].mean()),
            "Q3": float(D[h:,:h].mean()), "Q4": float(D[h:,h:].mean())}

def mi_center_cells(R, gr, gc, radius=50):
    n = R.shape[0]
    cy, cx = n//2, n//2
    center = R[cy-radius:cy+radius, cx-radius:cx+radius]
    if center.size == 0 or len(gr) < 2 or len(gc) < 2:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    def mi(a, b):
        a = np.clip(a, 0, 1).astype(np.uint8)
        b = np.clip(b, 0, 1).astype(np.uint8)
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                c[i,j] = np.mean((a == i) & (b == j))
        c /= c.sum()
        pa, pb = c.sum(axis=1), c.sum(axis=0)
        m = 0.0
        for i in range(2):
            for j in range(2):
                if c[i,j] > 0 and pa[i] > 0 and pb[j] > 0:
                    m += c[i,j] * np.log2(c[i,j] / (pa[i]*pb[j]))
        return m
    mis, dists = [], []
    for i in range(len(gr)-1):
        for j in range(len(gc)-1):
            cell = R[gr[i]:gr[i+1], gc[j]:gc[j+1]]
            if cell.size == 0:
                continue
            ccx = (gc[j]+gc[j+1])//2; ccy = (gr[i]+gr[i+1])//2
            dists.append(np.sqrt((ccx-cx)**2 + (ccy-cy)**2))
            mis.append(mi(center, cell))
    if len(mis) < 3:
        return {"mean_MI": float("nan"), "correlation_MI_distance": float("nan")}
    corr, _ = pearsonr(dists, mis)
    return {"mean_MI": float(np.mean(mis)), "correlation_MI_distance": float(corr)}

def analisis_completo(R, profile, gap=10, radius=20):
    gr_n, gc_n, cx, cy, qsize, gr, gc = grid_y_cruz(R, gap)
    sim, nc = similitud_celdas(R, gr, gc)
    return {
        "densidad": float(R.mean()),
        "grid": f"{gr_n}x{gc_n}",
        "grid_rows": gr_n, "grid_cols": gc_n,
        "cruz_abs": (cx, cy),
        "cruz_rel": (round(cx/qsize, 3), round(cy/qsize, 3)),
        "similitud_celdas": sim,
        "n_celdas": nc,
        "D_fractal": float(box_counting_2d(R)),
        "delta_alpha": multifractal_multiscale(R)["delta_alpha"],
        "pico_central": central_peak_ratio(R, radius),
        "directional": directional_quadrants(profile),
        "mi_center_cells": mi_center_cells(R, gr, gc, radius),
    }

# ============================================================================
# 4. CONTROLES
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
        x[i] = phi * x[i-1] + noise[i]
    return x - x.mean() + p.mean()

def worker_metrics(args):
    R, profile, gap, radius = args
    return analisis_completo(R, profile, gap, radius)

# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"factor": FACTOR, "imagen3_corregida": {}, "jeshua2_corregida": {},
              "controles_jeshua2": {}, "comparacion": {}}

    # Cargar y corregir
    i3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    j2 = cv2.imread(os.path.join(BASE, "Re_verificacion", "Jeshua2.jpg"), cv2.IMREAD_GRAYSCALE)
    j2_izq = j2[:, :j2.shape[1]//2]

    i3_c = corregir_iluminacion_highpass(i3)
    j2_c = corregir_iluminacion_highpass(j2_izq)
    cv2.imwrite(os.path.join(TMP, "jeshua2_izq_corregida.png"), j2_c)
    cv2.imwrite(os.path.join(TMP, "imagen3_corregida.png"), i3_c)

    # Verificar gradiente residual
    def grad(img, sigma=15.0):
        h, w = img.shape
        p = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=sigma)
        m, _ = np.polyfit(np.arange(len(p)), p, 1)
        return m, p.std()
    m3, s3 = grad(i3_c); mj, sj = grad(j2_c)
    print("=" * 70, flush=True)
    print("GRADIENTE RESIDUAL TRAS CORRECCION", flush=True)
    print("=" * 70, flush=True)
    print(f"  imagen3 corregida:  pendiente={m3:+.4f}/px | {abs(m3)*i3.shape[0]/s3*100:.0f}% del std", flush=True)
    print(f"  Jeshua2 corregida:  pendiente={mj:+.4f}/px | {abs(mj)*j2_izq.shape[0]/sj*100:.0f}% del std", flush=True)

    # Parametros
    ksize_orig, ksize_esc = 15, int(15*FACTOR) | 1
    sigma_orig, sigma_esc = 15.0, 15.0*FACTOR
    gap_orig, gap_esc = 10, int(10*FACTOR)
    radius_orig, radius_esc = 20, int(20*FACTOR)
    print(f"\n  Parametros: ksize {ksize_orig}->{ksize_esc} | sigma {sigma_orig:.0f}->{sigma_esc:.1f} | gap {gap_orig}->{gap_esc} | radius {radius_orig}->{radius_esc}", flush=True)

    # --- imagen3 corregida con parametros ORIGINALES (referencia) ---
    print("\n" + "=" * 70, flush=True)
    print("IMAGEN3 CORREGIDA (parametros originales) - REFERENCIA", flush=True)
    print("=" * 70, flush=True)
    R_chip3, p_chip3 = metodo_chip(i3_c, ksize_orig)
    a_chip3 = analisis_completo(R_chip3, p_chip3, gap_orig, radius_orig)
    print(f"  [CHIP] dens={a_chip3['densidad']:.4f} | grid={a_chip3['grid']} | D={a_chip3['D_fractal']:.4f} | "
          f"cruz={a_chip3['cruz_abs']} rel={a_chip3['cruz_rel']} | sim={a_chip3['similitud_celdas']:.4f} | "
          f"dalpha={a_chip3['delta_alpha']:.3f} | pico={a_chip3['pico_central']:.2f} | "
          f"Q2={a_chip3['directional']['Q2']:.3f} | MI={a_chip3['mi_center_cells']['correlation_MI_distance']:.4f}", flush=True)
    report["imagen3_corregida"]["metodo_chip"] = a_chip3
    R_d3, p_d3 = metodo_d(i3_c, sigma_orig)
    a_d3 = analisis_completo(R_d3, p_d3, gap_orig, radius_orig)
    print(f"  [D]     dens={a_d3['densidad']:.4f} | grid={a_d3['grid']} | D={a_d3['D_fractal']:.4f} | "
          f"cruz={a_d3['cruz_abs']} rel={a_d3['cruz_rel']} | sim={a_d3['similitud_celdas']:.4f} | "
          f"dalpha={a_d3['delta_alpha']:.3f} | pico={a_d3['pico_central']:.2f} | "
          f"Q2={a_d3['directional']['Q2']:.3f} | MI={a_d3['mi_center_cells']['correlation_MI_distance']:.4f}", flush=True)
    report["imagen3_corregida"]["metodo_d"] = a_d3

    # --- Jeshua2 corregida con parametros ESCALADOS ---
    print("\n" + "=" * 70, flush=True)
    print("JESHUA2-IZQ CORREGIDA (parametros escalados x2.149)", flush=True)
    print("=" * 70, flush=True)
    R_chip_j, p_chip_j = metodo_chip(j2_c, ksize_esc)
    a_chip_j = analisis_completo(R_chip_j, p_chip_j, gap_esc, radius_esc)
    print(f"  [CHIP] dens={a_chip_j['densidad']:.4f} | grid={a_chip_j['grid']} | D={a_chip_j['D_fractal']:.4f} | "
          f"cruz={a_chip_j['cruz_abs']} rel={a_chip_j['cruz_rel']} | sim={a_chip_j['similitud_celdas']:.4f} | "
          f"dalpha={a_chip_j['delta_alpha']:.3f} | pico={a_chip_j['pico_central']:.2f} | "
          f"Q2={a_chip_j['directional']['Q2']:.3f} | MI={a_chip_j['mi_center_cells']['correlation_MI_distance']:.4f}", flush=True)
    report["jeshua2_corregida"]["metodo_chip"] = a_chip_j
    R_d_j, p_d_j = metodo_d(j2_c, sigma_esc)
    a_d_j = analisis_completo(R_d_j, p_d_j, gap_esc, radius_esc)
    print(f"  [D]     dens={a_d_j['densidad']:.4f} | grid={a_d_j['grid']} | D={a_d_j['D_fractal']:.4f} | "
          f"cruz={a_d_j['cruz_abs']} rel={a_d_j['cruz_rel']} | sim={a_d_j['similitud_celdas']:.4f} | "
          f"dalpha={a_d_j['delta_alpha']:.3f} | pico={a_d_j['pico_central']:.2f} | "
          f"Q2={a_d_j['directional']['Q2']:.3f} | MI={a_d_j['mi_center_cells']['correlation_MI_distance']:.4f}", flush=True)
    report["jeshua2_corregida"]["metodo_d"] = a_d_j

    # --- Controles sobre Jeshua2 corregida (metodo D escalado) ---
    print(f"\nCONTROLES JESHUA2 CORREGIDA ({N_CONTROLS} x 3, metodo D escalado):", flush=True)
    kinds = ["permutation", "gaussian", "ar1"]
    control_metrics = {k: [] for k in kinds}
    for kind in kinds:
        tasks = []
        for i in range(N_CONTROLS):
            if kind == "permutation":
                pc = control_permutation(p_d_j)
            elif kind == "gaussian":
                pc = control_gaussian(p_d_j)
            else:
                pc = control_ar1(p_d_j)
            Rc = (np.abs(pc[:, None] - pc[None, :]) < 10.0).astype(float)
            tasks.append((Rc, pc, gap_esc, radius_esc))
        with Pool(N_WORKERS) as pool:
            res = list(pool.imap_unordered(worker_metrics, tasks))
        control_metrics[kind] = res
        print(f"  [{kind}] {N_CONTROLS} controles listos", flush=True)

    # --- Significancia ---
    print("\nSIGNIFICANCIA (Jeshua2 corregida, metodo D escalado):", flush=True)
    sig = {}
    for kind in kinds:
        sig[kind] = {}
        for metric, real_val in [("D_fractal", a_d_j["D_fractal"]),
                                 ("similitud_celdas", a_d_j["similitud_celdas"]),
                                 ("pico_central", a_d_j["pico_central"]),
                                 ("grid_rows", a_d_j["grid_rows"]),
                                 ("grid_cols", a_d_j["grid_cols"])]:
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
        vals = [mc["delta_alpha"] for mc in control_metrics[kind]
                if not np.isnan(mc["delta_alpha"])]
        if vals:
            vals = np.array(vals)
            real_da = a_d_j["delta_alpha"]
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
            real_mi = a_d_j["mi_center_cells"]["correlation_MI_distance"]
            z = (real_mi - vals.mean()) / vals.std() if vals.std() > 0 else float("nan")
            pct = float((vals < real_mi).mean() * 100)
            sig[kind]["mi_distance_corr"] = {"real": real_mi, "control_mean": float(vals.mean()),
                                             "control_std": float(vals.std()), "z_score": float(z),
                                             "percentile": pct}
            print(f"  [{kind}] MI-distancia: real={real_mi:.4f} | control={vals.mean():.4f}±{vals.std():.4f} | z={z:+.2f} | pct={pct:.1f}%", flush=True)
    report["controles_jeshua2"] = sig

    # --- Comparacion ---
    print("\nCOMPARACION (metodo D, corregidas):", flush=True)
    comp = {}
    for metric in ["D_fractal", "similitud_celdas", "pico_central", "grid_rows", "grid_cols"]:
        comp[metric] = {"imagen3": a_d3[metric], "jeshua2": a_d_j[metric]}
        print(f"  {metric}: imagen3={a_d3[metric]:.4f} | jeshua2={a_d_j[metric]:.4f}", flush=True)
    comp["delta_alpha"] = {"imagen3": a_d3["delta_alpha"], "jeshua2": a_d_j["delta_alpha"]}
    comp["mi_distance"] = {"imagen3": a_d3["mi_center_cells"]["correlation_MI_distance"],
                           "jeshua2": a_d_j["mi_center_cells"]["correlation_MI_distance"]}
    comp["directional_Q2"] = {"imagen3": a_d3["directional"]["Q2"], "jeshua2": a_d_j["directional"]["Q2"]}
    report["comparacion"] = comp
    print(f"  delta_alpha: imagen3={comp['delta_alpha']['imagen3']:.4f} | jeshua2={comp['delta_alpha']['jeshua2']:.4f}", flush=True)
    print(f"  MI-dist: imagen3={comp['mi_distance']['imagen3']:.4f} | jeshua2={comp['mi_distance']['jeshua2']:.4f}", flush=True)
    print(f"  Q2: imagen3={comp['directional_Q2']['imagen3']:.4f} | jeshua2={comp['directional_Q2']['jeshua2']:.4f}", flush=True)

    out_json = os.path.join(OUT, "analisis_final_corregido_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
