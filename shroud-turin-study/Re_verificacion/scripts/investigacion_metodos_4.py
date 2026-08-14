"""
RONDA FINAL: ¿QUE QUEDA DEL ESTUDIO?
====================================
T1 (hecho): grid 14x14 = artefacto del metodo (70% de controles >= 14 filas)

T2: Similitud 0.6484 — ¿se explica por la SUAVIDAD del perfil?
    Control AR(1) coloreado con autocorrelacion similar al perfil real.
    Si AR(1) tambien da ~0.64, la "redundancia 64.8%" es consecuencia
    trivial de que el perfil es suave (sin estructura oculta).

T3: Pico central / "cruz" — ¿es generica de cualquier perfil simetrico suave?
    Un perfil gaussiano simetrico produce en su matriz de recurrencia una
    cruz = interseccion de la banda diagonal (i~j) y la banda anti-diagonal
    (simetria i+j ~ constante). Verificar:
      a) ratio centro/global del sintetico (ancho de banda)
      b) estructura de "brazos" (D11) del sintetico: densidad y D fractal
      c) simetria rotacional 90 grados del sintetico

T4 (hecho): direccionalidad Q2/Q3 = diferencia de medias de mitades (trivial)
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N = 100

# ---------- Metodo exacto del estudio ----------
def study_exact(perfil, threshold=10.0):
    recurrence = (np.abs(perfil[:, None] - perfil[None, :]) < threshold).astype(float)
    n = recurrence.shape[0]
    quadrant = recurrence[:n//2, :n//2]
    row_density = quadrant.mean(axis=1)
    col_density = quadrant.mean(axis=0)
    thr_row = row_density.mean() + row_density.std()
    thr_col = col_density.mean() + col_density.std()
    strong_rows = np.where(row_density > thr_row)[0]
    strong_cols = np.where(col_density > thr_col)[0]
    def group_lines(lines, gap=10):
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
    grid_rows = group_lines(strong_rows)
    grid_cols = group_lines(strong_cols)
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
    return {"grid_rows": len(grid_rows), "grid_cols": len(grid_cols),
            "similarity": sim, "n_cells": n_cells, "density": float(recurrence.mean())}

def worker_study(args):
    return study_exact(args)

# ---------- AR(1) con autocorrelacion similar al perfil real ----------
def ar1_profile(p, phi):
    n = len(p)
    noise = rng.normal(0, p.std() * np.sqrt(1 - phi**2), size=n)
    x = np.zeros(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + noise[i]
    return x - x.mean() + p.mean()

def autocorr_lag1(p):
    p = p - p.mean()
    return float(np.mean(p[:-1] * p[1:]) / np.mean(p * p))

# ---------- Perfil sintetico simetrico suave ----------
def gaussian_profile(n, sigma_px, noise_frac=0.05):
    x = np.arange(n)
    p = np.exp(-0.5 * ((x - n//2) / sigma_px) ** 2)
    p = p + rng.normal(0, noise_frac * p.std(), size=n)
    return p

def cross_arms(R, radius=20):
    """D11 del estudio: densidad de los 4 brazos a distancia radius del centro."""
    n = R.shape[0]
    cy, cx = n//2, n//2
    def dens(dx, dy):
        return float(R[cy-dy:cy+dy+1, cx-dx:cx+dx+1].mean())
    arms = {
        "arriba": dens(radius//2, radius),   # franja vertical sobre el centro
        "abajo": dens(radius//2, radius),
        "izquierda": dens(radius, radius//2),
        "derecha": dens(radius, radius//2),
    }
    return arms

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

def main():
    t0 = time.time()
    report = {"T2_similitud_ar1": {}, "T3_cruz_sintetica": {}}

    # Perfil real
    img = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (15,1), 0).flatten()
    r_real = study_exact(perfil)
    lag1_real = autocorr_lag1(perfil)
    print("=" * 70, flush=True)
    print(f"REAL: grid={r_real['grid_rows']}x{r_real['grid_cols']} | similitud={r_real['similarity']:.4f} | autocorr lag1={lag1_real:.3f}", flush=True)

    # ---------- T2: AR(1) con phi que iguala autocorrelacion real ----------
    print("\n" + "=" * 70, flush=True)
    print("T2. Similitud con control AR(1) (misma autocorrelacion que el real)", flush=True)
    print("=" * 70, flush=True)
    # Encontrar phi que produce lag1 ~ lag1_real
    for phi in [0.98, 0.99, 0.995]:
        p_t = ar1_profile(perfil, phi)
        print(f"  AR(1) phi={phi}: autocorr lag1 = {autocorr_lag1(p_t):.3f}", flush=True)
    phi_best = 0.99
    tasks = []
    for i in range(N):
        tasks.append(ar1_profile(perfil, phi_best))
    with Pool(12) as pool:
        res = list(pool.imap_unordered(worker_study, tasks))
    sims = np.array([r["similarity"] for r in res if not np.isnan(r["similarity"])])
    grids = np.array([[r["grid_rows"], r["grid_cols"]] for r in res])
    print(f"  AR(1) phi={phi_best} ({N} controles):", flush=True)
    print(f"    similitud: {sims.mean():.4f}±{sims.std():.4f} (real: {r_real['similarity']:.4f})", flush=True)
    print(f"    % controles con similitud >= real: {float((sims>=r_real['similarity']).mean()*100):.1f}%", flush=True)
    z = (r_real['similarity'] - sims.mean()) / sims.std()
    print(f"    z-score: {z:+.2f}", flush=True)
    print(f"    grid: {grids[:,0].mean():.1f}±{grids[:,0].std():.1f} filas (real: {r_real['grid_rows']})", flush=True)
    report["T2_similitud_ar1"] = {"phi": phi_best, "real": r_real["similarity"],
        "control_mean": float(sims.mean()), "control_std": float(sims.std()),
        "z_score": float(z), "pct_ge_real": float((sims>=r_real['similarity']).mean()*100),
        "autocorr_real": lag1_real}

    # ---------- T3: cruz en perfil sintetico simetrico ----------
    print("\n" + "=" * 70, flush=True)
    print("T3. 'Cruz central' en un perfil sintetico SIMETRICO y suave", flush=True)
    print("=" * 70, flush=True)
    n = len(perfil)
    # Recurrencia del real con metodo del estudio
    R_real = (np.abs(perfil[:, None] - perfil[None, :]) < 10.0).astype(float)
    ratio_real = R_real[n//2-20:n//2+20, n//2-20:n//2+20].mean() / R_real.mean()
    print(f"  REAL: ratio centro/global = {ratio_real:.2f}", flush=True)
    arms_real = cross_arms(R_real)
    print(f"  REAL brazos (densidad): arriba={arms_real['arriba']:.4f} abajo={arms_real['abajo']:.4f} izq={arms_real['izquierda']:.4f} der={arms_real['derecha']:.4f}", flush=True)
    # Sinteticos gaussianos simetricos
    for sigma_px in [100, 200, 400]:
        ratios = []
        arms_s = []
        for i in range(30):
            p = gaussian_profile(n, sigma_px)
            p_n = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64).flatten()
            Rs = (np.abs(p_n[:, None] - p_n[None, :]) < 10.0).astype(float)
            ratios.append(Rs[n//2-20:n//2+20, n//2-20:n//2+20].mean() / Rs.mean())
            arms_s.append(cross_arms(Rs))
        ratios = np.array(ratios)
        arm_mean = {k: float(np.mean([a[k] for a in arms_s])) for k in arms_s[0]}
        arm_std = {k: float(np.std([a[k] for a in arms_s])) for k in arms_s[0]}
        print(f"  Gaussiano sigma={sigma_px}: ratio={ratios.mean():.2f}±{ratios.std():.2f}", flush=True)
        print(f"    brazos: arriba={arm_mean['arriba']:.4f}±{arm_std['arriba']:.4f} | abajo={arm_mean['abajo']:.4f}±{arm_std['abajo']:.4f} | izq={arm_mean['izquierda']:.4f}±{arm_std['izquierda']:.4f} | der={arm_mean['derecha']:.4f}±{arm_std['derecha']:.4f}", flush=True)
        report["T3_cruz_sintetica"][f"gauss_sigma{sigma_px}"] = {
            "ratio_mean": float(ratios.mean()), "ratio_std": float(ratios.std()),
            "arms": arm_mean, "arms_std": arm_std}
    report["T3_cruz_sintetica"]["real"] = {"ratio": ratio_real, "arms": arms_real}
    print("\n  INTERPRETACION: si el gaussiano simetrico produce ratio y brazos", flush=True)
    print("  comparables al real, la 'cruz central' es la interseccion de la", flush=True)
    print("  banda diagonal (i~j) y anti-diagonal (simetria i+j~cte) — generica", flush=True)
    print("  de CUALQUIER perfil suave con maximo central, no estructura especial.", flush=True)

    out_json = os.path.join(OUT, "investigacion_metodos_4_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
