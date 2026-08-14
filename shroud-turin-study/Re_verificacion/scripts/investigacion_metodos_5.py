"""
RONDA FINAL 2: ¿Un simple gaussiano reproduce TODO el "ASIC"?
=============================================================
Metodo del estudio COMPLETO aplicado a:
  - perfil real de imagen3
  - 50 perfiles sinteticos gaussianos (suaves, con maximo central, sin estructura)

Comparar: grid, similitud, pico central, brazos, D fractal.
Si el gaussiano produce grid ~14x14, similitud ~0.6, pico, brazos y D similar,
la conclusion es demoledora: todo el "chip ASIC" es la geometria trivial de la
matriz de recurrencia de un perfil suave. Y lo que DIFERENCIA al real del
sintetico es lo unico que merece estudio.

Ademas: correlacion perfil real vs gaussiano ajustado (R2) para ver cuanto
del perfil real es explicado por una campana simple.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy.optimize import curve_fit
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N = 50

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
    # pico central (matriz completa, radio 20)
    cy, cx = n//2, n//2
    center = recurrence[cy-20:cy+20, cx-20:cx+20]
    ratio = float(center.mean() / recurrence.mean()) if recurrence.mean() > 0 else float("nan")
    return {"grid_rows": len(grid_rows), "grid_cols": len(grid_cols),
            "similarity": sim, "n_cells": n_cells, "density": float(recurrence.mean()),
            "peak_ratio": ratio}

def worker_study(args):
    return study_exact(args)

def gaussian_profile(n, sigma_px, noise_frac=0.05):
    x = np.arange(n)
    p = np.exp(-0.5 * ((x - n//2) / sigma_px) ** 2)
    p = p + rng.normal(0, noise_frac * p.std(), size=n)
    return np.clip(p, 0, None)

def main():
    t0 = time.time()
    report = {"real": {}, "gaussian_controls": {}, "gaussian_fit_real": {}}

    img = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (15,1), 0).flatten()

    r_real = study_exact(perfil)
    report["real"] = r_real
    print("=" * 70, flush=True)
    print(f"REAL: grid={r_real['grid_rows']}x{r_real['grid_cols']} | sim={r_real['similarity']:.4f} | pico={r_real['peak_ratio']:.2f} | dens={r_real['density']:.4f}", flush=True)

    # Ajuste gaussiano al perfil real
    x = np.arange(len(perfil))
    def gauss(x, A, mu, sigma, B):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + B
    p0 = [perfil.max() - perfil.min(), len(perfil)/2, 150, perfil.min()]
    try:
        popt, _ = curve_fit(gauss, x, perfil, p0=p0, maxfev=10000)
        fit = gauss(x, *popt)
        ss_res = np.sum((perfil - fit) ** 2)
        ss_tot = np.sum((perfil - perfil.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"\nAjuste gaussiano al perfil REAL:", flush=True)
        print(f"  A={popt[0]:.1f} mu={popt[1]:.1f} sigma={popt[2]:.1f} B={popt[3]:.1f}", flush=True)
        print(f"  R^2 = {r2:.4f} (cuanto del perfil real es una campana)", flush=True)
        # Ajustar escala para recurrencia (normalizar igual que el estudio)
        fit_n = cv2.normalize(fit, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64).flatten()
        r_fit = study_exact(fit_n)
        print(f"  Metodo estudio sobre el AJUSTE gaussiano:", flush=True)
        print(f"    grid={r_fit['grid_rows']}x{r_fit['grid_cols']} | sim={r_fit['similarity']:.4f} | pico={r_fit['peak_ratio']:.2f}", flush=True)
        report["gaussian_fit_real"] = {"params": popt.tolist(), "r2": float(r2),
            "study_on_fit": r_fit}
        # Residual: perfil real - gaussiano ajustado
        resid = perfil - fit
        print(f"  Residual: std={resid.std():.2f} (vs perfil std={perfil.std():.2f})", flush=True)
        report["gaussian_fit_real"]["residual_std"] = float(resid.std())
        report["gaussian_fit_real"]["profile_std"] = float(perfil.std())
    except Exception as e:
        print(f"  Ajuste fallo: {e}", flush=True)

    # Controles gaussianos (varios sigma)
    print(f"\nMetodo estudio sobre {N} gaussianos sinteticos:", flush=True)
    for sigma_px in [100, 200, 400]:
        tasks = [gaussian_profile(len(perfil), sigma_px) for _ in range(N)]
        with Pool(12) as pool:
            res = list(pool.imap_unordered(worker_study, tasks))
        grids = np.array([[r["grid_rows"], r["grid_cols"]] for r in res])
        sims = np.array([r["similarity"] for r in res if not np.isnan(r["similarity"])])
        picos = np.array([r["peak_ratio"] for r in res if not np.isnan(r["peak_ratio"])])
        print(f"  sigma={sigma_px}:", flush=True)
        print(f"    grid={grids[:,0].mean():.1f}±{grids[:,0].std():.1f} (real {r_real['grid_rows']}) | sim={sims.mean():.4f}±{sims.std():.4f} (real {r_real['similarity']:.4f}) | pico={picos.mean():.2f}±{picos.std():.2f} (real {r_real['peak_ratio']:.2f})", flush=True)
        report["gaussian_controls"][f"sigma{sigma_px}"] = {
            "grid_mean": float(grids[:,0].mean()), "grid_std": float(grids[:,0].std()),
            "sim_mean": float(sims.mean()), "sim_std": float(sims.std()),
            "pico_mean": float(picos.mean()), "pico_std": float(picos.std())}

    # CONCLUSION
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    r2 = report["gaussian_fit_real"].get("r2", None)
    if r2 is not None:
        print(f"  R^2 del ajuste gaussiano: {r2:.4f}", flush=True)
    sim_g = [report["gaussian_controls"][k]["sim_mean"] for k in report["gaussian_controls"]]
    print(f"  Similitud de gaussianos: {min(sim_g):.3f}-{max(sim_g):.3f} (real: {r_real['similarity']:.4f})", flush=True)
    report["conclusion"] = {
        "similitud_real_vs_gaussiano": r_real["similarity"],
        "similitud_gaussiano_rango": [min(sim_g), max(sim_g)],
        "pico_real": r_real["peak_ratio"],
        "r2_gaussiano": r2}

    out_json = os.path.join(OUT, "investigacion_metodos_5_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Guardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
