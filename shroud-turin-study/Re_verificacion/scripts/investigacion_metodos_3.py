"""
TERCERA RONDA: V1 y V4 con el metodo EXACTO del estudio
=======================================================
El metodo exacto de analisis_chip_profundo.py:
  - perfil = GaussianBlur(img_norm[:, w//2], (15,1))  (SOLO un suavizado)
  - recurrence = |perfil - perfil.T| < 10.0
  - grid: cuadrante TL, umbral mean+std, group_lines(gap=10)
  - celdas: fraccion de pixeles iguales, resized a tamano comun

Pregunta clave: si aplicamos este metodo EXACTO a CONTROLES
(permutaciones y gaussianos del perfil real), tambien dan
14x14 y ~0.65? Si es asi, ambos hallazgos son artefactos del METODO.
"""

import os
import json
import time
import numpy as np
import cv2
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N = 100

def study_exact(perfil, threshold=10.0):
    """Metodo EXACTO del estudio. Devuelve dict con grid y similitud."""
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
            "similarity": sim, "n_cells": n_cells,
            "density": float(recurrence.mean())}

def worker_study(args):
    return study_exact(args)

def main():
    t0 = time.time()
    report = {}

    # Perfil real (metodo exacto del estudio)
    img = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (15,1), 0).flatten()

    r_real = study_exact(perfil)
    print("=" * 70, flush=True)
    print("Metodo EXACTO del estudio - REAL", flush=True)
    print("=" * 70, flush=True)
    print(f"  grid={r_real['grid_rows']}x{r_real['grid_cols']} | similitud={r_real['similarity']:.4f} | densidad={r_real['density']:.4f}", flush=True)

    # Controles
    print(f"\nMetodo EXACTO aplicado a {N} controles x 2 tipos", flush=True)
    for kind in ["permutation", "gaussian"]:
        tasks = []
        for i in range(N):
            if kind == "permutation":
                p = rng.permutation(perfil)
            else:
                p = rng.normal(perfil.mean(), perfil.std(), size=len(perfil))
            tasks.append(p)
        with Pool(12) as pool:
            res = list(pool.imap_unordered(worker_study, tasks))
        grids = np.array([[r["grid_rows"], r["grid_cols"]] for r in res])
        sims = np.array([r["similarity"] for r in res if not np.isnan(r["similarity"])])
        dens = np.array([r["density"] for r in res])
        print(f"\n  [{kind}]", flush=True)
        print(f"    grid filas: {grids[:,0].mean():.1f}±{grids[:,0].std():.1f} (real {r_real['grid_rows']})", flush=True)
        print(f"    grid cols:  {grids[:,1].mean():.1f}±{grids[:,1].std():.1f} (real {r_real['grid_cols']})", flush=True)
        print(f"    % controles con grid >= 14 filas: {float((grids[:,0]>=14).mean()*100):.1f}%", flush=True)
        print(f"    % controles con grid == 14 filas: {float((grids[:,0]==14).mean()*100):.1f}%", flush=True)
        print(f"    similitud: {sims.mean():.4f}±{sims.std():.4f} (real {r_real['similarity']:.4f})", flush=True)
        print(f"    % controles con similitud >= real: {float((sims>=r_real['similarity']).mean()*100):.1f}%", flush=True)
        print(f"    densidad: {dens.mean():.4f}±{dens.std():.4f} (real {r_real['density']:.4f})", flush=True)
        report[kind] = {"grid_rows_mean": float(grids[:,0].mean()), "grid_rows_std": float(grids[:,0].std()),
                        "pct_ge_14": float((grids[:,0]>=14).mean()*100),
                        "pct_eq_14": float((grids[:,0]==14).mean()*100),
                        "similarity_mean": float(sims.mean()), "similarity_std": float(sims.std()),
                        "pct_sim_ge_real": float((sims>=r_real['similarity']).mean()*100),
                        "density_mean": float(dens.mean())}
    report["real"] = r_real
    report["conclusion"] = (
        "Si los controles tambien producen grids ~14x14 y similitudes ~0.6-0.7, "
        "los hallazgos 'grid 14x14' y '64.8% redundancia' son artefactos del METODO "
        "(deteccion por umbral+agrupacion y fraccion de iguales en matriz sparse).")

    out_json = os.path.join(OUT, "investigacion_metodos_3_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
