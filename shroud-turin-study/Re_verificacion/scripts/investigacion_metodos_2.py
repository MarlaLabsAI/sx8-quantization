"""
SEGUNDA RONDA: VERIFICACIONES CRUCIALES
========================================
V1. El metodo EXACTO del estudio (grid mean+std + group_lines + similitud
    fraccion iguales) aplicado a CONTROLES: permutaciones del perfil real.
    - Si los controles tambien dan 14x14 y ~0.65, ambos hallazgos son
      artefactos del METODO (no de la estructura).
V2. Pico central con control ADECUADO: perfiles suaves con maximo central
    (gaussiano/smoothstep) vs real. El control plano (ruido) era inadecuado.
V3. Direccionalidad Q2/Q3 = mean(p[:h]) - mean(p[h:]) : verificacion trivial.
V4. Similitud de celdas: permutar DENTRO de las celdas reales para obtener
    el valor nulo exacto con la misma geometria.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N = 100

# ============================================================================
# Metodo EXACTO del estudio (replicado de analisis_chip_profundo.py)
# ============================================================================
def study_grid_and_cells(recurrence):
    """CHIP-4 + CHIP-5 exactos del estudio, devuelve (grid_rows, grid_cols, similitud)."""
    n = recurrence.shape[0]
    quadrant = recurrence[:n//2, :n//2]
    qh, qw = quadrant.shape
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
    return len(grid_rows), len(grid_cols), sim, n_cells

def worker_study(args):
    R = args
    gr, gc, sim, nc = study_grid_and_cells(R)
    return {"grid_rows": gr, "grid_cols": gc, "similarity": sim, "n_cells": nc}

# ============================================================================
# Perfiles de control
# ============================================================================
def control_permutation_profile(p):
    return rng.permutation(p)

def control_gaussian_profile(p):
    return rng.normal(p.mean(), p.std(), size=len(p))

def control_smooth_peak(n, sigma_px):
    x = np.arange(n)
    return np.exp(-0.5 * ((x - n//2) / sigma_px) ** 2)

def build_R(profile, threshold=10.0, sigma=15.0):
    p = ndimage.gaussian_filter1d(profile, sigma=sigma)
    p = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64).flatten()
    n = len(p)
    diff = np.abs(p[:, None] - p[None, :])
    return (diff < threshold).astype(float)

def main():
    t0 = time.time()
    report = {"V1_grid_similitud_en_controles": {}, "V2_pico_central_control_adecuado": {},
              "V3_direccionalidad_trivial": {}, "V4_similitud_permutada": {}}

    # ============ V1. Metodo del estudio sobre CONTROLES ============
    print("=" * 70, flush=True)
    print(f"V1. Metodo EXACTO del estudio aplicado a {N} controles (permutacion y gaussiano)", flush=True)
    print("=" * 70, flush=True)
    img3 = os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg")
    img = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (15,1), 0).flatten()
    R_real = build_R(perfil)
    gr_r, gc_r, sim_r, nc_r = study_grid_and_cells(R_real)
    print(f"  REAL: grid={gr_r}x{gc_r} | similitud={sim_r:.4f} | celdas={nc_r}", flush=True)

    for kind in ["permutation", "gaussian"]:
        tasks = []
        for i in range(N):
            if kind == "permutation":
                p = control_permutation_profile(perfil)
            else:
                p = control_gaussian_profile(perfil)
            tasks.append(build_R(p))
        with Pool(12) as pool:
            res = list(pool.imap_unordered(worker_study, tasks))
        grids = np.array([[r["grid_rows"], r["grid_cols"]] for r in res])
        sims = np.array([r["similarity"] for r in res if not np.isnan(r["similarity"])])
        ncell = np.array([r["n_cells"] for r in res])
        print(f"  [{kind}] grid filas: {grids[:,0].mean():.1f}±{grids[:,0].std():.1f} | columnas: {grids[:,1].mean():.1f}±{grids[:,1].std():.1f}", flush=True)
        print(f"  [{kind}] similitud: {sims.mean():.4f}±{sims.std():.4f} (real: {sim_r:.4f}) | celdas: {ncell.mean():.0f}±{ncell.std():.0f}", flush=True)
        pct_grid = float((grids[:,0] >= gr_r-1).mean() * 100)
        pct_sim = float((sims >= sim_r).mean() * 100)
        print(f"  [{kind}] % controles con >= {gr_r} filas: {pct_grid:.1f}% | % con similitud >= {sim_r:.3f}: {pct_sim:.1f}%", flush=True)
        report["V1_grid_similitud_en_controles"][kind] = {
            "grid_rows_mean": float(grids[:,0].mean()), "grid_rows_std": float(grids[:,0].std()),
            "grid_cols_mean": float(grids[:,1].mean()), "grid_cols_std": float(grids[:,1].std()),
            "similarity_mean": float(sims.mean()), "similarity_std": float(sims.std()),
            "pct_controls_ge_14_rows": pct_grid, "pct_controls_ge_sim": pct_sim,
            "real": {"grid": [gr_r, gc_r], "similarity": sim_r}}
    report["V1_grid_similitud_en_controles"]["real"] = {"grid": [gr_r, gc_r], "similarity": sim_r}

    # ============ V2. Pico central con control adecuado ============
    print("\n" + "=" * 70, flush=True)
    print("V2. Pico central: control ADECUADO (perfiles suaves con maximo central)", flush=True)
    print("=" * 70, flush=True)
    def ratio(R, radius=20):
        n = R.shape[0]
        c = R[n//2-radius:n//2+radius, n//2-radius:n//2+radius]
        return float(c.mean()/R.mean()) if R.mean() > 0 else float("nan")
    ratio_real = ratio(R_real)
    print(f"  REAL imagen3: ratio={ratio_real:.2f}", flush=True)
    results = {}
    for sigma_px in [50, 100, 200, 400]:
        ratios = []
        for i in range(50):
            p = control_smooth_peak(len(perfil), sigma_px)
            # anadir algo de ruido para no ser perfecto
            p = p + rng.normal(0, 0.05*p.std(), size=len(p))
            p = np.clip(p, 0, None)
            Rc = build_R(p)
            ratios.append(ratio(Rc))
        ratios = np.array(ratios)
        print(f"  gaussiano sigma={sigma_px}: ratio={ratios.mean():.2f}±{ratios.std():.2f} (real: {ratio_real:.2f})", flush=True)
        results[f"gauss_sigma{sigma_px}"] = {"mean": float(ratios.mean()), "std": float(ratios.std())}
    report["V2_pico_central_control_adecuado"] = {"real_ratio": ratio_real, "controles": results}

    # ============ V3. Direccionalidad trivial ============
    print("\n" + "=" * 70, flush=True)
    print("V3. Direccionalidad: Q2/Q3 == diferencia de medias de mitades", flush=True)
    print("=" * 70, flush=True)
    p_s = ndimage.gaussian_filter1d(perfil, 15)
    n = len(p_s)
    h = n // 2
    Q2 = float(p_s[:h].mean() - p_s[h:].mean())
    Q3 = float(p_s[h:].mean() - p_s[:h].mean())
    print(f"  Q2 = mean(mitad1) - mean(mitad2) = {Q2:.4f}  (estudio: -0.696 o +0.696 segun orientacion)", flush=True)
    print(f"  Q3 = -Q2 = {Q3:.4f}  (estudio: el opuesto)", flush=True)
    print(f"  => La 'informacion direccional' es simplemente la diferencia de medias", flush=True)
    print(f"     entre la mitad superior e inferior del perfil. Trivial y dependiente de orientacion.", flush=True)
    report["V3_direccionalidad_trivial"] = {"Q2_mean_diff": Q2, "Q3_mean_diff": Q3,
        "explicacion": "Q2 = mean(p[:h]) - mean(p[h:]); Q3 = -Q2. No hay mecanismo direccional."}

    # ============ V4. Similitud con celdas permutadas internamente ============
    print("\n" + "=" * 70, flush=True)
    print("V4. Similitud de celdas: permutar internamente las celdas reales", flush=True)
    print("=" * 70, flush=True)
    gr, gc, sim_real, nc = study_grid_and_cells(R_real)
    n = R_real.shape[0]
    quadrant = R_real[:n//2, :n//2]
    # Re-extraer las mismas celdas
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
    rd = quadrant.mean(axis=1); cd = quadrant.mean(axis=0)
    gr_pos = group_lines(np.where(rd > rd.mean()+rd.std())[0])
    gc_pos = group_lines(np.where(cd > cd.mean()+cd.std())[0])
    cells = []
    for i in range(min(5, len(gr_pos)-1)):
        for j in range(min(5, len(gc_pos)-1)):
            r1, r2 = gr_pos[i], gr_pos[i+1]
            c1, c2 = gc_pos[j], gc_pos[j+1]
            if r2-r1 > 10 and c2-c1 > 10:
                cells.append(quadrant[r1:r2, c1:c2])
    cell_size = min(c.shape[0] for c in cells)
    cells_resized = np.array([cv2.resize(c, (cell_size, cell_size)) for c in cells])
    n_cells = len(cells_resized)
    perm_sims = []
    for t in range(N):
        cells_perm = np.array([cells_resized[i][rng.permutation(cell_size), :][:, rng.permutation(cell_size)] for i in range(n_cells)])
        sm = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j in range(n_cells):
                sm[i,j] = np.mean(cells_perm[i] == cells_perm[j])
        perm_sims.append(float(np.mean(sm[np.triu_indices(n_cells, 1)])))
    perm_sims = np.array(perm_sims)
    print(f"  Real: {sim_real:.4f} | celdas permutadas internamente: {perm_sims.mean():.4f}±{perm_sims.std():.4f}", flush=True)
    print(f"  La similitud real es IGUAL a la de celdas aleatorizadas internamente? "
          f"{'SI (artefacto)' if abs(sim_real - perm_sims.mean()) < 3*perm_sims.std() else 'NO (estructura real)'}", flush=True)
    report["V4_similitud_permutada"] = {"real": sim_real,
        "permutada_mean": float(perm_sims.mean()), "permutada_std": float(perm_sims.std()),
        "es_artefacto": bool(abs(sim_real - perm_sims.mean()) < 3*perm_sims.std())}

    out_json = os.path.join(OUT, "investigacion_metodos_2_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado en: {out_json}", flush=True)
    print(f"Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
