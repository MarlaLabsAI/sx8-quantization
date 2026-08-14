"""
INVESTIGACION MATEMATICA: METODOS EXACTOS DEL ESTUDIO ORIGINAL
==============================================================
Objetivos:
  A. Reproducir EXACTAMENTE el metodo CHIP-4 (grid) y CHIP-5 (celdas)
     del estudio: cuadrante TL, umbral mean+std, group_lines gap=10,
     similitud = fraccion de pixeles iguales.
  B. Hipotesis del 64.8%: E[coincidencia] = 1 - 2p(1-p) con p=densidad.
     Si el 64.8% es ~esperado por azar, el hallazgo de "redundancia
     estructural" colapsa.
  C. Pico central vs perfil sintetico: un perfil suave con maximo central
     (gaussiano), al construir su matriz de recurrencia, genera un pico
     central de densidad? Si si, el "pico 4-7x" podria ser artefacto
     del perfil y no de la "cruz".
  D. Direccionalidad bajo rotacion/reflexion: es el signo Q2/Q3 invariante?

Usa GPU + multiprocessing. Progreso visible.
"""

import os
import json
import time
import numpy as np
import cv2
import torch
from scipy import ndimage
from scipy.signal import find_peaks

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# ============================================================================
# A. REPRODUCCION EXACTA DEL METODO DEL ESTUDIO
# ============================================================================
def reproduce_study_methods(img_path, threshold=10.0, sigma=15.0):
    """Replica analisis_chip_profundo.py: normaliza, perfil central,
    recurrencia, cuadrante TL, grid con mean+std y group_lines(10),
    celdas con similitud = fraccion de iguales."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (15,1), 0).flatten()
    recurrence = (np.abs(perfil[:, None] - perfil[None, :]) < threshold).astype(float)
    n = recurrence.shape[0]
    # Cuadrante superior izquierdo (como el estudio)
    quadrant = recurrence[:n//2, :n//2]
    qh, qw = quadrant.shape
    # Densidad por filas y columnas (promedio)
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
    # Celdas 5x5 y similitud fraccion de iguales
    cells = []
    for i in range(min(5, len(grid_rows)-1)):
        for j in range(min(5, len(grid_cols)-1)):
            r1, r2 = grid_rows[i], grid_rows[i+1]
            c1, c2 = grid_cols[j], grid_cols[j+1]
            if r2-r1 > 10 and c2-c1 > 10:
                cells.append(quadrant[r1:r2, c1:c2])
    cell_info = None
    if len(cells) > 1:
        cell_size = min(c.shape[0] for c in cells)
        cells_resized = np.array([cv2.resize(c, (cell_size, cell_size)) for c in cells])
        n_cells = len(cells_resized)
        sm = np.zeros((n_cells, n_cells))
        for i in range(n_cells):
            for j in range(n_cells):
                sm[i, j] = np.mean(cells_resized[i] == cells_resized[j])
        mean_sim = float(np.mean(sm[np.triu_indices(n_cells, 1)]))
        cell_info = {"n_cells": n_cells, "mean_similarity_iguales": mean_sim,
                     "expected_random": float(1 - 2*recurrence.mean()*(1-recurrence.mean()))}
    return {
        "density_global": float(recurrence.mean()),
        "density_quadrant_TL": float(quadrant.mean()),
        "grid_rows": len(grid_rows), "grid_cols": len(grid_cols),
        "grid_rows_pos": grid_rows[:20], "grid_cols_pos": grid_cols[:20],
        "cells": cell_info,
    }

# ============================================================================
# C. PERFIL SINTETICO CON MAXIMO CENTRAL -> PICO EN RECURRENCIA?
# ============================================================================
def profile_synthetic_peak(n, peak_pos=None, sigma_px=50, kind="gaussian"):
    """Perfil sintetico con maximo central suave (sin cruz, sin estructura)."""
    x = np.arange(n)
    if kind == "gaussian":
        p = np.exp(-0.5 * ((x - (peak_pos or n//2)) / sigma_px) ** 2)
    elif kind == "smoothstep":
        d = np.abs(x - (peak_pos or n//2)) / (n//2)
        p = 1 - 3*d**2 + 2*d**3  # smoothstep 1->0
    return p.astype(np.float64)

def central_peak_ratio_of_profile(profile, threshold=10.0, sigma=15.0, radius=20):
    """Construye recurrencia del perfil y mide ratio centro/global."""
    p = ndimage.gaussian_filter1d(profile, sigma=sigma)
    # Normalizar a 0-255 como hace el estudio (para perfil sintetico)
    p = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    n = len(p)
    diff = np.abs(p[:, None] - p[None, :])
    R = (diff < threshold).astype(float)
    cy, cx = n//2, n//2
    center = R[cy-radius:cy+radius, cx-radius:cx+radius]
    return float(center.mean() / R.mean()) if R.mean() > 0 else float("nan"), float(R.mean())

# ============================================================================
# D. DIRECCIONALIDAD BAJO ROTACION/REFLEXION
# ============================================================================
def directional_quadrants(profile):
    n = len(profile)
    D = profile[:, None] - profile[None, :]
    h = n // 2
    return {"Q1": float(D[:h,:h].mean()), "Q2": float(D[:h,h:].mean()),
            "Q3": float(D[h:,:h].mean()), "Q4": float(D[h:,h:].mean())}

def main():
    t0 = time.time()
    report = {"A_reproduccion_metodos": {}, "B_similitud_vs_azar": {},
              "C_pico_central_sintetico": {}, "D_direccionalidad": {}}

    # ============ A. Reproducir metodos del estudio ============
    print("=" * 70, flush=True)
    print("A. Reproduccion EXACTA del metodo del estudio (imagen3)", flush=True)
    print("=" * 70, flush=True)
    img3 = os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg")
    a = reproduce_study_methods(img3)
    report["A_reproduccion_metodos"]["imagen3"] = a
    print(f"  Densidad global: {a['density_global']:.4f}", flush=True)
    print(f"  Grid (metodo estudio): {a['grid_rows']}x{a['grid_cols']}  (estudio: 14x14)", flush=True)
    print(f"  Pos filas: {a['grid_rows_pos'][:12]}", flush=True)
    print(f"  Pos cols:  {a['grid_cols_pos'][:12]}", flush=True)
    if a["cells"]:
        print(f"  Celdas: {a['cells']['n_cells']} | similitud (fraccion iguales) = {a['cells']['mean_similarity_iguales']:.4f}  (estudio: 0.648)", flush=True)
        print(f"  Esperado por azar (1-2p(1-p)): {a['cells']['expected_random']:.4f}", flush=True)
        print(f"  Ratio medido/azar: {a['cells']['mean_similarity_iguales']/a['cells']['expected_random']:.3f}", flush=True)

    # Jeshua2 izquierda tambien
    j2 = cv2.imread(os.path.join(BASE, "Re_verificacion", "Jeshua2.jpg"), cv2.IMREAD_GRAYSCALE)
    half = j2[:, :j2.shape[1]//2]
    tmp = "/tmp/opencode/jeshua2_izq.png"
    cv2.imwrite(tmp, half)
    b = reproduce_study_methods(tmp)
    report["A_reproduccion_metodos"]["jeshua2_izq"] = b
    print(f"  Jeshua2 izq -> Grid: {b['grid_rows']}x{b['grid_cols']}", flush=True)
    if b["cells"]:
        print(f"  Jeshua2 izq -> similitud: {b['cells']['mean_similarity_iguales']:.4f} | esperado azar: {b['cells']['expected_random']:.4f} | ratio: {b['cells']['mean_similarity_iguales']/b['cells']['expected_random']:.3f}", flush=True)

    # ============ B. Similitud vs azar: barrido de densidades ============
    print("\n" + "=" * 70, flush=True)
    print("B. Similitud (fraccion de iguales) vs valor esperado por azar", flush=True)
    print("=" * 70, flush=True)
    # Simular matrices binarias aleatorias con densidad p, celdas de 40x40, comparar
    rng = np.random.default_rng(42)
    for p in [0.05, 0.10, 0.1338, 0.20, 0.30]:
        sims = []
        for _ in range(200):
            ca = (rng.random((40, 40)) < p).astype(float)
            cb = (rng.random((40, 40)) < p).astype(float)
            sims.append(np.mean(ca == cb))
        expected = 1 - 2*p*(1-p)
        print(f"  p={p:.4f}: E[iguales]={expected:.4f} | simulado={np.mean(sims):.4f}±{np.std(sims):.3f}", flush=True)
    # Pregunta clave: con densidad 0.134, el 0.648 medido esta por DEBAJO del azar?
    p = 0.1338
    expected = 1 - 2*p*(1-p)
    print(f"\n  CONCLUSION B: densidad={p:.4f} -> E[similitud azar]={expected:.4f}", flush=True)
    print(f"  Estudio midio 0.648 < 0.768 (azar) -> las celdas son MENOS similares que el azar", flush=True)
    report["B_similitud_vs_azar"] = {"density": p, "expected_random": expected,
        "study_measured": 0.648, "below_random": 0.648 < expected}

    # ============ C. Pico central en perfil sintetico ============
    print("\n" + "=" * 70, flush=True)
    print("C. Pico central: perfil sintetico con maximo central (sin cruz)", flush=True)
    print("=" * 70, flush=True)
    n = 1080
    results_c = {}
    for kind in ["gaussian", "smoothstep"]:
        for sigma_px in [30, 60, 120, 240]:
            p = profile_synthetic_peak(n, kind=kind, sigma_px=sigma_px)
            ratio, dens = central_peak_ratio_of_profile(p)
            key = f"{kind}_sigma{sigma_px}"
            results_c[key] = {"ratio_centro_global": ratio, "density": dens}
            print(f"  {key}: ratio_centro/global = {ratio:.2f} | densidad = {dens:.4f}", flush=True)
    # Perfil real de imagen3 para comparar
    img3g = cv2.imread(img3, cv2.IMREAD_GRAYSCALE)
    profile_real = img3g[:, img3g.shape[1]//2].astype(np.float64)
    ratio_real, dens_real = central_peak_ratio_of_profile(profile_real)
    print(f"  REAL imagen3: ratio = {ratio_real:.2f} | densidad = {dens_real:.4f}", flush=True)
    report["C_pico_central_sintetico"] = {"sinteticos": results_c,
        "real_imagen3": {"ratio": ratio_real, "density": dens_real}}

    # ============ D. Direccionalidad bajo reflexion ============
    print("\n" + "=" * 70, flush=True)
    print("D. Direccionalidad Q2/Q3 bajo reflexion (inversion de perfil)", flush=True)
    print("=" * 70, flush=True)
    p_real = ndimage.gaussian_filter1d(profile_real.astype(np.float64), 15)
    d_orig = directional_quadrants(p_real)
    d_flip = directional_quadrants(p_real[::-1])
    print(f"  Perfil original:     Q1={d_orig['Q1']:.3f} Q2={d_orig['Q2']:.3f} Q3={d_orig['Q3']:.3f} Q4={d_orig['Q4']:.3f}", flush=True)
    print(f"  Perfil reflejado:    Q1={d_flip['Q1']:.3f} Q2={d_flip['Q2']:.3f} Q3={d_flip['Q3']:.3f} Q4={d_flip['Q4']:.3f}", flush=True)
    # OJO: al reflejar, Q2<->Q3 se intercambian: Q2_flip = -Q3_orig
    print(f"  -Q3_original = {-d_orig['Q3']:.3f} vs Q2_reflejado = {d_flip['Q2']:.3f} (deben coincidir)", flush=True)
    report["D_direccionalidad"] = {"original": d_orig, "reflejado": d_flip,
        "Q2_flip_vs_neg_Q3_orig": {"Q2_flip": d_flip["Q2"], "neg_Q3_orig": -d_orig["Q3"]}}
    # Perfil sintetico plano+ruido para referencia
    noise_p = rng.normal(128, 30, n)
    d_noise = directional_quadrants(noise_p)
    print(f"  Perfil ruido:         Q1={d_noise['Q1']:.3f} Q2={d_noise['Q2']:.3f} Q3={d_noise['Q3']:.3f} Q4={d_noise['Q4']:.3f}", flush=True)

    # Guardar
    out_json = os.path.join(OUT, "investigacion_metodos_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado en: {out_json}", flush=True)
    print(f"Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
