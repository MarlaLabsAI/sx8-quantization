"""
APLICACION DE LOS METODOS EXACTOS DEL ESTUDIO A JESHUA2-IZQ
===========================================================
El estudio usa DOS metodos distintos para generar la matriz de recurrencia:

  METODO CHIP (analisis_chip_profundo.py, tests_A1_A6):
    - img_norm = normalize(img, 0, 255)
    - perfil = cv2.GaussianBlur(img_norm[:, w//2], (15,1), 0)  # sin suavizado efectivo
    - R = |perfil - perfil.T| < 10.0
    - densidad resultante: 0.0992

  METODO D (tests_D1_D10, tests_D2b_D13):
    - img cruda (SIN normalizar)
    - perfil = gaussian_filter1d(img[:, w//2], sigma=15)
    - R = |perfil - perfil.T| < 10.0
    - densidad resultante: 0.1338

Este script aplica AMBOS metodos a Jeshua2-izq (1185x2321), con parametros
originales y con parametros escalados por el factor de resolucion 2.149.

NO modifica ningun archivo original. Todo se guarda en Re_verificacion/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

FACTOR = 2321 / 1080  # 2.149

# ============================================================================
# METODO CHIP EXACTO
# ============================================================================
def metodo_chip(img, sigma_ksize=15, threshold=10.0):
    """Replica analisis_chip_profundo.py: normaliza + GaussianBlur((15,1),0)."""
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    h, w = img.shape
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (sigma_ksize,1), 0).flatten()
    R = (np.abs(perfil[:, None] - perfil[None, :]) < threshold).astype(float)
    return R, perfil

# ============================================================================
# METODO D EXACTO
# ============================================================================
def metodo_d(img, sigma=15.0, threshold=10.0):
    """Replica tests_D1_D10: imagen cruda + gaussian_filter1d(sigma=15)."""
    h, w = img.shape
    perfil = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=sigma)
    R = (np.abs(perfil[:, None] - perfil[None, :]) < threshold).astype(float)
    return R, perfil

# ============================================================================
# METRICAS DEL ESTUDIO (CHIP-4 grid, CHIP-5 celdas, CHIP-6 D fractal, CHIP-8 cruz)
# ============================================================================
def box_counting_2d(binary_img):
    """CHIP-6 exacto: box-counting con padding a potencia de 2."""
    p = max(binary_img.shape)
    n = 2**int(np.ceil(np.log2(p)))
    padded = np.pad(binary_img, ((0, n-binary_img.shape[0]), (0, n-binary_img.shape[1])), 'constant')
    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = []
    for s in sizes:
        reshaped = padded.reshape(n//s, s, n//s, s)
        c = np.sum(np.any(reshaped, axis=(1,3)))
        counts.append(c)
    counts = np.array(counts)
    valid = counts > 0
    if np.sum(valid) < 2:
        return 0.0
    return -np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)[0]

def grid_y_cruz(R, gap=10):
    """CHIP-4 + CHIP-8 exactos: grid en cuadrante TL + cruz por argmax."""
    n = R.shape[0]
    quadrant_size = n // 2
    quadrant_tl = R[:quadrant_size, :quadrant_size]
    row_density = np.mean(quadrant_tl, axis=1)
    col_density = np.mean(quadrant_tl, axis=0)
    thr_row = np.mean(row_density) + np.std(row_density)
    thr_col = np.mean(col_density) + np.std(col_density)
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
    # Cruz: argmax de proyecciones del cuadrante TL
    proj_h = np.mean(quadrant_tl, axis=0)
    proj_v = np.mean(quadrant_tl, axis=1)
    center_x = int(np.argmax(proj_h))
    center_y = int(np.argmax(proj_v))
    return len(grid_rows), len(grid_cols), center_x, center_y, quadrant_size

def similitud_celdas(R, grid_rows, grid_cols):
    """CHIP-5 exacto: fraccion de pixeles iguales entre celdas 5x5."""
    n = R.shape[0]
    quadrant = R[:n//2, :n//2]
    if len(grid_rows) < 2 or len(grid_cols) < 2:
        return float("nan"), 0
    cells = []
    for i in range(min(5, len(grid_rows)-1)):
        for j in range(min(5, len(grid_cols)-1)):
            r1, r2 = grid_rows[i], grid_rows[i+1]
            c1, c2 = grid_cols[j], grid_cols[j+1]
            if r2-r1 > 10 and c2-c1 > 10:
                cells.append(quadrant[r1:r2, c1:c2])
    if len(cells) < 2:
        return float("nan"), 0
    cell_size = min(c.shape[0] for c in cells)
    cells_resized = np.array([cv2.resize(c, (cell_size, cell_size)) for c in cells])
    n_cells = len(cells_resized)
    sm = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        for j in range(n_cells):
            sm[i, j] = np.mean(cells_resized[i] == cells_resized[j])
    return float(np.mean(sm[np.triu_indices(n_cells, 1)])), n_cells

def analisis_completo(R, gap=10):
    """Ejecuta las metricas CHIP sobre una matriz."""
    gr, gc, cx, cy, qsize = grid_y_cruz(R, gap)
    # Recalcular grid para celdas
    n = R.shape[0]
    quadrant = R[:n//2, :n//2]
    rd = np.mean(quadrant, axis=1); cd = np.mean(quadrant, axis=0)
    thr_r = np.mean(rd) + np.std(rd); thr_c = np.mean(cd) + np.std(cd)
    sr = np.where(rd > thr_r)[0]; sc = np.where(cd > thr_c)[0]
    def gl(lines, gap):
        if len(lines) == 0: return []
        groups = []; cur = [lines[0]]
        for line in lines[1:]:
            if line - cur[-1] <= gap: cur.append(line)
            else: groups.append(int(np.mean(cur))); cur = [line]
        groups.append(int(np.mean(cur)))
        return groups
    gr_pos = gl(sr, gap); gc_pos = gl(sc, gap)
    sim, nc = similitud_celdas(R, gr_pos, gc_pos)
    D = box_counting_2d(R)
    return {
        "densidad": float(R.mean()),
        "grid": f"{gr}x{gc}",
        "grid_rows": gr, "grid_cols": gc,
        "cruz_abs": (cx, cy),
        "cruz_rel": (round(cx/qsize, 3), round(cy/qsize, 3)),
        "similitud_celdas": sim,
        "n_celdas": nc,
        "D_fractal": float(D),
    }

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"factor": FACTOR, "imagen3_referencia": {}, "jeshua2": {}}

    # Cargar imagenes
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    j2 = cv2.imread(os.path.join(BASE, "Re_verificacion", "Jeshua2.jpg"), cv2.IMREAD_GRAYSCALE)
    j2_izq = j2[:, :j2.shape[1]//2]

    print("=" * 70, flush=True)
    print("REFERENCIA: imagen3 con metodos exactos del estudio", flush=True)
    print("=" * 70, flush=True)
    # Metodo CHIP en imagen3
    R_chip3, p_chip3 = metodo_chip(img3)
    a_chip3 = analisis_completo(R_chip3, gap=10)
    print(f"  [CHIP] densidad={a_chip3['densidad']:.4f} | grid={a_chip3['grid']} | D={a_chip3['D_fractal']:.4f} | cruz={a_chip3['cruz_abs']} rel={a_chip3['cruz_rel']} | sim={a_chip3['similitud_celdas']:.4f}", flush=True)
    report["imagen3_referencia"]["metodo_chip"] = a_chip3
    # Metodo D en imagen3
    R_d3, p_d3 = metodo_d(img3)
    a_d3 = analisis_completo(R_d3, gap=10)
    print(f"  [D]     densidad={a_d3['densidad']:.4f} | grid={a_d3['grid']} | D={a_d3['D_fractal']:.4f} | cruz={a_d3['cruz_abs']} rel={a_d3['cruz_rel']} | sim={a_d3['similitud_celdas']:.4f}", flush=True)
    report["imagen3_referencia"]["metodo_d"] = a_d3

    print("\n" + "=" * 70, flush=True)
    print("JESHUA2-IZQ: metodos exactos con parametros ORIGINALES", flush=True)
    print("=" * 70, flush=True)
    R_chip_j, p_chip_j = metodo_chip(j2_izq)
    a_chip_j = analisis_completo(R_chip_j, gap=10)
    print(f"  [CHIP] densidad={a_chip_j['densidad']:.4f} | grid={a_chip_j['grid']} | D={a_chip_j['D_fractal']:.4f} | cruz={a_chip_j['cruz_abs']} rel={a_chip_j['cruz_rel']} | sim={a_chip_j['similitud_celdas']:.4f}", flush=True)
    report["jeshua2"]["metodo_chip_original"] = a_chip_j
    R_d_j, p_d_j = metodo_d(j2_izq)
    a_d_j = analisis_completo(R_d_j, gap=10)
    print(f"  [D]     densidad={a_d_j['densidad']:.4f} | grid={a_d_j['grid']} | D={a_d_j['D_fractal']:.4f} | cruz={a_d_j['cruz_abs']} rel={a_d_j['cruz_rel']} | sim={a_d_j['similitud_celdas']:.4f}", flush=True)
    report["jeshua2"]["metodo_d_original"] = a_d_j

    print("\n" + "=" * 70, flush=True)
    print(f"JESHUA2-IZQ: metodos exactos con parametros ESCALADOS (factor {FACTOR:.3f})", flush=True)
    print("=" * 70, flush=True)
    # Escalar: kernel (15*f, 1), sigma 15*f, gap 10*f
    ksize = int(15 * FACTOR) | 1  # impar
    gap_s = int(10 * FACTOR)
    sigma_s = 15.0 * FACTOR
    print(f"  kernel={ksize} | sigma={sigma_s:.1f} | gap={gap_s}", flush=True)
    R_chip_js, p_chip_js = metodo_chip(j2_izq, sigma_ksize=ksize)
    a_chip_js = analisis_completo(R_chip_js, gap=gap_s)
    print(f"  [CHIP] densidad={a_chip_js['densidad']:.4f} | grid={a_chip_js['grid']} | D={a_chip_js['D_fractal']:.4f} | cruz={a_chip_js['cruz_abs']} rel={a_chip_js['cruz_rel']} | sim={a_chip_js['similitud_celdas']:.4f}", flush=True)
    report["jeshua2"]["metodo_chip_escalado"] = a_chip_js
    R_d_js, p_d_js = metodo_d(j2_izq, sigma=sigma_s)
    a_d_js = analisis_completo(R_d_js, gap=gap_s)
    print(f"  [D]     densidad={a_d_js['densidad']:.4f} | grid={a_d_js['grid']} | D={a_d_js['D_fractal']:.4f} | cruz={a_d_js['cruz_abs']} rel={a_d_js['cruz_rel']} | sim={a_d_js['similitud_celdas']:.4f}", flush=True)
    report["jeshua2"]["metodo_d_escalado"] = a_d_js

    out_json = os.path.join(OUT, "metodos_exactos_jeshua_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
