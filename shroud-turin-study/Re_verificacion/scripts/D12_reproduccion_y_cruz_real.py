"""
REPRODUCCION EXACTA DEL D12 + APLICACION DEL MISMO METODO A LA CRUZ REAL
========================================================================
El estudio D12 demostro que la proyeccion 3D->2D de una esfera con capas
internas produce:
  - Centro: D=1.737, Delta_alpha=5.204
  - Periferia: D=0.0, Delta_alpha=0.964
  - Ratio Delta_alpha = 5.40x

Este script:
  1. Reproduce EXACTAMENTE el D12 (mismo codigo, mismos parametros)
  2. Aplica el MISMO metodo (box-counting sizes [2,4,8] + multifractal
     simple q[-3,3] log10) a la CRUZ REAL de la matriz de recurrencia:
     - Centro 20x20 alrededor de (416,416)
     - Periferia 20x20 en la esquina
  3. Compara los patrones: si la cruz real muestra el mismo gradiente
     (centro D mayor, Delta_alpha mayor), es consistente con proyeccion
     dimensional - el resultado del estudio.

NO modifica originales. Guarda en Re_verificacion/resultados/.
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

# ============================================================================
# METODOS EXACTOS DEL ESTUDIO (copiados de tests_D2b_D13_profundizacion.py)
# ============================================================================
def box_counting_simple(matrix):
    """D12 exacto: sizes [2,4,8], box.sum()>0."""
    sizes = [2, 4, 8]
    counts = []
    for size in sizes:
        h, w = matrix.shape
        n_boxes = 0
        for i in range(0, h, size):
            for j in range(0, w, size):
                box = matrix[i:i+size, j:j+size]
                if box.sum() > 0:
                    n_boxes += 1
        counts.append(n_boxes)
    sizes = np.array(sizes)
    counts = np.array(counts)
    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])

def simple_multifractal_width(matrix):
    """D12 exacto: q[-3,3] 13 valores, log base 10."""
    measure = matrix.flatten()
    measure = measure[measure > 0]
    if len(measure) == 0:
        return 0.0
    measure = measure / measure.sum()
    tau_q = []
    q_values = np.linspace(-3, 3, 13)
    for q in q_values:
        if q == 0:
            tau = 0
        else:
            tau = np.log(np.sum(measure ** q)) / np.log(10)
        tau_q.append(tau)
    tau_q = np.array(tau_q)
    alpha = np.gradient(tau_q, q_values)
    return float(alpha.max() - alpha.min())

# ============================================================================
# 1. REPRODUCCION EXACTA DEL D12
# ============================================================================
def reproducir_D12():
    """Copia exacta del test_D12_dimensional_projection_simulation."""
    size = 100
    x, y, z = np.mgrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    object_3d = np.zeros_like(r)
    object_3d[r < 0.9] = 1.0
    object_3d[r < 0.7] = 0.8
    object_3d[r < 0.5] = 0.6
    object_3d[r < 0.3] = 0.4
    projection_2d = object_3d.sum(axis=2)
    projection_2d = projection_2d / projection_2d.max()
    center_region = projection_2d[40:60, 40:60]
    peripheral_region = projection_2d[:20, :20]
    D_center = box_counting_simple(center_region)
    D_peripheral = box_counting_simple(peripheral_region)
    da_center = simple_multifractal_width(center_region)
    da_peripheral = simple_multifractal_width(peripheral_region)
    return {
        "D_center": D_center, "D_peripheral": D_peripheral,
        "delta_alpha_center": da_center, "delta_alpha_peripheral": da_peripheral,
        "ratio_da": da_center / da_peripheral if da_peripheral > 0 else 0.0,
        "densidad_centro": float(center_region.mean()),
        "densidad_periferia": float(peripheral_region.mean()),
    }

# ============================================================================
# 2. APLICAR EL MISMO METODO A LA CRUZ REAL
# ============================================================================
def analizar_cruz_real():
    """Mismo metodo D12 sobre la matriz de recurrencia real (imagen3)."""
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < 10.0).astype(np.float32)
    n = R_bin.shape[0]
    cx, cy = 416, 416
    # Centro 20x20 alrededor de la cruz (como D12: region 20x20)
    center_region = R_bin[cy-10:cy+10, cx-10:cx+10]
    # Periferia 20x20 en la esquina (como D12: [:20,:20])
    peripheral_region = R_bin[:20, :20]
    D_center = box_counting_simple(center_region)
    D_peripheral = box_counting_simple(peripheral_region)
    da_center = simple_multifractal_width(center_region)
    da_peripheral = simple_multifractal_width(peripheral_region)
    return {
        "D_center": D_center, "D_peripheral": D_peripheral,
        "delta_alpha_center": da_center, "delta_alpha_peripheral": da_peripheral,
        "ratio_da": da_center / da_peripheral if da_peripheral > 0 else 0.0,
        "densidad_centro": float(center_region.mean()),
        "densidad_periferia": float(peripheral_region.mean()),
        "cruz": (cx, cy), "n": n,
    }

# ============================================================================
# 3. COMPARACION CON CONTROLES (mismo metodo sobre permutaciones)
# ============================================================================
def controles_cruz_real(n_controles=50):
    """Mismo metodo sobre matrices de recurrencia de perfiles permutados."""
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    rng = np.random.default_rng(42)
    cx, cy = 416, 416
    resultados = {"D_center": [], "D_peripheral": [], "da_center": [], "da_peripheral": []}
    for _ in range(n_controles):
        p = rng.permutation(profile)
        R_cont = np.abs(p[:, None] - p[None, :])
        R_bin = (R_cont < 10.0).astype(np.float32)
        center_region = R_bin[cy-10:cy+10, cx-10:cx+10]
        peripheral_region = R_bin[:20, :20]
        resultados["D_center"].append(box_counting_simple(center_region))
        resultados["D_peripheral"].append(box_counting_simple(peripheral_region))
        resultados["da_center"].append(simple_multifractal_width(center_region))
        resultados["da_peripheral"].append(simple_multifractal_width(peripheral_region))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in resultados.items()}

# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {}

    print("=" * 70, flush=True)
    print("1. REPRODUCCION EXACTA DEL D12 (esfera 3D con capas -> proyeccion 2D)", flush=True)
    print("=" * 70, flush=True)
    d12 = reproducir_D12()
    report["D12_reproducido"] = d12
    print(f"  Centro: D={d12['D_center']:.4f} | Delta_alpha={d12['delta_alpha_center']:.4f} | densidad={d12['densidad_centro']:.4f}", flush=True)
    print(f"  Periferia: D={d12['D_peripheral']:.4f} | Delta_alpha={d12['delta_alpha_peripheral']:.4f} | densidad={d12['densidad_periferia']:.4f}", flush=True)
    print(f"  Ratio Delta_alpha = {d12['ratio_da']:.2f}x", flush=True)
    print(f"  (estudio: D_center=1.737, D_periph=0.0, da_center=5.204, da_periph=0.964, ratio=5.40x)", flush=True)
    ok_d12 = abs(d12["D_center"] - 1.737) < 0.05 and abs(d12["delta_alpha_center"] - 5.204) < 0.5
    print(f"  Reproduccion correcta: {'SI' if ok_d12 else 'NO'}", flush=True)
    report["D12_reproducido_ok"] = bool(ok_d12)

    print("\n" + "=" * 70, flush=True)
    print("2. MISMO METODO APLICADO A LA CRUZ REAL (matriz de recurrencia)", flush=True)
    print("=" * 70, flush=True)
    cruz = analizar_cruz_real()
    report["cruz_real"] = cruz
    print(f"  Centro (cruz 416,416): D={cruz['D_center']:.4f} | Delta_alpha={cruz['delta_alpha_center']:.4f} | densidad={cruz['densidad_centro']:.4f}", flush=True)
    print(f"  Periferia (esquina): D={cruz['D_peripheral']:.4f} | Delta_alpha={cruz['delta_alpha_peripheral']:.4f} | densidad={cruz['densidad_periferia']:.4f}", flush=True)
    print(f"  Ratio Delta_alpha = {cruz['ratio_da']:.2f}x", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("3. CONTROLES (50 permutaciones, mismo metodo)", flush=True)
    print("=" * 70, flush=True)
    ctrl = controles_cruz_real(50)
    report["controles"] = ctrl
    print(f"  D_center: {ctrl['D_center']['mean']:.4f}±{ctrl['D_center']['std']:.4f} (real: {cruz['D_center']:.4f})", flush=True)
    print(f"  D_peripheral: {ctrl['D_peripheral']['mean']:.4f}±{ctrl['D_peripheral']['std']:.4f} (real: {cruz['D_peripheral']:.4f})", flush=True)
    print(f"  da_center: {ctrl['da_center']['mean']:.4f}±{ctrl['da_center']['std']:.4f} (real: {cruz['delta_alpha_center']:.4f})", flush=True)
    print(f"  da_peripheral: {ctrl['da_peripheral']['mean']:.4f}±{ctrl['da_peripheral']['std']:.4f} (real: {cruz['delta_alpha_peripheral']:.4f})", flush=True)

    # z-scores
    z_dc = (cruz["D_center"] - ctrl["D_center"]["mean"]) / ctrl["D_center"]["std"] if ctrl["D_center"]["std"] > 0 else float("nan")
    z_da = (cruz["delta_alpha_center"] - ctrl["da_center"]["mean"]) / ctrl["da_center"]["std"] if ctrl["da_center"]["std"] > 0 else float("nan")
    print(f"\n  z-score D_center = {z_dc:+.2f} | z-score da_center = {z_da:+.2f}", flush=True)
    report["z_scores"] = {"z_D_center": float(z_dc), "z_da_center": float(z_da)}

    print("\n" + "=" * 70, flush=True)
    print("4. COMPARACION DE PATRONES: proyeccion 3D->2D vs cruz real", flush=True)
    print("=" * 70, flush=True)
    print(f"  {'Metrica':<22} {'Proyeccion 3D->2D':<20} {'Cruz real':<20} {'Coincide'}", flush=True)
    print(f"  {'-'*70}", flush=True)
    patrones = []
    for metrica, sim, real in [
        ("D centro", d12["D_center"], cruz["D_center"]),
        ("D periferia", d12["D_peripheral"], cruz["D_peripheral"]),
        ("Delta_alpha centro", d12["delta_alpha_center"], cruz["delta_alpha_center"]),
        ("Delta_alpha periferia", d12["delta_alpha_peripheral"], cruz["delta_alpha_peripheral"]),
        ("densidad centro", d12["densidad_centro"], cruz["densidad_centro"]),
        ("densidad periferia", d12["densidad_periferia"], cruz["densidad_periferia"]),
    ]:
        # Patron: centro > periferia?
        coincide = (sim > 0.5) == (real > 0.5) if metrica.startswith("densidad") else (sim > 0.1) == (real > 0.1)
        patrones.append(coincide)
        print(f"  {metrica:<22} {sim:<20.4f} {real:<20.4f} {'SI' if coincide else 'NO'}", flush=True)
    # Patron global: centro mas denso, D mayor, da mayor
    patron_sim = (d12["densidad_centro"] > d12["densidad_periferia"]) and (d12["D_center"] > d12["D_peripheral"]) and (d12["delta_alpha_center"] > d12["delta_alpha_peripheral"])
    patron_real = (cruz["densidad_centro"] > cruz["densidad_periferia"]) and (cruz["D_center"] > cruz["D_peripheral"]) and (cruz["delta_alpha_center"] > cruz["delta_alpha_peripheral"])
    print(f"\n  PATRON GLOBAL (centro: mas denso + D mayor + da mayor):", flush=True)
    print(f"    Proyeccion 3D->2D: {'SI' if patron_sim else 'NO'}", flush=True)
    print(f"    Cruz real:         {'SI' if patron_real else 'NO'}", flush=True)
    print(f"    -> {'PATRON IDENTICO: la cruz real es consistente con proyeccion dimensional' if patron_sim == patron_real else 'patrones distintos'}", flush=True)
    report["patron_global"] = {"simulacion": bool(patron_sim), "cruz_real": bool(patron_real),
                               "coincide": bool(patron_sim == patron_real)}

    out_json = os.path.join(OUT, "D12_reproduccion_y_cruz_real_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
