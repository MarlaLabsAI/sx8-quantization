"""
VISUALIZACION 3D DE LA TOPOGRAFIA DEL MAPA DE BITS + CONECTIVIDAD ENTRE BITPLANES
==================================================================================
Parte 1: Relieve 3D del mapa de bits (Otsu -> suavizado -> topografia)
  - Renderiza la topografia 3D de la Sabana (imagen1 rostro)
  - Renderiza la topografia 3D de xray1 (comparacion)
  - Guarda PNG de alta resolucion

Parte 2: Conectividad entre bitplanes adyacentes
  - Para cada par (bit i, bit i+1): correlacion espacial, informacion mutua,
    co-ocurrencia, y "conectividad" (transiciones 0<->1 alineadas)
  - Compara con controles (bitplanes permutados) para significancia
  - La hipotesis del usuario: cada bit conecta con los de al lado

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N_CONTROLS = 50
N_WORKERS = 12

# ============================================================================
# 1. TOPOGRAFIA 3D
# ============================================================================
def topografia_mapa_bits(img_u8, downsample=8):
    """Mapa de bits -> relieve 3D (Otsu + suavizado)."""
    h, w = img_u8.shape
    small = cv2.resize(img_u8, (w//downsample, h//downsample))
    _, bits = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits_bin = (bits > 0).astype(np.float64)
    topo = cv2.GaussianBlur(bits_bin, (0, 0), 3)
    if topo.max() > 0:
        topo = topo / topo.max()
    return topo

def render_3d(topo, titulo, filename, elev=30, azim=-60):
    """Renderiza el relieve 3D y guarda PNG."""
    h, w = topo.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Muestrear para no saturar (cada 2 px)
    step = 2
    Xs, Ys, Zs = X[::step, ::step], Y[::step, ::step], topo[::step, ::step]
    surf = ax.plot_surface(Xs, Ys, Zs, cmap='terrain', linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.set_title(titulo, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_zlabel('Altura (densidad de bits)')
    ax.view_init(elev=elev, azim=azim)
    fig.colorbar(surf, ax=ax, shrink=0.5, label='Altura normalizada')
    plt.tight_layout()
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path

# ============================================================================
# 2. CONECTIVIDAD ENTRE BITPLANES
# ============================================================================
def bitplanes(img_u8):
    return [((img_u8 >> b) & 1).astype(np.uint8) for b in range(8)]

def conectividad_entre_bits(b1, b2):
    """Mide la conexion entre dos bitplanes:
    - Correlacion espacial (Pearson)
    - Informacion mutua (2x2)
    - Co-ocurrencia: P(ambos=1) / (P(1)*P(2)) -> >1 = conexion positiva
    - Alineacion de transiciones: bordes de b1 que coinciden con bordes de b2"""
    a = b1.astype(np.float64)
    b = b2.astype(np.float64)
    # Correlacion
    if a.std() == 0 or b.std() == 0:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    # MI
    c = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            c[i, j] = np.mean((b1 == i) & (b2 == j))
    c /= c.sum()
    pa, pb = c.sum(axis=1), c.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if c[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                mi += c[i, j] * np.log2(c[i, j] / (pa[i] * pb[j]))
    # Co-ocurrencia normalizada
    p1, p2 = b1.mean(), b2.mean()
    p11 = np.mean((b1 == 1) & (b2 == 1))
    co_occ = p11 / (p1 * p2) if p1 > 0 and p2 > 0 else float("nan")
    # Alineacion de bordes (transiciones)
    g1 = np.abs(cv2.Sobel(b1.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)) + \
         np.abs(cv2.Sobel(b1.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3))
    g2 = np.abs(cv2.Sobel(b2.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)) + \
         np.abs(cv2.Sobel(b2.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3))
    e1 = (g1 > 0).astype(np.float64)
    e2 = (g2 > 0).astype(np.float64)
    alineacion = float(np.mean(e1 * e2)) / (np.mean(e1) * np.mean(e2)) if np.mean(e1) > 0 and np.mean(e2) > 0 else float("nan")
    return {"corr": corr, "mi": float(mi), "co_occ": co_occ, "alineacion_bordes": alineacion}

def worker_par(args):
    b1, b2, nombre = args
    return {"nombre": nombre, **conectividad_entre_bits(b1, b2)}

# ============================================================================
# 3. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"topografia": {}, "conectividad_bits": {}, "controles_conectividad": {},
              "conclusion": {}}

    # Cargar imagenes
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    face = img1[100:1100, 1000:2000]
    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)
    xray_small = cv2.resize(xray, (1000, 1000))

    # ============ PARTE 1: TOPOGRAFIA 3D ============
    print("=" * 70, flush=True)
    print("PARTE 1: TOPOGRAFIA 3D DEL MAPA DE BITS", flush=True)
    print("=" * 70, flush=True)
    topo_sabana = topografia_mapa_bits(face)
    topo_xray = topografia_mapa_bits(xray_small)
    p1 = render_3d(topo_sabana, "Topografia 3D del Mapa de Bits - Sabana Santa (rostro)", "topografia_3d_sabana.png")
    p2 = render_3d(topo_xray, "Topografia 3D del Mapa de Bits - Radiografia (xray1)", "topografia_3d_xray1.png")
    print(f"  Sabana: {p1}", flush=True)
    print(f"  XRAY1:  {p2}", flush=True)
    report["topografia"] = {"sabana_png": p1, "xray1_png": p2,
                            "sabana_shape": list(topo_sabana.shape), "xray1_shape": list(topo_xray.shape)}

    # ============ PARTE 2: CONECTIVIDAD ENTRE BITPLANES ============
    print("\n" + "=" * 70, flush=True)
    print("PARTE 2: CONECTIVIDAD ENTRE BITPLANES ADYACENTES", flush=True)
    print("=" * 70, flush=True)
    planes = bitplanes(face)
    print("  Pares adyacentes (bit i vs bit i+1) en la Sabana:", flush=True)
    conect = {}
    for i in range(7):
        r = conectividad_entre_bits(planes[i], planes[i+1])
        conect[f"bit{i}-bit{i+1}"] = r
        print(f"    bit{i}-bit{i+1}: corr={r['corr']:+.4f} | MI={r['mi']:.4f} | co_occ={r['co_occ']:.3f} | bordes={r['alineacion_bordes']:.3f}", flush=True)
    report["conectividad_bits"]["sabana"] = conect

    # Controles: permutar cada bitplane y medir la conectividad esperada por azar
    print(f"\n  Controles ({N_CONTROLS} permutaciones de bitplanes):", flush=True)
    tasks = []
    for i in range(7):
        for k in range(N_CONTROLS):
            b1_perm = planes[i].flatten()[rng.permutation(planes[i].size)].reshape(planes[i].shape)
            b2_perm = planes[i+1].flatten()[rng.permutation(planes[i+1].size)].reshape(planes[i+1].shape)
            tasks.append((b1_perm, b2_perm, f"bit{i}-bit{i+1}_{k}"))
    with Pool(N_WORKERS) as pool:
        res = list(pool.imap_unordered(worker_par, tasks))
    print("  Conectividad por azar (media±std de permutaciones):", flush=True)
    for i in range(7):
        grupo = [r for r in res if r["nombre"].startswith(f"bit{i}-bit{i+1}_")]
        corrs = [r["corr"] for r in grupo if not np.isnan(r["corr"])]
        mis = [r["mi"] for r in grupo]
        co_occs = [r["co_occ"] for r in grupo if not np.isnan(r["co_occ"])]
        alins = [r["alineacion_bordes"] for r in grupo if not np.isnan(r["alineacion_bordes"])]
        real = conect[f"bit{i}-bit{i+1}"]
        z_corr = (real["corr"] - np.mean(corrs)) / np.std(corrs) if np.std(corrs) > 0 else float("nan")
        z_mi = (real["mi"] - np.mean(mis)) / np.std(mis) if np.std(mis) > 0 else float("nan")
        z_co = (real["co_occ"] - np.mean(co_occs)) / np.std(co_occs) if np.std(co_occs) > 0 else float("nan")
        z_al = (real["alineacion_bordes"] - np.mean(alins)) / np.std(alins) if np.std(alins) > 0 else float("nan")
        print(f"    bit{i}-bit{i+1}: corr azar={np.mean(corrs):+.4f}±{np.std(corrs):.4f} (z={z_corr:+.1f}) | "
              f"MI azar={np.mean(mis):.4f}±{np.std(mis):.4f} (z={z_mi:+.1f}) | "
              f"co_occ azar={np.mean(co_occs):.3f}±{np.std(co_occs):.3f} (z={z_co:+.1f}) | "
              f"bordes azar={np.mean(alins):.3f}±{np.std(alins):.3f} (z={z_al:+.1f})", flush=True)
        report["controles_conectividad"][f"bit{i}-bit{i+1}"] = {
            "corr_azar_mean": float(np.mean(corrs)), "corr_azar_std": float(np.std(corrs)), "z_corr": float(z_corr),
            "mi_azar_mean": float(np.mean(mis)), "mi_azar_std": float(np.std(mis)), "z_mi": float(z_mi),
            "co_occ_azar_mean": float(np.mean(co_occs)), "co_occ_azar_std": float(np.std(co_occs)), "z_co": float(z_co),
            "bordes_azar_mean": float(np.mean(alins)), "bordes_azar_std": float(np.std(alins)), "z_al": float(z_al),
        }

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    # Resumen: cuantos pares tienen z > 3 en cada metrica
    n_sig_corr = sum(1 for k, v in report["controles_conectividad"].items() if abs(v["z_corr"]) > 3)
    n_sig_mi = sum(1 for k, v in report["controles_conectividad"].items() if abs(v["z_mi"]) > 3)
    n_sig_co = sum(1 for k, v in report["controles_conectividad"].items() if abs(v["z_co"]) > 3)
    n_sig_al = sum(1 for k, v in report["controles_conectividad"].items() if abs(v["z_al"]) > 3)
    print(f"  Pares con |z|>3 (de 7): corr={n_sig_corr} | MI={n_sig_mi} | co_occ={n_sig_co} | bordes={n_sig_al}", flush=True)
    report["conclusion"] = {
        "pares_sig_corr": n_sig_corr, "pares_sig_mi": n_sig_mi,
        "pares_sig_co_occ": n_sig_co, "pares_sig_bordes": n_sig_al,
        "total_pares": 7,
        "interpretacion": "Si la mayoria de pares tienen |z|>3, los bitplanes adyacentes estan conectados"
    }

    out_json = os.path.join(OUT, "topografia_3d_y_conectividad_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
