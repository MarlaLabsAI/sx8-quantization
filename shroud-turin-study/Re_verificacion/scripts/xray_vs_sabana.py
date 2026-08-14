"""
COMPARACION RADIOGRAFIA REAL (xray1) vs SABANA SANTA + TOPOGRAFIA 3D DEL MAPA DE BITS
=====================================================================================
Hipotesis del usuario:
  1. La Sabana NO fue pintada: es como una radiografia pero MAS DEBIL
     (los rayos penetran menos, solo irradian las fibras superficiales).
  2. El mapa de bits, con escala de grises por bit, es una TOPOGRAFIA
     que se puede pasar a 3D.

Este script:
  A. Compara las firmas de proyeccion (suavidad I-|grad|, kurtosis de
     gradientes, curvatura) entre:
     - Sabana (imagen1 rostro, crop 1000x1000)
     - Radiografia REAL (xray1, 3000x4325)
     - Controles (pintura simulada, radiografia simulada, ruido)
  B. Analiza los BITPLANES de la Sabana: descompone en 8 planos de bits
     y mide la estructura de cada uno (densidad, D fractal, entropia).
  C. Construye la TOPOGRAFIA 3D del mapa de bits (relieve) y mide sus
     propiedades: rugosidad, dimension fractal del relieve, distribucion
     de alturas, y la relacion altura-pendiente (firma de proyeccion).

NO modifica originales. Guarda en Re_verificacion/resultados/.
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
N_CONTROLS = 30
N_WORKERS = 12

# ============================================================================
# 1. FIRMAS DE PROYECCION (reutilizadas del script anterior)
# ============================================================================
def relacion_intensidad_gradiente(img_norm, n_bins=40):
    dx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(img_norm, bins)
    grad_por_bin = []
    for i in range(1, n_bins + 1):
        mask = idx == i
        if mask.sum() > 100:
            grad_por_bin.append(float(grad[mask].mean()))
        else:
            grad_por_bin.append(float("nan"))
    vals = np.array([g for g in grad_por_bin if not np.isnan(g)])
    if len(vals) < 5:
        return {"suavidad": float("nan"), "monotonia": float("nan")}
    diffs = np.abs(np.diff(vals))
    suavidad = 1.0 - np.mean(diffs) / (np.max(vals) - np.min(vals) + 1e-9)
    signos = np.sign(np.diff(vals))
    monotonia = float(np.mean(signos == signos[0])) if len(signos) > 0 else float("nan")
    return {"suavidad": float(suavidad), "monotonia": monotonia}

def distribucion_gradientes(img_norm):
    dx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    top5 = np.percentile(grad, 95)
    kurt = float(((grad - grad.mean())**4).mean() / grad.std()**4) if grad.std() > 0 else float("nan")
    return {"frac_bordes_top5": float((grad > top5).mean()), "kurtosis": kurt,
            "grad_medio": float(grad.mean()), "grad_p95": float(top5)}

def curvatura_media(img_norm):
    img_s = cv2.GaussianBlur(img_norm, (5, 5), 0)
    dx = cv2.Sobel(img_s, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_s, cv2.CV_64F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
    H = (dxx + dyy) / 2.0
    return {"H_mean": float(np.abs(H).mean()), "H_std": float(H.std())}

def firma_proyeccion(img_norm):
    return {
        "relacion_IG": relacion_intensidad_gradiente(img_norm),
        "gradientes": distribucion_gradientes(img_norm),
        "curvatura": curvatura_media(img_norm),
    }

# ============================================================================
# 2. BITPLANES
# ============================================================================
def bitplanes(img_u8):
    """Descompone en 8 bitplanes (bit 0 = LSB, bit 7 = MSB)."""
    planes = []
    for b in range(8):
        plane = ((img_u8 >> b) & 1).astype(np.uint8)
        planes.append(plane)
    return planes

def box_counting_dimension(binary, min_size=2, max_size=None):
    h, w = binary.shape
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
        nh, nw = h // s, w // s
        if nh == 0 or nw == 0:
            continue
        blocks = binary[:nh*s, :nw*s].reshape(nh, s, nw, s)
        counts.append((blocks.sum(axis=(1,3)) > 0).sum())
    counts = np.array(counts)
    valid = counts > 0
    if valid.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0/sizes[valid]), np.log(counts[valid]), 1)
    return float(slope)

def entropia_shannon(binary):
    p = binary.mean()
    if p == 0 or p == 1:
        return 0.0
    return float(-(p*np.log2(p) + (1-p)*np.log2(1-p)))

def analisis_bitplanes(img_u8):
    planes = bitplanes(img_u8)
    res = {}
    for b, plane in enumerate(planes):
        res[f"bit{b}"] = {
            "densidad": float(plane.mean()),
            "D_fractal": box_counting_dimension(plane),
            "entropia": entropia_shannon(plane),
        }
    return res

# ============================================================================
# 3. TOPOGRAFIA 3D DEL MAPA DE BITS
# ============================================================================
def topografia_mapa_bits(img_u8, downsample=4):
    """Mapa de bits -> relieve 3D:
    - Binarizar con Otsu (mapa de bits)
    - Suavizar el mapa binario -> escala de grises (topografia)
    - Medir propiedades del relieve"""
    h, w = img_u8.shape
    small = cv2.resize(img_u8, (w//downsample, h//downsample))
    # Mapa de bits con Otsu
    _, bits = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits_bin = (bits > 0).astype(np.float64)
    # Topografia: suavizar el mapa binario (cada bit contribuye a la altura)
    topo = cv2.GaussianBlur(bits_bin, (0, 0), 3)
    topo = topo / topo.max() if topo.max() > 0 else topo
    # Rugosidad: std de la topografia
    rugosidad = float(topo.std())
    # D fractal del relieve (umbrales de altura)
    D_relieve = []
    for T in [0.2, 0.4, 0.6, 0.8]:
        D_relieve.append(box_counting_dimension((topo > T).astype(np.uint8)))
    # Relacion altura-pendiente (firma de proyeccion en el relieve)
    dy, dx = np.gradient(topo)
    grad = np.sqrt(dx**2 + dy**2)
    bins = np.linspace(0, 1, 20)
    idx = np.digitize(topo, bins)
    grad_por_altura = []
    for i in range(1, 21):
        mask = idx == i
        if mask.sum() > 50:
            grad_por_altura.append(float(grad[mask].mean()))
        else:
            grad_por_altura.append(float("nan"))
    vals = np.array([g for g in grad_por_altura if not np.isnan(g)])
    suavidad_relieve = float("nan")
    if len(vals) >= 5:
        diffs = np.abs(np.diff(vals))
        suavidad_relieve = 1.0 - np.mean(diffs) / (np.max(vals) - np.min(vals) + 1e-9)
    return {
        "rugosidad": rugosidad,
        "D_relieve_por_umbral": D_relieve,
        "D_relieve_medio": float(np.mean([d for d in D_relieve if not np.isnan(d)])) if any(not np.isnan(d) for d in D_relieve) else float("nan"),
        "suavidad_altura_pendiente": suavidad_relieve,
        "grad_por_altura": grad_por_altura,
    }

# ============================================================================
# 4. CONTROLES
# ============================================================================
def pintura_simulada(shape, n_manchas=40):
    h, w = shape
    img = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_manchas):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = rng.integers(20, 120), rng.integers(20, 120)
        ang = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        yy, xx = np.mgrid[0:h, 0:w]
        xr = (xx - cx) * cos_a + (yy - cy) * sin_a
        yr = -(xx - cx) * sin_a + (yy - cy) * cos_a
        mask = (xr / rx)**2 + (yr / ry)**2 < 1
        img[mask] = rng.uniform(0.2, 0.9)
    return np.clip(img, 0, 1)

def radiografia_simulada(shape, n_esferas=3):
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_esferas):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(80, 250)
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        espesor = np.clip(2 * np.sqrt(np.maximum(r**2 - d**2, 0)), 0, None)
        img += espesor * rng.uniform(0.3, 1.0)
    return img / img.max()

def ruido_suave(shape, sigma=3.0):
    h, w = shape
    noise = rng.normal(0, 1, size=(h, w))
    smooth = cv2.GaussianBlur(noise, (0, 0), sigma)
    return (smooth - smooth.min()) / (smooth.max() - smooth.min())

def worker_firma(args):
    img_norm, nombre = args
    return {"nombre": nombre, "firma": firma_proyeccion(img_norm)}

# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"sabana": {}, "xray1": {}, "controles": {}, "bitplanes_sabana": {},
              "topografia_sabana": {}, "topografia_xray1": {}, "conclusion": {}}

    # Cargar Sabana (imagen1 rostro) y xray1
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    face = img1[100:1100, 1000:2000]
    face_norm = face.astype(float) / 255.0

    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)
    print("=" * 70, flush=True)
    print(f"XRAY1: {xray.shape}", flush=True)
    print("=" * 70, flush=True)
    # Reducir xray a 1000x1000 (mismo tamano que el rostro) para comparacion justa
    xray_small = cv2.resize(xray, (1000, 1000))
    xray_norm = xray_small.astype(float) / 255.0

    # A. Firmas
    print("\n[FIRMAS DE PROYECCION]", flush=True)
    f_sabana = firma_proyeccion(face_norm)
    f_xray = firma_proyeccion(xray_norm)
    report["sabana"]["firma"] = f_sabana
    report["xray1"]["firma"] = f_xray
    print(f"  SABANA: suavidad_IG={f_sabana['relacion_IG']['suavidad']:.3f} | kurtosis={f_sabana['gradientes']['kurtosis']:.1f} | "
          f"grad_medio={f_sabana['gradientes']['grad_medio']:.5f} | |H|={f_sabana['curvatura']['H_mean']:.6f}", flush=True)
    print(f"  XRAY1:  suavidad_IG={f_xray['relacion_IG']['suavidad']:.3f} | kurtosis={f_xray['gradientes']['kurtosis']:.1f} | "
          f"grad_medio={f_xray['gradientes']['grad_medio']:.5f} | |H|={f_xray['curvatura']['H_mean']:.6f}", flush=True)

    # Controles
    print(f"\n[CONTROLES] ({N_CONTROLS} x 3)", flush=True)
    tipos = ["pintura", "radiografia", "ruido_suave"]
    tasks = []
    for tipo in tipos:
        for i in range(N_CONTROLS):
            if tipo == "pintura":
                img_c = pintura_simulada(face_norm.shape)
            elif tipo == "radiografia":
                img_c = radiografia_simulada(face_norm.shape)
            else:
                img_c = ruido_suave(face_norm.shape)
            tasks.append((img_c, f"{tipo}_{i}"))
    with Pool(N_WORKERS) as pool:
        res = list(pool.imap_unordered(worker_firma, tasks))
    for tipo in tipos:
        grupo = [r for r in res if r["nombre"].startswith(tipo)]
        suav = [r["firma"]["relacion_IG"]["suavidad"] for r in grupo]
        kurt = [r["firma"]["gradientes"]["kurtosis"] for r in grupo]
        gmed = [r["firma"]["gradientes"]["grad_medio"] for r in grupo]
        curv = [r["firma"]["curvatura"]["H_mean"] for r in grupo]
        print(f"  [{tipo}] suav_IG={np.mean(suav):.3f}±{np.std(suav):.3f} | kurt={np.mean(kurt):.1f}±{np.std(kurt):.1f} | "
              f"grad={np.mean(gmed):.5f}±{np.std(gmed):.5f} | |H|={np.mean(curv):.6f}±{np.std(curv):.6f}", flush=True)
        report["controles"][tipo] = {
            "suav_IG_mean": float(np.mean(suav)), "suav_IG_std": float(np.std(suav)),
            "kurt_mean": float(np.mean(kurt)), "kurt_std": float(np.std(kurt)),
            "grad_mean": float(np.mean(gmed)), "grad_std": float(np.std(gmed)),
            "curv_mean": float(np.mean(curv)), "curv_std": float(np.std(curv)),
        }

    # B. Bitplanes de la Sabana
    print("\n[BITPLANES DE LA SABANA (imagen1 rostro)]", flush=True)
    bp = analisis_bitplanes(face)
    report["bitplanes_sabana"] = bp
    for b in range(8):
        d = bp[f"bit{b}"]
        print(f"  bit{b}: densidad={d['densidad']:.4f} | D_fractal={d['D_fractal']:.3f} | entropia={d['entropia']:.3f}", flush=True)

    # C. Topografia 3D del mapa de bits
    print("\n[TOPOGRAFIA 3D DEL MAPA DE BITS]", flush=True)
    topo_sabana = topografia_mapa_bits(face)
    topo_xray = topografia_mapa_bits(xray_small)
    report["topografia_sabana"] = topo_sabana
    report["topografia_xray1"] = topo_xray
    print(f"  SABANA: rugosidad={topo_sabana['rugosidad']:.4f} | D_relieve={topo_sabana['D_relieve_medio']:.3f} | "
          f"suavidad_altura_pendiente={topo_sabana['suavidad_altura_pendiente']:.3f}", flush=True)
    print(f"  XRAY1:  rugosidad={topo_xray['rugosidad']:.4f} | D_relieve={topo_xray['D_relieve_medio']:.3f} | "
          f"suavidad_altura_pendiente={topo_xray['suavidad_altura_pendiente']:.3f}", flush=True)

    # D. Conclusion
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    # Comparar Sabana vs xray1 en suavidad_IG
    print(f"  Suavidad I-|grad|: Sabana={f_sabana['relacion_IG']['suavidad']:.3f} vs XRAY1={f_xray['relacion_IG']['suavidad']:.3f}", flush=True)
    print(f"  Kurtosis: Sabana={f_sabana['gradientes']['kurtosis']:.1f} vs XRAY1={f_xray['gradientes']['kurtosis']:.1f}", flush=True)
    print(f"  Grad medio: Sabana={f_sabana['gradientes']['grad_medio']:.5f} vs XRAY1={f_xray['gradientes']['grad_medio']:.5f}", flush=True)
    print(f"  Curvatura |H|: Sabana={f_sabana['curvatura']['H_mean']:.6f} vs XRAY1={f_xray['curvatura']['H_mean']:.6f}", flush=True)
    print(f"  Topografia: rugosidad Sabana={topo_sabana['rugosidad']:.4f} vs XRAY1={topo_xray['rugosidad']:.4f}", flush=True)
    print(f"  Topografia: D_relieve Sabana={topo_sabana['D_relieve_medio']:.3f} vs XRAY1={topo_xray['D_relieve_medio']:.3f}", flush=True)
    report["conclusion"] = {
        "suavidad_IG": {"sabana": f_sabana["relacion_IG"]["suavidad"], "xray1": f_xray["relacion_IG"]["suavidad"]},
        "kurtosis": {"sabana": f_sabana["gradientes"]["kurtosis"], "xray1": f_xray["gradientes"]["kurtosis"]},
        "grad_medio": {"sabana": f_sabana["gradientes"]["grad_medio"], "xray1": f_xray["gradientes"]["grad_medio"]},
        "curvatura": {"sabana": f_sabana["curvatura"]["H_mean"], "xray1": f_xray["curvatura"]["H_mean"]},
        "topografia": {"rugosidad_sabana": topo_sabana["rugosidad"], "rugosidad_xray1": topo_xray["rugosidad"],
                        "D_relieve_sabana": topo_sabana["D_relieve_medio"], "D_relieve_xray1": topo_xray["D_relieve_medio"]},
    }

    out_json = os.path.join(OUT, "xray_vs_sabana_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
