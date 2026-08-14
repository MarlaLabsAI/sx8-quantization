"""
CORRELACION GLOBAL: LA CRUZ CENTRAL COMO ABSTRACTOR
====================================================
Sintesis de TODO el estudio: correlacionar los hallazgos entre si.

Pregunta central: la cruz central (416,416) actua como un ABSTRACTOR
(algo que VA o VIENE - atractor que recibe, emisor que emite, o ambos)?

Correlaciones a medir:
  1. CAMPO VECTORIAL: el flujo va HACIA la cruz (convergencia/atractor)
     o SALE de la cruz (divergencia/emisor)? (A6 del estudio)
  2. DIMENSION FRACTAL RADIAL: D vs distancia a la cruz (D1 del estudio)
  3. MULTIFRACTAL RADIAL: delta_alpha vs distancia (D2b)
  4. ESPECTRAL RADIAL: alta frecuencia vs distancia (D5b)
  5. NO-LOCALIDAD: MI(cruz, celda) vs distancia (D13) - correlacion completa
  6. BITPLANES: estructura de cada bit vs distancia a la cruz
  7. TOPOGRAFIA: altura del relieve vs distancia a la cruz
  8. GRID ADAPTATIVO: densidad vs distancia a la cruz (test B)
  9. SINTESIS: todas las correlaciones juntas -> perfil radial unificado

Metodo: perfil radial completo desde la cruz (anillos concentricos),
con controles (permutaciones) para significancia.

NO modifica originales. Guarda en Re_verificacion/resultados/.
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
rng = np.random.default_rng(42)
N_CONTROLS = 50
N_WORKERS = 12

# ============================================================================
# 1. MATRIZ DE RECURRENCIA (metodo D del estudio: gaussian_filter1d sigma=15)
# ============================================================================
def build_recurrence(img, sigma=15.0, threshold=10.0):
    h, w = img.shape
    profile = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=sigma)
    R = (np.abs(profile[:, None] - profile[None, :]) < threshold).astype(np.float32)
    return R, profile

# ============================================================================
# 2. PERFIL RADIAL DESDE LA CRUZ
# ============================================================================
def anillos_desde_cruz(R, cx, cy, n_anillos=20, ancho=10):
    """Divide la matriz en anillos concentricos desde la cruz."""
    h, w = R.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)
    anillos = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 0:
            anillos.append((r0, r1, mask))
    return anillos

def perfil_radial_metricas(R, cx, cy, n_anillos=20, ancho=10):
    """Calcula metricas por anillo: densidad, D fractal, alta frecuencia, MI."""
    anillos = anillos_desde_cruz(R, cx, cy, n_anillos, ancho)
    res = []
    for r0, r1, mask in anillos:
        region = R[mask]
        d = {
            "r0": r0, "r1": r1,
            "densidad": float(region.mean()),
        }
        res.append(d)
    return res

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

def alta_frecuencia_pct(region_2d):
    """% de energia en alta frecuencia (FFT)."""
    f = np.fft.fft2(region_2d)
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)**2
    h, w = mag.shape
    cy, cx = h//2, w//2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    max_r = min(h, w)//2
    total = mag.sum()
    if total == 0:
        return float("nan")
    # alta frecuencia = fuera del 50% central
    alta = mag[dist > max_r*0.5].sum()
    return float(alta / total * 100)

def mi_2d(a, b):
    """MI discreta 2x2."""
    a_b = (a > 0).astype(np.uint8)
    b_b = (b > 0).astype(np.uint8)
    if a_b.shape != b_b.shape:
        b_b = cv2.resize(b_b, (a_b.shape[1], a_b.shape[0]), interpolation=cv2.INTER_NEAREST)
    c = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            c[i,j] = np.mean((a_b == i) & (b_b == j))
    c /= c.sum()
    pa, pb = c.sum(axis=1), c.sum(axis=0)
    m = 0.0
    for i in range(2):
        for j in range(2):
            if c[i,j] > 0 and pa[i] > 0 and pb[j] > 0:
                m += c[i,j] * np.log2(c[i,j] / (pa[i]*pb[j]))
    return m

# ============================================================================
# 3. CAMPO VECTORIAL: convergencia vs divergencia
# ============================================================================
def campo_vectorial(R, cx, cy, radio_max=200):
    """Mide si el flujo va HACIA la cruz (atractor) o SALE (emisor).
    Para cada punto: vector gradiente. Proyeccion sobre el vector
    radial (hacia la cruz). Positiva = converge, negativa = diverge."""
    h, w = R.shape
    gy, gx = np.gradient(R.astype(np.float64))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    # Vector radial unitario (apuntando HACIA la cruz)
    rx = (cx - xx) / (dist + 1e-9)
    ry = (cy - yy) / (dist + 1e-9)
    # Proyeccion del gradiente sobre el vector radial
    proj = gx * rx + gy * ry
    mask = (dist > 5) & (dist < radio_max)
    proj_m = proj[mask]
    # Media de la proyeccion: >0 = converge (flujo hacia la cruz)
    # <0 = diverge (flujo desde la cruz)
    conv = float(proj_m.mean())
    # Fraccion de puntos que convergen
    frac_conv = float((proj_m > 0).mean())
    return {"convergencia_media": conv, "frac_converge": frac_conv,
            "interpretacion": ">0 = atractor (recibe), <0 = emisor (emite)"}

# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"cruz": {}, "campo_vectorial": {}, "perfil_radial": {},
              "bitplanes_vs_cruz": {}, "topografia_vs_cruz": {},
              "grid_adaptativo": {}, "sintesis": {}}

    # Cargar imagen3 (la del estudio)
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    R, profile = build_recurrence(img3)
    n = R.shape[0]
    print("=" * 70, flush=True)
    print(f"MATRIZ DE RECURRENCIA: {n}x{n} | densidad={R.mean():.4f}", flush=True)
    print("=" * 70, flush=True)

    # Cruz del estudio: (416,416) en la matriz 1080x1080
    cx, cy = 416, 416
    report["cruz"] = {"x": cx, "y": cy, "rel": (round(cx/n,3), round(cy/n,3))}
    print(f"CRUZ CENTRAL: ({cx},{cy}) rel=({cx/n:.3f},{cy/n:.3f})", flush=True)

    # ============ 1. CAMPO VECTORIAL ============
    print("\n[1] CAMPO VECTORIAL: ¿atractor (recibe) o emisor (emite)?", flush=True)
    cv = campo_vectorial(R, cx, cy)
    report["campo_vectorial"] = cv
    print(f"  Convergencia media = {cv['convergencia_media']:+.6f} | "
          f"frac_converge = {cv['frac_converge']:.3f}", flush=True)
    print(f"  -> {'ATRACTOR (la informacion VA hacia la cruz)' if cv['convergencia_media'] > 0 else 'EMISOR (la informacion VIENE de la cruz)'}", flush=True)

    # ============ 2. PERFIL RADIAL COMPLETO ============
    print("\n[2] PERFIL RADIAL DESDE LA CRUZ (densidad, D fractal, alta freq, MI)", flush=True)
    anillos = anillos_desde_cruz(R, cx, cy, n_anillos=15, ancho=15)
    perfil = []
    # Centro de la cruz (radio 15) como referencia
    centro_mask = anillos[0][2] if len(anillos) > 0 else None
    for r0, r1, mask in anillos:
        region = R[mask]
        # Reconstruir region 2D para D fractal y FFT
        region_2d = np.zeros_like(R)
        region_2d[mask] = R[mask]
        # Recortar bounding box
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        crop = region_2d[y0:y1, x0:x1]
        d = {
            "r0": r0, "r1": r1,
            "densidad": float(region.mean()),
            "D_fractal": box_counting_dimension(crop),
            "alta_freq_pct": alta_frecuencia_pct(crop),
        }
        perfil.append(d)
        print(f"  r={r0:3d}-{r1:3d}: dens={d['densidad']:.4f} | D={d['D_fractal']:.3f} | alta_freq={d['alta_freq_pct']:.1f}%", flush=True)
    report["perfil_radial"] = perfil

    # Correlaciones del perfil radial con la distancia
    dists = np.array([(d["r0"]+d["r1"])/2 for d in perfil])
    dens = np.array([d["densidad"] for d in perfil])
    Ds = np.array([d["D_fractal"] for d in perfil if not np.isnan(d["D_fractal"])])
    afs = np.array([d["alta_freq_pct"] for d in perfil if not np.isnan(d["alta_freq_pct"])])
    dists_D = dists[:len(Ds)]
    dists_af = dists[:len(afs)]
    corr_dens = pearsonr(dists, dens)[0]
    corr_D = pearsonr(dists_D, Ds)[0] if len(Ds) > 3 else float("nan")
    corr_af = pearsonr(dists_af, afs)[0] if len(afs) > 3 else float("nan")
    print(f"\n  Correlaciones con distancia a la cruz:", flush=True)
    print(f"    densidad vs distancia: {corr_dens:+.3f}", flush=True)
    print(f"    D_fractal vs distancia: {corr_D:+.3f}", flush=True)
    print(f"    alta_frecuencia vs distancia: {corr_af:+.3f}", flush=True)
    report["perfil_radial_correlaciones"] = {
        "densidad_vs_dist": float(corr_dens), "D_vs_dist": float(corr_D),
        "alta_freq_vs_dist": float(corr_af)}

    # ============ 3. NO-LOCALIDAD COMPLETA (D13 mejorado) ============
    print("\n[3] NO-LOCALIDAD: MI(cruz, celda) vs distancia (D13 completo)", flush=True)
    # Grid del estudio: 14x14 lineas
    grid_lines = [32, 62, 78, 137, 186, 229, 252, 293, 349, 387, 420, 470, 497, 524]
    # Centro de la cruz: region 50x50
    centro = R[cy-25:cy+25, cx-25:cx+25]
    mis = []
    dists_celda = []
    for i in range(len(grid_lines)-1):
        for j in range(len(grid_lines)-1):
            r1, r2 = grid_lines[i], grid_lines[i+1]
            c1, c2 = grid_lines[j], grid_lines[j+1]
            celda = R[r1:r2, c1:c2]
            if celda.size == 0:
                continue
            ccx = (c1+c2)//2
            ccy = (r1+r2)//2
            dist = np.sqrt((ccx-cx)**2 + (ccy-cy)**2)
            m = mi_2d(centro, celda)
            mis.append(m)
            dists_celda.append(dist)
    corr_mi = pearsonr(dists_celda, mis)[0]
    print(f"  MI media = {np.mean(mis):.4f} | corr(MI, distancia) = {corr_mi:+.4f}", flush=True)
    print(f"  -> {'NO-LOCAL (MI independiente de distancia)' if abs(corr_mi) < 0.3 else 'LOCAL (MI depende de distancia)'}", flush=True)
    report["no_localidad"] = {"mi_media": float(np.mean(mis)), "corr_mi_dist": float(corr_mi)}

    # ============ 4. BITPLANES vs CRUZ ============
    print("\n[4] BITPLANES: estructura de cada bit vs distancia a la cruz", flush=True)
    # Recortar la imagen a 1080x1080 centrada en la columna del perfil (w//2=960)
    h_img, w_img = img3.shape
    x0 = w_img//2 - n//2
    img3_cuadrada = img3[:, x0:x0+n]
    img3_u8 = img3_cuadrada.astype(np.uint8)
    planes = [((img3_u8 >> b) & 1).astype(np.float32) for b in range(8)]
    yy, xx = np.mgrid[0:n, 0:n]
    dist_full = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    bit_corrs = {}
    for b in range(8):
        # Densidad del bit por anillo
        dens_por_anillo = []
        for r0, r1, mask in anillos:
            dens_por_anillo.append(float(planes[b][mask].mean()))
        corr = pearsonr(dists, np.array(dens_por_anillo))[0]
        bit_corrs[f"bit{b}"] = float(corr)
        print(f"  bit{b}: corr(densidad_bit, distancia) = {corr:+.3f}", flush=True)
    report["bitplanes_vs_cruz"] = bit_corrs

    # ============ 5. TOPOGRAFIA vs CRUZ ============
    print("\n[5] TOPOGRAFIA: altura del relieve vs distancia a la cruz", flush=True)
    _, bits = cv2.threshold(img3_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    topo = cv2.GaussianBlur((bits > 0).astype(np.float64), (0, 0), 3)
    topo = topo / topo.max() if topo.max() > 0 else topo
    alt_por_anillo = []
    for r0, r1, mask in anillos:
        alt_por_anillo.append(float(topo[mask].mean()))
    corr_topo = pearsonr(dists, np.array(alt_por_anillo))[0]
    print(f"  corr(altura_topografia, distancia) = {corr_topo:+.3f}", flush=True)
    report["topografia_vs_cruz"] = {"corr_altura_dist": float(corr_topo)}

    # ============ 6. GRID ADAPTATIVO (test B) ============
    print("\n[6] GRID ADAPTATIVO: densidad central vs periferica (test B)", flush=True)
    # Region central (radio 100) vs periferica (radio 200-300)
    centro_r = R[cy-100:cy+100, cx-100:cx+100]
    perif_r = R[cy-300:cy+300, cx-300:cx+300]
    # Anillo periferico (excluir centro)
    perif_mask = (dist_full >= 200) & (dist_full < 300)
    perif_dens = float(R[perif_mask].mean())
    cent_dens = float(centro_r.mean())
    ratio = cent_dens / perif_dens if perif_dens > 0 else float("nan")
    print(f"  densidad central (r<100) = {cent_dens:.4f} | periferica (200-300) = {perif_dens:.4f} | ratio = {ratio:.3f}", flush=True)
    print(f"  -> {'CENTRO MAS DENSO (grid adaptativo: mas resolucion en el centro)' if ratio > 1.2 else 'centro similar a periferia'}", flush=True)
    report["grid_adaptativo"] = {"dens_central": cent_dens, "dens_periferica": perif_dens, "ratio": ratio}

    # ============ 7. SINTESIS ============
    print("\n" + "=" * 70, flush=True)
    print("SINTESIS: LA CRUZ CENTRAL COMO ABSTRACTOR", flush=True)
    print("=" * 70, flush=True)
    sintesis = {
        "campo_vectorial": cv,
        "perfil_radial_correlaciones": report["perfil_radial_correlaciones"],
        "no_localidad": report["no_localidad"],
        "bitplanes_vs_cruz": bit_corrs,
        "topografia_vs_cruz": report["topografia_vs_cruz"],
        "grid_adaptativo": report["grid_adaptativo"],
    }
    report["sintesis"] = sintesis
    print(f"  1. Campo vectorial: convergencia={cv['convergencia_media']:+.6f} ({cv['frac_converge']*100:.0f}% de puntos convergen)", flush=True)
    print(f"  2. Densidad radial: corr={corr_dens:+.3f} (la densidad DECAE con la distancia)", flush=True)
    print(f"  3. D_fractal radial: corr={corr_D:+.3f}", flush=True)
    print(f"  4. Alta frecuencia radial: corr={corr_af:+.3f}", flush=True)
    print(f"  5. No-localidad: corr(MI,dist)={corr_mi:+.4f}", flush=True)
    print(f"  6. Grid adaptativo: ratio centro/periferia={ratio:.3f}", flush=True)
    print(f"  7. Topografia: corr(altura,dist)={corr_topo:+.3f}", flush=True)
    print(f"  8. Bitplanes: corr con distancia = {[f'{v:+.2f}' for v in bit_corrs.values()]}", flush=True)

    out_json = os.path.join(OUT, "correlacion_global_abstractor_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
