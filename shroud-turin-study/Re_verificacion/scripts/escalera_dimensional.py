"""
ESCALERA DIMENSIONAL: EL PUNTO CENTRAL COMO ESCALA A DIMENSIONES SUPERIORES
===========================================================================
Hipotesis del usuario: el punto central (416,416) es multidimensional,
una ESCALA que va pasando a dimensiones superiores.

Tests:
  S1. PERFIL DIMENSIONAL RADIAL FINO
      - D local (box-counting 2D) y da (multifractal) en anillos de 5px
        desde el punto central hasta 200px
      - Si hay una "escalera dimensional", D(r) debe mostrar MESETAS
        (niveles discretos) en vez de variacion continua

  S2. DETECCION DE PLATAFORMAS DIMENSIONALES
      - Histograma de D local por anillo: picos = niveles discretos
      - Analisis de mesetas: segmentos donde D es casi constante
      - Comparar con controles (permutaciones) y gaussianos

  S3. AUTO-SIMILITUD JERARQUICA (cada escala del centro = estructura completa)
      - Region central a escala s (radio s) vs matriz completa a escala s
      - Correlacion: si el centro es una "escala" del todo, cada nivel
        del centro debe correlacionar con el todo a esa escala

  S4. ESPECTRO MULTIFRACTAL LOCAL f(alpha) EN EL CENTRO
      - f(alpha) del punto central exacto (radio 10) vs anillos
      - La forma del espectro revela la estructura dimensional local

  S5. DIMENSION DE CORRELACION LOCAL (Grassberger-Procaccia 2D)
      - D2 local en ventanas centradas en el punto, de tamano creciente
      - Si D2 crece con el tamano de ventana -> el punto "abre" a
        dimensiones superiores al expandirse

  S6. CONTROLES: mismo analisis sobre permutaciones y gaussianos

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import pdist
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N_CONTROLES = 20
N_WORKERS = 12

# ============================================================================
# METODOS
# ============================================================================
def box_counting_simple(matrix):
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
    if np.any(counts <= 0):
        return float("nan")
    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])

def simple_multifractal_width(matrix):
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

def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < 10.0).astype(np.float32)
    return R_bin, profile

# ============================================================================
# S1: PERFIL DIMENSIONAL RADIAL FINO
# ============================================================================
def perfil_dimensional_radial(R, cx, cy, ancho=5, n_anillos=40):
    """D y da en anillos finos desde el punto central."""
    h, w = R.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() < 200:
            continue
        region = np.zeros_like(R)
        region[mask] = R[mask]
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        crop = region[y0:y1, x0:x1]
        D = box_counting_simple(crop)
        da = simple_multifractal_width(crop)
        res.append({"r0": r0, "r1": r1, "r_medio": (r0+r1)/2,
                    "D": D, "da": da, "densidad": float(R[mask].mean())})
    return res

# ============================================================================
# S2: DETECCION DE PLATAFORMAS
# ============================================================================
def detectar_mesetas(D_vals, r_vals, tol=0.05):
    """Detecta segmentos donde D es casi constante (mesetas)."""
    mesetas = []
    inicio = 0
    for i in range(1, len(D_vals)):
        if abs(D_vals[i] - D_vals[inicio]) > tol:
            if i - inicio >= 3:  # meseta de al menos 3 anillos
                mesetas.append({
                    "r_inicio": r_vals[inicio], "r_fin": r_vals[i-1],
                    "D_media": float(np.mean(D_vals[inicio:i])),
                    "n_anillos": i - inicio,
                })
            inicio = i
    if len(D_vals) - inicio >= 3:
        mesetas.append({
            "r_inicio": r_vals[inicio], "r_fin": r_vals[-1],
            "D_media": float(np.mean(D_vals[inicio:])),
            "n_anillos": len(D_vals) - inicio,
        })
    return mesetas

# ============================================================================
# S3: AUTO-SIMILITUD JERARQUICA
# ============================================================================
def auto_similitud_jerarquica(R, cx, cy, escalas=(10, 20, 40, 80, 160, 320)):
    """Correlacion entre la region central a escala s y la matriz a escala s."""
    n = R.shape[0]
    res = []
    for s in escalas:
        # Region central (cuadrada, centrada en la cruz)
        x0, x1 = max(0, cx-s), min(n, cx+s)
        y0, y1 = max(0, cy-s), min(n, cy+s)
        central = R[y0:y1, x0:x1]
        # Matriz completa reducida a la misma escala (downsample)
        completo = cv2.resize(R, (central.shape[1], central.shape[0]))
        # Correlacion
        if central.std() > 0 and completo.std() > 0:
            corr = float(np.corrcoef(central.flatten(), completo.flatten())[0, 1])
        else:
            corr = float("nan")
        res.append({"escala": s, "corr": corr,
                    "densidad_central": float(central.mean()),
                    "densidad_completa": float(completo.mean())})
    return res

# ============================================================================
# S4: ESPECTRO f(alpha) LOCAL
# ============================================================================
def espectro_f_alpha(matrix, q_values=None):
    """Espectro multifractal f(alpha) completo."""
    if q_values is None:
        q_values = np.linspace(-5, 5, 21)
    measure = matrix.flatten()
    measure = measure[measure > 0]
    if len(measure) == 0:
        return None
    measure = measure / measure.sum()
    tau_q = []
    for q in q_values:
        if q == 0:
            tau_q.append(0.0)
        else:
            tau_q.append(np.log(np.sum(measure ** q)) / np.log(10))
    tau_q = np.array(tau_q)
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q
    return {"q": q_values.tolist(), "alpha": alpha.tolist(), "f_alpha": f_alpha.tolist(),
            "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
            "delta_alpha": float(alpha.max() - alpha.min()),
            "alpha_pico": float(alpha[np.argmax(f_alpha)])}

# ============================================================================
# S5: D2 LOCAL CON VENTANA CRECIENTE
# ============================================================================
def D2_local_ventana(R, cx, cy, radios=(10, 20, 40, 80, 160)):
    """D2 (Grassberger-Procaccia) en ventanas centradas en el punto."""
    n = R.shape[0]
    res = []
    for r in radios:
        x0, x1 = max(0, cx-r), min(n, cx+r)
        y0, y1 = max(0, cy-r), min(n, cy+r)
        region = R[y0:y1, x0:x1]
        data = region.flatten().astype(np.float64)
        if len(data) > 1500:
            idx = rng.choice(len(data), 1500, replace=False)
            data = data[idx]
        if len(data) < 200:
            res.append({"radio": r, "D2": float("nan")})
            continue
        dists = pdist(data.reshape(-1, 1))
        if len(dists) == 0:
            res.append({"radio": r, "D2": float("nan")})
            continue
        dmin = dists[dists > 0].min() if np.any(dists > 0) else dists.max()
        dmax = dists.max()
        if dmin <= 0 or dmax <= dmin:
            res.append({"radio": r, "D2": float("nan")})
            continue
        radii = np.logspace(np.log10(dmin), np.log10(dmax), 25)
        counts = np.array([np.sum(dists < rr) for rr in radii], dtype=np.float64)
        valid = counts > 1
        if valid.sum() < 5:
            res.append({"radio": r, "D2": float("nan")})
            continue
        log_r = np.log(radii[valid])
        log_c = np.log(counts[valid])
        slopes = np.gradient(log_c, log_r)
        mid = slice(len(slopes)//4, 3*len(slopes)//4)
        res.append({"radio": r, "D2": float(np.mean(slopes[mid]))})
    return res

# ============================================================================
# WORKERS
# ============================================================================
def worker_perfil(args):
    R, cx, cy = args
    return perfil_dimensional_radial(R, cx, cy)

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"S1_perfil_radial": {}, "S2_mesetas": {}, "S3_auto_similitud": {},
              "S4_f_alpha": {}, "S5_D2_ventana": {}, "S6_controles": {},
              "conclusion": {}}

    R_real, profile = matriz_real()
    n = R_real.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"MATRIZ REAL: {n}x{n} | punto central=({cx},{cy})", flush=True)
    print("=" * 70, flush=True)

    # ============ S1: PERFIL DIMENSIONAL RADIAL FINO ============
    print("\n[S1] PERFIL DIMENSIONAL RADIAL FINO (anillos de 5px)", flush=True)
    perfil = perfil_dimensional_radial(R_real, cx, cy, ancho=5, n_anillos=40)
    print(f"  {'r':>5} {'D':>8} {'da':>8} {'dens':>7}", flush=True)
    for p in perfil:
        print(f"  {p['r_medio']:5.0f} {p['D']:8.4f} {p['da']:8.4f} {p['densidad']:7.4f}", flush=True)
    report["S1_perfil_radial"] = perfil

    # ============ S2: MESETAS DIMENSIONALES ============
    print("\n[S2] DETECCION DE PLATAFORMAS DIMENSIONALES (mesetas en D)", flush=True)
    r_vals = np.array([p["r_medio"] for p in perfil if not np.isnan(p["D"])])
    D_vals = np.array([p["D"] for p in perfil if not np.isnan(p["D"])])
    mesetas = detectar_mesetas(D_vals, r_vals, tol=0.05)
    print(f"  Mesetas detectadas ({len(mesetas)}):", flush=True)
    for m in mesetas:
        print(f"    r={m['r_inicio']:.0f}-{m['r_fin']:.0f}: D_media={m['D_media']:.4f} ({m['n_anillos']} anillos)", flush=True)
    report["S2_mesetas"] = mesetas
    # Niveles discretos: D medias de las mesetas
    niveles = [m["D_media"] for m in mesetas]
    print(f"  Niveles de D: {[f'{d:.3f}' for d in niveles]}", flush=True)
    if len(niveles) >= 2:
        diffs = np.diff(niveles)
        print(f"  Saltos entre niveles: {[f'{d:+.3f}' for d in diffs]}", flush=True)
        report["S2_saltos"] = diffs.tolist()

    # ============ S3: AUTO-SIMILITUD JERARQUICA ============
    print("\n[S3] AUTO-SIMILITUD JERARQUICA (centro a escala s vs todo a escala s)", flush=True)
    auto = auto_similitud_jerarquica(R_real, cx, cy)
    for a in auto:
        print(f"  escala={a['escala']:4d}: corr={a['corr']:+.4f} | dens_centro={a['densidad_central']:.4f} | dens_total={a['densidad_completa']:.4f}", flush=True)
    report["S3_auto_similitud"] = auto

    # ============ S4: ESPECTRO f(alpha) LOCAL ============
    print("\n[S4] ESPECTRO f(alpha) LOCAL (punto central vs anillos)", flush=True)
    # Centro exacto (radio 10)
    centro = R_real[cy-10:cy+10, cx-10:cx+10]
    fa_centro = espectro_f_alpha(centro)
    print(f"  CENTRO (r<10): delta_alpha={fa_centro['delta_alpha']:.4f} | alpha_pico={fa_centro['alpha_pico']:.4f}", flush=True)
    # Anillo 20-40
    h, w = R_real.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    mask_anillo = (dist >= 20) & (dist < 40)
    anillo = np.zeros_like(R_real)
    anillo[mask_anillo] = R_real[mask_anillo]
    ys, xs = np.where(mask_anillo)
    anillo_crop = anillo[ys.min():ys.max()+1, xs.min():xs.max()+1]
    fa_anillo = espectro_f_alpha(anillo_crop)
    print(f"  ANILLO (20-40): delta_alpha={fa_anillo['delta_alpha']:.4f} | alpha_pico={fa_anillo['alpha_pico']:.4f}", flush=True)
    # Periferia (100-150)
    mask_perif = (dist >= 100) & (dist < 150)
    perif = np.zeros_like(R_real)
    perif[mask_perif] = R_real[mask_perif]
    ys, xs = np.where(mask_perif)
    perif_crop = perif[ys.min():ys.max()+1, xs.min():xs.max()+1]
    fa_perif = espectro_f_alpha(perif_crop)
    print(f"  PERIFERIA (100-150): delta_alpha={fa_perif['delta_alpha']:.4f} | alpha_pico={fa_perif['alpha_pico']:.4f}", flush=True)
    report["S4_f_alpha"] = {"centro": fa_centro, "anillo": fa_anillo, "periferia": fa_perif}

    # ============ S5: D2 LOCAL CON VENTANA CRECIENTE ============
    print("\n[S5] D2 LOCAL: ventana creciente centrada en el punto", flush=True)
    d2v = D2_local_ventana(R_real, cx, cy)
    for d in d2v:
        print(f"  radio={d['radio']:4d}: D2={d['D2']:.4f}", flush=True)
    report["S5_D2_ventana"] = d2v
    # ¿D2 crece con el radio?
    radios_v = np.array([d["radio"] for d in d2v if not np.isnan(d["D2"])])
    D2s_v = np.array([d["D2"] for d in d2v if not np.isnan(d["D2"])])
    if len(radios_v) > 3:
        corr_D2 = float(np.corrcoef(radios_v, D2s_v)[0, 1])
        print(f"  corr(D2, radio) = {corr_D2:+.3f} -> {'D2 CRECE con ventana (el punto abre a dimensiones superiores)' if corr_D2 > 0.5 else 'D2 no crece'}", flush=True)
        report["S5_corr"] = corr_D2

    # ============ S6: CONTROLES ============
    print(f"\n[S6] CONTROLES ({N_CONTROLES} permutaciones): perfil radial", flush=True)
    ctrl_mesetas = []
    ctrl_corr_D2 = []
    for i in range(N_CONTROLES):
        p = rng.permutation(profile)
        Rc = (np.abs(p[:, None] - p[None, :]) < 10.0).astype(np.float32)
        perfil_c = perfil_dimensional_radial(Rc, cx, cy, ancho=5, n_anillos=40)
        r_c = np.array([pp["r_medio"] for pp in perfil_c if not np.isnan(pp["D"])])
        D_c = np.array([pp["D"] for pp in perfil_c if not np.isnan(pp["D"])])
        if len(r_c) > 5:
            mesetas_c = detectar_mesetas(D_c, r_c, tol=0.05)
            ctrl_mesetas.append(len(mesetas_c))
        # D2 ventana
        d2v_c = D2_local_ventana(Rc, cx, cy)
        rv_c = np.array([d["radio"] for d in d2v_c if not np.isnan(d["D2"])])
        d2_c = np.array([d["D2"] for d in d2v_c if not np.isnan(d["D2"])])
        if len(rv_c) > 3:
            ctrl_corr_D2.append(float(np.corrcoef(rv_c, d2_c)[0, 1]))
    print(f"  Mesetas en controles: {np.mean(ctrl_mesetas):.1f}±{np.std(ctrl_mesetas):.1f} (real: {len(mesetas)})", flush=True)
    print(f"  corr(D2,radio) controles: {np.mean(ctrl_corr_D2):+.3f}±{np.std(ctrl_corr_D2):.3f} (real: {corr_D2:+.3f})" if len(radios_v) > 3 else "", flush=True)
    report["S6_controles"] = {
        "mesetas_mean": float(np.mean(ctrl_mesetas)), "mesetas_std": float(np.std(ctrl_mesetas)),
        "mesetas_real": len(mesetas),
        "corr_D2_mean": float(np.mean(ctrl_corr_D2)) if ctrl_corr_D2 else float("nan"),
        "corr_D2_std": float(np.std(ctrl_corr_D2)) if ctrl_corr_D2 else float("nan"),
        "corr_D2_real": float(corr_D2) if len(radios_v) > 3 else float("nan"),
    }

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿ESCALERA DIMENSIONAL EN EL PUNTO CENTRAL?", flush=True)
    print("=" * 70, flush=True)
    z_mesetas = (len(mesetas) - np.mean(ctrl_mesetas)) / np.std(ctrl_mesetas) if np.std(ctrl_mesetas) > 0 else float("nan")
    print(f"  Mesetas: real={len(mesetas)} vs controles={np.mean(ctrl_mesetas):.1f}±{np.std(ctrl_mesetas):.1f} (z={z_mesetas:+.1f})", flush=True)
    if len(radios_v) > 3 and ctrl_corr_D2:
        z_d2 = (corr_D2 - np.mean(ctrl_corr_D2)) / np.std(ctrl_corr_D2) if np.std(ctrl_corr_D2) > 0 else float("nan")
        print(f"  corr(D2,radio): real={corr_D2:+.3f} vs controles={np.mean(ctrl_corr_D2):+.3f}±{np.std(ctrl_corr_D2):.3f} (z={z_d2:+.1f})", flush=True)
        report["conclusion"]["z_corr_D2"] = float(z_d2)
    report["conclusion"]["z_mesetas"] = float(z_mesetas)
    report["conclusion"]["niveles_D"] = niveles
    report["conclusion"]["auto_similitud"] = auto

    out_json = os.path.join(OUT, "escalera_dimensional_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
