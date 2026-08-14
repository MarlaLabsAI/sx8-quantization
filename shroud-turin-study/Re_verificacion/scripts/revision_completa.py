"""
REVISION COMPLETA: REPETICION DE TESTS CON METODOS CORREGIDOS
=============================================================
Errores identificados en la sesion:

  E1. box_counting_simple (sizes 2,4,8) SATURA: densidades 0.8 y 1.0
      dan el mismo D=1.737. Todos los objetos del T7 dieron D=1.737
      por esto. -> CORRECCION: usar sizes [1,2,4,8,16,32] con mas puntos
      de regresion y regiones mas grandes.

  E2. Espectro f(alpha) con q<0 daba alpha_pico NEGATIVO (-2.5 a -4.0)
      = artefacto matematico (log de medidas con ceros para q negativo).
      -> CORRECCION: solo q>=0 para espectro f(alpha), o regularizar.

  E3. S5 (D2 Grassberger-Procaccia) fallo con NaN en todas las ventanas
      porque la matriz es BINARIA. -> CORRECCION: usar matriz CONTINUA.

  E4. T3/T4 comparaban perfiles radiales de objetos 100x100 con la
      matriz real 1080x1080 - escalas incompatibles.
      -> CORRECCION: redimensionar proyecciones a 1080 antes de comparar.

  E5. D12 comparaba la cruz real (periferia LLENA, D=1.63) con la
      simulacion (periferia VACIA, D=0) - comparacion injusta.
      -> CORRECCION: simular objetos que ocupan todo el campo.

  E6. La cruz (416,416) esta FUERA de la diagonal (540,540) - punto de
      proyeccion excentrico. No se enfatizo esto correctamente.

  E7. No se verifico la cruz en Jeshua2 (alta resolucion) con D12.

  E8. La auto-similitud (S3) comparaba centro vs TODO - la correlacion
      correcta para 'escala' es centro a escala s vs centro a escala 2s.

Tests corregidos:
  R1. D fractal con box-counting CORREGIDO (no saturado): centro real
      vs objetos proyectados (esfera, elipsoide, cruz3d, hiperesfera4d)
  R2. Espectro f(alpha) con q>=0: centro vs anillos vs periferia
  R3. D2 (Grassberger-Procaccia) sobre matriz CONTINUA en ventanas
      centradas en la cruz, con controles
  R4. Perfiles radiales a escala COMPARABLE (proyecciones redimensionadas)
  R5. Simulacion con objeto que ocupa todo el campo (periferia no vacia)
  R6. Verificacion de la cruz en Jeshua2 con metodo D12
  R7. Auto-similitud local: centro a escala s vs centro a escala 2s
  R8. El punto excentrico: la cruz fuera de la diagonal como firma
      de punto de proyeccion (verificar con barrido de TODA la matriz:
      cuantos puntos fuera de la diagonal tienen densidad tan alta)

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
# R1: BOX-COUNTING CORREGIDO (no saturado)
# ============================================================================
def box_counting_corregido(matrix, min_size=1, max_size=None):
    """Box-counting con multiples escalas y regresion robusta."""
    h, w = matrix.shape
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
        blocks = matrix[:nh*s, :nw*s].reshape(nh, s, nw, s)
        counts.append((blocks.sum(axis=(1,3)) > 0).sum())
    counts = np.array(counts, dtype=np.float64)
    valid = counts > 0
    if valid.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0/sizes[valid]), np.log(counts[valid]), 1)
    return float(slope)

# ============================================================================
# R2: ESPECTRO f(alpha) con q>=0 (sin artefactos)
# ============================================================================
def espectro_f_alpha_q0(matrix, q_values=None):
    """Espectro multifractal f(alpha) con q>=0 (evita log de ceros)."""
    if q_values is None:
        q_values = np.linspace(0.1, 5, 20)
    measure = matrix.flatten()
    measure = measure[measure > 0]
    if len(measure) == 0:
        return None
    measure = measure / measure.sum()
    tau_q = []
    for q in q_values:
        if abs(q - 1) < 1e-6:
            tau_q.append(-np.sum(measure * np.log(measure)))
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
# R3: D2 Grassberger-Procaccia sobre matriz CONTINUA
# ============================================================================
def D2_continua(region_cont, subsample=1200):
    """D2 sobre valores continuos (no binarios)."""
    data = region_cont.flatten().astype(np.float64)
    if len(data) > subsample:
        idx = rng.choice(len(data), subsample, replace=False)
        data = data[idx]
    if len(data) < 200:
        return float("nan")
    # Asegurar variacion
    if data.std() < 1e-6:
        return float("nan")
    dists = pdist(data.reshape(-1, 1))
    if len(dists) == 0:
        return float("nan")
    dmin = dists[dists > 0].min() if np.any(dists > 0) else dists.max()
    dmax = dists.max()
    if dmin <= 0 or dmax <= dmin:
        return float("nan")
    radii = np.logspace(np.log10(dmin), np.log10(dmax), 25)
    counts = np.array([np.sum(dists < r) for r in radii], dtype=np.float64)
    valid = counts > 1
    if valid.sum() < 5:
        return float("nan")
    log_r = np.log(radii[valid])
    log_c = np.log(counts[valid])
    slopes = np.gradient(log_c, log_r)
    mid = slice(len(slopes)//4, 3*len(slopes)//4)
    return float(np.mean(slopes[mid]))

# ============================================================================
# R4/R5: OBJETOS Y PROYECCIONES (con campo completo)
# ============================================================================
def objeto_3d(tipo, size=100):
    x, y, z = np.mgrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    if tipo == "esfera":
        obj = (r < 0.9).astype(np.float64)
        obj[r < 0.7] = 0.8
        obj[r < 0.5] = 0.6
        obj[r < 0.3] = 0.4
    elif tipo == "elipsoide":
        obj = ((x/0.9)**2 + (y/0.6)**2 + (z/0.4)**2 < 1).astype(np.float64)
    elif tipo == "cruz3d":
        obj = ((np.abs(x) < 0.3) | (np.abs(y) < 0.3) | (np.abs(z) < 0.3)).astype(np.float64)
        obj = obj * (r < 1.0).astype(np.float64)
    elif tipo == "hiperesfera4d":
        w_max = np.sqrt(np.maximum(0.9**2 - r**2, 0))
        obj = 2.0 * w_max
    elif tipo == "esfera_campo_completo":
        # Esfera que ocupa TODO el campo (periferia no vacia)
        obj = (r < 1.0).astype(np.float64)
        obj[r < 0.75] = 0.8
        obj[r < 0.5] = 0.6
        obj[r < 0.25] = 0.4
    else:
        obj = (r < 0.9).astype(np.float64)
    return obj

def proyectar(obj, modo="suma"):
    if modo == "suma":
        proj = obj.sum(axis=2)
    else:
        proj = obj.max(axis=2)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def perfil_radial_2d(proj, n_anillos=20, ancho=3):
    """Perfil radial de una proyeccion (densidad por anillo, normalizado)."""
    h, w = proj.shape
    cx, cy = w//2, h//2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 0:
            res.append(float(proj[mask].mean()))
    return np.array(res)

# ============================================================================
# MATRICES
# ============================================================================
def matrices_de_imagen(img):
    h, w = img.shape
    profile = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=15)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < 10.0).astype(np.float32)
    return R_cont, R_bin, profile

# ============================================================================
# R8: PUNTO EXCENTRICO (fuera de la diagonal)
# ============================================================================
def puntos_fuera_diagonal(R, umbral_densidad=0.5, ancho_diag=15):
    """Cuantos puntos FUERA de la banda diagonal tienen densidad alta."""
    n = R.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    # Banda diagonal: |i-j| < ancho_diag
    en_diagonal = np.abs(yy - xx) < ancho_diag
    fuera = ~en_diagonal
    # Densidad local (suavizada)
    dens_map = ndimage.gaussian_filter(R, sigma=5)
    puntos_alta_densidad_fuera = np.sum((dens_map > umbral_densidad) & fuera)
    puntos_alta_densidad_total = np.sum(dens_map > umbral_densidad)
    print(f"  Puntos con densidad>{umbral_densidad}: total={puntos_alta_densidad_total}")
    print(f"  De los cuales FUERA de la diagonal: {puntos_alta_densidad_fuera} "
          f"({puntos_alta_densidad_fuera/puntos_alta_densidad_total*100:.1f}%)")
    # La cruz esta entre ellos?
    dens_cruz = dens_map[416, 416]
    print(f"  Densidad en la cruz (416,416): {dens_cruz:.3f} (umbral {umbral_densidad})")
    return {"total": int(puntos_alta_densidad_total), "fuera_diagonal": int(puntos_alta_densidad_fuera),
            "pct_fuera": float(puntos_alta_densidad_fuera/puntos_alta_densidad_total*100),
            "densidad_cruz": float(dens_cruz)}

# ============================================================================
# R7: AUTO-SIMILITUD LOCAL (centro a escala s vs centro a escala 2s)
# ============================================================================
def auto_similitud_local(R, cx, cy, escalas=(10, 20, 40, 80, 160)):
    n = R.shape[0]
    res = []
    for s in escalas:
        x0, x1 = max(0, cx-s), min(n, cx+s)
        y0, y1 = max(0, cy-s), min(n, cy+s)
        r_s = R[y0:y1, x0:x1]
        s2 = s * 2
        x0b, x1b = max(0, cx-s2), min(n, cx+s2)
        y0b, y1b = max(0, cy-s2), min(n, cy+s2)
        r_2s = R[y0b:y1b, x0b:x1b]
        r_2s_red = cv2.resize(r_2s, (r_s.shape[1], r_s.shape[0]))
        if r_s.std() > 0 and r_2s_red.std() > 0:
            corr = float(np.corrcoef(r_s.flatten(), r_2s_red.flatten())[0, 1])
        else:
            corr = float("nan")
        res.append({"escala_s": s, "corr_s_vs_2s": corr})
    return res

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"R1_D_corregido": {}, "R2_f_alpha_q0": {}, "R3_D2_continua": {},
              "R4_perfiles_comparables": {}, "R5_campo_completo": {},
              "R6_cruz_jeshua2": {}, "R7_auto_similitud_local": {},
              "R8_punto_excentrico": {}, "conclusion": {}}

    # Cargar imagen3
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    R3_cont, R3_bin, profile3 = matrices_de_imagen(img3)
    n = R3_bin.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"MATRIZ REAL: {n}x{n} | cruz=({cx},{cy}) | fuera de diagonal: {cx != n//2}", flush=True)
    print("=" * 70, flush=True)

    # ============ R1: D corregido (no saturado) ============
    print("\n[R1] D FRACTAL CORREGIDO (box-counting multi-escala)", flush=True)
    centro_real = R3_bin[cy-20:cy+20, cx-20:cx+20]
    D_centro_real = box_counting_corregido(centro_real)
    print(f"  Cruz real (40x40): D={D_centro_real:.4f}", flush=True)
    report["R1_D_corregido"]["cruz_real"] = D_centro_real
    # Objetos proyectados
    for tipo in ["esfera", "elipsoide", "cruz3d", "hiperesfera4d"]:
        obj = objeto_3d(tipo)
        proj = proyectar(obj, "suma")
        hh, ww = proj.shape
        cyy, cxx = hh//2, ww//2
        centro_obj = proj[cyy-20:cyy+20, cxx-20:cxx+20]
        D_obj = box_counting_corregido(centro_obj)
        print(f"  {tipo} proyectado: D={D_obj:.4f}", flush=True)
        report["R1_D_corregido"][tipo] = D_obj

    # ============ R2: f(alpha) con q>=0 ============
    print("\n[R2] ESPECTRO f(alpha) CON q>=0 (sin artefactos)", flush=True)
    centro_R = R3_bin[cy-20:cy+20, cx-20:cx+20]
    perif_R = R3_bin[:100, :100]
    fa_c = espectro_f_alpha_q0(centro_R)
    fa_p = espectro_f_alpha_q0(perif_R)
    print(f"  Centro: delta_alpha={fa_c['delta_alpha']:.4f} | alpha_pico={fa_c['alpha_pico']:.4f}", flush=True)
    print(f"  Periferia: delta_alpha={fa_p['delta_alpha']:.4f} | alpha_pico={fa_p['alpha_pico']:.4f}", flush=True)
    report["R2_f_alpha_q0"] = {"centro": fa_c, "periferia": fa_p}

    # ============ R3: D2 continua en ventanas ============
    print("\n[R3] D2 GRASSBERGER-PROCACCIA (matriz CONTINUA)", flush=True)
    d2_ventanas = []
    for radio in [20, 40, 80, 160, 320]:
        x0, x1 = max(0, cx-radio), min(n, cx+radio)
        y0, y1 = max(0, cy-radio), min(n, cy+radio)
        region = R3_cont[y0:y1, x0:x1]
        d2 = D2_continua(region)
        d2_ventanas.append({"radio": radio, "D2": d2})
        print(f"  radio={radio:4d}: D2={d2:.4f}", flush=True)
    report["R3_D2_continua"]["real"] = d2_ventanas
    # Controles
    print(f"  Controles ({N_CONTROLES} permutaciones):", flush=True)
    ctrl_corrs = []
    for i in range(N_CONTROLES):
        p = rng.permutation(profile3)
        Rc = np.abs(p[:, None] - p[None, :])
        vals = []
        for radio in [20, 40, 80, 160, 320]:
            x0, x1 = max(0, cx-radio), min(n, cx+radio)
            y0, y1 = max(0, cy-radio), min(n, cy+radio)
            vals.append(D2_continua(Rc[y0:y1, x0:x1]))
        vals = np.array(vals)
        radios = np.array([20, 40, 80, 160, 320])
        valid = ~np.isnan(vals)
        if valid.sum() > 3:
            ctrl_corrs.append(float(np.corrcoef(radios[valid], vals[valid])[0, 1]))
    d2_real = np.array([d["D2"] for d in d2_ventanas if not np.isnan(d["D2"])])
    r_real = np.array([d["radio"] for d in d2_ventanas if not np.isnan(d["D2"])])
    corr_real = float(np.corrcoef(r_real, d2_real)[0, 1]) if len(r_real) > 3 else float("nan")
    print(f"  corr(D2, radio) real={corr_real:+.3f} vs controles={np.mean(ctrl_corrs):+.3f}±{np.std(ctrl_corrs):.3f}", flush=True)
    report["R3_D2_continua"]["corr_real"] = corr_real
    report["R3_D2_continua"]["controles"] = {"mean": float(np.mean(ctrl_corrs)), "std": float(np.std(ctrl_corrs))}

    # ============ R4: Perfiles a escala comparable ============
    print("\n[R4] PERFILES RADIALES A ESCALA COMPARABLE", flush=True)
    # Perfil radial real (densidad) desde la cruz
    yy, xx = np.mgrid[0:n, 0:n]
    dist_real = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    perfil_real = []
    for i in range(20):
        r0, r1 = i*5, (i+1)*5
        mask = (dist_real >= r0) & (dist_real < r1)
        if mask.sum() > 100:
            perfil_real.append(float(R3_bin[mask].mean()))
    perfil_real = np.array(perfil_real)
    print(f"  Perfil real (20 anillos de 5px): {perfil_real[:8].round(3)}...", flush=True)
    report["R4_perfiles_comparables"]["perfil_real"] = perfil_real.tolist()
    # Proyecciones redimensionadas a 1080
    for tipo in ["esfera", "elipsoide", "cruz3d", "hiperesfera4d"]:
        obj = objeto_3d(tipo)
        proj = proyectar(obj, "suma")
        proj_1080 = cv2.resize(proj, (n, n))
        # Perfil radial desde el centro de la proyeccion
        cyy, cxx = n//2, n//2
        dist_p = np.sqrt((xx-cxx)**2 + (yy-cyy)**2)
        perfil_p = []
        for i in range(20):
            r0, r1 = i*5, (i+1)*5
            mask = (dist_p >= r0) & (dist_p < r1)
            if mask.sum() > 100:
                perfil_p.append(float(proj_1080[mask].mean()))
        perfil_p = np.array(perfil_p)
        if len(perfil_p) == len(perfil_real):
            corr = float(np.corrcoef(perfil_p, perfil_real)[0, 1])
            print(f"  {tipo}: corr(perfil, real) = {corr:+.3f}", flush=True)
            report["R4_perfiles_comparables"][tipo] = corr

    # ============ R5: Objeto que ocupa todo el campo ============
    print("\n[R5] OBJETO CAMPO COMPLETO (periferia NO vacia)", flush=True)
    obj_full = objeto_3d("esfera_campo_completo")
    proj_full = proyectar(obj_full, "suma")
    hh, ww = proj_full.shape
    cyy, cxx = hh//2, ww//2
    centro_f = proj_full[cyy-20:cyy+20, cxx-20:cxx+20]
    perif_f = proj_full[:20, :20]
    D_c = box_counting_corregido(centro_f)
    D_p = box_counting_corregido(perif_f)
    print(f"  Esfera campo completo: D_centro={D_c:.4f} | D_periferia={D_p:.4f}", flush=True)
    report["R5_campo_completo"] = {"D_centro": D_c, "D_periferia": D_p}

    # ============ R6: Cruz en Jeshua2 ============
    print("\n[R6] VERIFICACION EN JESHUA2 (alta resolucion)", flush=True)
    j2 = cv2.imread(os.path.join(BASE, "Re_verificacion", "Jeshua2.jpg"), cv2.IMREAD_GRAYSCALE)
    j2_izq = j2[:, :j2.shape[1]//2]
    Rj_cont, Rj_bin, profile_j = matrices_de_imagen(j2_izq)
    nj = Rj_bin.shape[0]
    print(f"  Matriz Jeshua2: {nj}x{nj}", flush=True)
    # Detectar la cruz en Jeshua2 (metodo CHIP-8: argmax proyecciones cuadrante TL)
    qs = nj // 2
    q = Rj_bin[:qs, :qs]
    cxj = int(np.argmax(np.mean(q, axis=0)))
    cyj = int(np.argmax(np.mean(q, axis=1)))
    print(f"  Cruz Jeshua2: ({cxj},{cyj}) rel=({cxj/qs:.3f},{cyj/qs:.3f})", flush=True)
    # Firma D12 en Jeshua2
    centro_j = Rj_bin[cyj-10:cyj+10, cxj-10:cxj+10]
    perif_j = Rj_bin[:20, :20]
    D_cj = box_counting_corregido(centro_j)
    D_pj = box_counting_corregido(perif_j)
    print(f"  Jeshua2: D_centro={D_cj:.4f} | D_periferia={D_pj:.4f} | "
          f"dens_centro={centro_j.mean():.3f} | dens_perif={perif_j.mean():.3f}", flush=True)
    report["R6_cruz_jeshua2"] = {"cruz": (int(cxj), int(cyj)), "D_centro": D_cj, "D_periferia": D_pj,
                                  "dens_centro": float(centro_j.mean()), "dens_perif": float(perif_j.mean())}

    # ============ R7: Auto-similitud local ============
    print("\n[R7] AUTO-SIMILITUD LOCAL (centro a escala s vs 2s)", flush=True)
    auto = auto_similitud_local(R3_bin, cx, cy)
    for a in auto:
        print(f"  escala={a['escala_s']:4d}: corr(s vs 2s) = {a['corr_s_vs_2s']:+.4f}", flush=True)
    report["R7_auto_similitud_local"] = auto

    # ============ R8: Punto excentrico ============
    print("\n[R8] PUNTO EXCENTRICO (fuera de la diagonal)", flush=True)
    r8 = puntos_fuera_diagonal(R3_bin)
    report["R8_punto_excentrico"] = r8

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION DE LA REVISION", flush=True)
    print("=" * 70, flush=True)
    report["conclusion"] = {
        "R1_D_cruz_real": D_centro_real,
        "R3_corr_real": corr_real, "R3_controles": report["R3_D2_continua"]["controles"],
        "R4_mejor": max(report["R4_perfiles_comparables"], key=lambda k: report["R4_perfiles_comparables"][k] if isinstance(report["R4_perfiles_comparables"][k], float) else -1),
        "R6_jeshua2": report["R6_cruz_jeshua2"],
        "R8": r8,
    }
    print(f"  R1: D cruz real (corregido) = {D_centro_real:.4f}", flush=True)
    print(f"  R3: corr(D2,radio) real = {corr_real:+.3f}", flush=True)
    print(f"  R8: {r8['pct_fuera']:.1f}% de puntos de alta densidad estan FUERA de la diagonal", flush=True)

    out_json = os.path.join(OUT, "revision_completa_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
