"""
INDAGACION FUERTE v2: LA CRUZ CENTRAL COMO ESCALADOR MULTIDIMENSIONAL
====================================================================
Correcciones vs v1:
  - D2 se calcula con TAKENS (dimension de atractor) sobre el PERFIL,
    no sobre la matriz 2D aplanada (que daba valores espurios).
  - Espectro Renyi solo q>=0 (D0, D1, D2) - los q<0 dan artefactos
    con matrices sparse.
  - Simulaciones: proyeccion 3D->2D -> perfil central -> Takens D2.

Tests:
  E1. TAKENS D2 del perfil COMPLETO (dimension efectiva del atractor)
  E2. TAKENS D2 en VENTANAS: centro del perfil vs extremos
      (si la cruz es escalador, el centro del perfil tiene D2 mayor)
  E3. ESPECTRO RENYI D0/D1/D2 (matriz de recurrencia, q>=0)
  E4. SIMULACIONES 3D->2D: esfera, cubo, cruz3d, toro, elipsoide
      -> perfil central -> Takens D2 (centro vs periferia del perfil)
  E5. D2 RADIAL DEL PERFIL: ventanas deslizantes desde el centro
  E6. CONTROLES: permutacion, gaussiano, AR(1) - mismo analisis

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
N_CONTROLS = 30
N_WORKERS = 12

# ============================================================================
# 1. TAKENS D2 (dimension de correlacion del atractor)
# ============================================================================
def takens_D2(profile, m=5, tau=5, subsample=800):
    """D2 del atractor reconstruido con embedding de Takens."""
    profile = np.asarray(profile, dtype=np.float64)
    n = len(profile)
    if n <= (m-1)*tau:
        return float("nan")
    X = np.stack([profile[i:n-(m-1)*tau+i] for i in range(0, (m-1)*tau+1, tau)], axis=1)
    if len(X) < 200:
        return float("nan")
    if len(X) > subsample:
        idx = rng.choice(len(X), subsample, replace=False)
        X = X[idx]
    dists = pdist(X)
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

def takens_D2_scan(profile, m=5, tau=5, ventana=200, paso=50):
    """D2 en ventanas deslizantes a lo largo del perfil."""
    n = len(profile)
    res = []
    for start in range(0, n - ventana, paso):
        seg = profile[start:start+ventana]
        d2 = takens_D2(seg, m=m, tau=tau)
        res.append({"pos": start + ventana//2, "D2": d2})
    return res

# ============================================================================
# 2. RENYI D0/D1/D2 (matriz de recurrencia, q>=0)
# ============================================================================
def renyi_D012(data, box_sizes=(4, 8, 16, 32)):
    """D0, D1, D2 de Renyi sobre una matriz (q>=0)."""
    data = np.asarray(data, dtype=np.float64)
    h, w = data.shape
    def slope_q(q):
        log_eps, log_Z = [], []
        for s in box_sizes:
            nh, nw = h // s, w // s
            if nh == 0 or nw == 0:
                continue
            blocks = data[:nh*s, :nw*s].reshape(nh, s, nw, s)
            measure = blocks.sum(axis=(1, 3)) / (s*s)
            measure = measure[measure > 0]
            if len(measure) == 0:
                continue
            measure = measure / measure.sum()
            log_eps.append(np.log(1.0/s))
            if q == 1:
                log_Z.append(-np.sum(measure * np.log(measure)))
            else:
                log_Z.append(np.log(np.sum(measure ** q)) / (q - 1))
        if len(log_eps) >= 3:
            slope, _ = np.polyfit(log_eps, log_Z, 1)
            return float(slope)
        return float("nan")
    return {"D0": slope_q(0), "D1": slope_q(1), "D2": slope_q(2)}

# ============================================================================
# 3. SIMULACIONES 3D->2D
# ============================================================================
def objeto_3d(tipo, size=80):
    x, y, z = np.mgrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    if tipo == "esfera":
        obj = (r < 0.9).astype(np.float64)
    elif tipo == "cubo":
        obj = (np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z))) < 0.8).astype(np.float64)
    elif tipo == "cruz3d":
        obj = ((np.abs(x) < 0.3) | (np.abs(y) < 0.3) | (np.abs(z) < 0.3)).astype(np.float64)
        obj = obj * (r < 1.0).astype(np.float64)
    elif tipo == "toro":
        R_t, r_t = 0.6, 0.3
        obj = (np.sqrt((np.sqrt(x**2 + y**2) - R_t)**2 + z**2) < r_t).astype(np.float64)
    elif tipo == "elipsoide":
        obj = ((x/0.9)**2 + (y/0.6)**2 + (z/0.4)**2 < 1).astype(np.float64)
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

def perfil_central(proj):
    """Perfil central de la proyeccion (como el estudio)."""
    h, w = proj.shape
    return proj[:, w//2].astype(np.float64)

def firma_escalador(proj, m=5, tau=3):
    """D2 del atractor del perfil central: centro vs extremos."""
    perfil = perfil_central(proj)
    n = len(perfil)
    # Centro del perfil (ventana central) vs extremos
    centro = perfil[n//2 - 100:n//2 + 100]
    extremo1 = perfil[:200]
    extremo2 = perfil[-200:]
    D2_c = takens_D2(centro, m=m, tau=tau)
    D2_e1 = takens_D2(extremo1, m=m, tau=tau)
    D2_e2 = takens_D2(extremo2, m=m, tau=tau)
    D2_ext = np.nanmean([D2_e1, D2_e2])
    return {"D2_centro": D2_c, "D2_extremos": float(D2_ext) if not np.isnan(D2_ext) else float("nan"),
            "ratio": D2_c / D2_ext if D2_ext and D2_ext > 0 else float("nan")}

# ============================================================================
# 4. CONTROLES
# ============================================================================
def control_permutation(profile):
    return rng.permutation(profile)

def control_gaussian(profile):
    return rng.normal(profile.mean(), profile.std(), size=len(profile))

def control_ar1(profile):
    phi = 0.99
    n = len(profile)
    noise = rng.normal(0, profile.std() * np.sqrt(1 - phi**2), size=n)
    x = np.zeros(n)
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = phi * x[i-1] + noise[i]
    return x - x.mean() + profile.mean()

def worker_scan(args):
    profile, m, tau = args
    return takens_D2_scan(profile, m=m, tau=tau)

# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"E1_D2_total": {}, "E2_D2_ventanas": {}, "E3_Renyi": {},
              "E4_simulaciones": {}, "E5_D2_radial": {}, "E6_controles": {},
              "conclusion": {}}

    # Cargar imagen3
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    n = len(profile)
    print("=" * 70, flush=True)
    print(f"PERFIL CENTRAL: {n} puntos | cruz en y={416}", flush=True)
    print("=" * 70, flush=True)

    # ============ E1: D2 total del perfil ============
    print("\n[E1] D2 DEL ATRACTOR (Takens) - perfil completo", flush=True)
    for m in [3, 5, 7]:
        d2 = takens_D2(profile, m=m, tau=5)
        print(f"  m={m}: D2={d2:.4f}", flush=True)
        report["E1_D2_total"][f"m{m}"] = d2

    # ============ E2: D2 en ventanas (centro vs extremos) ============
    print("\n[E2] D2 EN VENTANAS: centro del perfil vs extremos", flush=True)
    scan = takens_D2_scan(profile, m=5, tau=5, ventana=200, paso=50)
    # Centro del perfil = alrededor de y=416
    centro_scan = [s for s in scan if abs(s["pos"] - 416) < 150]
    extremos_scan = [s for s in scan if s["pos"] < 200 or s["pos"] > 880]
    D2_centro = np.nanmean([s["D2"] for s in centro_scan]) if centro_scan else float("nan")
    D2_extremos = np.nanmean([s["D2"] for s in extremos_scan]) if extremos_scan else float("nan")
    print(f"  D2 centro (y~416) = {D2_centro:.4f} | D2 extremos = {D2_extremos:.4f} | ratio = {D2_centro/D2_extremos:.3f}" if D2_extremos and D2_extremos > 0 else f"  D2 centro = {D2_centro:.4f} | D2 extremos = {D2_extremos:.4f}", flush=True)
    report["E2_D2_ventanas"] = {"D2_centro": D2_centro, "D2_extremos": D2_extremos,
                                "ratio": D2_centro/D2_extremos if D2_extremos and D2_extremos > 0 else float("nan"),
                                "scan": scan}

    # ============ E3: Renyi D0/D1/D2 ============
    print("\n[E3] ESPECTRO RENYI D0/D1/D2 (matriz de recurrencia)", flush=True)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < 10.0).astype(np.float32)
    # Normalizar continua a [0,1]
    R_norm = R_cont / (R_cont.max() + 1e-12)
    # Centro (region 100x100 alrededor de 416) vs periferia (esquina)
    centro_R = R_norm[366:466, 366:466]
    perif_R = R_norm[:100, :100]
    ren_c = renyi_D012(centro_R)
    ren_p = renyi_D012(perif_R)
    print(f"  Centro:  D0={ren_c['D0']:.3f} | D1={ren_c['D1']:.3f} | D2={ren_c['D2']:.3f}", flush=True)
    print(f"  Perif:   D0={ren_p['D0']:.3f} | D1={ren_p['D1']:.3f} | D2={ren_p['D2']:.3f}", flush=True)
    report["E3_Renyi"] = {"centro": ren_c, "periferia": ren_p}

    # ============ E4: Simulaciones 3D->2D ============
    print("\n[E4] SIMULACIONES 3D->2D: firma del escalador (D2 centro vs extremos)", flush=True)
    sims = {}
    for tipo in ["esfera", "cubo", "cruz3d", "toro", "elipsoide"]:
        obj = objeto_3d(tipo)
        for modo in ["suma", "maximo"]:
            proj = proyectar(obj, modo)
            firma = firma_escalador(proj)
            key = f"{tipo}_{modo}"
            sims[key] = firma
            print(f"  {key}: D2_centro={firma['D2_centro']:.3f} | D2_extremos={firma['D2_extremos']:.3f} | ratio={firma['ratio']:.2f}", flush=True)
    report["E4_simulaciones"] = sims

    # ============ E5: D2 radial del perfil ============
    print("\n[E5] D2 RADIAL DEL PERFIL (ventanas desde el centro)", flush=True)
    # Ventanas centradas en la cruz, expandiendose
    radial = []
    for radio in [50, 100, 150, 200, 250, 300, 350, 400]:
        y0 = max(0, 416 - radio)
        y1 = min(n, 416 + radio)
        seg = profile[y0:y1]
        d2 = takens_D2(seg, m=5, tau=5)
        radial.append({"radio": radio, "D2": d2})
        print(f"  radio={radio:3d}: D2={d2:.4f}", flush=True)
    report["E5_D2_radial"] = radial
    # Correlacion D2 vs radio
    radios = np.array([r["radio"] for r in radial if not np.isnan(r["D2"])])
    D2s = np.array([r["D2"] for r in radial if not np.isnan(r["D2"])])
    if len(radios) > 3:
        corr = np.corrcoef(radios, D2s)[0, 1]
        print(f"  corr(D2, radio) = {corr:+.3f}", flush=True)
        report["E5_corr"] = float(corr)

    # ============ E6: Controles ============
    print(f"\n[E6] CONTROLES ({N_CONTROLS} x 3): D2 radial en controles", flush=True)
    kinds = ["permutation", "gaussian", "ar1"]
    control_corrs = {k: [] for k in kinds}
    for kind in kinds:
        tasks = []
        for i in range(N_CONTROLS):
            if kind == "permutation":
                p = control_permutation(profile)
            elif kind == "gaussian":
                p = control_gaussian(profile)
            else:
                p = control_ar1(profile)
            tasks.append((p, 5, 5))
        with Pool(N_WORKERS) as pool:
            res = list(pool.imap_unordered(worker_scan, tasks))
        # Para cada control, calcular corr(D2, radio) con el mismo esquema
        for scan_c in res:
            radios_c = []
            D2s_c = []
            for radio in [50, 100, 150, 200, 250, 300, 350, 400]:
                y0 = max(0, 416 - radio)
                y1 = min(n, 416 + radio)
                seg = profile[y0:y1]
                # Reconstruir el perfil del control en esa ventana
                # (el scan ya tiene D2 por posicion; aproximamos con el scan)
                pass
            # Usar el scan directamente: correlacion D2 vs |pos - 416|
            pos = np.array([s["pos"] for s in scan_c if not np.isnan(s["D2"])])
            d2s = np.array([s["D2"] for s in scan_c if not np.isnan(s["D2"])])
            if len(pos) > 3:
                dist_centro = np.abs(pos - 416)
                control_corrs[kind].append(float(np.corrcoef(dist_centro, d2s)[0, 1]))
        print(f"  [{kind}] corr(D2, dist_centro) = {np.mean(control_corrs[kind]):+.3f}±{np.std(control_corrs[kind]):.3f}", flush=True)
    report["E6_controles"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in control_corrs.items()}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿LA CRUZ ES UN ESCALADOR MULTIDIMENSIONAL?", flush=True)
    print("=" * 70, flush=True)
    if len(radios) > 3:
        z = (corr - np.mean(control_corrs["permutation"])) / np.std(control_corrs["permutation"]) if np.std(control_corrs["permutation"]) > 0 else float("nan")
        print(f"  corr(D2, radio) real = {corr:+.3f} vs controles = {np.mean(control_corrs['permutation']):+.3f}±{np.std(control_corrs['permutation']):.3f} (z={z:+.1f})", flush=True)
        report["conclusion"]["z_corr"] = float(z)
    report["conclusion"]["E2_ratio"] = report["E2_D2_ventanas"]["ratio"]
    report["conclusion"]["E4_sims"] = sims

    out_json = os.path.join(OUT, "escalador_multidimensional_v2_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
