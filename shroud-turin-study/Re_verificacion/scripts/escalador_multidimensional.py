"""
INDAGACION FUERTE: LA CRUZ CENTRAL COMO ESCALADOR MULTIDIMENSIONAL
==================================================================
Hipotesis: la cruz central actua como un escalador que lleva la
estructura a dimensiones superiores (proyeccion de objeto 3D+ en 2D).

Tests NUEVOS (no modifican los originales D1-D13):

  E1. DIMENSION DE CORRELACION RIGUROSA (Grassberger-Procaccia)
      - D2 en anillos concentricos desde la cruz (centro -> periferia)
      - Matriz continua (sin umbralizar) y binaria
      - Con controles (permutaciones) para significancia

  E2. ESPECTRO DE DIMENSIONES DE RENYI (Dq completo)
      - Dq para q = -10..10 en centro vs periferia
      - El espectro Dq revela multifractalidad dimensional real

  E3. EMBEDDING DE TAKENS (reconstruccion de espacio de fases)
      - Embedding del perfil en dimensiones m = 1..10
      - D2 del atractor reconstruido: si el centro "escala" a
        dimensiones superiores, D2 debe crecer con m

  E4. SIMULACION DE PROYECCION 3D->2D MEJORADA
      - Objetos: esfera, cubo, cruz 3D, toro, elipsoide
      - Proyeccion por suma (radiografia) y por maximo (sombra)
      - Comparar firma dimensional (D2 centro vs periferia) con la real

  E5. TEST DEL ESCALADOR: D2 radial completo
      - D2 en cada anillo desde la cruz
      - Si la cruz es un escalador, D2 debe DECRECER con la distancia
        (el centro tiene dimension efectiva mayor)

  E6. CONTROLES NEGATIVOS
      - Permutaciones del perfil, gaussianos, AR(1)
      - Mismo analisis E1/E5 sobre controles

Usa GPU + multiprocessing. NO modifica archivos originales.
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
# 1. MATRICES (metodo D del estudio: gaussian_filter1d sigma=15)
# ============================================================================
def build_matrices(img, sigma=15.0, threshold=10.0):
    h, w = img.shape
    profile = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=sigma)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < threshold).astype(np.float32)
    return R_cont, R_bin, profile

# ============================================================================
# 2. E1: DIMENSION DE CORRELACION (Grassberger-Procaccia) rigurosa
# ============================================================================
def correlation_dimension(data, radii=None, n_radii=30, subsample=2000):
    """D2 = d(log C(r)) / d(log r) en la zona de escalado."""
    data = np.asarray(data, dtype=np.float64).flatten()
    if len(data) > subsample:
        idx = rng.choice(len(data), subsample, replace=False)
        data = data[idx]
    N = len(data)
    if N < 100:
        return float("nan")
    # Distancias por pares (submuestreo de pares si N grande)
    if N > 1500:
        idx = rng.choice(N, 1500, replace=False)
        data = data[idx]
        N = len(data)
    dists = pdist(data.reshape(-1, 1))
    if len(dists) == 0:
        return float("nan")
    if radii is None:
        dmin, dmax = dists.min() + 1e-12, dists.max()
        radii = np.logspace(np.log10(dmin), np.log10(dmax), n_radii)
    counts = np.array([np.sum(dists < r) for r in radii], dtype=np.float64)
    valid = counts > 1
    if valid.sum() < 5:
        return float("nan")
    log_r = np.log(radii[valid])
    log_c = np.log(counts[valid])
    # Zona de escalado: pendiente local (derivada)
    slopes = np.gradient(log_c, log_r)
    # D2 = pendiente media en la zona central (evitar saturacion)
    mid = slice(len(slopes)//4, 3*len(slopes)//4)
    D2 = float(np.mean(slopes[mid]))
    return D2

# ============================================================================
# 3. E2: ESPECTRO DE RENYI Dq
# ============================================================================
def renyi_spectrum(data, q_values=None, box_sizes=(4, 8, 16, 32)):
    """Dq para q en [-10, 10] usando box-counting generalizado."""
    if q_values is None:
        q_values = np.linspace(-10, 10, 21)
    data = np.asarray(data, dtype=np.float64)
    h, w = data.shape
    Dq = []
    for q in q_values:
        if abs(q - 1) < 1e-6:
            # D1: entropia de informacion
            log_eps, log_I = [], []
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
                log_I.append(-np.sum(measure * np.log(measure)))
            if len(log_eps) >= 3:
                slope, _ = np.polyfit(log_eps, log_I, 1)
                Dq.append(float(slope))
            else:
                Dq.append(float("nan"))
        else:
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
                log_Z.append(np.log(np.sum(measure ** q)) / (q - 1))
            if len(log_eps) >= 3:
                slope, _ = np.polyfit(log_eps, log_Z, 1)
                Dq.append(float(slope))
            else:
                Dq.append(float("nan"))
    return {"q": q_values.tolist(), "Dq": Dq}

# ============================================================================
# 4. E3: EMBEDDING DE TAKENS
# ============================================================================
def takens_embedding(profile, m, tau=5):
    """Reconstruye el atractor en dimension m."""
    n = len(profile)
    if n <= (m-1)*tau:
        return None
    X = np.stack([profile[i:n-(m-1)*tau+i] for i in range(0, (m-1)*tau+1, tau)], axis=1)
    return X

def takens_D2(profile, m_max=10, tau=5):
    """D2 del atractor reconstruido para m = 1..m_max."""
    results = {}
    for m in range(1, m_max+1):
        X = takens_embedding(profile, m, tau)
        if X is None or len(X) < 200:
            results[m] = float("nan")
            continue
        # Submuestrear
        if len(X) > 800:
            idx = rng.choice(len(X), 800, replace=False)
            X = X[idx]
        # Distancias por pares
        dists = pdist(X)
        if len(dists) == 0:
            results[m] = float("nan")
            continue
        dmin, dmax = dists.min() + 1e-12, dists.max()
        radii = np.logspace(np.log10(dmin), np.log10(dmax), 25)
        counts = np.array([np.sum(dists < r) for r in radii], dtype=np.float64)
        valid = counts > 1
        if valid.sum() < 5:
            results[m] = float("nan")
            continue
        log_r = np.log(radii[valid])
        log_c = np.log(counts[valid])
        slopes = np.gradient(log_c, log_r)
        mid = slice(len(slopes)//4, 3*len(slopes)//4)
        results[m] = float(np.mean(slopes[mid]))
    return results

# ============================================================================
# 5. E4: SIMULACION DE PROYECCION 3D->2D
# ============================================================================
def objeto_3d(tipo, size=80):
    """Genera objeto 3D volumetrico."""
    x, y, z = np.mgrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    if tipo == "esfera":
        obj = (r < 0.9).astype(np.float64)
    elif tipo == "cubo":
        obj = (np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z))) < 0.8).astype(np.float64)
    elif tipo == "cruz3d":
        obj = ((np.abs(x) < 0.3) | (np.abs(y) < 0.3) | (np.abs(z) < 0.3)).astype(np.float64)
        obj = obj & (r < 1.0)
    elif tipo == "toro":
        R_t, r_t = 0.6, 0.3
        obj = ((np.sqrt((np.sqrt(x**2 + y**2) - R_t)**2 + z**2) < r_t)).astype(np.float64)
    elif tipo == "elipsoide":
        obj = ((x/0.9)**2 + (y/0.6)**2 + (z/0.4)**2 < 1).astype(np.float64)
    else:
        obj = (r < 0.9).astype(np.float64)
    return obj

def proyectar(obj, modo="suma"):
    """Proyeccion 3D->2D: suma (radiografia) o maximo (sombra)."""
    if modo == "suma":
        proj = obj.sum(axis=2)
    else:
        proj = obj.max(axis=2)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def firma_proyeccion(proj):
    """D2 centro vs periferia de una proyeccion."""
    h, w = proj.shape
    cy, cx = h//2, w//2
    centro = proj[cy-15:cy+15, cx-15:cx+15]
    perif = proj[:15, :15]
    D2_c = correlation_dimension(centro)
    D2_p = correlation_dimension(perif)
    return {"D2_centro": D2_c, "D2_periferia": D2_p,
            "ratio": D2_c / D2_p if D2_p and D2_p > 0 else float("nan")}

# ============================================================================
# 6. E5: D2 RADIAL COMPLETO
# ============================================================================
def D2_radial(R_cont, cx, cy, n_anillos=10, ancho=15):
    """D2 en anillos concentricos desde la cruz."""
    h, w = R_cont.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() < 500:
            continue
        region = R_cont[mask]
        D2 = correlation_dimension(region)
        res.append({"r0": r0, "r1": r1, "D2": D2, "densidad": float((R_cont[mask] < 10).mean())})
    return res

# ============================================================================
# 7. CONTROLES
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

def worker_D2_radial(args):
    R_cont, cx, cy = args
    return D2_radial(R_cont, cx, cy)

# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"E1_D2_anillos": {}, "E2_Renyi": {}, "E3_Takens": {},
              "E4_simulaciones": {}, "E5_D2_radial": {}, "E6_controles": {},
              "conclusion": {}}

    # Cargar imagen3
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    R_cont, R_bin, profile = build_matrices(img3)
    n = R_cont.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"MATRIZ: {n}x{n} | cruz=({cx},{cy}) | densidad={R_bin.mean():.4f}", flush=True)
    print("=" * 70, flush=True)

    # ============ E1: D2 en anillos ============
    print("\n[E1] DIMENSION DE CORRELACION POR ANILLOS (matriz continua)", flush=True)
    anillos = D2_radial(R_cont, cx, cy, n_anillos=10, ancho=15)
    for a in anillos:
        print(f"  r={a['r0']:3d}-{a['r1']:3d}: D2={a['D2']:.4f} | densidad={a['densidad']:.4f}", flush=True)
    report["E1_D2_anillos"] = anillos
    # Correlacion D2 vs distancia
    dists = np.array([(a["r0"]+a["r1"])/2 for a in anillos if not np.isnan(a["D2"])])
    D2s = np.array([a["D2"] for a in anillos if not np.isnan(a["D2"])])
    if len(dists) > 3:
        corr = np.corrcoef(dists, D2s)[0, 1]
        print(f"  corr(D2, distancia) = {corr:+.3f}", flush=True)
        report["E1_corr_D2_dist"] = float(corr)

    # ============ E2: Espectro de Renyi ============
    print("\n[E2] ESPECTRO DE RENYI Dq (centro vs periferia)", flush=True)
    centro = R_cont[cy-50:cy+50, cx-50:cx+50]
    perif = R_cont[:100, :100]
    # Normalizar a [0,1] para box-counting
    def norm01(x):
        x = x - x.min()
        return x / (x.max() + 1e-12)
    ren_centro = renyi_spectrum(norm01(centro))
    ren_perif = renyi_spectrum(norm01(perif))
    qs = np.array(ren_centro["q"])
    Dq_c = np.array(ren_centro["Dq"])
    Dq_p = np.array(ren_perif["Dq"])
    print(f"  q:      {' '.join(f'{q:6.0f}' for q in qs[::4])}", flush=True)
    print(f"  Dq centro: {' '.join(f'{d:6.2f}' if not np.isnan(d) else '   nan' for d in Dq_c[::4])}", flush=True)
    print(f"  Dq perif:  {' '.join(f'{d:6.2f}' if not np.isnan(d) else '   nan' for d in Dq_p[::4])}", flush=True)
    # D0, D1, D2
    for label, arr in [("centro", Dq_c), ("periferia", Dq_p)]:
        d0 = arr[qs == 0][0] if np.any(qs == 0) else float("nan")
        d1 = arr[np.argmin(np.abs(qs - 1))] if len(arr) > 0 else float("nan")
        d2 = arr[np.argmin(np.abs(qs - 2))] if len(arr) > 0 else float("nan")
        print(f"  {label}: D0={d0:.3f} | D1={d1:.3f} | D2={d2:.3f}", flush=True)
    report["E2_Renyi"] = {"q": qs.tolist(), "Dq_centro": Dq_c.tolist(), "Dq_periferia": Dq_p.tolist()}

    # ============ E3: Takens ============
    print("\n[E3] EMBEDDING DE TAKENS (D2 del atractor vs dimension m)", flush=True)
    tak = takens_D2(profile, m_max=8, tau=5)
    for m, d2 in tak.items():
        print(f"  m={m}: D2={d2:.4f}" if not np.isnan(d2) else f"  m={m}: D2=nan", flush=True)
    report["E3_Takens"] = tak
    # ¿D2 satura o crece con m?
    vals = [d for d in tak.values() if not np.isnan(d)]
    if len(vals) >= 3:
        crece = vals[-1] > vals[0] * 1.2
        print(f"  D2(m=1)={vals[0]:.3f} -> D2(m=8)={vals[-1]:.3f} | {'CRECE (escalador)' if crece else 'satura'}", flush=True)
        report["E3_crece"] = bool(crece)

    # ============ E4: Simulaciones 3D->2D ============
    print("\n[E4] SIMULACION DE PROYECCION 3D->2D (firma dimensional)", flush=True)
    sims = {}
    for tipo in ["esfera", "cubo", "cruz3d", "toro", "elipsoide"]:
        obj = objeto_3d(tipo)
        for modo in ["suma", "maximo"]:
            proj = proyectar(obj, modo)
            firma = firma_proyeccion(proj)
            key = f"{tipo}_{modo}"
            sims[key] = firma
            print(f"  {key}: D2_centro={firma['D2_centro']:.3f} | D2_perif={firma['D2_periferia']:.3f} | ratio={firma['ratio']:.2f}", flush=True)
    report["E4_simulaciones"] = sims

    # ============ E5: D2 radial completo (ya en E1) ============
    print("\n[E5] D2 RADIAL: si la cruz es escalador, D2 decae con distancia", flush=True)
    if len(dists) > 3:
        print(f"  corr(D2, distancia) = {corr:+.3f}", flush=True)
        if corr < -0.5:
            print(f"  -> D2 DECAE con distancia: el centro tiene dimension efectiva MAYOR (escalador)", flush=True)
        elif corr > 0.5:
            print(f"  -> D2 CRECE con distancia: el centro tiene dimension MENOR", flush=True)
        else:
            print(f"  -> D2 constante: sin gradiente dimensional", flush=True)
    report["E5_D2_radial"] = {"corr_D2_dist": float(corr) if len(dists) > 3 else float("nan")}

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
            Rc = np.abs(p[:, None] - p[None, :])
            tasks.append((Rc, cx, cy))
        with Pool(N_WORKERS) as pool:
            res = list(pool.imap_unordered(worker_D2_radial, tasks))
        for r in res:
            dists_c = np.array([(a["r0"]+a["r1"])/2 for a in r if not np.isnan(a["D2"])])
            D2s_c = np.array([a["D2"] for a in r if not np.isnan(a["D2"])])
            if len(dists_c) > 3:
                control_corrs[kind].append(float(np.corrcoef(dists_c, D2s_c)[0, 1]))
        print(f"  [{kind}] corr(D2,dist) = {np.mean(control_corrs[kind]):+.3f}±{np.std(control_corrs[kind]):.3f}", flush=True)
    report["E6_controles"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in control_corrs.items()}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿LA CRUZ ES UN ESCALADOR MULTIDIMENSIONAL?", flush=True)
    print("=" * 70, flush=True)
    if len(dists) > 3:
        z = (corr - np.mean(control_corrs["permutation"])) / np.std(control_corrs["permutation"]) if np.std(control_corrs["permutation"]) > 0 else float("nan")
        print(f"  corr(D2,dist) real = {corr:+.3f} vs controles = {np.mean(control_corrs['permutation']):+.3f}±{np.std(control_corrs['permutation']):.3f} (z={z:+.1f})", flush=True)
        report["conclusion"]["z_corr_D2"] = float(z)
    report["conclusion"]["E3_crece"] = report.get("E3_crece", None)
    report["conclusion"]["E4_sims"] = sims

    out_json = os.path.join(OUT, "escalador_multidimensional_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
