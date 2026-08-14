"""
TESTS T1-T5: LA CRUZ CENTRAL COMO PROYECCION DIMENSIONAL
========================================================
Premisa del usuario: la cruz NO es generica - en el CENTRO de la cruz
esta la PROYECCION. El punto (416,416) es el punto de proyeccion.

Tests (metodo 2D exacto del estudio D12: box-counting [2,4,8] +
multifractal simple q[-3,3] log10):

  T1. CRUZ GENERICA vs CRUZ REAL (firma radial completa)
      - Perfiles gaussianos suaves -> su matriz de recurrencia -> cruz
      - Comparar firma radial (D, da, densidad en anillos) con la real
      - Si la real tiene estructura EXTRA (asimetria, pico mas agudo),
        la cruz no es generica

  T2. GRADIENTE RADIAL 2D COMPLETO
      - D y Delta_alpha en anillos concentricos desde (416,416)
      - Con controles (permutaciones) para significancia
      - Si D y da DECAEN con distancia -> proyeccion confirmada

  T3. TIPO DE PROYECCION (misma esfera con capas)
      - Suma (radiografia), Maximo (sombra), Perspectiva
      - Comparar perfil radial con el real (correlacion)

  T4. OBJETO PROYECTADO
      - esfera, elipsoide (proporciones humanas), cruz3d, toro, cubo,
        hiperesfera 4D (S3)
      - Proyeccion por suma -> perfil radial -> correlacion con real

  T5. LEY DE BEER-LAMBERT
      - I = I0 * exp(-mu * espesor) -> ln(I) proporcional a -espesor
      - Verificar si la intensidad del perfil real sigue esta ley

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
N_CONTROLES = 30
N_WORKERS = 12

# ============================================================================
# METODOS EXACTOS DEL ESTUDIO (D12)
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

# ============================================================================
# MATRIZ REAL
# ============================================================================
def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < 10.0).astype(np.float32)
    return R_bin, profile

# ============================================================================
# T1: CRUZ GENERICA (perfil gaussiano suave) vs REAL
# ============================================================================
def perfil_gaussiano(n, sigma_px, ruido=0.05):
    x = np.arange(n)
    p = np.exp(-0.5 * ((x - n//2) / sigma_px) ** 2)
    p = p + rng.normal(0, ruido * p.std(), size=n)
    return np.clip(p, 0, None)

def firma_radial(R, cx, cy, n_anillos=8, ancho=15):
    """D, da, densidad en anillos concentricos desde (cx,cy)."""
    h, w = R.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() < 100:
            continue
        # Reconstruir region 2D del anillo
        region = np.zeros_like(R)
        region[mask] = R[mask]
        ys, xs = np.where(mask)
        y0, y1 = ys.min(), ys.max()+1
        x0, x1 = xs.min(), xs.max()+1
        crop = region[y0:y1, x0:x1]
        res.append({
            "r0": r0, "r1": r1,
            "densidad": float(R[mask].mean()),
            "D": box_counting_simple(crop),
            "da": simple_multifractal_width(crop),
        })
    return res

def firma_radial_centro(R, cx, cy, radio=15):
    """Firma del CENTRO (region cuadrada alrededor del punto de proyeccion)."""
    h, w = R.shape
    region = R[max(0,cy-radio):cy+radio, max(0,cx-radio):cx+radio]
    return {
        "densidad": float(region.mean()),
        "D": box_counting_simple(region),
        "da": simple_multifractal_width(region),
    }

# ============================================================================
# T3/T4: OBJETOS 3D Y PROYECCIONES
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
        obj[obj > 0] = 1.0
    elif tipo == "cruz3d":
        obj = ((np.abs(x) < 0.3) | (np.abs(y) < 0.3) | (np.abs(z) < 0.3)).astype(np.float64)
        obj = obj * (r < 1.0).astype(np.float64)
    elif tipo == "toro":
        R_t, r_t = 0.6, 0.3
        obj = (np.sqrt((np.sqrt(x**2 + y**2) - R_t)**2 + z**2) < r_t).astype(np.float64)
    elif tipo == "cubo":
        obj = (np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z))) < 0.8).astype(np.float64)
    elif tipo == "hiperesfera4d":
        # S3: proyeccion 4D -> 3D. Para cada voxel (x,y,z), el valor es la
        # longitud del intervalo de w que satisface x^2+y^2+z^2+w^2 < R^2:
        #   w_max = sqrt(R^2 - r3^2) -> valor = 2*w_max (simetrico en w)
        r3 = np.sqrt(x**2 + y**2 + z**2)
        w_max = np.sqrt(np.maximum(0.9**2 - r3**2, 0))
        obj = 2.0 * w_max  # objeto 3D con densidad radial suave
        return obj
    else:
        obj = (r < 0.9).astype(np.float64)
    return obj

def proyectar(obj, modo="suma"):
    if modo == "suma":
        proj = obj.sum(axis=2)
    elif modo == "maximo":
        proj = obj.max(axis=2)
    elif modo == "perspectiva":
        # Proyeccion perspectiva: dividir por (1 + z/d)
        d = 2.0
        zz = np.linspace(-1, 1, obj.shape[2])
        proj = np.zeros((obj.shape[0], obj.shape[1]))
        for zi in range(obj.shape[2]):
            factor = 1.0 / (1.0 + zz[zi] / d)
            proj += obj[:, :, zi] * factor
    else:
        proj = obj.sum(axis=2)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def perfil_central(proj):
    h, w = proj.shape
    return proj[:, w//2].astype(np.float64)

def perfil_radial_2d(proj, cx=None, cy=None, n_anillos=8, ancho=5):
    """Perfil radial de una proyeccion 2D (densidad por anillo)."""
    h, w = proj.shape
    if cx is None:
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

def corr_seguro(a, b):
    """Correlacion de Pearson con manejo de constantes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0 or len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================================
# T5: BEER-LAMBERT
# ============================================================================
def test_beer_lambert(profile):
    """I = I0 * exp(-mu * t). Si el perfil es una radiografia:
    ln(I) debe ser proporcional a -espesor.
    Para una esfera: espesor t(r) = 2*sqrt(R^2 - r^2).
    Verificamos si ln(I) correlaciona con -t para alguna R."""
    n = len(profile)
    x = np.arange(n)
    # Normalizar perfil
    I = profile / (profile.max() + 1e-12)
    lnI = np.log(I + 1e-12)
    # Probar varios radios R y centros
    mejores = []
    for R in [100, 150, 200, 250, 300, 400, 500]:
        for c in [416, 540, n//2]:
            t = 2 * np.sqrt(np.maximum(R**2 - (x - c)**2, 0))
            mask = t > 0
            if mask.sum() < 50:
                continue
            corr = np.corrcoef(lnI[mask], -t[mask])[0, 1]
            mejores.append({"R": R, "centro": c, "corr": float(corr)})
    mejores.sort(key=lambda m: -m["corr"])
    return mejores[:5]

# ============================================================================
# WORKERS
# ============================================================================
def worker_firma_radial(args):
    R, cx, cy = args
    return firma_radial(R, cx, cy)

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"T1_cruz_generica": {}, "T2_gradiente_radial": {}, "T3_tipo_proyeccion": {},
              "T4_objeto": {}, "T5_beer_lambert": {}, "conclusion": {}}

    R_real, profile = matriz_real()
    n = R_real.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"MATRIZ REAL: {n}x{n} | punto de proyeccion=({cx},{cy})", flush=True)
    print("=" * 70, flush=True)

    # ============ T1: CRUZ GENERICA vs REAL ============
    print("\n[T1] CRUZ GENERICA (gaussiano suave) vs CRUZ REAL", flush=True)
    # Firma del centro de la cruz real
    centro_real = firma_radial_centro(R_real, cx, cy)
    print(f"  REAL centro: densidad={centro_real['densidad']:.4f} | D={centro_real['D']:.4f} | da={centro_real['da']:.4f}", flush=True)
    # Firma radial real
    radial_real = firma_radial(R_real, cx, cy, n_anillos=8, ancho=15)
    print(f"  REAL radial:", flush=True)
    for a in radial_real:
        print(f"    r={a['r0']:3d}-{a['r1']:3d}: dens={a['densidad']:.4f} | D={a['D']:.3f} | da={a['da']:.3f}", flush=True)
    report["T1_cruz_generica"]["real_centro"] = centro_real
    report["T1_cruz_generica"]["real_radial"] = radial_real

    # Cruces genericas: gaussianos con varios sigma
    print(f"  Cruces genericas (gaussianos):", flush=True)
    gen_centros = []
    for sigma_px in [50, 100, 200, 400]:
        p = perfil_gaussiano(n, sigma_px)
        Rc = (np.abs(p[:, None] - p[None, :]) < 10.0).astype(np.float32)
        fc = firma_radial_centro(Rc, n//2, n//2)
        gen_centros.append({"sigma": sigma_px, **fc})
        print(f"    gauss sigma={sigma_px}: densidad={fc['densidad']:.4f} | D={fc['D']:.4f} | da={fc['da']:.4f}", flush=True)
    report["T1_cruz_generica"]["genericas"] = gen_centros
    # Comparacion: la real tiene da mucho mayor que las genericas?
    da_gen = np.mean([g["da"] for g in gen_centros])
    print(f"  da real={centro_real['da']:.3f} vs da genericas={da_gen:.3f} | ratio={centro_real['da']/da_gen:.2f}x", flush=True)
    report["T1_cruz_generica"]["ratio_da_real_vs_generica"] = float(centro_real["da"]/da_gen) if da_gen > 0 else float("nan")

    # ============ T2: GRADIENTE RADIAL 2D + CONTROLES ============
    print("\n[T2] GRADIENTE RADIAL 2D (D y da en anillos) + controles", flush=True)
    # Correlacion D y da con distancia (real)
    dists = np.array([(a["r0"]+a["r1"])/2 for a in radial_real])
    Ds = np.array([a["D"] for a in radial_real])
    das = np.array([a["da"] for a in radial_real])
    corr_D = np.corrcoef(dists, Ds)[0, 1]
    corr_da = np.corrcoef(dists, das)[0, 1]
    print(f"  REAL: corr(D, distancia)={corr_D:+.3f} | corr(da, distancia)={corr_da:+.3f}", flush=True)
    report["T2_gradiente_radial"]["real"] = {"corr_D": float(corr_D), "corr_da": float(corr_da)}

    # Controles: permutaciones
    print(f"  Controles ({N_CONTROLES} permutaciones):", flush=True)
    ctrl_corr_D = []
    ctrl_corr_da = []
    for i in range(N_CONTROLES):
        p = rng.permutation(profile)
        Rc = (np.abs(p[:, None] - p[None, :]) < 10.0).astype(np.float32)
        fr = firma_radial(Rc, cx, cy, n_anillos=8, ancho=15)
        if len(fr) < 4:
            continue
        d_c = np.array([(a["r0"]+a["r1"])/2 for a in fr])
        D_c = np.array([a["D"] for a in fr])
        da_c = np.array([a["da"] for a in fr])
        ctrl_corr_D.append(float(np.corrcoef(d_c, D_c)[0, 1]))
        ctrl_corr_da.append(float(np.corrcoef(d_c, da_c)[0, 1]))
    z_D = (corr_D - np.mean(ctrl_corr_D)) / np.std(ctrl_corr_D) if np.std(ctrl_corr_D) > 0 else float("nan")
    z_da = (corr_da - np.mean(ctrl_corr_da)) / np.std(ctrl_corr_da) if np.std(ctrl_corr_da) > 0 else float("nan")
    print(f"  corr(D,dist) controles={np.mean(ctrl_corr_D):+.3f}±{np.std(ctrl_corr_D):.3f} (z={z_D:+.1f})", flush=True)
    print(f"  corr(da,dist) controles={np.mean(ctrl_corr_da):+.3f}±{np.std(ctrl_corr_da):.3f} (z={z_da:+.1f})", flush=True)
    report["T2_gradiente_radial"]["controles"] = {
        "corr_D_mean": float(np.mean(ctrl_corr_D)), "corr_D_std": float(np.std(ctrl_corr_D)), "z_D": float(z_D),
        "corr_da_mean": float(np.mean(ctrl_corr_da)), "corr_da_std": float(np.std(ctrl_corr_da)), "z_da": float(z_da)}

    # ============ T3: TIPO DE PROYECCION ============
    print("\n[T3] TIPO DE PROYECCION (esfera con capas: suma/maximo/perspectiva)", flush=True)
    obj = objeto_3d("esfera")
    # Perfil radial real (densidad por anillo desde la cruz)
    radial_real_dens = np.array([a["densidad"] for a in radial_real])
    tipos = {}
    for modo in ["suma", "maximo", "perspectiva"]:
        proj = proyectar(obj, modo)
        pr = perfil_radial_2d(proj, n_anillos=8, ancho=5)
        # Correlacion con el perfil real (normalizados)
        if len(pr) == len(radial_real_dens):
            corr = corr_seguro(pr, radial_real_dens)
            tipos[modo] = {"corr": corr, "perfil": pr.tolist()}
            print(f"  {modo}: corr(perfil, real) = {corr:+.3f}", flush=True)
    report["T3_tipo_proyeccion"] = tipos
    mejor_tipo = max(tipos, key=lambda k: tipos[k]["corr"] if not np.isnan(tipos[k]["corr"]) else -1) if tipos else None
    print(f"  MEJOR TIPO: {mejor_tipo}", flush=True)

    # ============ T4: OBJETO PROYECTADO ============
    print("\n[T4] OBJETO PROYECTADO (proyeccion por suma)", flush=True)
    objetos = {}
    for tipo in ["esfera", "elipsoide", "cruz3d", "toro", "cubo", "hiperesfera4d"]:
        obj_t = objeto_3d(tipo)
        proj = proyectar(obj_t, "suma")
        pr = perfil_radial_2d(proj, n_anillos=8, ancho=5)
        if len(pr) == len(radial_real_dens):
            corr = corr_seguro(pr, radial_real_dens)
            objetos[tipo] = {"corr": corr, "perfil": pr.tolist()}
            print(f"  {tipo}: corr(perfil, real) = {corr:+.3f}", flush=True)
    report["T4_objeto"] = objetos
    mejor_objeto = max(objetos, key=lambda k: objetos[k]["corr"] if not np.isnan(objetos[k]["corr"]) else -1) if objetos else None
    print(f"  MEJOR OBJETO: {mejor_objeto}", flush=True)

    # ============ T5: BEER-LAMBERT ============
    print("\n[T5] LEY DE BEER-LAMBERT (radiografia)", flush=True)
    bl = test_beer_lambert(profile)
    for m in bl:
        print(f"  R={m['R']} centro={m['centro']}: corr(lnI, -espesor) = {m['corr']:+.3f}", flush=True)
    report["T5_beer_lambert"] = bl

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    report["conclusion"] = {
        "T1_ratio_da": report["T1_cruz_generica"]["ratio_da_real_vs_generica"],
        "T2_z_D": float(z_D), "T2_z_da": float(z_da),
        "T3_mejor_tipo": mejor_tipo,
        "T4_mejor_objeto": mejor_objeto,
        "T5_mejor_beer": bl[0] if bl else None,
    }
    print(f"  T1: da real vs genericas = {report['conclusion']['T1_ratio_da']:.2f}x", flush=True)
    print(f"  T2: z(D)={z_D:+.1f} | z(da)={z_da:+.1f}", flush=True)
    print(f"  T3: mejor tipo = {mejor_tipo}", flush=True)
    print(f"  T4: mejor objeto = {mejor_objeto}", flush=True)
    print(f"  T5: mejor Beer-Lambert = {bl[0] if bl else None}", flush=True)

    out_json = os.path.join(OUT, "tests_T1_T5_proyeccion_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
