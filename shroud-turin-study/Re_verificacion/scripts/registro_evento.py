"""
ANALISIS DEL REGISTRO DEL EVENTO (marco correcto)
=================================================
MARCO: la imagen NO es un cuerpo pintado. Es el REGISTRO de un evento
de emision/proyeccion — como una radiografia registra el proceso que
la atraveso. Analizamos la firma del SUCESO registrado, no la anatomia.

Propiedades fisicas del registro de un evento:
  R1. PERFIL RADIAL DE INTENSIDAD desde el punto de proyeccion:
      ¿como decae la densidad registrada con la distancia a la fuente?
      - Ley de potencia: I ~ r^-alpha (emision puntual)
      - Exponencial: I ~ exp(-beta*r) (absorcion Beer-Lambert)
      - ¿Hay estructura (capas) en el decaimiento? (evento con estructura)

  R2. COMPARACION CON REGISTRO RADIOGRAFICO REAL (xray1):
      - Mismo analisis radial sobre xray1
      - ¿El registro de la Sabana tiene la misma firma que un registro
        radiografico real? (si si: misma clase de evento)

  R3. DIMENSIONALIDAD DEL EVENTO:
      - Decaimiento radial en N dimensiones: I(r) ~ r^-(N-1) para emision
        isotropica en N dims
      - Ajustar N desde el perfil de intensidad
      - Comparar con los ~6.4 obtenidos de la redundancia

  R4. ISOTROPIA DEL REGISTRO:
      - ¿El evento emitio igual en todas direcciones? (perfil radial
        por sectores angulares)

  R5. ESTRUCTURA INTERNA DEL REGISTRO (capas del evento):
      - Detectar capas concentricas en el registro (escalones de
        densidad) -> estructura interna del evento

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.optimize import curve_fit

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    return R, profile, img3

def perfil_radial(R, cx, cy, n_anillos=40, ancho=5):
    """Perfil radial de densidad del registro desde el punto de proyeccion."""
    n = R.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res_r, res_d = [], []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 200:
            res_r.append((r0+r1)/2)
            res_d.append(float(R[mask].mean()))
    return np.array(res_r), np.array(res_d)

def ajustar_leyes(r, d):
    """Ajusta leyes fisicas de decaimiento al perfil radial del registro."""
    resultados = {}
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    valid = (r > 0) & (d > 0)
    r, d = r[valid], d[valid]
    if len(r) < 8:
        return resultados
    # 1. Ley de potencia: I = A * r^-alpha
    try:
        log_r, log_d = np.log(r), np.log(d)
        A, alpha = np.polyfit(log_r, log_d, 1)
        # R^2
        pred = A + alpha * log_r
        ss_res = np.sum((log_d - pred)**2)
        ss_tot = np.sum((log_d - log_d.mean())**2)
        r2 = 1 - ss_res/ss_tot
        resultados["potencia"] = {"alpha": float(alpha), "A": float(np.exp(A)), "r2": float(r2)}
    except Exception:
        pass
    # 2. Exponencial: I = I0 * exp(-beta*r)
    try:
        popt, _ = curve_fit(lambda x, I0, beta: I0*np.exp(-beta*x), r, d, p0=[d[0], 0.01], maxfev=5000)
        pred = popt[0]*np.exp(-popt[1]*r)
        ss_res = np.sum((d - pred)**2)
        ss_tot = np.sum((d - d.mean())**2)
        r2 = 1 - ss_res/ss_tot
        resultados["exponencial"] = {"I0": float(popt[0]), "beta": float(popt[1]), "r2": float(r2)}
    except Exception:
        pass
    # 3. Gaussiana: I = A*exp(-r^2/(2*sigma^2))
    try:
        popt, _ = curve_fit(lambda x, A, sigma: A*np.exp(-x**2/(2*sigma**2)), r, d, p0=[d[0], 50], maxfev=5000)
        pred = popt[0]*np.exp(-r**2/(2*popt[1]**2))
        ss_res = np.sum((d - pred)**2)
        ss_tot = np.sum((d - d.mean())**2)
        r2 = 1 - ss_res/ss_tot
        resultados["gaussiana"] = {"A": float(popt[0]), "sigma": float(popt[1]), "r2": float(r2)}
    except Exception:
        pass
    return resultados

def dimensionalidad_desde_decaimiento(r, d):
    """Estima N de la ley de potencia I ~ r^-(N-1) -> N = alpha + 1."""
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    valid = (r > 0) & (d > 0)
    r, d = r[valid], d[valid]
    if len(r) < 8:
        return float("nan")
    log_r, log_d = np.log(r), np.log(d)
    alpha = np.polyfit(log_r, log_d, 1)[0]
    return float(alpha + 1)

def isotropia_sectores(R, cx, cy, n_sectores=8, r_max=200):
    """Isotropia del registro: densidad por sector angular."""
    n = R.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    ang = np.arctan2(yy-cy, xx-cx) % (2*np.pi)
    dens_sectores = []
    for s in range(n_sectores):
        mask = (dist < r_max) & (dist > 20) & (ang >= s*2*np.pi/n_sectores) & (ang < (s+1)*2*np.pi/n_sectores)
        if mask.sum() > 0:
            dens_sectores.append(float(R[mask].mean()))
    dens_sectores = np.array(dens_sectores)
    return {
        "sectores": dens_sectores.tolist(),
        "media": float(dens_sectores.mean()),
        "std": float(dens_sectores.std()),
        "cv": float(dens_sectores.std()/(dens_sectores.mean()+1e-9)),
    }

def capas_registro(r, d, umbral_cambio=0.05):
    """Detecta capas concentricas (escalones) en el perfil radial."""
    d = np.asarray(d, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    capas = []
    for i in range(1, len(d)):
        if abs(d[i] - d[i-1]) > umbral_cambio:
            capas.append({"r": float(r[i]), "densidad_antes": float(d[i-1]), "densidad_despues": float(d[i])})
    return capas

def main():
    t0 = time.time()
    report = {}

    R, profile, img3 = matriz_real()
    n = R.shape[0]
    # Punto de proyeccion del evento: la cruz central del estudio (416,416)
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print("ANALISIS DEL REGISTRO DEL EVENTO (marco correcto)", flush=True)
    print(f"Punto de proyeccion: ({cx},{cy}) | matriz {n}x{n}", flush=True)
    print("=" * 70, flush=True)

    # ============ R1: PERFIL RADIAL DEL REGISTRO ============
    print("\n[R1] PERFIL RADIAL DE INTENSIDAD DEL REGISTRO (desde el punto de proyeccion)", flush=True)
    r, d = perfil_radial(R, cx, cy, n_anillos=40, ancho=5)
    print("  r | densidad registrada")
    for i in range(0, len(r), 2):
        print(f"  {r[i]:5.0f} | {d[i]:.4f}")
    # Ajustes de leyes fisicas
    ajustes = ajustar_leyes(r, d)
    print("\n  Ajuste de leyes de decaimiento del registro:")
    for ley, params in ajustes.items():
        print(f"    {ley}: {params}")
    # Mejor ley
    if ajustes:
        mejor = max(ajustes, key=lambda k: ajustes[k]["r2"])
        print(f"  MEJOR LEY: {mejor} (R2={ajustes[mejor]['r2']:.4f})")
    report["R1_perfil_radial"] = {"r": r.tolist(), "d": d.tolist(), "ajustes": ajustes,
                                  "mejor_ley": mejor if ajustes else None}

    # ============ R3: DIMENSIONALIDAD DEL EVENTO ============
    print("\n[R3] DIMENSIONALIDAD DEL EVENTO desde el decaimiento", flush=True)
    N_est = dimensionalidad_desde_decaimiento(r, d)
    print(f"  Ley de potencia: I ~ r^-alpha con alpha = N-1 -> N estimado = {N_est:.2f}", flush=True)
    print(f"  (comparar con N=6.43 de la redundancia 64.8%)", flush=True)
    report["R3_dimensionalidad"] = {"N_estimada": N_est}

    # ============ R4: ISOTROPIA DEL REGISTRO ============
    print("\n[R4] ISOTROPIA DEL REGISTRO (emision en todas direcciones)", flush=True)
    iso = isotropia_sectores(R, cx, cy, n_sectores=8)
    print(f"  Densidad por sector: {[f'{s:.3f}' for s in iso['sectores']]}", flush=True)
    print(f"  CV = {iso['cv']:.4f} -> {'ISOTROPICO (CV<0.1)' if iso['cv'] < 0.1 else 'anisotropico'}", flush=True)
    report["R4_isotropia"] = iso

    # ============ R5: CAPAS DEL REGISTRO ============
    print("\n[R5] ESTRUCTURA INTERNA DEL REGISTRO (capas del evento)", flush=True)
    capas = capas_registro(r, d, umbral_cambio=0.05)
    print(f"  Capas (escalones de densidad) detectadas: {len(capas)}")
    for c in capas:
        print(f"    r={c['r']:.0f}: densidad {c['densidad_antes']:.3f} -> {c['densidad_despues']:.3f}")
    report["R5_capas"] = capas

    # ============ R2: COMPARACION CON XRAY1 (registro radiografico real) ============
    print("\n[R2] COMPARACION CON REGISTRO RADIOGRAFICO REAL (xray1)", flush=True)
    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)
    if xray is not None:
        # Construir matriz de recurrencia del perfil central de la radiografia
        hx, wx = xray.shape
        perfil_x = ndimage.gaussian_filter1d(xray[:, wx//2].astype(np.float32), sigma=15)
        Rx = (np.abs(perfil_x[:, None] - perfil_x[None, :]) < 10.0).astype(np.float32)
        nx = Rx.shape[0]
        cxx, cyx = nx//2, nx//2
        rx, dx = perfil_radial(Rx, cxx, cyx, n_anillos=30, ancho=5)
        ajustes_x = ajustar_leyes(rx, dx)
        N_x = dimensionalidad_desde_decaimiento(rx, dx)
        print(f"  XRAY1: perfil radial desde su centro ({nx}x{nx})", flush=True)
        for ley, params in ajustes_x.items():
            print(f"    {ley}: {params}")
        if ajustes_x:
            mejor_x = max(ajustes_x, key=lambda k: ajustes_x[k]["r2"])
            print(f"  XRAY1 MEJOR LEY: {mejor_x} (R2={ajustes_x[mejor_x]['r2']:.4f})", flush=True)
        print(f"  XRAY1 N estimada: {N_x:.2f}", flush=True)
        # Comparar: misma clase de registro?
        if ajustes and ajustes_x:
            misma_ley = (max(ajustes, key=lambda k: ajustes[k]["r2"]) ==
                         max(ajustes_x, key=lambda k: ajustes_x[k]["r2"]))
            print(f"  Misma ley de decaimiento: {'SI' if misma_ley else 'NO'}", flush=True)
        report["R2_xray1"] = {"ajustes": ajustes_x, "N_estimada": N_x, "mejor_ley": mejor_x if ajustes_x else None}
    else:
        print("  XRAY1 no encontrada", flush=True)

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: FIRMA DEL EVENTO REGISTRADO", flush=True)
    print("=" * 70, flush=True)
    print(f"  1. Ley de decaimiento del registro: {mejor if ajustes else 'N/A'} (R2={ajustes[mejor]['r2']:.4f})", flush=True)
    print(f"  2. Dimensionalidad del evento (decaimiento): N={N_est:.2f}", flush=True)
    print(f"  3. Isotropia del registro: CV={iso['cv']:.4f}", flush=True)
    print(f"  4. Capas internas del evento: {len(capas)}", flush=True)
    report["conclusion"] = {
        "mejor_ley_sabana": mejor if ajustes else None,
        "N_sabana": N_est,
        "isotropia_cv": iso["cv"],
        "n_capas": len(capas),
    }

    out_json = os.path.join(OUT, "registro_evento_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
