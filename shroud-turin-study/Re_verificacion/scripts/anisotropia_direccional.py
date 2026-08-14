"""
ANISOTROPIA DIRECCIONAL DEL REGISTRO DEL EVENTO (profundizacion)
================================================================
El registro es anisotropico (CV=0.81). Profundizamos:
  D1. MAPA ANGULAR COMPLETO: densidad por angulo (36 sectores de 10°)
      -> direccion preferente del evento
  D2. PERFIL RADIAL POR DIRECCIONES CARDINALES: N/S/E/O y diagonales
      -> como decae el registro en cada direccion
  D3. COMPARACION CON ASIMETRIA DE BRAZOS DE LA CRUZ:
      - estudio D11: brazos arriba=izq densidad 0.149, abajo=der 0.477
      - verificar si la anisotropia del registro coincide
  D4. COMPARACION CON XRAY1: misma anisotropia? (misma clase de evento?)
  D5. CONTROL: ¿la anisotropia es artefacto de la posicion EXCENTRICA
      del punto de proyeccion (416,416 cerca de la esquina)?
      -> limitar radio al minimo disponible (416) para que todos los
         sectores tengan la misma cobertura
  D6. GRADIENTE DIRECCIONAL: pendiente de la densidad a lo largo de
      cada eje (evento con direccion preferente vs emision simetrica)

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

def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    return R

def mapa_angular(R, cx, cy, n_sectores=36, r_min=10, r_max=None):
    """Densidad por sector angular (anillo completo)."""
    n = R.shape[0]
    if r_max is None:
        # Radio maximo comun: minimo disponible en todas direcciones
        r_max = min(cx, cy, n-1-cx, n-1-cy) - 1
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    ang = np.degrees(np.arctan2(yy-cy, xx-cx)) % 360
    # Mascara anillo completo
    anillo = (dist >= r_min) & (dist < r_max)
    densidades = []
    for s in range(n_sectores):
        a0 = s * 360 / n_sectores
        a1 = (s+1) * 360 / n_sectores
        mask = anillo & (ang >= a0) & (ang < a1)
        if mask.sum() > 50:
            densidades.append(float(R[mask].mean()))
        else:
            densidades.append(float("nan"))
    angulos = np.array([(s+0.5)*360/n_sectores for s in range(n_sectores)])
    return angulos, np.array(densidades), r_max

def direccion_preferente(angulos, densidades):
    """Angulo de maxima densidad (direccion preferente del evento)."""
    valid = ~np.isnan(densidades)
    a, d = angulos[valid], densidades[valid]
    # Interpolar para encontrar el maximo con precision
    from scipy.interpolate import interp1d
    try:
        f = interp1d(a, d, kind='cubic', bounds_error=False, fill_value='extrapolate')
        a_fine = np.linspace(0, 360, 720)
        d_fine = f(a_fine)
        idx = np.argmax(d_fine)
        return float(a_fine[idx]), float(d_fine[idx])
    except Exception:
        idx = np.argmax(d)
        return float(a[idx]), float(d[idx])

def perfil_direcciones(R, cx, cy, r_max=None, n_puntos=20):
    """Perfil radial a lo largo de direcciones cardinales + diagonales."""
    n = R.shape[0]
    if r_max is None:
        r_max = min(cx, cy, n-1-cx, n-1-cy) - 1
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    ang = np.degrees(np.arctan2(yy-cy, xx-cx)) % 360
    direcciones = {
        "N (arriba, 270°)": 270, "S (abajo, 90°)": 90,
        "E (derecha, 0°)": 0, "O (izquierda, 180°)": 180,
        "NE (diag, 315°)": 315, "NO (diag, 225°)": 225,
        "SE (diag, 45°)": 45, "SO (diag, 135°)": 135,
    }
    resultados = {}
    for nombre, a_dir in direcciones.items():
        # Ventana angular de ±5°
        tol = 5
        mask_ang = ((ang >= a_dir - tol) & (ang < a_dir + tol)) | \
                   ((a_dir - tol < 0) & (ang >= 360 + a_dir - tol)) | \
                   ((a_dir + tol >= 360) & (ang < (a_dir + tol) % 360))
        radios = []
        densidades = []
        for i in range(n_puntos):
            r0 = i * r_max / n_puntos
            r1 = (i+1) * r_max / n_puntos
            mask = mask_ang & (dist >= r0) & (dist < r1)
            if mask.sum() > 10:
                radios.append((r0+r1)/2)
                densidades.append(float(R[mask].mean()))
        if len(radios) >= 5:
            # Pendiente del decaimiento (exponencial)
            log_d = np.log(np.array(densidades) + 1e-9)
            beta = np.polyfit(radios, log_d, 1)[0]
            resultados[nombre] = {
                "densidad_media": float(np.mean(densidades)),
                "densidad_inicial": float(densidades[0]),
                "densidad_final": float(densidades[-1]),
                "beta_decaimiento": float(beta),
                "perfil": densidades,
            }
    return resultados

def asimetria_brazos_cruz(R, cx, cy):
    """Replica D11 del estudio: densidad de los 4 brazos de la cruz."""
    # Brazos: regiones rectangulares a lo largo de los ejes
    arm_length = 80
    arm_width = 10
    brazos = {
        'arriba': R[cx-arm_length:cx, cy-arm_width:cy+arm_width],
        'abajo': R[cx:cx+arm_length, cy-arm_width:cy+arm_width],
        'izquierda': R[cx-arm_width:cx+arm_width, cy-arm_length:cy],
        'derecha': R[cx-arm_width:cx+arm_width, cy:cy+arm_length],
    }
    return {k: float(v.mean()) for k, v in brazos.items()}

def main():
    t0 = time.time()
    report = {}

    R = matriz_real()
    n = R.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"ANISOTROPIA DIRECCIONAL DEL REGISTRO | punto ({cx},{cy}) | matriz {n}x{n}", flush=True)
    print("=" * 70, flush=True)

    # ============ D1: MAPA ANGULAR COMPLETO ============
    print("\n[D1] MAPA ANGULAR (36 sectores de 10°, anillo completo)", flush=True)
    angulos, dens, r_max = mapa_angular(R, cx, cy, n_sectores=36)
    print(f"  Radio maximo comun: {r_max}px (limitado por la posicion excéntrica)", flush=True)
    print("  Sector | angulo | densidad")
    for i in range(0, 36, 3):
        print(f"    {i:3d} | {angulos[i]:5.1f}° | {dens[i]:.4f}")
    # Direccion preferente
    ang_pref, dens_pref = direccion_preferente(angulos, dens)
    print(f"\n  DIRECCION PREFERENTE: {ang_pref:.1f}° (densidad {dens_pref:.4f})", flush=True)
    # Direccion menos densa
    valid = ~np.isnan(dens)
    ang_min = angulos[valid][np.argmin(dens[valid])]
    print(f"  DIRECCION MINIMA: {ang_min:.1f}° (densidad {np.nanmin(dens):.4f})", flush=True)
    report["D1_mapa_angular"] = {"angulos": angulos.tolist(), "densidades": dens.tolist(),
                                  "r_max": int(r_max),
                                  "direccion_preferente": ang_pref,
                                  "densidad_preferente": dens_pref,
                                  "direccion_minima": float(ang_min)}

    # ============ D2: PERFIL POR DIRECCIONES ============
    print("\n[D2] PERFIL RADIAL POR DIRECCIONES (decaimiento del registro)", flush=True)
    direcciones = perfil_direcciones(R, cx, cy, r_max=r_max)
    for nombre, datos in direcciones.items():
        print(f"  {nombre}: dens_media={datos['densidad_media']:.4f} | "
              f"inicial={datos['densidad_inicial']:.4f} -> final={datos['densidad_final']:.4f} | "
              f"beta={datos['beta_decaimiento']:+.5f}", flush=True)
    report["D2_direcciones"] = direcciones

    # ============ D3: ASIMETRIA DE BRAZOS (D11 del estudio) ============
    print("\n[D3] ASIMETRIA DE BRAZOS DE LA CRUZ (D11 del estudio)", flush=True)
    brazos = asimetria_brazos_cruz(R, cx, cy)
    for k, v in brazos.items():
        print(f"  brazo {k}: densidad={v:.4f}")
    ratio_v = brazos['abajo'] / (brazos['arriba'] + 1e-9)
    ratio_h = brazos['derecha'] / (brazos['izquierda'] + 1e-9)
    print(f"  Ratio abajo/arriba: {ratio_v:.2f} | Ratio derecha/izquierda: {ratio_h:.2f}", flush=True)
    print(f"  (estudio D11: arriba=izq 0.149, abajo=der 0.477 -> ratio 3.2)", flush=True)
    report["D3_brazos"] = brazos

    # ============ D4: COMPARACION CON XRAY1 ============
    print("\n[D4] COMPARACION CON XRAY1 (misma anisotropia?)", flush=True)
    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)
    if xray is not None:
        hx, wx = xray.shape
        perfil_x = ndimage.gaussian_filter1d(xray[:, wx//2].astype(np.float32), sigma=15)
        Rx = (np.abs(perfil_x[:, None] - perfil_x[None, :]) < 10.0).astype(np.float32)
        nx = Rx.shape[0]
        cxx, cyx = nx//2, nx//2
        ang_x, dens_x, rmax_x = mapa_angular(Rx, cxx, cyx, n_sectores=36)
        cv_x = float(np.nanstd(dens_x) / (np.nanmean(dens_x) + 1e-9))
        cv_s = float(np.nanstd(dens) / (np.nanmean(dens) + 1e-9))
        print(f"  CV anisotropia: SABANA={cv_s:.4f} vs XRAY1={cv_x:.4f}", flush=True)
        print(f"  -> {'SABANA MAS ANISOTROPICA' if cv_s > cv_x else 'XRAY1 MAS ANISOTROPICA'}", flush=True)
        report["D4_xray1"] = {"cv_sabana": cv_s, "cv_xray1": cv_x}

    # ============ D5: CONTROL - artefacto de posicion? ============
    print("\n[D5] CONTROL: ¿anisotropia = artefacto de posicion excéntrica?", flush=True)
    # El radio maximo comun (416) garantiza cobertura igual en todos los sectores.
    # Pero la posicion (416,416) esta en el cuadrante TL. Comprobar con un
    # punto CENTRADO (540,540): ¿tambien es anisotropico?
    ang_c, dens_c, _ = mapa_angular(R, 540, 540, n_sectores=36)
    cv_c = float(np.nanstd(dens_c) / (np.nanmean(dens_c) + 1e-9))
    print(f"  CV en punto EXCENTRICO (416,416): {cv_s:.4f}", flush=True)
    print(f"  CV en punto CENTRADO (540,540): {cv_c:.4f}", flush=True)
    print(f"  -> {'la anisotropia NO es solo artefacto de posicion' if cv_s > cv_c else 'la anisotropia es similar en el centro (estructura global)'}", flush=True)
    # Tambien: punto aleatorio
    rng = np.random.default_rng(42)
    cvs_ctrl = []
    for _ in range(20):
        px = rng.integers(200, n-200)
        py = rng.integers(200, n-200)
        a_t, d_t, _ = mapa_angular(R, px, py, n_sectores=36)
        cvs_ctrl.append(float(np.nanstd(d_t) / (np.nanmean(d_t) + 1e-9)))
    cvs_ctrl = np.array(cvs_ctrl)
    z = (cv_s - cvs_ctrl.mean()) / cvs_ctrl.std() if cvs_ctrl.std() > 0 else float("nan")
    print(f"  CV en 20 puntos aleatorios: {cvs_ctrl.mean():.4f}±{cvs_ctrl.std():.4f} | z={z:+.2f}", flush=True)
    report["D5_control"] = {"cv_excentrico": cv_s, "cv_centrado": cv_c,
                            "cv_aleatorios_mean": float(cvs_ctrl.mean()),
                            "cv_aleatorios_std": float(cvs_ctrl.std()),
                            "z": float(z)}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: GEOMETRIA DEL EVENTO REGISTRADO", flush=True)
    print("=" * 70, flush=True)
    print(f"  1. Direccion preferente del registro: {ang_pref:.0f}°", flush=True)
    print(f"  2. Asimetria brazos: abajo/arriba={ratio_v:.2f}, der/izq={ratio_h:.2f}", flush=True)
    print(f"  3. CV anisotropia: {cv_s:.4f} (z={z:+.2f} vs aleatorios)", flush=True)
    report["conclusion"] = {
        "direccion_preferente": ang_pref,
        "asimetria_brazos": brazos,
        "cv_anisotropia": cv_s,
        "z_control": float(z),
    }

    out_json = os.path.join(OUT, "anisotropia_direccional_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
