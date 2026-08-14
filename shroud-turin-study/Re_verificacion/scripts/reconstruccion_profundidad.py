"""
RECONSTRUCCION DEL MAPA DE PROFUNDIDAD 3D DESDE EL REGISTRO
============================================================
Nexo: si I = I0 * exp(-beta * Z), entonces Z = -ln(I/I0) / beta.
Podemos RECONSTRUIR el mapa de profundidad 3D del cuerpo desde la
escala de grises del registro, usando la beta medida (0.005).

Tests:
  N1. RECONSTRUIR Z: aplicar Z = -ln(I)/beta a la figura.
      ¿La superficie 3D resultante es coherente (suave, simetrica)?
  N2. VALIDACION CON EL ROSTRO: el rostro reconstruido debe mostrar
      la anatomia esperada (nariz prominente, ojos hundidos, frente,
      menton). Un mapa de profundidad REAL de un rostro tiene estas
      caracteristicas TOPOGRAFICAS.
  N3. CONSISTENCIA DE BETA: variar beta y ver cual produce el mapa
      de profundidad MAS suave (beta optimo = el del mecanismo real).
  N4. COMPARACION FIGURA/FONDO: el fondo NO debe producir un mapa de
      profundidad coherente (validacion de que la estructura es del
      cuerpo, no del metodo).
  N5. RELIEVE 3D: guardar el mapa de profundidad reconstruido como
      superficie 3D para visualizacion.

CPU. NO modifica originales. Guarda en Re_verificacion/resultados/.
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

def suavidad_superficie(Z):
    Z = Z.astype(np.float64)
    d2x = np.abs(np.gradient(np.gradient(Z, axis=1), axis=1))
    d2y = np.abs(np.gradient(np.gradient(Z, axis=0), axis=0))
    curvatura = d2x + d2y
    return float(curvatura.mean())

def simetria_rostro(Z):
    h, w = Z.shape
    izq = Z[:, :w//2]
    der = np.fliplr(Z[:, w//2:])
    mw = min(izq.shape[1], der.shape[1])
    if izq[:, :mw].std() == 0 or der[:, :mw].std() == 0:
        return float("nan")
    return float(np.corrcoef(izq[:, :mw].flatten(), der[:, :mw].flatten())[0, 1])

def main():
    t0 = time.time()
    report = {}

    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    print("=" * 70, flush=True)
    print("RECONSTRUCCION DEL MAPA DE PROFUNDIDAD 3D (Z = -ln(I)/beta)", flush=True)
    print("=" * 70, flush=True)

    if img1 is None:
        print("imagen1 no disponible")
        return
    face = img1[100:1100, 1000:2000].astype(np.float64)
    h, w = face.shape
    print(f"Rostro: {w}x{h}", flush=True)

    # Beta medida en el registro: 0.005 (ley exponencial del perfil radial)
    beta_medida = 0.005

    # ============ N1: RECONSTRUIR Z ============
    print("\n[N1] Reconstruccion con beta = 0.005", flush=True)
    # En el negativo: mayor intensidad = mas cerca de la tela
    # Z = -ln(I_normalizado) / beta
    I = face / 255.0
    I = np.clip(I, 1e-6, 1.0)
    Z = -np.log(I) / beta_medida
    # Normalizar Z a 0-1 para visualizacion
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    suav = suavidad_superficie(Z_norm)
    sim = simetria_rostro(Z_norm)
    print(f"  Suavidad del mapa reconstruido: {suav:.4f}", flush=True)
    print(f"  Simetria del mapa reconstruido: {sim:+.3f}", flush=True)
    report["N1_beta005"] = {"suavidad": suav, "simetria": sim}

    # ============ N3: BETA OPTIMO ============
    print("\n[N3] BARRIDO DE BETA (el optimo produce el mapa mas suave)", flush=True)
    betas = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.02, 0.05]
    resultados_beta = []
    for beta in betas:
        Z_b = -np.log(I) / beta
        Z_bn = (Z_b - Z_b.min()) / (Z_b.max() - Z_b.min() + 1e-9)
        s = suavidad_superficie(Z_bn)
        sim_b = simetria_rostro(Z_bn)
        resultados_beta.append({"beta": beta, "suavidad": s, "simetria": sim_b})
        print(f"  beta={beta:.3f}: suavidad={s:.4f} | simetria={sim_b:+.3f}", flush=True)
    # Beta optimo: maxima suavidad (menor curvatura) con buena simetria
    mejor = min(resultados_beta, key=lambda r: r["suavidad"])
    print(f"  BETA OPTIMO (menor curvatura): {mejor['beta']:.3f}", flush=True)
    report["N3_barrido_beta"] = resultados_beta
    report["N3_mejor"] = mejor

    # ============ N2: TOPOGRAFIA DEL ROSTRO RECONSTRUIDO ============
    print("\n[N2] TOPOGRAFIA DEL ROSTRO RECONSTRUIDO (anatomia)", flush=True)
    # Usar beta optimo o la medida
    beta_uso = mejor["beta"] if mejor["suavidad"] < suav else beta_medida
    Z_opt = -np.log(I) / beta_uso
    Z_optn = (Z_opt - Z_opt.min()) / (Z_opt.max() - Z_opt.min() + 1e-9)
    # Perfil vertical medio (topografia: frente -> nariz -> boca -> menton)
    perfil_medio = ndimage.gaussian_filter1d(Z_optn[:, w//2], sigma=10)
    print(f"  Perfil medio vertical (frente->menton), beta={beta_uso}:", flush=True)
    for i in range(0, h, 100):
        print(f"    y={i:4d} ({i/h*100:.0f}%): Z={perfil_medio[i]:.3f}", flush=True)
    # Detectar picos/valles (nariz prominente = pico, ojos = valles)
    from scipy.signal import find_peaks
    picos, _ = find_peaks(perfil_medio, distance=50)
    valles, _ = find_peaks(-perfil_medio, distance=50)
    print(f"  Picos de profundidad (protuberancias): {picos.tolist()}", flush=True)
    print(f"  Valles (hundimientos): {valles.tolist()}", flush=True)
    report["N2_topografia"] = {"perfil": perfil_medio[::50].tolist(),
                               "picos": picos.tolist(), "valles": valles.tolist()}

    # ============ N4: FONDO (validacion) ============
    print("\n[N4] FONDO: ¿produce mapa de profundidad coherente?", flush=True)
    # Region de tela pura (si existe en imagen1... es el rostro, poco fondo)
    # Usar la esquina si hay tela
    esquina = face[:100, :100]
    if esquina.std() > 5:
        I_e = esquina / 255.0
        I_e = np.clip(I_e, 1e-6, 1.0)
        Z_e = -np.log(I_e) / beta_uso
        Z_en = (Z_e - Z_e.min()) / (Z_e.max() - Z_e.min() + 1e-9)
        s_e = suavidad_superficie(Z_en)
        sim_e = simetria_rostro(Z_en)
        print(f"  Esquina (fondo): suavidad={s_e:.4f} | simetria={sim_e:+.3f}", flush=True)
        print(f"  (compare: rostro suavidad={suav:.4f}, simetria={sim:+.3f})", flush=True)
        report["N4_fondo"] = {"suavidad": s_e, "simetria": sim_e}

    # ============ N5: GUARDAR RELIEVE ============
    print("\n[N5] Guardando mapa de profundidad reconstruido", flush=True)
    # Guardar como imagen de profundidad (8-bit)
    Z_img = (Z_optn * 255).astype(np.uint8)
    p1 = os.path.join(OUT, "mapa_profundidad_reconstruido.png")
    cv2.imwrite(p1, Z_img)
    print(f"  Guardado: {p1}", flush=True)
    report["N5_relieve"] = {"path": p1}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: EL MECANISMO DEL EVENTO", flush=True)
    print("=" * 70, flush=True)
    print(f"  N1: mapa reconstruido (beta=0.005): suavidad={suav:.4f}, simetria={sim:+.3f}", flush=True)
    print(f"  N3: beta optimo = {mejor['beta']:.3f} (curvatura minima {mejor['suavidad']:.4f})", flush=True)
    print(f"  N2: topografia del rostro: picos={picos.tolist()}, valles={valles.tolist()}", flush=True)
    report["conclusion"] = {
        "beta_medida": beta_medida,
        "beta_optimo": mejor["beta"],
        "suavidad_optima": mejor["suavidad"],
        "simetria_optima": mejor["simetria"],
        "picos_topografia": picos.tolist(),
        "valles_topografia": valles.tolist(),
    }

    out_json = os.path.join(OUT, "reconstruccion_profundidad_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
