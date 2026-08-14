"""
INDAGACION PROFUNDA: ESTRUCTURA PERIODICA DEL REGISTRO
======================================================
Hallazgo previo: el perfil de densidad de la figura tiene 5 picos
frecuenciales con 24.8% de energia concentrada.

Profundizacion:
  F1. LOCALIZAR LOS PICOS EXACTOS: frecuencia y periodicidad espacial
      (px/ciclo) de cada pico del perfil 1D.
  F2. ESPECTRO 2D COMPLETO: FFT 2D de la imagen -> picos en el plano
      frecuencial -> periodicidad en x e y.
  F3. ORIGEN DE LA PERIODICIDAD: ¿viene de la FIGURA (cuerpo) o del
      FONDO (tela/tejido)? Separar y comparar espectros.
  F4. COMPARACION CON XRAY1: ¿la radiografia real tiene la misma
      estructura periodica? (si no: es especifica del registro)
  F5. CONTROL: permutaciones espaciales -> los picos son significativos?
  F6. INTERPRETACION FISICA: los picos corresponden a:
      - Tejido del lino (hilos: periodicidad ~1mm)
      - Anatomia del cuerpo
      - Estructura del evento registrado

Usa GPU (FFT con torch). NO modifica originales.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

print("MODO CPU (GPU ocupada)", flush=True)

def fft_1d_picos(senal, nombre, prominence=0.04, n_mostrar=10):
    """FFT 1D y deteccion de picos con periodicidad en px."""
    s = senal - senal.mean()
    N = len(s)
    f = np.fft.rfft(s)
    mag = np.abs(f)
    freqs = np.fft.rfftfreq(N)
    # Normalizar
    mag_n = mag / mag.max()
    picos, props = find_peaks(mag_n, prominence=prominence)
    # Periodicidad: px/ciclo = 1/freq
    resultados = []
    for p in picos:
        freq = freqs[p]
        if freq > 0:
            periodicidad = 1.0 / freq
        else:
            periodicidad = float("inf")
        resultados.append({
            "idx": int(p),
            "frecuencia": float(freq),
            "periodicidad_px": float(periodicidad),
            "magnitud_normalizada": float(mag_n[p]),
        })
    resultados.sort(key=lambda r: -r["magnitud_normalizada"])
    energia_picos = sum(r["magnitud_normalizada"] for r in resultados[:5])
    energia_total = mag_n.sum()
    print(f"\n  {nombre}: {len(picos)} picos | energia top5 = {energia_picos/energia_total*100:.1f}%", flush=True)
    for r in resultados[:n_mostrar]:
        print(f"    freq={r['frecuencia']:.4f} | periodicidad={r['periodicidad_px']:7.1f} px/ciclo | mag={r['magnitud_normalizada']:.3f}", flush=True)
    return resultados

def fft_2d_picos(img, nombre, prominence_rel=0.05, n_mostrar=10):
    """FFT 2D y picos en el plano frecuencial."""
    img_f = img.astype(np.float32)
    img_f = (img_f - img_f.mean()) / (img_f.std() + 1e-9)
    # Padding a potencia de 2
    h, w = img_f.shape
    n = 2**int(np.ceil(np.log2(max(h, w))))
    img_p = np.zeros((n, n), dtype=np.float32)
    img_p[:h, :w] = img_f
    # FFT en numpy (CPU)
    F = np.fft.fft2(img_p)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    mag = mag / mag.max()
    # Ignorar DC (centro)
    cy, cx = n//2, n//2
    mag[cy-2:cy+2, cx-2:cx+2] = 0
    # Picos (local maxima con un radio)
    picos = []
    # Buscar maximos locales con ventana
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(mag, size=7) == mag
    ys, xs = np.where(local_max & (mag > prominence_rel))
    # Ordenar por magnitud
    inds = np.argsort(mag[ys, xs])[::-1]
    print(f"\n  {nombre}: FFT 2D ({n}x{n})", flush=True)
    print(f"    Top picos frecuenciales (dx, dy desde centro):", flush=True)
    vistos = set()
    for k in inds[:50]:
        y, x = ys[k], xs[k]
        dy, dx = y - cy, x - cx
        # Periodicidad: n/|dy| px
        if dx == 0 and dy == 0:
            continue
        per_x = n/abs(dx) if dx != 0 else float("inf")
        per_y = n/abs(dy) if dy != 0 else float("inf")
        key = (dx//3, dy//3)
        if key in vistos:
            continue
        vistos.add(key)
        picos.append({
            "dx": int(dx), "dy": int(dy),
            "periodicidad_x_px": float(per_x), "periodicidad_y_px": float(per_y),
            "magnitud": float(mag[y, x]),
            "angulo_grados": float(np.degrees(np.arctan2(dy, dx))),
        })
        if len(picos) >= n_mostrar:
            break
    for p in picos:
        print(f"      (dx={p['dx']:+4d}, dy={p['dy']:+4d}) | per_x={p['periodicidad_x_px']:7.1f} | "
              f"per_y={p['periodicidad_y_px']:7.1f} | ang={p['angulo_grados']:6.1f}° | mag={p['magnitud']:.3f}", flush=True)
    return picos

def main():
    t0 = time.time()
    report = {}

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    print("=" * 70, flush=True)
    print(f"INDAGACION ESTRUCTURA PERIODICA | imagen {w}x{h}", flush=True)
    print("=" * 70, flush=True)

    # Separar figura y fondo
    fig = (img3 < 114).astype(np.float64)
    fondo = (img3 >= 114).astype(np.float64)

    # ============ F1: PICOS 1D DEL PERFIL DE FIGURA ============
    print("\n[F1] PICOS 1D: perfil de densidad de figura (vertical)", flush=True)
    perfil_fig = fig.mean(axis=1)
    picos_1d_fig = fft_1d_picos(perfil_fig, "perfil_figura")
    report["F1_perfil_figura"] = picos_1d_fig[:10]

    # ============ F2: ESPECTRO 2D COMPLETO ============
    print("\n[F2] ESPECTRO 2D COMPLETO de la imagen", flush=True)
    picos_2d = fft_2d_picos(img3, "imagen_completa")
    report["F2_espectro_2d"] = picos_2d[:10]

    # ============ F3: ORIGEN (figura vs fondo) ============
    print("\n[F3] ORIGEN DE LA PERIODICIDAD: figura vs fondo", flush=True)
    # Imagen de figura (solo donde hay cuerpo) y de fondo (solo tela)
    img_fig = img3 * fig
    img_fon = img3 * fondo
    # Promediar las dos regiones para tener senal comparable
    # (la periodicidad de la TELA estaria en el fondo)
    perfil_fon = fondo.mean(axis=1)
    picos_1d_fon = fft_1d_picos(perfil_fon, "perfil_fondo(tela)")
    report["F3_fondo"] = picos_1d_fon[:8]
    # Espectro 2D del fondo (tela) - separar la periodicidad del tejido
    # Hacer la tela uniforme: region lejos del cuerpo
    # Esquina superior izquierda (donde hay tela)
    region_tela = img3[:300, :300]
    picos_tela = fft_2d_picos(region_tela, "tela(esquina 300x300)")
    report["F3_tela"] = picos_tela[:8]

    # ============ F4: COMPARACION CON XRAY1 ============
    print("\n[F4] COMPARACION CON XRAY1 (radiografia real)", flush=True)
    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)
    if xray is not None:
        xray_r = cv2.resize(xray, (w, h))
        # Perfil de intensidad (radiografia: sin binarizar)
        perfil_x = xray_r.mean(axis=1)
        picos_x = fft_1d_picos(perfil_x, "xray1_perfil")
        report["F4_xray1"] = picos_x[:8]
        # Espectro 2D
        picos_x2d = fft_2d_picos(xray_r, "xray1_2d")
        report["F4_xray1_2d"] = picos_x2d[:8]

    # ============ F5: CONTROL (permutaciones) ============
    print(f"\n[F5] CONTROL: significancia de los picos (20 permutaciones)", flush=True)
    # Permutar el perfil de figura y ver cuantos picos salen por azar
    n_picos_azar = []
    energia_azar = []
    for _ in range(20):
        p_perm = rng.permutation(perfil_fig)
        s = p_perm - p_perm.mean()
        N = len(s)
        f = np.fft.rfft(s)
        mag = np.abs(f)
        mag_n = mag / mag.max()
        picos_p, _ = find_peaks(mag_n, prominence=0.04)
        n_picos_azar.append(len(picos_p))
        energia_azar.append(mag_n[picos_p].sum() / mag_n.sum() if len(picos_p) else 0)
    n_picos_azar = np.array(n_picos_azar)
    energia_azar = np.array(energia_azar)
    # Real
    s = perfil_fig - perfil_fig.mean()
    f = np.fft.rfft(s)
    mag = np.abs(f)
    mag_n = mag / mag.max()
    picos_r, _ = find_peaks(mag_n, prominence=0.04)
    energia_real = mag_n[picos_r].sum() / mag_n.sum()
    print(f"  Picos reales: {len(picos_r)} | azar: {n_picos_azar.mean():.1f}±{n_picos_azar.std():.1f}", flush=True)
    print(f"  Energia picos real: {energia_real*100:.1f}% | azar: {energia_azar.mean()*100:.1f}±{energia_azar.std()*100:.1f}%", flush=True)
    z = (energia_real - energia_azar.mean()) / energia_azar.std() if energia_azar.std() > 0 else float("nan")
    print(f"  z-score energia: {z:+.2f} {'(SIGNIFICATIVO: la periodicidad no es azar)' if z > 2 else '(no significativo)'}", flush=True)
    report["F5_control"] = {"n_picos_real": int(len(picos_r)), "n_picos_azar_mean": float(n_picos_azar.mean()),
                            "energia_real": float(energia_real), "energia_azar_mean": float(energia_azar.mean()),
                            "z": float(z)}

    # ============ F6: INTERPRETACION ============
    print("\n[F6] INTERPRETACION FISICA DE LA PERIODICIDAD", flush=True)
    # Si los picos estan en ~30-60 px/ciclo podrian ser hilos de tela
    # Si en ~100-300 px/ciclo podrian ser anatomia
    # Si en muy pocos ciclos (estructura global) = evento
    periodicidades = [r["periodicidad_px"] for r in picos_1d_fig[:5] if r["periodicidad_px"] != float("inf")]
    if periodicidades:
        print(f"  Periodicidades principales del registro: {[f'{p:.0f}' for p in periodicidades]} px/ciclo", flush=True)
        # Clasificar
        cortas = [p for p in periodicidades if p < 20]
        medias = [p for p in periodicidades if 20 <= p < 80]
        largas = [p for p in periodicidades if p >= 80]
        print(f"  Cortas (<20px, tejido fino): {len(cortas)}", flush=True)
        print(f"  Medias (20-80px, estructura): {len(medias)}", flush=True)
        print(f"  Largas (>=80px, anatomia/evento): {len(largas)}", flush=True)
        report["F6_interpretacion"] = {"periodicidades": periodicidades,
                                       "n_cortas": len(cortas), "n_medias": len(medias), "n_largas": len(largas)}

    out_json = os.path.join(OUT, "estructura_periodica_profunda_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
