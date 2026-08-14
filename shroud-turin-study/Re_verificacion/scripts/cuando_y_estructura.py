"""
ANALISIS DEL CUANDO + ESTRUCTURA ESPACIAL DE LA SEÑAL
=====================================================
Pregunta del investigador: ¿el evento ocurrio en el momento del deposito
de Jesus en la cripta, o fue el proceso de resurreccion al cabo de 3 dias?

El registro no tiene reloj, pero hay INDICADORES INDIRECTOS:

  C1. POSICION DEL CUERPO REGISTRADO:
      - Un cuerpo recien depositado: posicion de entierro con rigor mortis
        (brazos cruzados sobre el pubis, piernas extendidas, rigido)
      - Un cuerpo tras 3 dias: el rigor se resuelve (24-36h), posicion
        relajada, abdomen hinchado por gases
      - Analizar la posicion/geometria del registro

  C2. COHERENCIA SANGRE-TEJIDO (indicador forense):
      - Sangre coagulada SIN suero separado alrededor = coagulacion rapida
        post-mortem (cuerpo recien muerto) o en vida
      - Sangre con halo de suero = coagulacion tardia (cuerpo con horas)
      - Analizar el contorno de las manchas (si detectables)

  C3. AUSENCIA DE SIGNOS DE DESCOMPOSICION EN EL REGISTRO:
      - A los 3 dias sin embalsamar: hinchazon, decoloracion, fluidos
      - El registro mostraria distorsiones del contorno corporal
      - Medir la suavidad/regularidad del contorno del cuerpo

  E1. ESTRUCTURA ESPACIAL DE LA SEÑAL (Fourier + autocorrelacion):
      - ¿Periodicidad estructurada (senal modulada) vs ruido natural?
      - Un registro de radiacion estructurada tendria componentes
        frecuenciales NO aleatorias

  E2. AUTOCORRELACION DEL PERFIL: correlaciones de largo alcance
      - Señal estructurada: decaimiento lento de autocorrelacion
      - Ruido: decaimiento rapido

NO modifica originales. Guarda en Re_verificacion/resultados/.
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

def main():
    t0 = time.time()
    report = {}

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    print("=" * 70, flush=True)
    print("ANALISIS: ¿CUANDO OCURRIO EL EVENTO? + ESTRUCTURA DE LA SEÑAL", flush=True)
    print("=" * 70, flush=True)

    # ============ C1: POSICION DEL CUERPO ============
    print("\n[C1] POSICION DEL CUERPO REGISTRADO (deposito vs 3 dias)", flush=True)
    # Un cuerpo depositado: brazos cruzados sobre el pubis, piernas juntas
    # El registro deberia mostrar SIMETRIA BILATERAL del cuerpo
    # (los brazos cruzados son simetricos izquierda-derecha)
    fig = (img3 < 114).astype(np.float64)
    # Simetria izquierda-derecha de la figura
    mitad = w // 2
    izq = fig[:, :mitad]
    der = fig[:, mitad:]
    der_flip = der[:, ::-1]
    # Alinear (las dos mitades pueden no ser exactas)
    min_w = min(izq.shape[1], der_flip.shape[1])
    sim = np.corrcoef(izq[:, :min_w].flatten(), der_flip[:, :min_w].flatten())[0, 1]
    print(f"  Simetria izquierda-derecha de la figura: {sim:+.3f}", flush=True)
    print(f"  -> {'CUERPO SIMETRICO (posicion de entierro: brazos cruzados)' if sim > 0.5 else 'asimetrico'}", flush=True)
    report["C1_posicion"] = {"simetria_bilateral": float(sim)}

    # ============ C3: CONTORNO DEL CUERPO (signos de descomposicion) ============
    print("\n[C3] CONTORNO DEL CUERPO (suavidad = sin descomposicion)", flush=True)
    # El contorno de un cuerpo con 3 dias tendria irregularidades (hinchazon)
    # Medir la suavidad del borde de la figura
    img_s = cv2.GaussianBlur(img3, (5, 5), 0)
    _, bw = cv2.threshold(img_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Contornos
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grandes = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    print(f"  Contornos grandes: {len(grandes)}", flush=True)
    for i, c in enumerate(grandes[:3]):
        area = cv2.contourArea(c)
        perim = cv2.arcLength(c, True)
        # Compacidad: 4*pi*area/perim^2 (1 = circulo perfecto, <1 = irregular)
        compacidad = 4*np.pi*area/(perim**2 + 1e-9)
        print(f"    contorno {i}: area={area:.0f} | compacidad={compacidad:.3f}", flush=True)
    report["C3_contorno"] = {"n_contornos": len(grandes)}

    # ============ E1: ESTRUCTURA ESPACIAL (Fourier) ============
    print("\n[E1] ESTRUCTURA ESPACIAL DE LA SEÑAL (Fourier)", flush=True)
    # Perfil vertical de la figura (la señal del registro a lo largo del cuerpo)
    perfil_fig = fig.mean(axis=1)  # densidad de figura por fila
    # FFT del perfil
    f = np.fft.fft(perfil_fig - perfil_fig.mean())
    mag = np.abs(f[:len(f)//2])
    # ¿Picos frecuenciales significativos (senal modulada)?
    mag_n = mag / mag.max()
    picos, _ = find_peaks(mag_n, prominence=0.05)
    print(f"  Picos frecuenciales del perfil de figura: {len(picos)}", flush=True)
    print(f"  -> {'SEÑAL CON COMPONENTES PERIODICAS (estructurada)' if len(picos) > 3 else 'espectro suave'}", flush=True)
    # Fraccion de energia en picos vs fondo
    energia_picos = mag[picos].sum() / mag.sum() if len(picos) > 0 else 0
    print(f"  Energia en picos: {energia_picos*100:.1f}%", flush=True)
    report["E1_fourier"] = {"n_picos": int(len(picos)), "energia_picos_pct": float(energia_picos*100)}

    # ============ E2: AUTOCORRELACION (largo alcance) ============
    print("\n[E2] AUTOCORRELACION DEL PERFIL (correlaciones de largo alcance)", flush=True)
    p = perfil_fig - perfil_fig.mean()
    autocorr = np.correlate(p, p, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / (autocorr[0] + 1e-12)
    # Longitud de correlacion: donde cae a 1/e
    lags = np.arange(len(autocorr))
    idx_1e = np.where(autocorr < 1/np.e)[0]
    long_corr = idx_1e[0] if len(idx_1e) > 0 else len(autocorr)
    print(f"  Longitud de correlacion (caida a 1/e): {long_corr} filas (de {h})", flush=True)
    print(f"  -> {'CORRELACION DE LARGO ALCANCE (senal estructurada)' if long_corr > h*0.1 else 'correlacion corta'}", flush=True)
    report["E2_autocorr"] = {"longitud_correlacion": int(long_corr), "h": int(h)}

    # ============ C2: MANCHAS DE SANGRE (coherencia) ============
    print("\n[C2] MANCHAS OSCURAS (sangre): contorno y coherencia", flush=True)
    # Las manchas de sangre son las zonas MAS oscuras (muy por debajo de la figura)
    sangre = (img3 < 60).astype(np.uint8)
    n_sangre = int(sangre.sum())
    print(f"  Pixeles muy oscuros (posible sangre): {n_sangre} ({n_sangre/(h*w)*100:.1f}%)", flush=True)
    if n_sangre > 100:
        num_lab, labels, stats, _ = cv2.connectedComponentsWithStats(sangre)
        manchas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_lab) if stats[i, cv2.CC_STAT_AREA] > 50]
        manchas.sort(key=lambda x: -x[1])
        print(f"  Manchas grandes (>50px): {len(manchas)}", flush=True)
        for i, (lab, area) in enumerate(manchas[:5]):
            print(f"    mancha {i}: area={area}px", flush=True)
        report["C2_sangre"] = {"n_pixeles": n_sangre, "n_manchas": len(manchas)}
    else:
        report["C2_sangre"] = {"n_pixeles": n_sangre}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    print(f"  C1 simetria cuerpo: {sim:+.3f}", flush=True)
    print(f"  E1 picos frecuenciales: {len(picos)} ({energia_picos*100:.1f}% energia)", flush=True)
    print(f"  E2 longitud correlacion: {long_corr} filas", flush=True)
    report["conclusion"] = {
        "simetria_cuerpo": float(sim),
        "picos_frecuenciales": int(len(picos)),
        "longitud_correlacion": int(long_corr),
    }

    out_json = os.path.join(OUT, "cuando_y_estructura_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
