"""
CARACTERIZACION DEL EVENTO: ESTRUCTURA INTERNA + TIPO DE RADIACION
==================================================================
Si el registro es de un proceso tipo TAC/RMN, deberia contener estructura
INTERNA (huesos, dientes, densidades de tejido) — no solo la superficie.

Tests:
  G1. ESTRUCTURA INTERNA DEL ROSTRO (imagen1 - negativo):
      - El rostro en negativo muestra detalles tipo hueso/dientes?
      - Analizar la zona de la boca/dientes (alta densidad optica)
      - Comparar con una radiografia real de craneo (xray1 es de cuerpo?)
      - ¿El registro contiene informacion que NO es visible en la
        superficie del cuerpo (estructura interna)?

  G2. PERFIL DE ABSORCION (Beer-Lambert generalizado):
      - I = I0 * exp(-mu * espesor) -> ¿el registro codifica espesor?
      - Si hay estructura interna: el registro NO es una simple sombra
        de superficie, sino una proyeccion de densidad volumetrica

  G3. RANGO DINAMICO INTERNO:
      - ¿Cuantos niveles de densidad distinguibles hay en el cuerpo
        (no en el fondo)? Un TAC tiene muchos; una sombra superficial pocos

  G4. NITIDEZ DE BORDES (temporalidad del evento):
      - Bordes nitidos = pulso corto (sin desenfoque por movimiento)
      - Bordes difusos = exposicion larga o fuente extensa
      - Medir la transicion de intensidad en los bordes del cuerpo

  G5. COMPARACION CON XRAY1 REAL (referencia de registro volumetrico)

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    report = {}

    print("=" * 70, flush=True)
    print("CARACTERIZACION DEL EVENTO: ESTRUCTURA INTERNA + RADIACION", flush=True)
    print("=" * 70, flush=True)

    # Cargar imagenes
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)

    # ============ G1: ESTRUCTURA INTERNA DEL ROSTRO ============
    print("\n[G1] ESTRUCTURA INTERNA: ¿el negativo del rostro muestra detalle interno?", flush=True)
    if img1 is not None:
        h1, w1 = img1.shape
        print(f"  imagen1 (negativo): {w1}x{h1}", flush=True)
        # El negativo del rostro: la zona de los dientes/boca tiene
        # densidad optica distinta (los dientes aparecen en el negativo
        # como zonas oscuras detras de los labios)
        # Analizar el rango de intensidades DENTRO del rostro
        # Recortar rostro (como la otra IA: [100:1100, 1000:2000])
        face = img1[100:1100, 1000:2000]
        print(f"  Rostro crop: {face.shape}", flush=True)
        # Estadisticas del rostro
        print(f"  Rango: [{face.min()}, {face.max()}] | media={face.mean():.1f} | std={face.std():.1f}", flush=True)
        # Histograma: cuantos modos (niveles de densidad interna)
        hist, edges = np.histogram(face, bins=32, range=(0, 256))
        # Suavizar y contar picos
        from scipy import ndimage
        hist_s = ndimage.gaussian_filter1d(hist.astype(float), 2)
        from scipy.signal import find_peaks
        picos, _ = find_peaks(hist_s, prominence=hist_s.max()*0.02)
        print(f"  Modos del histograma del rostro: {len(picos)} (en posiciones {[int(edges[p]) for p in picos]})", flush=True)
        report["G1_rostro"] = {"n_modos": int(len(picos)), "posiciones": [int(edges[p]) for p in picos],
                               "rango": [int(face.min()), int(face.max())]}

        # Zona de la boca: buscar la region con variacion fina (dientes)
        # La boca esta en la parte inferior del rostro
        # Usar varianza local para encontrar zonas de detalle fino
        lap = np.abs(cv2.Laplacian(face, cv2.CV_64F))
        lap_s = cv2.GaussianBlur(lap, (15, 15), 0)
        # Zonas de maximo detalle fino
        flat = lap_s.flatten()
        top = np.argsort(flat)[-200:]
        ys = [idx // face.shape[1] for idx in top]
        xs = [idx % face.shape[1] for idx in top]
        print(f"  Zonas de maximo detalle fino (200 px): y media={np.mean(ys):.0f}, "
              f"distribucion y: [{np.min(ys)}, {np.max(ys)}]", flush=True)
        report["G1_detalle"] = {"y_media_detalle": float(np.mean(ys)), "y_min": int(np.min(ys)), "y_max": int(np.max(ys))}
    else:
        print("  imagen1 no disponible", flush=True)

    # ============ G2: BEER-LAMBERT (el registro codifica espesor?) ============
    print("\n[G2] ¿El registro codifica ESPESOR (proyeccion volumetrica)?", flush=True)
    # Para una proyeccion volumetrica: I = I0 * exp(-mu * espesor)
    # ln(I) ~ -mu * espesor. En un cuerpo, el espesor varia suavemente
    # y la densidad optica del registro deberia correlacionar con la
    # "profundidad" de la estructura interna
    # Metodo: en el rostro, medir si la intensidad tiene gradaciones
    # internas (no solo 2 niveles: piel/no-piel)
    if img1 is not None:
        face = img1[100:1100, 1000:2000]
        # Niveles de gris distintos dentro del rostro (estructura interna)
        # Cuantificar con cuantiles
        cuantiles = np.percentile(face, [5, 25, 50, 75, 95])
        print(f"  Cuantiles de intensidad del rostro: {[int(c) for c in cuantiles]}", flush=True)
        rango_interno = cuantiles[3] - cuantiles[1]
        print(f"  Rango intercuartil (estructura interna): {rango_interno:.0f} niveles", flush=True)
        # Comparar con el fondo
        fondo = img1[:100, :100]
        rango_fondo = np.percentile(fondo, 75) - np.percentile(fondo, 25)
        print(f"  Rango intercuartil del fondo: {rango_fondo:.0f}", flush=True)
        print(f"  -> El rostro tiene {rango_interno/rango_fondo:.1f}x mas estructura interna que el fondo", flush=True)
        report["G2_beer"] = {"cuantiles": [int(c) for c in cuantiles],
                             "rango_interno": float(rango_interno), "rango_fondo": float(rango_fondo)}

    # ============ G3: NITIDEZ DE BORDES (temporalidad) ============
    print("\n[G3] NITIDEZ DE BORDES (pulso corto vs exposicion larga)", flush=True)
    if img3 is not None:
        # Bordes del cuerpo: transicion abrupta = pulso corto
        # Usar la columna central del cuerpo y medir la transicion
        h3, w3 = img3.shape
        perfil_col = img3[:, w3//2].astype(float)
        # Normalizar
        perfil_col = (perfil_col - perfil_col.min()) / (perfil_col.max() - perfil_col.min() + 1e-9)
        # Derivada (bordes)
        deriv = np.abs(np.gradient(perfil_col))
        # Anchura de los bordes: cuantos px tarda en subir/bajar
        # Encontrar el borde mas fuerte
        borde_idx = np.argmax(deriv)
        # Medir la transicion alrededor del borde
        ventana = 30
        x0 = max(0, borde_idx-ventana)
        x1 = min(len(perfil_col), borde_idx+ventana)
        seg = perfil_col[x0:x1]
        # Pendiente maxima (nitidez del borde)
        pend_max = deriv[x0:x1].max()
        # Anchura de transicion: distancia entre 10% y 90% del cambio
        v_min, v_max = seg.min(), seg.max()
        if v_max - v_min > 0.05:
            cruz_10 = np.where(seg > v_min + 0.1*(v_max-v_min))[0]
            cruz_90 = np.where(seg > v_min + 0.9*(v_max-v_min))[0]
            if len(cruz_10) and len(cruz_90):
                ancho_trans = cruz_90[0] - cruz_10[0]
            else:
                ancho_trans = -1
        else:
            ancho_trans = -1
        print(f"  Borde mas fuerte en y={borde_idx} (de {len(perfil_col)})", flush=True)
        print(f"  Pendiente maxima del borde: {pend_max:.4f}/px", flush=True)
        print(f"  Ancho de transicion (10%->90%): {ancho_trans} px", flush=True)
        print(f"  -> {'BORDE NITIDO (pulso corto, <10px)' if 0 <= ancho_trans < 10 else 'borde difuso (>10px)'}", flush=True)
        report["G3_nitidez"] = {"borde_idx": int(borde_idx), "pend_max": float(pend_max),
                                "ancho_transicion": int(ancho_trans)}

    # ============ G5: COMPARACION CON XRAY1 ============
    print("\n[G5] REFERENCIA: xray1 (registro volumetrico real)", flush=True)
    if xray is not None:
        hx, wx = xray.shape
        print(f"  xray1: {wx}x{hx}", flush=True)
        # Misma analisis de estructura interna
        hist_x, edges_x = np.histogram(xray, bins=32, range=(0, 256))
        from scipy import ndimage
        from scipy.signal import find_peaks
        hist_xs = ndimage.gaussian_filter1d(hist_x.astype(float), 2)
        picos_x, _ = find_peaks(hist_xs, prominence=hist_xs.max()*0.02)
        print(f"  Modos del histograma de xray1: {len(picos_x)}", flush=True)
        report["G5_xray1"] = {"n_modos": int(len(picos_x))}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿REGISTRO VOLUMETRICO (estructura interna)?", flush=True)
    print("=" * 70, flush=True)
    if img1 is not None:
        n_modos_rostro = report["G1_rostro"]["n_modos"]
        print(f"  G1: el rostro tiene {n_modos_rostro} modos de densidad interna", flush=True)
    if img3 is not None:
        print(f"  G3: borde con transicion de {ancho_trans}px -> "
              f"{'PULSO CORTO (evento instantaneo)' if 0 <= ancho_trans < 10 else 'exposicion larga'}", flush=True)
    report["conclusion"] = {
        "n_modos_rostro": report.get("G1_rostro", {}).get("n_modos"),
        "ancho_borde": ancho_trans,
        "pulso_corto": bool(0 <= ancho_trans < 10),
    }

    out_json = os.path.join(OUT, "caracterizacion_evento_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
