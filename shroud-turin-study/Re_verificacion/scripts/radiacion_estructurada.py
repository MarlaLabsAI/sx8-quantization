"""
ANALISIS DE LA RADIACION ESTRUCTURADA: ¿DIGITAL + FRACTAL?
==========================================================
Hipotesis del investigador: la radiacion que produjo el registro esta
ESTRUCTURADA — es como digital y a la vez tiene un patron fractal.
Esto seria tecnologia mucho mas superior a la nuestra.

Una senal NATURAL (foto, pintura, radiacion difusa) es:
  - Continua (histograma suave, sin niveles discretos)
  - No cuantizada (gradaciones infinitas)
  - No periodicidad en los niveles

Una senal DIGITAL (PNG, JPEG, datos) es:
  - Cuantizada (niveles discretos)
  - Periodica en los niveles (saltos regulares)
  - Entropia de niveles distinta a una senal continua

Un FRACTAL natural es:
  - Auto-similar a multiples escalas
  - Pero NO cuantizado digitalmente

Tests:
  H1. CUANTIZACION: ¿el histograma del registro tiene NIVELES DISCRETOS
      (picos separados) o es continuo? Comparar con xray1 (analogico)
      y con una foto natural.
  H2. PERIODICIDAD DE NIVELES: ¿los picos del histograma estan separados
      regularmente (saltos cuantizados) como una senal digital?
  H3. ESTRUCTURA DE BITPLANES: ¿los bitplanes altos (MSB) tienen
      estructura no trivial? (ya vimos: bit7 D=1.79, no ruido)
  H4. COMBINACION DIGITAL+FRACTAL: cuantizacion detectable Y fractalidad
      simultanea. La combinacion es lo inusual.
  H5. COMPARACION CON XRAY1 (analogico) y con la IMAGEN REAL del rostro.

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

def cuantizacion(imagen, n_bins=256):
    """Detecta niveles discretos en el histograma.
    Una senal cuantizada tiene picos SEPARADOS (modos con valles entre ellos).
    Una senal continua tiene histograma suave sin valles profundos."""
    hist, edges = np.histogram(imagen, bins=n_bins, range=(0, 256))
    hist_s = ndimage.gaussian_filter1d(hist.astype(float), 2)
    # Normalizar
    hist_s = hist_s / hist_s.sum()
    # Picos
    picos, props = find_peaks(hist_s, prominence=hist_s.max() * 0.01)
    # Valles entre picos (separacion de niveles)
    if len(picos) > 1:
        valles = []
        for i in range(len(picos)-1):
            entre = hist_s[picos[i]:picos[i+1]]
            if len(entre) > 0:
                valles.append(entre.min())
        # Profundidad de los valles relativa a los picos
        profundidad = np.mean([1 - v/(hist_s[picos[i]]+1e-9) for i, v in enumerate(valles)])
    else:
        profundidad = 0.0
    # Periodicidad de los picos (separacion regular = cuantizacion)
    if len(picos) > 2:
        separaciones = np.diff(picos)
        cv_sep = separaciones.std() / (separaciones.mean() + 1e-9)
    else:
        cv_sep = float("nan")
    return {
        "n_picos": int(len(picos)),
        "posiciones_picos": [int(edges[p]) for p in picos],
        "profundidad_valles": float(profundidad),
        "cv_separacion_picos": float(cv_sep),
    }

def entropia_niveles(imagen, n_bins=64):
    """Entropia de Shannon de la distribucion de intensidades."""
    hist, _ = np.histogram(imagen, bins=n_bins, range=(0, 256))
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def box_counting_dimension(binary, min_size=2, max_size=None):
    h, w = binary.shape
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
        blocks = binary[:nh*s, :nw*s].reshape(nh, s, nw, s)
        counts.append((blocks.sum(axis=(1,3)) > 0).sum())
    counts = np.array(counts)
    valid = counts > 0
    if valid.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(1.0/sizes[valid]), np.log(counts[valid]), 1)
    return float(slope)

def fractalidad_multiescala(imagen, escalas=(50, 25, 10)):
    """D fractal a multiples escalas (auto-similitud)."""
    img_u8 = imagen.astype(np.uint8)
    _, bits = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    Ds = []
    for s in escalas:
        h, w = bits.shape
        y0, y1 = max(0, h//2 - s), min(h, h//2 + s)
        x0, x1 = max(0, w//2 - s), min(w, w//2 + s)
        region = bits[y0:y1, x0:x1]
        if region.size > 0:
            Ds.append(box_counting_dimension(region))
    Ds = np.array([d for d in Ds if not np.isnan(d)])
    if len(Ds) < 2 or Ds.mean() == 0:
        return {"D_media": float("nan"), "CV": float("nan")}
    return {"D_media": float(Ds.mean()), "CV": float(Ds.std()/Ds.mean())}

def main():
    t0 = time.time()
    report = {}

    print("=" * 70, flush=True)
    print("ANALISIS: ¿RADIACION ESTRUCTURADA (DIGITAL + FRACTAL)?", flush=True)
    print("=" * 70, flush=True)

    # Cargar imagenes
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    xray = cv2.imread(os.path.join(BASE, "Re_verificacion", "xray1.avif"), cv2.IMREAD_GRAYSCALE)

    # Referencia: foto natural (para comparar cuantizacion)
    # Usar una region de la sabana que sea SOLO tela (fondo) como control
    # y una region de imagen natural si existe
    print("\n[H1/H2] CUANTIZACION DE NIVELES (discretos vs continuos)", flush=True)
    muestras = {}
    if img3 is not None:
        # La imagen completa de la sabana
        muestras["sabana_completa"] = img3
        # Rostro
    if img1 is not None:
        muestras["rostro_negativo"] = img1[100:1100, 1000:2000]
    if xray is not None:
        muestras["xray1_radiografia"] = cv2.resize(xray, (1000, 1000))

    resultados = {}
    for nombre, img in muestras.items():
        q = cuantizacion(img)
        ent = entropia_niveles(img)
        frac = fractalidad_multiescala(img)
        resultados[nombre] = {"cuantizacion": q, "entropia_niveles": ent, "fractalidad": frac}
        print(f"\n  {nombre}:", flush=True)
        print(f"    Picos histograma: {q['n_picos']} en {q['posiciones_picos'][:10]}", flush=True)
        print(f"    Profundidad valles: {q['profundidad_valles']:.3f} | CV separacion picos: {q['cv_separacion_picos']:.3f}", flush=True)
        print(f"    Entropia de niveles (64 bins): {ent:.3f} (maximo 6.0)", flush=True)
        print(f"    Fractalidad: D_media={frac['D_media']:.3f} | CV={frac['CV']:.4f}", flush=True)
    report["H1_H2_cuantizacion"] = resultados

    # ============ H3: BITPLANES DEL ROSTRO (estructura MSB) ============
    print("\n[H3] BITPLANES DEL ROSTRO: estructura de cada bit", flush=True)
    if img1 is not None:
        face = img1[100:1100, 1000:2000]
        face_u8 = face.astype(np.uint8)
        for b in range(8):
            plane = ((face_u8 >> b) & 1).astype(np.uint8)
            D = box_counting_dimension(plane)
            dens = plane.mean()
            print(f"  bit{b}: densidad={dens:.4f} | D_fractal={D:.3f} "
                  f"({'ESTRUCTURA' if abs(D - 2.0) > 0.05 and dens > 0.01 else 'ruido'})", flush=True)
        report["H3_bitplanes"] = {
            b: {"densidad": float(((face_u8 >> b) & 1).mean()),
                "D_fractal": box_counting_dimension(((face_u8 >> b) & 1).astype(np.uint8))}
            for b in range(8)
        }

    # ============ H4: COMBINACION DIGITAL + FRACTAL ============
    print("\n[H4] COMBINACION DIGITAL + FRACTAL", flush=True)
    for nombre, r in resultados.items():
        q = r["cuantizacion"]
        f = r["fractalidad"]
        # Digital: picos separados con valles profundos (cuantizacion)
        digital = q["n_picos"] > 3 and q["profundidad_valles"] > 0.2
        # Fractal: CV < 0.2 (auto-similitud)
        fractal = f["CV"] < 0.2 and not np.isnan(f["CV"])
        combo = digital and fractal
        print(f"  {nombre}: digital={digital} (picos={q['n_picos']}, valles={q['profundidad_valles']:.2f}) | "
              f"fractal={fractal} (CV={f['CV']:.4f}) | COMBINACION={combo}", flush=True)
        r["combinacion_digital_fractal"] = {"digital": bool(digital), "fractal": bool(fractal),
                                             "combinacion": bool(combo)}
    report["H4_combinacion"] = {k: v["combinacion_digital_fractal"] for k, v in resultados.items()}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿SENAL ESTRUCTURADA (DIGITAL+FRACTAL)?", flush=True)
    print("=" * 70, flush=True)
    for nombre, r in resultados.items():
        combo = r.get("combinacion_digital_fractal", {})
        print(f"  {nombre}: digital={combo.get('digital')} + fractal={combo.get('fractal')} = {combo.get('combinacion')}", flush=True)
    report["conclusion"] = {k: v.get("combinacion_digital_fractal", {}) for k, v in resultados.items()}

    out_json = os.path.join(OUT, "radiacion_estructurada_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
