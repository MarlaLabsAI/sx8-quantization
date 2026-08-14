"""
ANALISIS SOBRE EL CONJUNTO DE PIXELES (no promedios 1D)
========================================================
El analisis previo de periodicidad uso perfiles 1D (promedios de
filas/columnas). Aqui analizamos DIRECTAMENTE el conjunto de pixeles:

  P1. AUTOCORRELACION 2D COMPLETA: FFT 2D de la autocorrelacion de
      TODOS los pixeles -> periodicidad espacial real pixel a pixel
  P2. SEMIVARIOGRAMA RADIAL: como decae la correlacion entre pixeles
      con la distancia (estructura espacial del conjunto)
  P3. GLCM (matriz de co-ocurrencia): patrones de pares de pixeles
      vecinos -> textura estructurada?
  P4. MULTIESCALA (piramide Gaussiana): estructura de los pixeles
      a diferentes resoluciones
  P5. ENTROPIA LOCAL: mapa de entropia de los pixeles (estructura
      de informacion espacial)
  P6. FIGURA vs FONDO: mismas pruebas separando cuerpo y tela
  P7. VECINDAD: relacion de cada pixel con sus vecinos inmediatos
      (patron tipo 'celda' o digital?)

CPU (GPU ocupada). NO modifica originales.
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

def autocorrelacion_2d(img, max_lag=100):
    """Autocorrelacion 2D del conjunto de pixeles (via FFT)."""
    img_f = img.astype(np.float64)
    img_f = img_f - img_f.mean()
    # FFT
    F = np.fft.fft2(img_f)
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-12)
    h, w = ac.shape
    cy, cx = h//2, w//2
    # Perfil radial de la autocorrelacion (decaimiento con distancia)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    max_r = min(max_lag, cy, cx)
    perfil = []
    for r in range(max_r):
        mask = (dist >= r) & (dist < r+1)
        if mask.sum() > 0:
            perfil.append(float(ac[mask].mean()))
    return np.array(perfil)

def semivariograma(img, max_lag=80):
    """Semivariograma: gamma(h) = 0.5*E[(Z(x+h)-Z(x))^2]."""
    img_f = img.astype(np.float64)
    gamma = []
    # Desplazamientos horizontales
    for h_off in range(1, max_lag+1):
        diff = img_f[:, h_off:] - img_f[:, :-h_off]
        gamma.append(float(0.5 * np.mean(diff**2)))
    return np.array(gamma)

def glcm_textura(img, d=1, n_bins=32):
    """Matriz de co-ocurrencia (pares de pixeles vecinos)."""
    img_b = (img // (256 // n_bins)).astype(np.uint8)
    # Pares horizontales (derecha)
    glcm_h = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            glcm_h[i, j] = np.sum((img_b[:, :-d] == i) & (img_b[:, d:] == j))
    glcm_h = glcm_h / glcm_h.sum()
    # Metricas
    # Contraste: sum (i-j)^2 * P(i,j)
    ii, jj = np.mgrid[0:n_bins, 0:n_bins]
    contraste = float(np.sum((ii-jj)**2 * glcm_h))
    # Homogeneidad: sum P/(1+|i-j|)
    homogeneidad = float(np.sum(glcm_h / (1 + np.abs(ii-jj))))
    # Energia: sum P^2
    energia = float(np.sum(glcm_h**2))
    # Correlacion
    mi = np.sum(ii * glcm_h)
    mj = np.sum(jj * glcm_h)
    si = np.sqrt(np.sum((ii-mi)**2 * glcm_h))
    sj = np.sqrt(np.sum((jj-mj)**2 * glcm_h))
    corr = float(np.sum((ii-mi)*(jj-mj)*glcm_h) / (si*sj+1e-12))
    return {"contraste": contraste, "homogeneidad": homogeneidad,
            "energia": energia, "correlacion": corr}

def entropia_local(img, ventana=9):
    """Mapa de entropia local de Shannon."""
    img_s = cv2.resize(img, (200, 200))
    ent = np.zeros_like(img_s, dtype=np.float64)
    half = ventana // 2
    h, w = img_s.shape
    for r in range(half, h-half):
        for c in range(half, w-half):
            patch = img_s[r-half:r+half+1, c-half:c+half+1]
            hist, _ = np.histogram(patch, bins=16, range=(0, 256))
            p = hist[hist > 0] / hist.sum()
            ent[r, c] = -np.sum(p * np.log2(p))
    return ent

def multiescala_piramide(img, niveles=5):
    """Piramide Gaussiana: energia de cada nivel."""
    img_p = img.astype(np.float64)
    energias = []
    for i in range(niveles):
        energias.append(float(np.mean(img_p**2)))
        img_p = cv2.pyrDown(img_p)
    return energias

def main():
    t0 = time.time()
    report = {}

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    print("=" * 70, flush=True)
    print(f"ANALISIS SOBRE EL CONJUNTO DE PIXELES | {w}x{h}", flush=True)
    print("=" * 70, flush=True)

    # Figura vs fondo
    fig_mask = (img3 < 114)
    fondo_mask = ~fig_mask
    img_fig = np.where(fig_mask, img3, 0).astype(np.uint8)
    img_fon = np.where(fondo_mask, img3, 0).astype(np.uint8)

    # ============ P1: AUTOCORRELACION 2D ============
    print("\n[P1] AUTOCORRELACION 2D del conjunto de pixeles", flush=True)
    for nombre, img in [("completa", img3), ("figura", img_fig), ("fondo", img_fon)]:
        perfil_ac = autocorrelacion_2d(img, max_lag=60)
        # Longitud de correlacion (1/e)
        idx_1e = np.where(perfil_ac < 1/np.e)[0]
        long = idx_1e[0] if len(idx_1e) > 0 else len(perfil_ac)
        # Oscilaciones (periodicidad en la autocorrelacion)
        # Detectar si la autocorrelacion oscila (periodicidad real)
        desde = max(3, long//2)
        seg = perfil_ac[desde:desde+40]
        oscilaciones = float(np.std(seg)) if len(seg) > 5 else 0.0
        print(f"  {nombre}: long_correlacion={long}px | oscilaciones_post_corr={oscilaciones:.4f}", flush=True)
        # Si la autocorrelacion oscila tras caer, hay periodicidad real
        print(f"    Perfil AC primeros 15: {[f'{v:.3f}' for v in perfil_ac[:15]]}", flush=True)
        report[f"P1_{nombre}"] = {"long_correlacion": int(long), "perfil": perfil_ac.tolist()}

    # ============ P2: SEMIVARIOGRAMA ============
    print("\n[P2] SEMIVARIOGRAMA (correlacion entre pixeles vs distancia)", flush=True)
    for nombre, img in [("completa", img3), ("figura", img_fig), ("fondo", img_fon)]:
        gamma = semivariograma(img, max_lag=50)
        # Meseta (sill): donde gamma se estabiliza
        # Periodo: oscilaciones de gamma (si hay periodicidad, gamma oscila)
        desde = 5
        seg = gamma[desde:]
        osc = float(np.std(seg) / (np.mean(seg)+1e-9))
        # Detectar picos en gamma (periodicidad)
        from scipy.signal import find_peaks
        picos, _ = find_peaks(gamma, prominence=gamma.std()*0.3)
        print(f"  {nombre}: gamma_media={gamma.mean():.0f} | oscilacion_rel={osc:.3f} | picos={len(picos)}", flush=True)
        if len(picos) > 0:
            print(f"    Picos en lags: {picos[:10].tolist()}", flush=True)
        report[f"P2_{nombre}"] = {"gamma": gamma.tolist(), "picos": picos[:10].tolist()}

    # ============ P3: GLCM ============
    print("\n[P3] GLCM (textura de pares de pixeles vecinos)", flush=True)
    for nombre, img in [("completa", img3), ("figura", img_fig), ("fondo", img_fon)]:
        g = glcm_textura(img)
        print(f"  {nombre}: contraste={g['contraste']:.3f} | homogeneidad={g['homogeneidad']:.3f} | "
              f"energia={g['energia']:.4f} | correlacion={g['correlacion']:+.3f}", flush=True)
        report[f"P3_{nombre}"] = g

    # ============ P4: MULTIESCALA ============
    print("\n[P4] PIRAMIDE MULTIESCALA (energia por nivel)", flush=True)
    for nombre, img in [("figura", img_fig), ("fondo", img_fon)]:
        energias = multiescala_piramide(img, niveles=5)
        ratios = [energias[i]/energias[i+1] if i+1 < len(energias) else 0 for i in range(len(energias)-1)]
        print(f"  {nombre}: energias={[f'{e:.0f}' for e in energias]} | ratios={[f'{r:.2f}' for r in ratios]}", flush=True)
        # Decaimiento regular = estructura auto-similar
        report[f"P4_{nombre}"] = {"energias": energias, "ratios": ratios}

    # ============ P5: ENTROPIA LOCAL ============
    print("\n[P5] ENTROPIA LOCAL (mapa de informacion)", flush=True)
    for nombre, img in [("figura", img_fig), ("fondo", img_fon)]:
        ent = entropia_local(img, ventana=9)
        print(f"  {nombre}: entropia media={ent.mean():.3f} | std={ent.std():.3f} | max={ent.max():.3f}", flush=True)
        report[f"P5_{nombre}"] = {"media": float(ent.mean()), "std": float(ent.std()), "max": float(ent.max())}

    # ============ P7: VECINDAD ============
    print("\n[P7] RELACION PIXEL-VECINO (patron de celda?)", flush=True)
    for nombre, img in [("figura", img_fig), ("fondo", img_fon)]:
        img_f = img.astype(np.float64)
        # Diferencia con vecino derecho e inferior
        diff_h = np.abs(np.diff(img_f, axis=1))
        diff_v = np.abs(np.diff(img_f, axis=0))
        # Distribucion de diferencias: si es 'digital', habria picos en 0 y valores grandes
        # (transiciones binarias). Si es continua, distribucion suave.
        hist_h, edges = np.histogram(diff_h, bins=20, range=(0, 100))
        hist_h = hist_h / hist_h.sum()
        # Fraccion de diferencias ~0 (pixeles iguales)
        frac_cero = float((diff_h < 1).mean())
        # Fraccion de diferencias grandes (>50, transiciones duras)
        frac_grande = float((diff_h > 50).mean())
        print(f"  {nombre}: dif<1 (iguales)={frac_cero*100:.1f}% | dif>50 (transiciones)={frac_grande*100:.1f}%", flush=True)
        report[f"P7_{nombre}"] = {"frac_iguales": frac_cero, "frac_transiciones": frac_grande}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ESTRUCTURA DEL CONJUNTO DE PIXELES", flush=True)
    print("=" * 70, flush=True)
    out_json = os.path.join(OUT, "pixeles_estructura_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Guardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
