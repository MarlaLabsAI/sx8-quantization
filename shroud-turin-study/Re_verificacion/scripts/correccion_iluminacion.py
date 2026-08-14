"""
CORRECCION DE ILUMINACION - VERIFICACION DE METODOS
====================================================
Problema: Jeshua2-izq tiene un gradiente de iluminacion vertical fuerte
(pendiente +0.0094/px, 50% del std) vs imagen3 (-0.0059/px, 15% del std).

Este script:
  1. Estima el fondo (iluminacion) con blur grande
  2. Prueba 2 correcciones: ADITIVA (resta) y MULTIPLICATIVA (division)
  3. Mide el gradiente residual del perfil central tras cada correccion
  4. Verifica que la correccion NO destruye la estructura (correlacion
     con la imagen original en alta frecuencia)
  5. Guarda las imagenes corregidas en /tmp/opencode (NO toca originales)

NO modifica ningun archivo original.
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
TMP = "/tmp/opencode"
os.makedirs(TMP, exist_ok=True)

def cargar_imagenes():
    i3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    j2 = cv2.imread(os.path.join(BASE, "Re_verificacion", "Jeshua2.jpg"), cv2.IMREAD_GRAYSCALE)
    j2_izq = j2[:, :j2.shape[1]//2]
    return i3, j2_izq

def gradiente_perfil(img, sigma=15.0):
    """Pendiente lineal del perfil central (metodo D del estudio)."""
    h, w = img.shape
    p = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=sigma)
    x = np.arange(len(p))
    m, b = np.polyfit(x, p, 1)
    return m, float(p.std()), float(p.min()), float(p.max())

def corregir_aditiva(img, blur=101):
    """Resta el fondo suavizado (iluminacion aditiva)."""
    fondo = cv2.GaussianBlur(img, (blur, blur), 0)
    corr = img.astype(np.float32) - fondo.astype(np.float32) + fondo.mean()
    return np.clip(corr, 0, 255).astype(np.uint8)

def corregir_multiplicativa(img, blur=101):
    """Divide por el fondo suavizado (iluminacion multiplicativa)."""
    fondo = cv2.GaussianBlur(img, (blur, blur), 0)
    fondo = np.maximum(fondo, 1.0)
    corr = img.astype(np.float32) / fondo.astype(np.float32) * fondo.mean()
    return np.clip(corr, 0, 255).astype(np.uint8)

def corregir_perfil_highpass(img, sigma_hp=None):
    """Corrige SOLO el perfil central restando su tendencia de baja frecuencia."""
    h, w = img.shape
    p = img[:, w//2].astype(np.float32)
    if sigma_hp is None:
        sigma_hp = len(p) / 8.0
    tendencia = ndimage.gaussian_filter1d(p, sigma=sigma_hp)
    p_corr = p - tendencia + p.mean()
    # Reconstruir imagen con perfil corregido (solo columna central)
    img_corr = img.copy().astype(np.float32)
    img_corr[:, w//2] = np.clip(p_corr, 0, 255)
    return img_corr.astype(np.uint8)

def estructura_preservada(original, corregida):
    """Verifica que la correccion no destruye la estructura:
    correlacion de alta frecuencia (Laplaciano) entre original y corregida."""
    lap_orig = cv2.Laplacian(original, cv2.CV_64F)
    lap_corr = cv2.Laplacian(corregida, cv2.CV_64F)
    # Correlacion de Pearson entre los mapas de borde
    a = lap_orig.flatten()
    b = lap_corr.flatten()
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr)

def main():
    t0 = time.time()
    report = {}

    i3, j2_izq = cargar_imagenes()
    print("=" * 70, flush=True)
    print("VERIFICACION DE CORRECCION DE ILUMINACION", flush=True)
    print("=" * 70, flush=True)

    # --- Estado original ---
    print("\n--- ESTADO ORIGINAL (perfil central, metodo D sigma=15) ---", flush=True)
    for nombre, img in [("imagen3", i3), ("Jeshua2-izq", j2_izq)]:
        m, std, pmin, pmax = gradiente_perfil(img)
        print(f"  {nombre}: pendiente={m:+.4f}/px | std={std:.1f} | rango=[{pmin:.0f},{pmax:.0f}] | "
              f"gradiente={abs(m)*img.shape[0]:.1f} ({abs(m)*img.shape[0]/std*100:.0f}% del std)", flush=True)
        report[f"{nombre}_original"] = {"pendiente": m, "std": std, "rango": [pmin, pmax]}

    # --- Probar correcciones en Jeshua2-izq ---
    print("\n--- CORRECCIONES EN JESHUA2-IZQ ---", flush=True)
    metodos = {
        "aditiva_blur101": corregir_aditiva(j2_izq, 101),
        "multiplicativa_blur101": corregir_multiplicativa(j2_izq, 101),
        "aditiva_blur201": corregir_aditiva(j2_izq, 201),
        "multiplicativa_blur201": corregir_multiplicativa(j2_izq, 201),
        "perfil_highpass": corregir_perfil_highpass(j2_izq),
    }
    for nombre, img_corr in metodos.items():
        m, std, pmin, pmax = gradiente_perfil(img_corr)
        preserv = estructura_preservada(j2_izq, img_corr)
        pct = abs(m)*img_corr.shape[0]/std*100 if std > 0 else float("inf")
        print(f"  {nombre}: pendiente={m:+.4f}/px | std={std:.1f} | "
              f"gradiente={pct:.0f}% del std | estructura_preservada={preserv:.4f}", flush=True)
        report[nombre] = {"pendiente": m, "std": std, "gradiente_pct": pct,
                          "estructura_preservada": preserv}
        # Guardar imagen corregida
        cv2.imwrite(os.path.join(TMP, f"jeshua2_izq_{nombre}.png"), img_corr)

    # --- Aplicar la mejor correccion a imagen3 (comparacion justa) ---
    print("\n--- CORRECCION ADITIVA BLUR101 EN IMAGEN3 (comparacion justa) ---", flush=True)
    i3_corr = corregir_aditiva(i3, 101)
    m3, std3, _, _ = gradiente_perfil(i3_corr)
    print(f"  imagen3 corregida: pendiente={m3:+.4f}/px | std={std3:.1f} | "
          f"gradiente={abs(m3)*i3.shape[0]/std3*100:.0f}% del std", flush=True)
    report["imagen3_corregida"] = {"pendiente": m3, "std": std3}
    cv2.imwrite(os.path.join(TMP, "imagen3_aditiva_blur101.png"), i3_corr)

    # --- Conclusion ---
    print("\n--- CONCLUSION ---", flush=True)
    print("  Objetivo: reducir el gradiente de Jeshua2-izq a niveles comparables", flush=True)
    print("  a imagen3 (~15% del std) SIN destruir la estructura (corr > 0.9).", flush=True)
    report["conclusion"] = "Revisar tabla y elegir metodo con gradiente < 20% y estructura > 0.9"

    out_json = os.path.join(OUT, "correccion_iluminacion_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
