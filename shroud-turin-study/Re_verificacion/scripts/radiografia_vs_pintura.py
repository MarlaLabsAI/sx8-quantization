"""
VERIFICACION: ¿LA IMAGEN ES CONSISTENTE CON PROYECCION (RADIOGRAFIA) O PINTURA?
===============================================================================
Hipotesis del usuario: la imagen no fue pintada; es como una radiografia
(proyeccion de un volumen 3D sobre el plano).

Firmas medibles en el mapa de bits:
  A. RELACION INTENSIDAD-GRADIENTE (Z vs |grad Z|):
     - Radiografia: I ∝ espesor integrado. El gradiente |∇I| es la pendiente
       del espesor. En un cuerpo 3D suave, |∇I| crece con I hasta un maximo
       y luego decae (forma de campana) - la relacion es CONTINUA y suave.
     - Pintura: bordes duros de pincelada -> gradientes altos concentrados
       en pocas intensidades, relacion DISCONTINUA.
  B. DISTRIBUCION DE GRADIENTES (bordes duros vs penumbra):
     - Radiografia: gradientes distribuidos suavemente (muchos valores medios)
     - Pintura: pico de gradientes muy altos (bordes) + fondo plano
  C. SUAVIDAD DEL CAMPO (curvatura):
     - Radiografia: curvatura media pequena y continua
     - Pintura: curvatura alta en bordes de pincelada
  D. COMPARACION CON CONTROLES:
     - Pintura simulada (bordes duros, manchas con contorno definido)
     - Radiografia simulada (proyeccion de esfera/elipsoide con penumbra)
     - Ruido

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N_CONTROLS = 30
N_WORKERS = 12

# ============================================================================
# 1. METRICAS DE FIRMA
# ============================================================================
def relacion_intensidad_gradiente(img_norm, n_bins=40):
    """I vs |∇I|: gradiente medio por bin de intensidad."""
    dx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(img_norm, bins)
    grad_por_bin = []
    for i in range(1, n_bins + 1):
        mask = idx == i
        if mask.sum() > 100:
            grad_por_bin.append(float(grad[mask].mean()))
        else:
            grad_por_bin.append(float("nan"))
    # Suavidad de la relacion: si es continua (radiografia) el perfil es suave
    vals = np.array([g for g in grad_por_bin if not np.isnan(g)])
    if len(vals) < 5:
        return {"perfil": grad_por_bin, "suavidad": float("nan"), "monotonia": float("nan")}
    # Suavidad = 1 - (variacion de vecinos normalizada)
    diffs = np.abs(np.diff(vals))
    suavidad = 1.0 - np.mean(diffs) / (np.max(vals) - np.min(vals) + 1e-9)
    # Monotonia: fraccion de pasos en la misma direccion
    signos = np.sign(np.diff(vals))
    monotonia = float(np.mean(signos == signos[0])) if len(signos) > 0 else float("nan")
    return {"perfil": grad_por_bin, "suavidad": float(suavidad), "monotonia": monotonia}

def distribucion_gradientes(img_norm):
    """Histograma de |∇I|: pico de bordes duros vs distribucion suave."""
    dx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dx**2 + dy**2)
    hist, edges = np.histogram(grad, bins=50, range=(0, grad.max() * 1.01))
    hist = hist / hist.sum()
    # Fraccion de gradientes muy altos (bordes duros) - top 5%
    top5 = np.percentile(grad, 95)
    frac_bordes = float((grad > top5).mean())
    # Kurtosis de la distribucion (picos = pintura, plana = radiografia)
    kurt = float(((grad - grad.mean())**4).mean() / grad.std()**4) if grad.std() > 0 else float("nan")
    return {"hist": hist.tolist(), "frac_bordes_top5": frac_bordes, "kurtosis": kurt,
            "grad_medio": float(grad.mean()), "grad_p95": float(top5)}

def curvatura_media(img_norm):
    """Curvatura media del campo (suavidad global)."""
    img_s = cv2.GaussianBlur(img_norm, (5, 5), 0)
    dx = cv2.Sobel(img_s, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_s, cv2.CV_64F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
    H = (dxx + dyy) / 2.0
    return {"H_mean": float(np.abs(H).mean()), "H_std": float(H.std())}

def firma_completa(img_norm):
    return {
        "relacion_IG": relacion_intensidad_gradiente(img_norm),
        "gradientes": distribucion_gradientes(img_norm),
        "curvatura": curvatura_media(img_norm),
    }

# ============================================================================
# 2. CONTROLES
# ============================================================================
def pintura_simulada(shape, n_manchas=40):
    """Pintura simulada: manchas con bordes duros (pinceladas)."""
    h, w = shape
    img = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_manchas):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        rx, ry = rng.integers(20, 120), rng.integers(20, 120)
        ang = rng.uniform(0, np.pi)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        yy, xx = np.mgrid[0:h, 0:w]
        xr = (xx - cx) * cos_a + (yy - cy) * sin_a
        yr = -(xx - cx) * sin_a + (yy - cy) * cos_a
        mask = (xr / rx)**2 + (yr / ry)**2 < 1
        val = rng.uniform(0.2, 0.9)
        img[mask] = val
    # Borde duro: sin suavizado
    return np.clip(img, 0, 1)

def radiografia_simulada(shape, n_esferas=3):
    """Radiografia simulada: proyeccion de esferas/elipsoides con penumbra.
    I = suma de espesores proyectados (suave, continuo)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float64)
    for _ in range(n_esferas):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        r = rng.integers(80, 250)
        # Espesor de esfera: 2*sqrt(r^2 - d^2) (proyeccion de volumen)
        d = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        espesor = np.clip(2 * np.sqrt(np.maximum(r**2 - d**2, 0)), 0, None)
        img += espesor * rng.uniform(0.3, 1.0)
    img = img / img.max()
    return img

def ruido_suave(shape, sigma=3.0):
    h, w = shape
    noise = rng.normal(0, 1, size=(h, w))
    smooth = cv2.GaussianBlur(noise, (0, 0), sigma)
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth

def worker_firma(args):
    img_norm, nombre = args
    return {"nombre": nombre, "firma": firma_completa(img_norm)}

# ============================================================================
# 3. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"real": {}, "controles": {}, "conclusion": {}}

    # Rostro real (imagen1 crop 1000x1000)
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    face = img1[100:1100, 1000:2000]
    face_norm = face.astype(float) / 255.0
    print("=" * 70, flush=True)
    print("FIRMA DEL ROSTRO REAL (imagen1, crop 1000x1000)", flush=True)
    print("=" * 70, flush=True)

    f_real = firma_completa(face_norm)
    report["real"] = f_real
    print(f"  Relacion I-|grad|: suavidad={f_real['relacion_IG']['suavidad']:.3f} | monotonia={f_real['relacion_IG']['monotonia']:.3f}", flush=True)
    print(f"  Gradientes: kurtosis={f_real['gradientes']['kurtosis']:.1f} | grad_medio={f_real['gradientes']['grad_medio']:.5f} | p95={f_real['gradientes']['grad_p95']:.5f}", flush=True)
    print(f"  Curvatura: |H|_media={f_real['curvatura']['H_mean']:.6f}", flush=True)

    # Controles
    print(f"\nCONTROLES ({N_CONTROLS} x 3 tipos):", flush=True)
    tipos = ["pintura", "radiografia", "ruido_suave"]
    tasks = []
    for tipo in tipos:
        for i in range(N_CONTROLS):
            if tipo == "pintura":
                img_c = pintura_simulada(face_norm.shape)
            elif tipo == "radiografia":
                img_c = radiografia_simulada(face_norm.shape)
            else:
                img_c = ruido_suave(face_norm.shape)
            tasks.append((img_c, f"{tipo}_{i}"))
    with Pool(N_WORKERS) as pool:
        res = list(pool.imap_unordered(worker_firma, tasks))

    for tipo in tipos:
        grupo = [r for r in res if r["nombre"].startswith(tipo)]
        suav = [r["firma"]["relacion_IG"]["suavidad"] for r in grupo]
        kurt = [r["firma"]["gradientes"]["kurtosis"] for r in grupo]
        gmed = [r["firma"]["gradientes"]["grad_medio"] for r in grupo]
        curv = [r["firma"]["curvatura"]["H_mean"] for r in grupo]
        print(f"  [{tipo}] suavidad_IG={np.mean(suav):.3f}±{np.std(suav):.3f} | kurtosis={np.mean(kurt):.1f}±{np.std(kurt):.1f} | "
              f"grad_medio={np.mean(gmed):.5f}±{np.std(gmed):.5f} | |H|={np.mean(curv):.6f}±{np.std(curv):.6f}", flush=True)
        report["controles"][tipo] = {
            "suavidad_IG_mean": float(np.mean(suav)), "suavidad_IG_std": float(np.std(suav)),
            "kurtosis_mean": float(np.mean(kurt)), "kurtosis_std": float(np.std(kurt)),
            "grad_medio_mean": float(np.mean(gmed)), "grad_medio_std": float(np.std(gmed)),
            "curvatura_mean": float(np.mean(curv)), "curvatura_std": float(np.std(curv)),
        }

    # Conclusion
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    # Comparar con pintura y radiografia
    for metrica, real_val, key in [
        ("suavidad_IG", f_real["relacion_IG"]["suavidad"], "suavidad_IG_mean"),
        ("kurtosis", f_real["gradientes"]["kurtosis"], "kurtosis_mean"),
        ("grad_medio", f_real["gradientes"]["grad_medio"], "grad_medio_mean"),
        ("curvatura", f_real["curvatura"]["H_mean"], "curvatura_mean"),
    ]:
        for tipo in ["pintura", "radiografia"]:
            c_mean = report["controles"][tipo][key]
            c_std = report["controles"][tipo][key.replace("mean", "std")]
            z = (real_val - c_mean) / c_std if c_std > 0 else float("nan")
            print(f"  {metrica}: real={real_val:.5f} vs {tipo}={c_mean:.5f}±{c_std:.5f} (z={z:+.2f})", flush=True)
    report["conclusion"] = {
        "real": {"suavidad_IG": f_real["relacion_IG"]["suavidad"], "kurtosis": f_real["gradientes"]["kurtosis"],
                 "grad_medio": f_real["gradientes"]["grad_medio"], "curvatura": f_real["curvatura"]["H_mean"]},
        "nota": "z pequeno = indistinguible del control; z grande = mas cercano a ese proceso"
    }

    out_json = os.path.join(OUT, "radiografia_vs_pintura_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
