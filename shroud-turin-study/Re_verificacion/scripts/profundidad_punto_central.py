"""
PROFUNDIDAD DEL PUNTO CENTRAL Y DE LA CRUZ
==========================================
Idea del usuario: el punto central NO es un punto plano 2D — tiene
PROFUNDIDAD (estructura interna en una tercera dimension). Quizas la
cruz tambien, pero el punto central mas.

Interpretaciones de "profundidad":
  P1. VECTOR DE RECURRENCIA del punto central (fila 416 de la matriz):
      cuantas zonas del perfil recurren con el punto -> bloques de
      recurrencia = "profundidad" (cuanta informacion del perfil pasa
      por el punto). Comparar con otros puntos de la matriz.
  P2. BITPLANES COMO CAPAS: el mapa de bits apilado (8 bitplanes) forma
      un VOLUMEN 3D. La "profundidad" del punto central = cuantos bits
      estan activos en cada (x,y) -> el punto central deberia tener
      mas profundidad (mas capas activas).
  P3. TOPOGRAFIA DEL PUNTO: el relieve (densidad de bits suavizada) en
      el punto central vs periferia.
  P4. LA CRUZ COMO VOLUMEN: la cruz en el volumen de bitplanes (cortes
      verticales/horizontales).
  P5. COMPARACION: punto central vs otros puntos de la matriz (misma
      profundidad? o el central es especial?).

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
rng = np.random.default_rng(42)

def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    return R, profile, img3

def bloques_recurrencia(fila):
    """Bloques contiguos de 1s en una fila."""
    bloques = []
    en_bloque = False
    inicio = 0
    n = len(fila)
    for j in range(n):
        if fila[j] == 1 and not en_bloque:
            en_bloque = True; inicio = j
        elif fila[j] == 0 and en_bloque:
            en_bloque = False
            bloques.append((inicio, j-1))
    if en_bloque:
        bloques.append((inicio, n-1))
    return bloques

def profundidad_punto(R, i, j):
    """Profundidad del punto (i,j): bloques de recurrencia de fila y columna."""
    fila = R[i, :]
    col = R[:, j]
    bf = bloques_recurrencia(fila)
    bc = bloques_recurrencia(col)
    return {
        "n_bloques_fila": len(bf),
        "n_bloques_col": len(bc),
        "suma_bloques_fila": sum(b[1]-b[0]+1 for b in bf),
        "suma_bloques_col": sum(b[1]-b[0]+1 for b in bc),
        "densidad_fila": float(fila.mean()),
        "densidad_col": float(col.mean()),
        "profundidad_total": len(bf) + len(bc),
        "bloques_fila": [(int(a), int(b)) for a, b in bf],
        "bloques_col": [(int(a), int(b)) for a, b in bc],
    }

def main():
    t0 = time.time()
    report = {"P1_vector_recurrencia": {}, "P2_bitplanes_volumen": {},
              "P3_topografia": {}, "P5_comparacion": {}, "conclusion": {}}

    R, profile, img3 = matriz_real()
    n = R.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"MATRIZ: {n}x{n} | punto central=({cx},{cy})", flush=True)
    print("=" * 70, flush=True)

    # ============ P1: VECTOR DE RECURRENCIA ============
    print("\n[P1] VECTOR DE RECURRENCIA DEL PUNTO CENTRAL (profundidad)", flush=True)
    prof_cruz = profundidad_punto(R, cx, cy)
    print(f"  Punto central ({cx},{cy}):", flush=True)
    print(f"    Bloques fila: {prof_cruz['n_bloques_fila']} | bloques col: {prof_cruz['n_bloques_col']}", flush=True)
    print(f"    Profundidad total: {prof_cruz['profundidad_total']}", flush=True)
    print(f"    Densidad fila: {prof_cruz['densidad_fila']:.4f} | col: {prof_cruz['densidad_col']:.4f}", flush=True)
    print(f"    Bloques fila: {prof_cruz['bloques_fila']}", flush=True)
    report["P1_vector_recurrencia"]["punto_central"] = prof_cruz

    # ============ P5: COMPARACION CON OTROS PUNTOS ============
    print("\n[P5] COMPARACION: profundidad del punto central vs otros puntos", flush=True)
    # Puntos de referencia: diagonal (540,540), esquinas, puntos aleatorios fuera de la cruz
    puntos_ref = {
        "diagonal_540": (540, 540),
        "esquina_0": (0, 0),
        "esquina_1000": (1000, 1000),
        "aleatorio_1": (200, 700),
        "aleatorio_2": (800, 300),
    }
    profundidades = {}
    for nombre, (pi, pj) in puntos_ref.items():
        p = profundidad_punto(R, pi, pj)
        profundidades[nombre] = p
        print(f"  {nombre} ({pi},{pj}): profundidad={p['profundidad_total']} | "
              f"fila={p['n_bloques_fila']} bloques | col={p['n_bloques_col']} bloques", flush=True)
    # Barrido: profundidad media de puntos aleatorios
    n_aleatorios = 100
    profs_aleatorias = []
    for _ in range(n_aleatorios):
        pi = rng.integers(50, n-50)
        pj = rng.integers(50, n-50)
        p = profundidad_punto(R, pi, pj)
        profs_aleatorias.append(p["profundidad_total"])
    profs_aleatorias = np.array(profs_aleatorias)
    z = (prof_cruz["profundidad_total"] - profs_aleatorias.mean()) / profs_aleatorias.std()
    print(f"  Puntos aleatorios: profundidad media={profs_aleatorias.mean():.1f}±{profs_aleatorias.std():.1f}", flush=True)
    print(f"  Punto central: profundidad={prof_cruz['profundidad_total']} (z={z:+.2f})", flush=True)
    report["P5_comparacion"] = {
        "puntos_ref": profundidades,
        "aleatorios_mean": float(profs_aleatorias.mean()),
        "aleatorios_std": float(profs_aleatorias.std()),
        "cruz_profundidad": prof_cruz["profundidad_total"],
        "z_cruz": float(z),
    }

    # ============ P2: BITPLANES COMO VOLUMEN ============
    print("\n[P2] BITPLANES COMO VOLUMEN 3D (profundidad = capas activas)", flush=True)
    # Recortar imagen a 1080x1080 centrada en el perfil
    h_img, w_img = img3.shape
    x0 = w_img//2 - n//2
    img_cuadrada = img3[:, x0:x0+n]
    img_u8 = img_cuadrada.astype(np.uint8)
    planes = [((img_u8 >> b) & 1).astype(np.uint8) for b in range(8)]
    volumen = np.stack(planes, axis=-1)  # (1080,1080,8) - profundidad = 8 capas
    print(f"  Volumen de bitplanes: {volumen.shape}", flush=True)
    # Profundidad del punto central en el volumen: cuantos bits activos en (x,y)
    # El punto central de la matriz (416,416) -> en la imagen cuadrada, (416, 416+x0)?
    # La matriz de recurrencia se construye del perfil (columna central). El punto (416,416)
    # de la matriz corresponde a y=416 del perfil. En la imagen cuadrada, la columna central
    # es x = n//2 = 540. El punto (416, 540) en la imagen cuadrada.
    py_img = 416
    px_img = n//2
    profundidad_bit = int(volumen[py_img, px_img, :].sum())
    print(f"  Punto central (y=416, x=540 en imagen): profundidad de bits = {profundidad_bit}/8", flush=True)
    # Profundidad media de todos los puntos
    prof_media_bits = float(volumen.sum(axis=-1).mean())
    print(f"  Profundidad media de bits (toda la imagen): {prof_media_bits:.2f}/8", flush=True)
    # Perfil de profundidad a lo largo de y (columna x=540)
    col_prof = volumen[:, px_img, :].sum(axis=1)
    print(f"  Profundidad en y=416: {col_prof[416]}/8 | media columna: {col_prof.mean():.2f}/8", flush=True)
    # La cruz en el volumen: cortes
    print(f"  Corte vertical (x=540) en y=416: bits={volumen[416, 540, :].tolist()}", flush=True)
    report["P2_bitplanes_volumen"] = {
        "profundidad_punto_central": int(profundidad_bit),
        "profundidad_media_imagen": float(prof_media_bits),
        "profundidad_col_416": int(col_prof[416]),
        "profundidad_media_col": float(col_prof.mean()),
    }

    # ============ P3: TOPOGRAFIA DEL PUNTO ============
    print("\n[P3] TOPOGRAFIA (relieve) DEL PUNTO CENTRAL", flush=True)
    _, bits = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    topo = cv2.GaussianBlur((bits > 0).astype(np.float64), (0, 0), 5)
    topo = topo / (topo.max() + 1e-12)
    # Altura en el punto central vs media
    altura_punto = float(topo[416, px_img])
    altura_media = float(topo.mean())
    print(f"  Altura topografica en punto central: {altura_punto:.3f} | media: {altura_media:.3f} | ratio: {altura_punto/(altura_media+1e-9):.2f}", flush=True)
    report["P3_topografia"] = {
        "altura_punto": altura_punto, "altura_media": altura_media,
        "ratio": float(altura_punto/(altura_media+1e-9)),
    }

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: PROFUNDIDAD DEL PUNTO CENTRAL", flush=True)
    print("=" * 70, flush=True)
    print(f"  P1: profundidad de recurrencia del punto central = {prof_cruz['profundidad_total']} (z={z:+.2f} vs aleatorios)", flush=True)
    print(f"  P2: profundidad de bits del punto central = {profundidad_bit}/8 (media {prof_media_bits:.2f})", flush=True)
    print(f"  P3: altura topografica punto = {altura_punto:.3f} (media {altura_media:.3f})", flush=True)
    report["conclusion"] = {
        "P1_z": float(z),
        "P1_profundidad": prof_cruz["profundidad_total"],
        "P2_bits": int(profundidad_bit),
        "P2_media": float(prof_media_bits),
        "P3_altura": float(altura_punto),
    }

    out_json = os.path.join(OUT, "profundidad_punto_central_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
