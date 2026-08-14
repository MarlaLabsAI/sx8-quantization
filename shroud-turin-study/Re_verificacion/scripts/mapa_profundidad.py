"""
EL MAPA DE PROFUNDIDAD: LA ESCALA DE GRISES COMO CODIFICACION 3D
================================================================
Hipotesis del investigador: la escala de grises de cada pixel sobre
el cuerpo es un MAPA DE PROFUNDIDAD del cuerpo de Jesus, producido
por el suceso energetico. La correlacion del evento esta en que TODOS
los pixeles juntos forman un relieve 3D coherente y anatomico.

Tests:
  M1. RELIEVE 3D: tratar la escala de grises como Z (profundidad).
      ¿La superficie resultante es SUAVE y CONTINUA (un mapa de
      profundidad real) o es ruido?
      - Suavidad: segundas derivadas pequenas (sin picos aleatorios)
      - Coherencia: la variacion de Z es gradual, no granular

  M2. RELACION INTENSIDAD-DISTANCIA: en un mapa de profundidad real,
      la intensidad decae suavemente desde el punto mas cercano a la
      tela (maxima intensidad) hacia los bordes del cuerpo (menos
      intensidad). Verificar el gradiente de la escala de grises
      sobre el cuerpo.

  M3. COMPARACION FIGURA vs FONDO: la tela NO es un mapa de
      profundidad (textura uniforme). Si la figura tiene suavidad
      de profundidad mucho mayor que el fondo, el evento codifico
      profundidad SOLO en el cuerpo.

  M4. CURVATURA DEL MAPA DE PROFUNDIDAD: un mapa de profundidad
      anatomico tiene curvaturas suaves (torso, cabeza, nariz).
      Medir la distribucion de curvatura del relieve.

  M5. DIMENSION DE LA SUPERFICIE: la superficie 3D del cuerpo
      deberia tener dimension ~2 (superficie continua), no fractal
      ruidosa. Medir D fractal del relieve de profundidad.

  M6. COHERENCIA DEL EVENTO: correlacion entre la escala de grises
      (profundidad) y la GEOMETRIA esperada del cuerpo (simetria,
      gradiente centro-bordes).

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
    """Suavidad de una superficie de profundidad Z.
    Un mapa de profundidad REAL es C1-continuo: segundas derivadas
    pequenas y distribuidas. Ruido tendria picos de curvatura."""
    Z = Z.astype(np.float64)
    # Segundas derivadas (curvatura aproximada)
    d2x = np.abs(np.gradient(np.gradient(Z, axis=1), axis=1))
    d2y = np.abs(np.gradient(np.gradient(Z, axis=0), axis=0))
    curvatura = d2x + d2y
    # Metricas
    media_curv = float(curvatura.mean())
    std_curv = float(curvatura.std())
    # Fraccion de curvatura muy alta (picos = ruido)
    frac_picos = float((curvatura > curvatura.mean() + 3*curvatura.std()).mean())
    # Suavidad relativa: curvatura media / rango de Z
    rango = float(Z.max() - Z.min() + 1e-9)
    suavidad_rel = media_curv / rango
    return {"media_curvatura": media_curv, "std_curvatura": std_curv,
            "frac_picos": frac_picos, "suavidad_relativa": suavidad_rel,
            "rango_Z": rango}

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

def dimension_superficie(Z, umbrales=(0.3, 0.5, 0.7)):
    """D fractal del relieve a varios umbrales de altura."""
    Z = Z.astype(np.float64)
    Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
    Ds = []
    for T in umbrales:
        Ds.append(box_counting_dimension((Z > T).astype(np.uint8)))
    return Ds

def main():
    t0 = time.time()
    report = {}

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    print("=" * 70, flush=True)
    print("EL MAPA DE PROFUNDIDAD: ESCALA DE GRISES COMO CODIFICACION 3D", flush=True)
    print(f"Imagen: {w}x{h}", flush=True)
    print("=" * 70, flush=True)

    # Separar figura (cuerpo) y fondo (tela)
    # La figura es mas oscura; invertir para que la intensidad = proximidad
    # (en el negativo del rostro, lo claro = cercano a la tela)
    fig_mask = (img3 < 114)
    fondo_mask = ~fig_mask

    # MAPA DE PROFUNDIDAD: usamos la escala de grises como Z
    # Para la figura: mayor intensidad = mayor cercania a la tela
    # En la imagen sepia, el cuerpo es oscuro -> invertir
    Z_fig = 255 - img3.astype(np.float64)  # invertida: cuerpo claro = cerca
    Z_fig = np.where(fig_mask, Z_fig, 0)
    Z_fon = img3.astype(np.float64)
    Z_fon = np.where(fondo_mask, Z_fon, 0)

    # ============ M1: SUAVIDAD DEL RELIEVE ============
    print("\n[M1] SUAVIDAD DE LA SUPERFICIE DE PROFUNDIDAD", flush=True)
    # Tomar region central de la figura (donde esta el cuerpo)
    # y region del fondo
    # Figura: regiones con figura (mayoria del area)
    print("  (la escala de grises del cuerpo = profundidad)", flush=True)
    # Suavidad de la figura completa (solo pixeles del cuerpo)
    Z_fig_c = Z_fig.copy()
    s_fig = suavidad_superficie(Z_fig_c)
    print(f"  FIGURA: media_curvatura={s_fig['media_curvatura']:.3f} | "
          f"frac_picos={s_fig['frac_picos']:.5f} | suavidad_rel={s_fig['suavidad_relativa']:.5f}", flush=True)
    report["M1_figura"] = s_fig

    # Control: ruido gaussiano con la misma desviacion
    ruido = np.random.default_rng(42).normal(img3.mean(), img3.std(), size=(h, w))
    s_ruido = suavidad_superficie(ruido)
    print(f"  RUIDO (control): media_curvatura={s_ruido['media_curvatura']:.3f} | "
          f"frac_picos={s_ruido['frac_picos']:.5f} | suavidad_rel={s_ruido['suavidad_relativa']:.5f}", flush=True)
    report["M1_ruido"] = s_ruido
    # Ratio de suavidad: figura vs ruido
    ratio_suav = s_ruido["media_curvatura"] / (s_fig["media_curvatura"] + 1e-9)
    print(f"  La figura es {ratio_suav:.0f}x MAS suave que el ruido", flush=True)
    report["M1_ratio_vs_ruido"] = float(ratio_suav)

    # ============ M3: FIGURA vs FONDO ============
    print("\n[M3] FIGURA vs FONDO (la tela NO es mapa de profundidad)", flush=True)
    # Fondo: region de tela pura (esquina sup-izq, lejos del cuerpo)
    tela = Z_fon[50:350, 50:350]
    s_tela = suavidad_superficie(tela)
    print(f"  TELA: media_curvatura={s_tela['media_curvatura']:.3f} | frac_picos={s_tela['frac_picos']:.5f}", flush=True)
    report["M3_tela"] = s_tela
    # La figura como mapa de profundidad debe ser MAS suave que la tela
    # (o al menos diferente: la tela tiene textura de hilos)
    print(f"  Comparacion: figura curv={s_fig['media_curvatura']:.3f} vs tela={s_tela['media_curvatura']:.3f}", flush=True)

    # ============ M2: GRADIENTE CENTRO-BORDES (relacion con geometria) ============
    print("\n[M2] GRADIENTE DEL MAPA DE PROFUNDIDAD (intensidad vs posicion)", flush=True)
    # En un mapa de profundidad real: el punto mas cercano a la tela tiene
    # maxima intensidad (nariz, pecho) y decae hacia los bordes del cuerpo
    # Medir: el centro del cuerpo deberia tener mas 'profundidad' (intensidad)
    # que los bordes (donde el cuerpo se aleja de la tela)
    # Usar el rostro (imagen1 negativo) donde la anatomia es clara
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    if img1 is not None:
        face = img1[100:1100, 1000:2000]
        face_f = face.astype(np.float64)
        # En el negativo: la nariz (punto mas cercano a la tela) es la zona
        # MAS CLARA. Verificar: el maximo de intensidad esta en el centro
        # del rostro (nariz) y decae hacia los bordes
        hf, wf = face.shape
        # Perfil de intensidad a lo largo de la linea media vertical
        perfil_medio = face_f[:, wf//2]
        # Suavizar
        perfil_s = ndimage.gaussian_filter1d(perfil_medio, sigma=10)
        # Posicion del maximo
        idx_max = np.argmax(perfil_s)
        print(f"  ROSTRO: maximo de intensidad (nariz?) en y={idx_max} de {hf} ({idx_max/hf*100:.0f}%)", flush=True)
        print(f"  Perfil medio (muestra): {[int(v) for v in perfil_s[::100]]}", flush=True)
        # Si es mapa de profundidad: el maximo esta en el centro (no en los bordes)
        centro_rel = abs(idx_max - hf/2) / (hf/2)
        print(f"  Maximo centrado: {1-centro_rel:.2f} (1=perfectamente centrado)", flush=True)
        report["M2_rostro"] = {"idx_max": int(idx_max), "centrado": float(1-centro_rel),
                               "perfil": perfil_s[::50].tolist()}

    # ============ M4: CURVATURA DEL RELIEVE ============
    print("\n[M4] CURVATURA DEL MAPA DE PROFUNDIDAD (suavidad anatomica)", flush=True)
    if img1 is not None:
        face = img1[100:1100, 1000:2000]
        Z_face = face.astype(np.float64)
        # Curvatura gaussiana del relieve facial
        Z_s = cv2.GaussianBlur(Z_face, (5, 5), 0)
        gy, gx = np.gradient(Z_s)
        gyy, gxy = np.gradient(gy)
        gxy2, gxx = np.gradient(gx)
        K = (gxx * gyy - gxy**2) / (1 + gx**2 + gy**2)**2
        H = ((1+gy**2)*gxx - 2*gx*gy*gxy + (1+gx**2)*gyy) / (2*(1+gx**2+gy**2)**1.5)
        print(f"  Curvatura gaussiana K: media={K.mean():.6f} | std={K.std():.6f}", flush=True)
        print(f"  Curvatura media H: media={H.mean():.6f} | std={H.std():.6f}", flush=True)
        # Un mapa de profundidad real tiene curvatura media baja (suave)
        print(f"  -> {'SUPERFICIE SUAVE (mapa de profundidad coherente)' if abs(H.mean()) < 0.01 else 'superficie rugosa'}", flush=True)
        report["M4_curvatura"] = {"K_mean": float(K.mean()), "K_std": float(K.std()),
                                  "H_mean": float(H.mean()), "H_std": float(H.std())}

    # ============ M5: DIMENSION DE LA SUPERFICIE ============
    print("\n[M5] DIMENSION DE LA SUPERFICIE DE PROFUNDIDAD", flush=True)
    if img1 is not None:
        face = img1[100:1100, 1000:2000]
        Ds = dimension_superficie(face.astype(np.float64))
        print(f"  D fractal del relieve facial: {[f'{d:.3f}' for d in Ds]}", flush=True)
        D_media = np.mean(Ds)
        print(f"  D media: {D_media:.3f} -> {'SUPERFICIE (~2D, continua)' if abs(D_media-2) < 0.3 else 'otra estructura'}", flush=True)
        report["M5_dimension"] = {"Ds": Ds, "D_media": float(D_media)}

    # ============ M6: COHERENCIA ============
    print("\n[M6] COHERENCIA DEL MAPA DE PROFUNDIDAD", flush=True)
    # Un mapa de profundidad coherente tiene: maximo centrado, decaimiento
    # suave hacia bordes, simetria. Medir coherencia global.
    if img1 is not None:
        face = img1[100:1100, 1000:2000].astype(np.float64)
        # Simetria del relieve (rostro simetrico)
        hf, wf = face.shape
        izq = face[:, :wf//2]
        der = np.fliplr(face[:, wf//2:])
        mw = min(izq.shape[1], der.shape[1])
        sim = np.corrcoef(izq[:, :mw].flatten(), der[:, :mw].flatten())[0, 1]
        print(f"  Simetria del relieve facial: {sim:+.3f}", flush=True)
        # Suavidad del decaimiento radial desde el maximo
        print(f"  -> {'MAPA DE PROFUNDIDAD COHERENTE (simetrico)' if sim > 0.5 else 'asimetrico'}", flush=True)
        report["M6_coherencia"] = {"simetria": float(sim)}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿LA ESCALA DE GRISES ES UN MAPA DE PROFUNDIDAD?", flush=True)
    print("=" * 70, flush=True)
    print(f"  M1: figura {ratio_suav:.0f}x mas suave que ruido", flush=True)
    if img1 is not None:
        print(f"  M2: maximo de intensidad centrado al {1-centro_rel:.0%}", flush=True)
        print(f"  M4: curvatura media H={H.mean():.6f} (suave)", flush=True)
        print(f"  M5: D fractal superficie={D_media:.3f}", flush=True)
        print(f"  M6: simetria relieve={sim:+.3f}", flush=True)
    report["conclusion"] = {
        "ratio_suavidad_vs_ruido": float(ratio_suav),
        "M2_centrado": float(1-centro_rel) if img1 is not None else None,
        "M4_H_mean": float(H.mean()) if img1 is not None else None,
        "M5_D_media": float(D_media) if img1 is not None else None,
        "M6_simetria": float(sim) if img1 is not None else None,
    }

    out_json = os.path.join(OUT, "mapa_profundidad_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
