"""
FILTRADO DEL LINO Y QUEMADURAS DEL ROSTRO
==========================================
El lino tiene textura periodica (hilos): periodicidades ~64px y ~21px
medidas en FFT 2D de la tela. La UV afecta al lino de forma conocida
(oxidacion superficial). El rostro esta contaminado por la textura del
tejido + quemaduras de 1532.

Proceso:
  F1. Medir frecuencias del lino en zona de tela pura
  F2. Filtro NOTCH: eliminar las frecuencias del lino del rostro
  F3. Excluir quemaduras (zonas extremas de intensidad)
  F4. Reconstruir mapa de profundidad sobre imagen FILTRADA
  F5. Validar: simetria, estructura facial (nariz central, ojos)
  F6. Comparar: con lino vs sin lino
"""
import os, json, time
import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import find_peaks

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def frecuencias_lino(img, n=512):
    """FFT 2D de la tela y picos no-DC dominantes."""
    img_f = img.astype(np.float32)
    img_f = (img_f - img_f.mean()) / (img_f.std() + 1e-9)
    h, w = img_f.shape
    # Recortar a cuadrado
    s = min(h, w, n)
    img_c = img_f[:s, :s]
    F = np.fft.fft2(img_c)
    F = np.fft.fftshift(F)
    mag = np.abs(F)
    mag = mag / mag.max()
    cy, cx = s//2, s//2
    mag[cy-3:cy+3, cx-3:cx+3] = 0  # quitar DC
    # Picos
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(mag, size=5) == mag
    ys, xs = np.where(local_max & (mag > 0.15))
    inds = np.argsort(mag[ys, xs])[::-1]
    picos = []
    for k in inds[:15]:
        y, x = ys[k], xs[k]
        dy, dx = y - cy, x - cx
        if dx == 0 and dy == 0:
            continue
        per_x = s/abs(dx) if dx != 0 else float('inf')
        per_y = s/abs(dy) if dy != 0 else float('inf')
        picos.append({"dx": int(dx), "dy": int(dy),
                      "per_x": float(per_x), "per_y": float(per_y),
                      "mag": float(mag[y, x])})
    return picos

def notch_filter(img, frecuencias, radio=4):
    """Elimina frecuencias del lino de la imagen."""
    img_f = img.astype(np.float32)
    h, w = img_f.shape
    n = 2**int(np.ceil(np.log2(max(h, w))))
    img_p = np.zeros((n, n), dtype=np.float32)
    img_p[:h, :w] = img_f
    F = np.fft.fft2(img_p)
    F = np.fft.fftshift(F)
    cy, cx = n//2, n//2
    # Crear mascara de notch
    mascara = np.ones((n, n))
    for p in frecuencias:
        dx, dy = p["dx"], p["dy"]
        if p["per_x"] == float('inf') and p["per_y"] == float('inf'):
            continue
        # Escalar dx, dy al tamano n (los picos se midieron en s=s del analisis)
        # Normalizar: la frecuencia relativa es la misma
        # En el analisis se uso s = min(h,w,n=512). El dx medido corresponde a n_orig.
        # Recalcular dx,dy para el tamano n usando la periodicidad
        if p["per_x"] != float('inf'):
            fx = n / p["per_x"]
        else:
            fx = 0
        if p["per_y"] != float('inf'):
            fy = n / p["per_y"]
        else:
            fy = 0
        # Notch en +f y -f
        for sgn in [1, -1]:
            x0, y0 = cx + sgn*fx, cy + sgn*fy
            # Region circular
            yy, xx = np.mgrid[0:n, 0:n]
            mask = (xx-x0)**2 + (yy-y0)**2 < radio**2
            mascara[mask] = 0
    F_f = F * mascara
    F_f = np.fft.ifftshift(F_f)
    img_fil = np.fft.ifft2(F_f).real
    img_fil = img_fil[:h, :w]
    return img_fil

def main():
    t0 = time.time()
    report = {}
    print("="*70, flush=True)
    print("FILTRADO DEL LINO Y QUEMADURAS DEL ROSTRO", flush=True)
    print("="*70, flush=True)

    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    face = img1[100:1100, 1000:2000].astype(np.float64)
    h, w = face.shape
    print(f"Rostro: {w}x{h}", flush=True)

    # F1: Frecuencias del lino desde imagen3 (tela)
    print("\n[F1] FRECUENCIAS DEL LINO (de la tela en imagen3)", flush=True)
    if img3 is not None:
        tela = img3[50:562, 50:562]  # region de tela (esquina)
        picos_lino = frecuencias_lino(tela, n=512)
        print(f"  Picos del lino: {len(picos_lino)}", flush=True)
        for p in picos_lino[:8]:
            print(f"    (dx={p['dx']:+3d}, dy={p['dy']:+3d}) per_x={p['per_x']:.0f} per_y={p['per_y']:.0f} mag={p['mag']:.3f}", flush=True)
        # Seleccionar los picos significativos (mag > 0.2)
        lino_principal = [p for p in picos_lino if p["mag"] > 0.2]
        print(f"  Picos principales (mag>0.2): {len(lino_principal)}", flush=True)
        report["F1_lino"] = picos_lino[:10]
    else:
        lino_principal = []

    # F2: Aplicar notch filter al rostro
    print("\n[F2] FILTRO NOTCH del lino sobre el rostro", flush=True)
    if lino_principal:
        face_filtrada = notch_filter(face, lino_principal, radio=4)
        # Guardar
        face_fil_img = np.clip(face_filtrada, 0, 255).astype(np.uint8)
        p_fil = os.path.join(OUT, "rostro_sin_lino.png")
        cv2.imwrite(p_fil, face_fil_img)
        print(f"  Rostro filtrado guardado: {p_fil}", flush=True)
        report["F2_path"] = p_fil
    else:
        face_filtrada = face.copy()
        print("  Sin picos de lino significativos, uso original", flush=True)

    # F3: Excluir quemaduras (1532)
    print("\n[F3] EXCLUSION DE QUEMADURAS (1532)", flush=True)
    # Las quemaduras son zonas muy oscuras en el positivo (o muy claras en negativo)
    # En el negativo (imagen1): las quemaduras aparecen como zonas extremas
    # Detectar outliers extremos de intensidad
    media = face.mean()
    std = face.std()
    # Quemaduras: zonas MUY por encima del promedio (quemadas = claras en negativo)
    quemadura_mask = face > media + 2.5*std
    frac_q = quemadura_mask.mean()
    print(f"  Zonas de posible quemadura (>2.5std): {frac_q*100:.1f}% del rostro", flush=True)
    report["F3_quemaduras"] = {"frac": float(frac_q), "media": float(media), "std": float(std)}

    # F4: Reconstruir mapa de profundidad sobre imagen FILTRADA (sin lino, sin quemaduras)
    print("\n[F4] MAPA DE PROFUNDIDAD SOBRE IMAGEN FILTRADA", flush=True)
    Z_limpio = face_filtrada.copy()
    # Excluir quemaduras del mapa de profundidad
    Z_limpio[quemadura_mask] = 0
    # Normalizar a 0-1
    Z_min, Z_max = Z_limpio[~quemadura_mask].min(), Z_limpio[~quemadura_mask].max()
    Z_norm = (Z_limpio - Z_min) / (Z_max - Z_min + 1e-9)
    Z_norm[quemadura_mask] = 0
    # Guardar relieve limpio
    Z_img = (Z_norm * 255).astype(np.uint8)
    p_lim = os.path.join(OUT, "relieve_3d_rostro_sin_lino.png")
    cv2.imwrite(p_lim, Z_img)
    print(f"  Relieve limpio guardado: {p_lim}", flush=True)
    report["F4_path"] = p_lim

    # F5: Validacion - simetria y estructura
    print("\n[F5] VALIDACION DEL RELIEVE LIMPIO", flush=True)
    # Simetria
    izq = Z_norm[:, :w//2]; der = np.fliplr(Z_norm[:, w//2:])
    mw = min(izq.shape[1], der.shape[1])
    m_i = (izq[:, :mw] > 0) & (der[:, :mw] > 0)
    sim_limpio = float(np.corrcoef(izq[:, :mw][m_i], der[:, :mw][m_i])[0, 1]) if m_i.sum() > 100 else float("nan")
    # Simetria del original (para comparar)
    Z_orig = face / 255.0
    izq_o = Z_orig[:, :w//2]; der_o = np.fliplr(Z_orig[:, w//2:])
    sim_orig = float(np.corrcoef(izq_o[:, :mw].flatten(), der_o[:, :mw].flatten())[0, 1])
    print(f"  Simetria ORIGINAL: {sim_orig:+.3f}", flush=True)
    print(f"  Simetria SIN LINO: {sim_limpio:+.3f}", flush=True)
    report["F5_simetria"] = {"original": float(sim_orig), "sin_lino": float(sim_limpio)}

    # Estructura facial en el relieve limpio
    Z_s = cv2.GaussianBlur(Z_norm, (9, 9), 0)
    perfil_v = ndimage.gaussian_filter1d(Z_s.mean(axis=1), sigma=8)
    picos_v, _ = find_peaks(perfil_v, distance=50, prominence=0.02)
    valles_v, _ = find_peaks(-perfil_v, distance=50, prominence=0.02)
    print(f"  Picos verticales (limpio): {picos_v.tolist()} ({[f'{p/h*100:.0f}%' for p in picos_v]})", flush=True)
    print(f"  Valles verticales (limpio): {valles_v.tolist()} ({[f'{v/h*100:.0f}%' for v in valles_v]})", flush=True)
    report["F5_estructura"] = {"picos": picos_v.tolist(), "valles": valles_v.tolist()}

    # Perfil horizontal a la altura del pico principal
    if len(picos_v) > 0:
        idx_n = picos_v[np.argmax(perfil_v[picos_v])]
        perfil_h = ndimage.gaussian_filter1d(Z_s[idx_n, :], sigma=8)
        hpicos, _ = find_peaks(perfil_h, distance=30, prominence=0.01)
        print(f"  Altura del pico principal: y={idx_n} ({idx_n/h*100:.0f}%)", flush=True)
        print(f"  Picos horizontales: {hpicos.tolist()} ({[f'{p/w*100:.0f}%' for p in hpicos]})", flush=True)
        report["F5_horizontal"] = {"y": int(idx_n), "picos_x": hpicos.tolist()}

    # Conclusion
    print("\n" + "="*70, flush=True)
    print("CONCLUSION: EFECTO DEL FILTRADO DEL LINO", flush=True)
    print("="*70, flush=True)
    print(f"  Simetria: original={sim_orig:+.3f} -> sin lino={sim_limpio:+.3f}", flush=True)
    print(f"  {'MEJORA' if abs(sim_limpio) > abs(sim_orig) else 'NO mejora o igual'} tras filtrar lino", flush=True)
    report["conclusion"] = {"sim_orig": float(sim_orig), "sim_limpio": float(sim_limpio),
                            "mejora": bool(abs(sim_limpio) > abs(sim_orig))}

    out_json = os.path.join(OUT, "filtrar_lino_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Guardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
