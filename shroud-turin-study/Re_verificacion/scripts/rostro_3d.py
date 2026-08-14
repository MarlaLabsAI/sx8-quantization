"""
RECONSTRUCCION 3D DEL ROSTRO (imagen1, negativo) - VALIDACION ANATOMICA
=======================================================================
El rostro en negativo es la region mas limpia para validar el mapa de
profundidad. En el negativo: MAS CLARO = MAS CERCA de la tela (nariz,
pomulos prominentes = claros; ojos, mejillas hundidas = oscuros).

Enfoques:
  R1. Relieve 3D del rostro: Z = intensidad del negativo (claro=cerca)
  R2. Topografia: nariz (pico), ojos (valles), frente, menton
      - Localizar el maximo (nariz) y verificar que esta en la zona
        central del rostro (~50-65% del alto)
      - Verificar valles (ojos) alrededor de la nariz
  R3. Simetria bilateral del relieve facial
  R4. Perfil horizontal a la altura de la nariz: nariz = pico central
      con valles de ojos a los lados
  R5. Perfil vertical: frente -> nariz -> boca -> menton
"""
import os, json, time
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
    print("="*70, flush=True)
    print("RECONSTRUCCION 3D DEL ROSTRO (mapa de profundidad facial)", flush=True)
    print("="*70, flush=True)

    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        print("imagen1 no disponible")
        return
    face = img1[100:1100, 1000:2000].astype(np.float64)
    h, w = face.shape
    print(f"Rostro: {w}x{h}", flush=True)

    # R1: Relieve (claro = cerca = pico)
    Z = face / 255.0
    Z_s = cv2.GaussianBlur(Z, (9, 9), 0)
    Z_img = (Z * 255).astype(np.uint8)
    p_rel = os.path.join(OUT, "relieve_3d_rostro.png")
    cv2.imwrite(p_rel, Z_img)
    print(f"  Relieve guardado: {p_rel}", flush=True)
    report["relieve_path"] = p_rel

    # R3: Simetria bilateral
    izq = Z[:, :w//2]; der = np.fliplr(Z[:, w//2:])
    mw = min(izq.shape[1], der.shape[1])
    sim = float(np.corrcoef(izq[:, :mw].flatten(), der[:, :mw].flatten())[0, 1])
    print(f"  Simetria bilateral: {sim:+.3f}", flush=True)
    report["simetria"] = float(sim)

    # R5: Perfil vertical medio (frente -> menton)
    perfil_v = ndimage.gaussian_filter1d(Z_s.mean(axis=1), sigma=8)
    # Normalizar perfil para encontrar estructura
    print("\n[R5] PERFIL VERTICAL (frente -> nariz -> boca -> menton):", flush=True)
    for i in range(0, h, 80):
        print(f"    y={i:4d} ({i/h*100:5.1f}%): Z={perfil_v[i]:.3f}", flush=True)
    # Picos y valles
    picos, _ = find_peaks(perfil_v, distance=50, prominence=0.02)
    valles, _ = find_peaks(-perfil_v, distance=50, prominence=0.02)
    print(f"  Picos (protuberancias) en y: {picos.tolist()} ({[f'{p/h*100:.0f}%' for p in picos]})", flush=True)
    print(f"  Valles (hundimientos) en y: {valles.tolist()} ({[f'{v/h*100:.0f}%' for v in valles]})", flush=True)
    report["perfil_vertical"] = {"picos": picos.tolist(), "valles": valles.tolist()}

    # R4: Perfil horizontal a la altura de la nariz (si la encontramos)
    print("\n[R4] PERFIL HORIZONTAL (a la altura del pico principal):", flush=True)
    if len(picos) > 0:
        # El pico mas alto del perfil vertical = nariz
        idx_nariz = picos[np.argmax(perfil_v[picos])]
        print(f"  Pico principal en y={idx_nariz} ({idx_nariz/h*100:.0f}%) -> candidato a NARIZ", flush=True)
        perfil_h = ndimage.gaussian_filter1d(Z_s[idx_nariz, :], sigma=8)
        for i in range(0, w, 100):
            print(f"    x={i:4d} ({i/w*100:5.1f}%): Z={perfil_h[i]:.3f}", flush=True)
        # Picos del perfil horizontal (nariz central + posiblemente pomulos)
        hpicos, _ = find_peaks(perfil_h, distance=30, prominence=0.01)
        print(f"  Picos horizontales en x: {hpicos.tolist()} ({[f'{p/w*100:.0f}%' for p in hpicos]})", flush=True)
        report["perfil_horizontal"] = {"y_nariz": int(idx_nariz),
                                       "picos_x": hpicos.tolist()}

        # R2: Verificar estructura facial: nariz central con ojos a los lados
        print("\n[R2] ESTRUCTURA FACIAL:", flush=True)
        if len(hpicos) >= 1:
            cx_nariz = hpicos[np.argmax(perfil_h[hpicos])]
            centralidad = 1 - abs(cx_nariz - w/2)/(w/2)
            print(f"  Nariz en x={cx_nariz} ({cx_nariz/w*100:.0f}%): centralidad={centralidad:.2f}", flush=True)
            # Ojos: valles a los lados de la nariz
            ojos = [p for p in hpicos if abs(p - cx_nariz) > 30 and abs(p - cx_nariz) < 200]
            print(f"  Estructuras a los lados de la nariz: {[f'x={p} ({p/w*100:.0f}%)' for p in ojos]}", flush=True)
            report["estructura_facial"] = {"nariz_x": int(cx_nariz), "centralidad": float(centralidad),
                                           "laterales": ojos}
        else:
            print("  Sin picos horizontales claros", flush=True)

    # Conclusion
    print("\n" + "="*70, flush=True)
    print("CONCLUSION: MAPA DE PROFUNDIDAD DEL ROSTRO", flush=True)
    print("="*70, flush=True)
    print(f"  Simetria: {sim:+.3f}", flush=True)
    if len(picos) > 0:
        print(f"  Picos verticales (protuberancias): {len(picos)}", flush=True)
        print(f"  Valles (hundimientos): {len(valles)}", flush=True)
    report["conclusion"] = {"simetria": float(sim), "n_picos_v": len(picos), "n_valles_v": len(valles)}

    out_json = os.path.join(OUT, "rostro_3d_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Guardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
