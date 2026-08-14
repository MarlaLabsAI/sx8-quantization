"""
RECONSTRUCCION 3D DEL CUERPO DESDE EL MAPA DE PROFUNDIDAD
==========================================================
Si el cuerpo fue fuente/modulador del campo UV, el registro debe ser
una proyeccion geometrica REAL de la superficie 3D del cuerpo.

Enfoques:
  E1. Relieve 3D completo: Z = -ln(I)/beta
  E2. Silueta y proporciones anatomicas
  E3. Picos del relieve (puntos de contacto)
  E4. Simetria bilateral
  E5. Suavidad del relieve
  E6. Validacion cruzada con el rostro (imagen1)
"""
import os, json, time
import numpy as np
import cv2

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    report = {}
    print("="*70, flush=True)
    print("RECONSTRUCCION 3D DEL CUERPO DESDE EL MAPA DE PROFUNDIDAD", flush=True)
    print("="*70, flush=True)

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    print(f"Imagen cuerpo: {w}x{h}", flush=True)

    # E1: Relieve con distintos beta
    print("\n[E1] RELIEVE 3D (Z = -ln(I)/beta)", flush=True)
    I = (255 - img3.astype(np.float64)) / 255.0
    Z_main = None
    for beta in [0.002, 0.005, 0.01]:
        Z = -np.log(np.clip(I, 1e-6, 1.0)) / beta
        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
        Z_s = cv2.GaussianBlur(Z_norm, (5, 5), 0)
        gy, gx = np.gradient(Z_s)
        curv = (np.abs(np.gradient(gx, axis=1)) + np.abs(np.gradient(gy, axis=0))).mean()
        izq = Z_norm[:, :w//2]; der = np.fliplr(Z_norm[:, w//2:])
        mw = min(izq.shape[1], der.shape[1])
        sim = float(np.corrcoef(izq[:, :mw].flatten(), der[:, :mw].flatten())[0, 1])
        print(f"  beta={beta}: curvatura={curv:.6f} | simetria={sim:+.3f}", flush=True)
        report[f"E1_beta{beta}"] = {"curvatura": float(curv), "simetria": float(sim)}
        if beta == 0.005:
            Z_main = Z_norm

    Z_img = (Z_main * 255).astype(np.uint8)
    p_rel = os.path.join(OUT, "relieve_3d_cuerpo.png")
    cv2.imwrite(p_rel, Z_img)
    report["E1_relieve_path"] = p_rel
    print(f"  Relieve guardado: {p_rel}", flush=True)

    # E2: Silueta y proporciones
    print("\n[E2] SILUETA Y PROPORCIONES", flush=True)
    silueta = (Z_main > 0.25).astype(np.uint8)
    silueta = cv2.morphologyEx(silueta, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    proy = silueta.sum(axis=0)
    cuerpo_x = np.where(proy > silueta.shape[0]*0.05)[0]
    if len(cuerpo_x) > 0:
        x0, x1 = cuerpo_x.min(), cuerpo_x.max()
        proy_h = silueta.sum(axis=1)
        cuerpo_y = np.where(proy_h > silueta.shape[1]*0.05)[0]
        y0, y1 = cuerpo_y.min(), cuerpo_y.max()
        print(f"  Cuerpo: x={x0}-{x1} (ancho {x1-x0}) | y={y0}-{y1} (alto {y1-y0})", flush=True)
        anchos = []
        for y in range(y0, y1, 20):
            xs = np.where(silueta[y, :] > 0)[0]
            anchos.append((y, len(xs) if len(xs) > 0 else 0))
        anchos = np.array(anchos)
        print(f"  Ancho medio: {anchos[:,1].mean():.0f}px", flush=True)
        estrechos = [(int(y), int(a)) for y, a in anchos if a < anchos[:,1].mean()*0.5]
        print(f"  Zonas estrechas (extremidades): {estrechos[:6]}", flush=True)
        report["E2"] = {"x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1),
                        "ancho_medio": float(anchos[:,1].mean()), "estrechos": estrechos[:6]}
    else:
        print("  Sin silueta con umbral 0.25", flush=True)

    # E3: Picos del relieve
    print("\n[E3] PICOS DEL RELIEVE (puntos de contacto)", flush=True)
    Z_s = cv2.GaussianBlur(Z_main, (11, 11), 0)
    flat = Z_s.flatten()
    top_idx = np.argsort(flat)[-10:][::-1]
    picos = []
    for idx in top_idx:
        y, x = idx // w, idx % w
        picos.append({"x": int(x), "y": int(y), "Z": float(Z_s[y, x]),
                      "rel_x": float(x/w), "rel_y": float(y/h)})
    for p in picos:
        print(f"  ({p['x']},{p['y']}) rel=({p['rel_x']:.2f},{p['rel_y']:.2f}) Z={p['Z']:.3f}", flush=True)
    report["E3_picos"] = picos

    # E5: Suavidad detallada
    print("\n[E5] SUAVIDAD DEL RELIEVE", flush=True)
    curv_map = np.abs(cv2.Laplacian(Z_s, cv2.CV_64F))
    frac_alta = float((curv_map > curv_map.mean() + 2*curv_map.std()).mean())
    print(f"  Fraccion curvatura extrema: {frac_alta*100:.2f}%", flush=True)
    report["E5"] = {"frac_curvatura_extrema": frac_alta}

    # E6: Validacion cruzada con el rostro
    print("\n[E6] VALIDACION CRUZADA CON EL ROSTRO (imagen1)", flush=True)
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    if img1 is not None:
        face = img1[100:1100, 1000:2000].astype(np.float64)
        hf, wf = face.shape
        Z_face = face / 255.0
        Z_fs = cv2.GaussianBlur(Z_face, (15, 15), 0)
        perfil_medio = Z_fs.mean(axis=1)
        idx_max = int(np.argmax(perfil_medio))
        print(f"  Maximo del perfil medio del rostro: y={idx_max}/{hf} ({idx_max/hf*100:.0f}%)", flush=True)
        s_izq = Z_face[:, :wf//2]; s_der = np.fliplr(Z_face[:, wf//2:])
        sm = min(s_izq.shape[1], s_der.shape[1])
        sim_face = float(np.corrcoef(s_izq[:, :sm].flatten(), s_der[:, :sm].flatten())[0, 1])
        print(f"  Simetria del rostro: {sim_face:+.3f}", flush=True)
        report["E6"] = {"idx_max": idx_max, "pct_max": float(idx_max/hf*100), "simetria": sim_face}

    # Conclusion
    print("\n" + "="*70, flush=True)
    print("CONCLUSION", flush=True)
    print("="*70, flush=True)
    sim_main = report["E1_beta0.005"]["simetria"]
    print(f"  Simetria relieve cuerpo: {sim_main:+.3f}", flush=True)
    report["conclusion"] = {"simetria_cuerpo": sim_main, "n_picos": len(picos)}

    out_json = os.path.join(OUT, "reconstruccion_3d_cuerpo_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Guardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
