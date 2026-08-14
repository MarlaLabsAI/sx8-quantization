"""
RECONSTRUCCION 3D v2: SEGMENTACION MEJORADA + VALIDACION MULTIENFOQUE
=====================================================================
La imagen3 tiene 2 figuras superpuestas + quemaduras. Estrategia:
  - Aislar la figura principal por componentes conectados
  - Reconstruir el relieve SOLO sobre esa figura
  - Validar con 3 enfoques: simetria, picos anatomicos, proporciones
"""
import os, json, time
import numpy as np
import cv2
from scipy import ndimage

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    report = {}
    print("="*70, flush=True)
    print("RECONSTRUCCION 3D v2: SEGMENTACION + VALIDACION MULTIENFOQUE", flush=True)
    print("="*70, flush=True)

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape

    # ---- Segmentar la figura: umbral adaptativo + componentes ----
    img_eq = cv2.equalizeHist(img3)
    # Umbral: la figura es mas oscura que la tela
    thresh_val, bin_img = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Limpiar
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((7,7), np.uint8))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, np.ones((21,21), np.uint8))
    # Componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
    # Ordenar por area, tomar los 2 mas grandes (las 2 figuras)
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > 50000]
    areas.sort(key=lambda x: -x[1])
    print(f"Componentes grandes: {len(areas)}", flush=True)
    for i, area in areas[:4]:
        x, y, cw, ch = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        print(f"  comp {i}: area={area}, bbox=({x},{y},{cw}x{ch}), aspecto={cw/ch:.2f}", flush=True)

    if len(areas) >= 2:
        # Tomar la figura mas grande (primera)
        i_fig = areas[0][0]
        fig_mask = (labels == i_fig)
        x0, y0 = stats[i_fig, cv2.CC_STAT_LEFT], stats[i_fig, cv2.CC_STAT_TOP]
        cw, ch = stats[i_fig, cv2.CC_STAT_WIDTH], stats[i_fig, cv2.CC_STAT_HEIGHT]
        print(f"\nFigura principal: bbox=({x0},{y0},{cw}x{ch})", flush=True)

        # ---- Relieve sobre la figura ----
        # La intensidad de la figura (oscuridad = proximidad)
        fig_region = img3[y0:y0+ch, x0:x0+cw].astype(np.float64)
        # Mascara local
        mask_local = fig_mask[y0:y0+ch, x0:x0+cw]
        # Intensidad de proximidad: oscuro = cerca
        I = (255 - fig_region) / 255.0
        I = np.where(mask_local, I, 0)
        Z = -np.log(np.clip(I, 1e-6, 1.0)) / 0.005
        Z = (Z - Z.min()) / (Z.max() - Z.min() + 1e-9)
        # Enmascarar fondo
        Z = np.where(mask_local, Z, 0)

        # Guardar relieve
        Z_img = (Z * 255).astype(np.uint8)
        p_rel = os.path.join(OUT, "relieve_3d_figura.png")
        cv2.imwrite(p_rel, Z_img)
        print(f"  Relieve figura guardado: {p_rel}", flush=True)
        report["relieve_path"] = p_rel

        # ---- Simetria bilateral de la figura ----
        fh, fw = Z.shape
        izq = Z[:, :fw//2]; der = np.fliplr(Z[:, fw//2:])
        mw = min(izq.shape[1], der.shape[1])
        # Solo donde hay figura en ambas mitades
        m_i = (izq[:, :mw] > 0) & (der[:, :mw] > 0)
        if m_i.sum() > 100:
            sim = float(np.corrcoef(izq[:, :mw][m_i], der[:, :mw][m_i])[0, 1])
        else:
            sim = float("nan")
        print(f"  Simetria bilateral figura: {sim:+.3f}", flush=True)
        report["simetria_figura"] = float(sim)

        # ---- Picos del relieve (puntos de contacto) ----
        Z_s = cv2.GaussianBlur(Z, (15, 15), 0)
        # Picos dentro de la figura
        mask_s = cv2.GaussianBlur(mask_local.astype(np.float32), (15,15), 0) > 0.5
        flat_idx = np.where(mask_s.flatten() & (Z_s.flatten() > 0.3))[0]
        if len(flat_idx) > 100:
            top = flat_idx[np.argsort(Z_s.flatten()[flat_idx])[-10:][::-1]]
            picos = []
            for idx in top:
                fy, fx = idx // fw, idx % fw
                picos.append({"x_fig": int(fx), "y_fig": int(fy),
                              "x_abs": int(x0+fx), "y_abs": int(y0+fy),
                              "Z": float(Z_s[fy, fx]),
                              "rel_x": float((x0+fx)/w), "rel_y": float((y0+fy)/h)})
            print("  Picos del relieve (puntos de contacto):", flush=True)
            for p in picos:
                print(f"    abs=({p['x_abs']},{p['y_abs']}) rel=({p['rel_x']:.2f},{p['rel_y']:.2f}) Z={p['Z']:.3f}", flush=True)
            report["picos"] = picos

        # ---- Perfil de profundidad a lo largo del eje largo ----
        # La figura puede estar horizontal o vertical; usar la mayor dimension
        if cw > ch:
            eje = "horizontal"
            perfil = Z.mean(axis=0)  # a lo largo de x
        else:
            eje = "vertical"
            perfil = Z.mean(axis=1)  # a lo largo de y
        print(f"\n  Orientacion figura: {eje}", flush=True)
        # Perfil de profundidad (0-100%)
        perfil_s = ndimage.gaussian_filter1d(perfil, sigma=5)
        print("  Perfil de profundidad a lo largo del cuerpo:", flush=True)
        n_p = len(perfil_s)
        for i in range(0, n_p, max(1, n_p//12)):
            pct = i/n_p*100
            print(f"    {pct:5.1f}%: Z={perfil_s[i]:.3f}", flush=True)
        # Pico del perfil (punto de maximo contacto)
        idx_pico = int(np.argmax(perfil_s))
        print(f"  Maximo contacto en {idx_pico/n_p*100:.0f}% del eje", flush=True)
        report["eje"] = eje
        report["perfil"] = perfil_s[::max(1,n_p//12)].tolist()
        report["pico_pct"] = float(idx_pico/n_p*100)

        # ---- Proporciones: zonas estrechas vs anchas ----
        # Perfil de ancho de la figura a lo largo del eje
        if cw > ch:
            anchos = mask_local.sum(axis=0)
        else:
            anchos = mask_local.sum(axis=1)
        anchos_s = ndimage.gaussian_filter1d(anchos.astype(float), sigma=5)
        ancho_medio = anchos_s.mean()
        print(f"  Ancho medio: {ancho_medio:.0f}px", flush=True)
        estrechos = np.where(anchos_s < ancho_medio*0.4)[0]
        print(f"  Zonas estrechas (<40% del ancho): {len(estrechos)} posiciones", flush=True)
        report["ancho_medio"] = float(ancho_medio)
        report["n_estrechos"] = int(len(estrechos))

    out_json = os.path.join(OUT, "reconstruccion_3d_v2_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
