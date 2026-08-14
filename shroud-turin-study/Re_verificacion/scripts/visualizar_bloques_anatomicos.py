"""
VISUALIZACION: 12 BLOQUES ANATOMICOS DEL PUNTO CENTRAL SOBRE LA SABANA
=====================================================================
Muestra como el punto central (416,416) conecta con 12 regiones
anatomicas del cuerpo a lo largo del perfil vertical.

Salidas:
  - bloques_anatomicos_sabana.png  (cuerpo + bloques + tuneles)
  - matriz_recurrencia_fila416.png (matriz + fila 416 resaltada)

NO modifica originales.
"""

import os
import json
import numpy as np
import cv2
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    return R, profile, img3

def bloques_recurrencia(fila):
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

# Anatomia asignada (de Y6)
ANATOMIA = [
    ("Cabeza alta", 0.03),
    ("Frente", 0.07),
    ("Ojos/Nariz", 0.12),
    ("Cuello", 0.24),
    ("Hombros", 0.27),
    ("PECHO (punto central)", 0.38),
    ("Torso medio", 0.45),
    ("Cintura", 0.51),
    ("Caderas", 0.63),
    ("Rodillas", 0.71),
    ("Piernas bajas", 0.77),
    ("Tobillos/Pies", 0.96),
]

def main():
    R, profile, img3 = matriz_real()
    n = R.shape[0]
    cx = 416
    fila = R[cx, :]
    bloques = bloques_recurrencia(fila)
    print(f"Bloques de la fila {cx}: {len(bloques)}")
    for b in bloques:
        print(f"  {b}")

    # Colores: degradado viridis (cabeza=azul, torso=verde, piernas=amarillo)
    cmap = plt.cm.viridis
    colores = [cmap(i / max(1, len(bloques)-1)) for i in range(len(bloques))]

    # ============ VIZ 1: CUERPO + BLOQUES + TUNELES ============
    h_img, w_img = img3.shape
    # Imagen a color para visualizacion
    img_color = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    # Aclarar un poco para que se vean las marcas
    img_color = cv2.convertScaleAbs(img_color, alpha=0.8, beta=40)

    fig, ax = plt.subplots(1, 1, figsize=(10, 16))
    ax.imshow(img_color)
    ax.set_title("SABANA SANTA - 12 BLOQUES DE RECURRENCIA DEL PUNTO CENTRAL (y=416)", fontsize=13, fontweight='bold')

    # Columna del perfil (x=960)
    ax.axvline(x=w_img//2, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Punto central: y=416, x=960
    px = w_img // 2
    py = 416
    ax.plot(px, py, 'o', markersize=14, color='red', markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(px, py, '+', markersize=20, color='white', markeredgewidth=3, zorder=11)

    # Bloques como bandas + tuneles desde el punto central
    for i, (b0, b1) in enumerate(bloques):
        color = colores[i]
        y_centro_bloque = (b0 + b1) / 2
        # Banda horizontal en el centro de la columna (ancha 30px alrededor de x=960)
        ax.axhspan(b0, b1, xmin=(px-15)/w_img, xmax=(px+15)/w_img,
                   color=color, alpha=0.55, zorder=5)
        # Tunel: linea desde el punto central al bloque
        ax.plot([px, px], [py, y_centro_bloque], '-', color='yellow', linewidth=1.0,
                alpha=0.7, zorder=4)
        # Etiqueta
        nombre, rel = ANATOMIA[i] if i < len(ANATOMIA) else (f"Bloque {i}", 0)
        ax.text(px + 40, y_centro_bloque, nombre, fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=to_rgba(color, 0.8), edgecolor='none'),
                va='center', zorder=12)

    # Leyenda de colores por region
    handles = [mpatches.Patch(color=colores[i], label=f"{ANATOMIA[i][0]} (y={bloques[i][0]}-{bloques[i][1]})")
               for i in range(len(bloques))]
    ax.legend(handles=handles, loc='upper left', fontsize=8, framealpha=0.8)

    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.set_xlabel("x (px) - columna central del perfil en x=960")
    ax.set_ylabel("y (px) - posicion vertical")
    plt.tight_layout()
    p1 = os.path.join(OUT, "bloques_anatomicos_sabana.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGuardado: {p1}")

    # ============ VIZ 2: MATRIZ + FILA 416 ============
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    # Matriz completa (submuestreada para visualizacion)
    step = 2
    R_viz = R[::step, ::step]
    ax[0].imshow(R_viz, cmap='gray_r', interpolation='nearest')
    ax[0].set_title("Matriz de recurrencia completa", fontsize=12)
    ax[0].axhline(y=416//step, color='red', linewidth=1.5)
    ax[0].axvline(x=416//step, color='red', linewidth=1.5)
    ax[0].plot(416//step, 416//step, 'o', color='yellow', markersize=6)
    ax[0].set_xlabel("j")
    ax[0].set_ylabel("i")

    # Zoom en la fila 416
    fila_viz = fila[::1]
    ax[1].plot(fila_viz, 'k-', linewidth=1)
    ax[1].fill_between(range(n), fila_viz, alpha=0.3, color='blue')
    # Marcar bloques
    for i, (b0, b1) in enumerate(bloques):
        ax[1].axvspan(b0, b1, color=colores[i], alpha=0.5)
        ax[1].text((b0+b1)/2, 1.05, f"{i}", fontsize=8, ha='center')
    ax[1].set_title("Fila 416: 12 bloques de recurrencia (tuneles del punto central)", fontsize=12)
    ax[1].set_xlabel("j (posicion en el perfil)")
    ax[1].set_ylabel("R(416, j)")
    ax[1].set_ylim(-0.1, 1.3)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    p2 = os.path.join(OUT, "matriz_recurrencia_fila416.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {p2}")

    # Guardar datos
    data = {
        "punto_central": [416, 416],
        "columna_perfil_imagen": 960,
        "bloques": [{"y0": int(b[0]), "y1": int(b[1]), "anatomia": ANATOMIA[i][0] if i < len(ANATOMIA) else f"bloque_{i}"} for i, b in enumerate(bloques)],
        "viz1": p1, "viz2": p2,
    }
    with open(os.path.join(OUT, "visualizacion_bloques_resultados.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
