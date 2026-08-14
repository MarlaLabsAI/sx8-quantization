"""
VISUALIZACION CORREGIDA: 13 BLOQUES ANATOMICOS (PERFIL HORIZONTAL)
=================================================================
CORRECCION: el cuerpo en la imagen3 (1920x1080 apaisada) esta orientado
HORIZONTALMENTE (acostado: cabeza a un lado, pies al otro). El perfil
correcto para mapear anatomia cabeza->pies es la FILA central (y=540),
NO la columna vertical.

Salidas:
  - bloques_anatomicos_corregidos.png (cuerpo + bloques horizontales)
  - matriz_recurrencia_horizontal.png (matriz + fila 960 resaltada)

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

def main():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    print(f"Imagen: {w}x{h}")

    # PERFIL HORIZONTAL (fila central y=540) - anatomia cabeza->pies
    y_perfil = h // 2
    perfil_h = ndimage.gaussian_filter1d(img3[y_perfil, :].astype(np.float32), sigma=15)
    R_h = (np.abs(perfil_h[:, None] - perfil_h[None, :]) < 10.0).astype(np.float32)
    n = R_h.shape[0]
    cx_h, cy_h = n // 2, n // 2
    fila = R_h[cy_h, :]
    bloques = bloques_recurrencia(fila)
    print(f"Matriz horizontal: {n}x{n} | punto central: ({cx_h},{cy_h}) | bloques: {len(bloques)}")

    # Anatomia (orientacion horizontal: izquierda -> derecha = cabeza -> pies)
    # Las posiciones relativas de los bloques:
    ANATOMIA = [
        "Extremo izq", "Cabeza", "Cuello", "Hombros", "Pecho alto",
        "Pecho", "Torso sup", "PUNTO CENTRAL (pecho)", "Torso inf",
        "Caderas", "Muslos", "Rodillas", "Piernas", "Tobillos", "Pies"
    ]
    # Asignar etiquetas segun posicion relativa (0=izquierda, 1=derecha)
    def etiqueta_anatomica(rel):
        if rel < 0.05: return "Extremo izq"
        if rel < 0.15: return "Cabeza"
        if rel < 0.20: return "Cuello"
        if rel < 0.30: return "Hombros"
        if rel < 0.38: return "Pecho alto"
        if rel < 0.45: return "Pecho"
        if rel < 0.50: return "Torso sup"
        if rel < 0.55: return "PUNTO CENTRAL"
        if rel < 0.62: return "Torso inf"
        if rel < 0.68: return "Caderas"
        if rel < 0.75: return "Muslos"
        if rel < 0.80: return "Rodillas"
        if rel < 0.88: return "Piernas"
        if rel < 0.95: return "Tobillos"
        return "Pies"

    cmap = plt.cm.viridis
    colores = [cmap(i / max(1, len(bloques)-1)) for i in range(len(bloques))]

    # ============ VIZ 1: CUERPO + BLOQUES HORIZONTALES ============
    img_color = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    img_color = cv2.convertScaleAbs(img_color, alpha=0.8, beta=40)

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.imshow(img_color)
    ax.set_title(f"SABANA SANTA - BLOQUES DE RECURRENCIA DEL PUNTO CENTRAL (perfil HORIZONTAL y={y_perfil})",
                 fontsize=13, fontweight='bold')

    # Fila del perfil
    ax.axhline(y=y_perfil, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Punto central en la imagen: (x=cx_h, y=y_perfil)
    px, py = cx_h, y_perfil
    ax.plot(px, py, 'o', markersize=14, color='red', markeredgecolor='white', markeredgewidth=2, zorder=10)
    ax.plot(px, py, '+', markersize=20, color='white', markeredgewidth=3, zorder=11)

    # Bloques como bandas VERTICALES (a lo largo de x) + tuneles desde el punto central
    for i, (b0, b1) in enumerate(bloques):
        color = colores[i]
        rel = (b0 + b1) / 2 / n
        nombre = etiqueta_anatomica(rel)
        x_centro_bloque = (b0 + b1) / 2
        # Banda vertical (ancha 15px alrededor de y_perfil)
        ax.axvspan(b0, b1, ymin=(y_perfil-15)/h, ymax=(y_perfil+15)/h,
                   color=color, alpha=0.55, zorder=5)
        # Tunel: linea desde el punto central al bloque
        ax.plot([px, x_centro_bloque], [py, py], '-', color='yellow', linewidth=1.0,
                alpha=0.7, zorder=4)
        # Etiqueta
        ax.text(x_centro_bloque, y_perfil + 30, nombre, fontsize=8, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=to_rgba(color, 0.8), edgecolor='none'),
                ha='center', va='bottom', zorder=12, rotation=45)

    # Leyenda
    handles = [mpatches.Patch(color=colores[i],
               label=f"{etiqueta_anatomica((bloques[i][0]+bloques[i][1])/2/n)} (x={bloques[i][0]}-{bloques[i][1]})")
               for i in range(len(bloques))]
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.8)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("x (px) - a lo largo del cuerpo (izquierda->derecha = cabeza->pies)")
    ax.set_ylabel("y (px)")
    plt.tight_layout()
    p1 = os.path.join(OUT, "bloques_anatomicos_corregidos.png")
    plt.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {p1}")

    # ============ VIZ 2: MATRIZ HORIZONTAL + FILA 960 ============
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    step = 2
    R_viz = R_h[::step, ::step]
    ax[0].imshow(R_viz, cmap='gray_r', interpolation='nearest')
    ax[0].set_title("Matriz de recurrencia (perfil HORIZONTAL)", fontsize=12)
    ax[0].axhline(y=cy_h//step, color='red', linewidth=1.5)
    ax[0].axvline(x=cx_h//step, color='red', linewidth=1.5)
    ax[0].plot(cx_h//step, cy_h//step, 'o', color='yellow', markersize=6)
    ax[0].set_xlabel("j (a lo largo del cuerpo)")
    ax[0].set_ylabel("i")

    fila_viz = fila
    ax[1].plot(fila_viz, 'k-', linewidth=1)
    ax[1].fill_between(range(n), fila_viz, alpha=0.3, color='blue')
    for i, (b0, b1) in enumerate(bloques):
        ax[1].axvspan(b0, b1, color=colores[i], alpha=0.5)
        ax[1].text((b0+b1)/2, 1.05, f"{i}", fontsize=8, ha='center')
    ax[1].set_title(f"Fila {cy_h}: {len(bloques)} bloques de recurrencia (anatomia cabeza->pies)", fontsize=12)
    ax[1].set_xlabel("j (posicion a lo largo del cuerpo)")
    ax[1].set_ylabel(f"R({cy_h}, j)")
    ax[1].set_ylim(-0.1, 1.3)
    ax[1].grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(OUT, "matriz_recurrencia_horizontal.png")
    plt.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {p2}")

    # Datos
    data = {
        "perfil": "horizontal",
        "y_perfil": y_perfil,
        "punto_central_matriz": [int(cx_h), int(cy_h)],
        "punto_central_imagen": [int(px), int(py)],
        "bloques": [{"x0": int(b[0]), "x1": int(b[1]),
                     "rel": float((b[0]+b[1])/2/n),
                     "anatomia": etiqueta_anatomica((b[0]+b[1])/2/n)}
                    for b in bloques],
        "viz1": p1, "viz2": p2,
    }
    with open(os.path.join(OUT, "visualizacion_bloques_corregida_resultados.json"), "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
