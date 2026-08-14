"""
¿QUE ES EL OBJETO DE 6-7 DIMENSIONES? INDAGACION CONCRETA
=========================================================
El punto central de la Sabana se comporta como la proyeccion de un
objeto de ~6.4-7 dimensiones. ¿Que podria ser concretamente?

Hipotesis concretas (no abstractas):
  H1. CUERPO CON N TEJIDOS INTERNOS: cada tejido (piel, musculo,
      hueso, grasa, cartilago, sangre, organos...) tiene un coeficiente
      de absorcion distinto. La proyeccion suma las contribuciones:
      I(x,y) = sum_i mu_i * t_i(x,y). El numero de tejidos = numero de
      "dimensiones" de densidad. TEST: histograma de intensidades ->
      cuantos modos (tejidos) tiene la imagen?

  H2. CAMPO DE EMISION CON N PARAMETROS: el proceso dependia de
      posicion 3D + direccion (2) + espectro/t (1-2) = 6-7 parametros.
      TEST: PCA de la matriz de recurrencia -> cuantos autovalores
      significativos? Si ~6-7, el campo tiene 6-7 dimensiones efectivas.

  H3. ESTRUCTURA JERARQUICA: 7 niveles de organizacion (molecula,
      fibra, hilo, tejido, imagen, estructura, proceso).
      TEST: analisis multiescala (ya hecho: CV=0.052 fractal).

  H4. SIMULACION: cuerpo 3D con N tejidos proyectado -> redundancia
      entre celdas vs N tejidos. El N que reproduce 64.8% indica
      cuantos tejidos internos tiene el "objeto".

Tests:
  Y1. HISTOGRAMA DE INTENSIDADES: numero de modos (picos) de la imagen
      del cuerpo -> cuantos tejidos/materials distinguibles.
      Con controles (gaussiano, suavizado) para no confundir ruido.
  Y2. PCA DE LA MATRIZ DE RECURRENCIA: autovalores significativos
      (explican >95% varianza) -> dimensionalidad efectiva del campo.
  Y3. PCA DEL PERFIL CENTRAL: componentes principales.
  Y4. CLUSTERING: K-means en intensidades con criterio (elbow/BIC)
      -> numero optimo de "materiales".
  Y5. SIMULACION: cuerpo 3D con N tejidos (N=3..9) proyectado ->
      redundancia entre celdas vs N tejidos. Comparar con 64.8%.
  Y6. LOS 12 BLOQUES DE RECURRENCIA del punto central: corresponden
      a zonas del perfil -> mapear a la imagen (anatomia).

Usa GPU. NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
import torch
from scipy import ndimage
from scipy.signal import find_peaks

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
rng = np.random.default_rng(42)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# ============================================================================
# 1. CARGA DE IMAGENES
# ============================================================================
def cargar_imagenes():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    return img3, img1

def matriz_real(img):
    h, w = img.shape
    profile = ndimage.gaussian_filter1d(img[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    return R, profile

# ============================================================================
# Y1: HISTOGRAMA -> MODOS (tejidos)
# ============================================================================
def histograma_modos(img, n_bins=256, suavizado_sigma=5, prominencia=0.01):
    """Histograma de intensidades y deteccion de picos (modos = tejidos)."""
    hist, edges = np.histogram(img, bins=n_bins, range=(0, 256))
    hist_s = ndimage.gaussian_filter1d(hist.astype(np.float64), suavizado_sigma)
    # Normalizar
    hist_s = hist_s / hist_s.sum()
    # Picos con prominencia relativa
    picos, props = find_peaks(hist_s, prominence=prominencia * hist_s.max())
    return len(picos), picos, hist_s, edges

# ============================================================================
# Y2/Y3: PCA -> dimensionalidad efectiva
# ============================================================================
def dimensionalidad_pca(matriz, umbral_var=0.95):
    """Numero de componentes que explican umbral_var de la varianza."""
    # Submuestrear filas para PCA (matriz grande)
    n = matriz.shape[0]
    if n > 1500:
        idx = rng.choice(n, 1500, replace=False)
        sub = matriz[idx, :][:, idx]
    else:
        sub = matriz
    sub = sub - sub.mean(axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(sub, full_matrices=False)
    var = S**2 / (S**2).sum()
    cum = np.cumsum(var)
    n_comp = int(np.searchsorted(cum, umbral_var) + 1)
    return n_comp, var

def dimensionalidad_pca_perfil(profile, umbral_var=0.95):
    """PCA del perfil (ventanas deslizantes como features)."""
    # Embedding por retardo (Takens-like) para PCA
    m, tau = 10, 5
    n = len(profile)
    if n <= (m-1)*tau:
        return 1, None
    X = np.stack([profile[i:n-(m-1)*tau+i] for i in range(0, (m-1)*tau+1, tau)], axis=1)
    X = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2 / (S**2).sum()
    cum = np.cumsum(var)
    n_comp = int(np.searchsorted(cum, umbral_var) + 1)
    return n_comp, var

# ============================================================================
# Y4: CLUSTERING K-MEANS -> numero de materiales
# ============================================================================
def clustering_materiales(img, k_max=12):
    """K-means en intensidades con criterio de inercia (elbow via BIC aprox)."""
    # Muestrear pixeles
    pixeles = img.flatten().astype(np.float64).reshape(-1, 1)
    if len(pixeles) > 50000:
        idx = rng.choice(len(pixeles), 50000, replace=False)
        pixeles = pixeles[idx]
    # Normalizar
    pixeles = pixeles / 255.0
    # BIC para cada k
    bics = []
    for k in range(1, k_max+1):
        # K-means simple (unas pocas iteraciones)
        centroids = np.linspace(0.05, 0.95, k)
        for _ in range(20):
            dists = np.abs(pixeles - centroids[None, :])
            asign = np.argmin(dists, axis=1)
            for c in range(k):
                mask = asign == c
                if mask.sum() > 0:
                    centroids[c] = pixeles[mask].mean()
        # Inercia
        dists = np.abs(pixeles - centroids[None, :])
        asign = np.argmin(dists, axis=1)
        inercia = sum(((pixeles[asign == c] - centroids[c])**2).sum() for c in range(k) if (asign == c).sum() > 0)
        # BIC aprox: -2*logL + k*log(N)
        sigma2 = inercia / len(pixeles) + 1e-9
        bic = len(pixeles) * np.log(sigma2) + k * np.log(len(pixeles))
        bics.append(bic)
    bics = np.array(bics)
    k_opt = int(np.argmin(bics) + 1)
    return k_opt, bics

# ============================================================================
# Y5: SIMULACION cuerpo 3D con N tejidos
# ============================================================================
def cuerpo_tejidos_nd(n_tejidos, size=96):
    """Cuerpo humanoide simplificado con N tejidos internos.
    Cada tejido es una capa con densidad distinta (como un cuerpo real:
    piel, musculo, hueso, grasa, organos...)."""
    # Cuerpo simplificado: elipsoide central (torso) + cabeza (esfera)
    lin = torch.linspace(-1, 1, size, device=DEVICE)
    coords = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1)
    pts = coords.reshape(-1, 3).float()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Torso: elipsoide
    torso = (x/0.5)**2 + (y/0.8)**2 + (z/0.4)**2
    # Cabeza: esfera en la parte superior
    cabeza = ((x-0.0)**2 + (y-0.95)**2 + (z-0.0)**2) / 0.25**2
    # Combinar
    dentro_cuerpo = (torso < 1) | (cabeza < 1)
    # Distancia normalizada al centro del cuerpo (para capas)
    r_cuerpo = torch.sqrt(torso.clamp(0, 2))
    r_cabeza = torch.sqrt(cabeza.clamp(0, 2))
    r_comb = torch.where(cabeza < torso, r_cabeza, r_cuerpo)
    # N tejidos: capas concentricas con densidades distintas
    vals = torch.zeros(len(pts), device=DEVICE)
    for i in range(n_tejidos):
        radio = 1.0 - i * (0.9 / n_tejidos)
        densidad = 1.0 - i * (0.7 / n_tejidos)
        vals[r_comb < radio] = densidad
    # Solo dentro del cuerpo
    vals[~dentro_cuerpo] = 0.0
    return pts, vals

def proyectar_3d(pts, vals, M, size=96):
    pts_rot = pts @ M.T
    idx = ((pts_rot[:, :2] + 1) / 2 * (size - 1)).long().clamp(0, size-1)
    flat_idx = idx[:, 0] * size + idx[:, 1]
    proj = torch.zeros(size * size, device=DEVICE, dtype=torch.float32)
    proj.scatter_add_(0, flat_idx, vals)
    proj = proj.view(size, size)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def rotacion_so3_aleatoria():
    planos = [(0,1),(0,2),(1,2)]
    M = torch.eye(3, device=DEVICE)
    for k in rng.choice(3, 2, replace=False):
        i, j = planos[k]
        theta = rng.uniform(0, 2*np.pi)
        R = torch.eye(3, device=DEVICE)
        c, s = np.cos(theta), np.sin(theta)
        R[i, i] = c; R[i, j] = -s
        R[j, i] = s; R[j, j] = c
        M = R @ M
    return M

def redundancia_celdas_proyeccion(proj, n_celdas=5, tam=10):
    """Redundancia entre celdas de una proyeccion (como CHIP-5)."""
    h, w = proj.shape
    # Grid de celdas
    cy, cx = h//2, w//2
    cells = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            y0 = cy + i*tam - tam//2
            x0 = cx + j*tam - tam//2
            if 0 <= y0 < h-tam and 0 <= x0 < w-tam:
                cells.append(proj[y0:y0+tam, x0:x0+tam])
    if len(cells) < 2:
        return float("nan")
    # Binarizar como el estudio (0/1 con umbral)
    cells_b = [(c > 0.3).float() for c in cells]
    sims = []
    for i in range(len(cells_b)):
        for j in range(i+1, len(cells_b)):
            sims.append(float((cells_b[i] == cells_b[j]).float().mean()))
    return float(np.mean(sims))

# ============================================================================
# Y6: BLOQUES DE RECURRENCIA -> ANATOMIA
# ============================================================================
def bloques_a_anatomia(img, profile, R, cx=416):
    """Mapea los bloques de recurrencia del punto central a la imagen."""
    fila = R[cx, :]
    # Bloques
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
    # Para cada bloque, la posicion y en la imagen es el centro del bloque
    # La imagen es h x w, el perfil es la columna central (w//2)
    h_img, w_img = img.shape
    resultado = []
    for b0, b1 in bloques:
        y_centro = (b0 + b1) // 2
        # Posicion relativa en la imagen
        rel = y_centro / n
        # Brillo medio del perfil en el bloque
        brillo = float(profile[b0:b1+1].mean())
        resultado.append({"y0": int(b0), "y1": int(b1), "y_centro": int(y_centro),
                          "rel": float(rel), "brillo": brillo})
    return resultado

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"Y1_histograma": {}, "Y2_pca_matriz": {}, "Y3_pca_perfil": {},
              "Y4_clustering": {}, "Y5_cuerpo_tejidos": {}, "Y6_anatomia": {},
              "conclusion": {}}

    img3, img1 = cargar_imagenes()
    R3, profile3 = matriz_real(img3)
    n = R3.shape[0]
    cx = 416

    # ============ Y1: HISTOGRAMA ============
    print("=" * 70, flush=True)
    print("[Y1] HISTOGRAMA DE INTENSIDADES -> modos (tejidos/materiales)", flush=True)
    print("=" * 70, flush=True)
    for nombre, img in [("imagen3_completa", img3), ("imagen1_rostro", img1[100:1100, 1000:2000])]:
        n_picos, picos, hist_s, edges = histograma_modos(img)
        pos_picos = [float(edges[p]) for p in picos]
        print(f"  {nombre}: {n_picos} modos en posiciones {[f'{p:.0f}' for p in pos_picos]}", flush=True)
        report["Y1_histograma"][nombre] = {"n_modos": int(n_picos), "posiciones": pos_picos}
    # Control: gaussiano con misma media/std
    ctrl = np.clip(rng.normal(img3.mean(), img3.std(), size=img3.shape), 0, 255).astype(np.uint8)
    n_picos_c, picos_c, _, edges_c = histograma_modos(ctrl)
    print(f"  Control gaussiano: {n_picos_c} modos", flush=True)
    report["Y1_histograma"]["control_gaussiano"] = {"n_modos": int(n_picos_c)}

    # ============ Y2: PCA MATRIZ ============
    print("\n[Y2] PCA DE LA MATRIZ DE RECURRENCIA (dimensionalidad efectiva)", flush=True)
    n_comp95, var = dimensionalidad_pca(R3, 0.95)
    print(f"  Componentes para 95% varianza: {n_comp95}", flush=True)
    # Autovalores > 1% de varianza
    n_comp1pct = int((var > 0.01).sum())
    print(f"  Componentes con >1% varianza: {n_comp1pct}", flush=True)
    report["Y2_pca_matriz"] = {"n_comp_95pct": int(n_comp95), "n_comp_1pct": int(n_comp1pct),
                                "var_top10": var[:10].tolist()}

    # ============ Y3: PCA PERFIL ============
    print("\n[Y3] PCA DEL PERFIL CENTRAL (embedding Takens)", flush=True)
    n_comp_p, var_p = dimensionalidad_pca_perfil(profile3)
    print(f"  Componentes para 95% varianza: {n_comp_p}", flush=True)
    report["Y3_pca_perfil"] = {"n_comp_95pct": int(n_comp_p), "var_top10": var_p[:10].tolist() if var_p is not None else None}

    # ============ Y4: CLUSTERING ============
    print("\n[Y4] CLUSTERING K-MEANS -> numero optimo de materiales", flush=True)
    k_opt, bics = clustering_materiales(img3)
    print(f"  Numero optimo de clusters (BIC minimo): {k_opt}", flush=True)
    print(f"  BIC por k: {[f'{b:.1f}' for b in bics]}", flush=True)
    report["Y4_clustering"] = {"k_optimo": int(k_opt), "bics": bics.tolist()}

    # ============ Y5: CUERPO CON N TEJIDOS ============
    print("\n[Y5] SIMULACION: cuerpo 3D con N tejidos -> redundancia entre celdas", flush=True)
    print(f"  (Objetivo: redundancia ~64.8% como la Sabana)", flush=True)
    # Redundancia real
    red_sabana, _ = (0.6484, 0.0)  # del estudio CHIP-5
    print(f"  Redundancia Sabana (CHIP-5): {red_sabana:.4f}", flush=True)
    resultados_tejidos = {}
    for n_tej in [3, 4, 5, 6, 7, 8, 9]:
        pts, vals = cuerpo_tejidos_nd(n_tej)
        reds = []
        for s in range(10):
            M = rotacion_so3_aleatoria()
            proj = proyectar_3d(pts, vals, M)
            red = redundancia_celdas_proyeccion(proj)
            if not np.isnan(red):
                reds.append(red)
        if reds:
            reds = np.array(reds)
            diff = abs(np.mean(reds) - red_sabana)
            resultados_tejidos[n_tej] = {"redundancia": float(np.mean(reds)),
                                          "std": float(np.std(reds)), "diff": float(diff)}
            print(f"  N_tejidos={n_tej}: redundancia={np.mean(reds):.4f}±{np.std(reds):.4f} | diff={diff:.4f}", flush=True)
    # Mejor N_tejidos
    mejor_tej = min(resultados_tejidos, key=lambda k: resultados_tejidos[k]["diff"])
    print(f"  MEJOR N de tejidos: {mejor_tej}", flush=True)
    report["Y5_cuerpo_tejidos"] = resultados_tejidos
    report["Y5_mejor"] = int(mejor_tej)

    # ============ Y6: ANATOMIA ============
    print("\n[Y6] BLOQUES DE RECURRENCIA DEL PUNTO CENTRAL -> ANATOMIA", flush=True)
    anatomia = bloques_a_anatomia(img3, profile3, R3, cx)
    for a in anatomia:
        print(f"  Bloque y={a['y0']}-{a['y1']} (centro={a['y_centro']}, rel={a['rel']:.2f}, brillo={a['brillo']:.0f})", flush=True)
    report["Y6_anatomia"] = anatomia

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿QUE ES EL OBJETO DE 6-7 DIMENSIONES?", flush=True)
    print("=" * 70, flush=True)
    print(f"  Y1 modos histograma: ver reporte", flush=True)
    print(f"  Y2 PCA matriz: {n_comp95} comp (95%), {n_comp1pct} comp (>1%)", flush=True)
    print(f"  Y3 PCA perfil: {n_comp_p} comp (95%)", flush=True)
    print(f"  Y4 clustering: {k_opt} materiales optimos", flush=True)
    print(f"  Y5 cuerpo con N tejidos: mejor N={mejor_tej}", flush=True)
    report["conclusion"] = {
        "Y1": report["Y1_histograma"],
        "Y2_n_comp": int(n_comp95),
        "Y4_k_opt": int(k_opt),
        "Y5_mejor_tejidos": int(mejor_tej),
    }

    out_json = os.path.join(OUT, "que_es_el_objeto_nd_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
