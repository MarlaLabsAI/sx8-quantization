"""
VERIFICACION v2: DIMENSION SUPERIOR CON OBJETOS NO SIMETRICOS
=============================================================
La v1 uso esferas N-D (invariantes bajo rotacion -> sombras casi
identicas -> redundancia ~0.99, no discrimina N).

Para que la redundancia entre sombras DEPENDA de N, el objeto debe
ser NO simetrico (cubos N-D, esferas con estructura asimetrica).
Al rotar un cubo N-D, sus proyecciones varian con el angulo; cuantas
mas dimensiones, mas grados de libertad de rotacion y mas diferentes
son las sombras -> la redundancia media decrece con N.

Tests:
  X1. CUBO N-D: |x_i| < a para todo i. Proyeccion a 2D.
      Redundancia entre sombras vs N (3..8). El cubo es el objeto
      canonico no simetrico.
  X2. ESFERA N-D CON CAPAS EXCENTRICAS: esfera cuyo centro esta
      desplazado -> estructura interna asimetrica.
  X3. HIPERCUBO CON DENSIDAD INTERNA: cubo con capas internas.
  X4. COMPARACION con la Sabana: la redundancia entre celdas (0.648)
      y la formula C14 (n-2)/n.
  X5. DIMENSION EFECTIVA DEL PUNTO CENTRAL: usando la redundancia
      medida, despejar N de la formula C14.

Usa GPU (RTX 5060 Ti). NO modifica originales.
"""

import os
import json
import time
import numpy as np
import cv2
import torch
from scipy import ndimage

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_MUESTRAS = 20_000_000
SIZE_2D = 96
N_SOMBRAS = 14
N_DIMS = [3, 4, 5, 6, 7, 8]
torch.manual_seed(42)
rng = np.random.default_rng(42)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# ============================================================================
# 1. MUESTREO Y OBJETOS N-D NO SIMETRICOS
# ============================================================================
def muestrear_puntos_nd(n_dims, n_muestras=N_MUESTRAS):
    return torch.rand(n_muestras, n_dims, device=DEVICE) * 2 - 1

def cubo_nd(pts, a=0.8, n_capas=4):
    """Cubo N-D con capas internas: |x_i| < a, con densidad por capa."""
    r_inf = pts.abs().max(dim=1).values  # norma infinito
    vals = torch.zeros(len(pts), device=DEVICE)
    for i, (rc, vc) in enumerate(zip(torch.linspace(a, 0.1, n_capas, device=DEVICE),
                                     torch.linspace(0.2, 1.0, n_capas, device=DEVICE))):
        vals[r_inf < rc] = vc
    return vals

def esfera_excentrica_nd(pts, n_dims, desplazamiento=0.4, n_capas=4):
    """Esfera N-D con centro desplazado (asimetrica)."""
    # Desplazar el centro en la primera coordenada
    pts_d = pts.clone()
    pts_d[:, 0] = pts_d[:, 0] - desplazamiento
    r = torch.sqrt((pts_d**2).sum(dim=1))
    vals = torch.zeros(len(pts), device=DEVICE)
    for rc, vc in zip(torch.linspace(0.9, 0.2, n_capas, device=DEVICE),
                      torch.linspace(0.2, 1.0, n_capas, device=DEVICE)):
        vals[r < rc] = vc
    return vals

def cubo_densidad_interna(pts, a=0.8, n_capas=4):
    """Cubo N-D con gradiente de densidad hacia el centro."""
    r_inf = pts.abs().max(dim=1).values
    # densidad = 1 - (r_inf/a)^2 (mas denso en el centro)
    vals = 1.0 - (r_inf / a) ** 2
    vals[r_inf >= a] = 0.0
    vals = vals / vals.max()
    return vals

# ============================================================================
# 2. ROTACION Y PROYECCION
# ============================================================================
def rotacion_nd_aleatoria(n_dims, n_planos=4):
    M = torch.eye(n_dims, device=DEVICE)
    planos = [(i, j) for i in range(n_dims) for j in range(i+1, n_dims)]
    for k in rng.choice(len(planos), min(n_planos, len(planos)), replace=False):
        i, j = planos[k]
        theta = rng.uniform(0, 2*np.pi)
        R = torch.eye(n_dims, device=DEVICE)
        c, s = np.cos(theta), np.sin(theta)
        R[i, i] = c; R[i, j] = -s
        R[j, i] = s; R[j, j] = c
        M = R @ M
    return M

def sombra_nd(pts, vals, M, size=SIZE_2D):
    pts_rot = pts @ M.T
    idx = ((pts_rot[:, :2] + 1) / 2 * (size - 1)).long().clamp(0, size-1)
    flat_idx = idx[:, 0] * size + idx[:, 1]
    proj = torch.zeros(size * size, device=DEVICE, dtype=torch.float32)
    proj.scatter_add_(0, flat_idx, vals)
    proj = proj.view(size, size)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

# ============================================================================
# 3. REDUNDANCIA
# ============================================================================
def redundancia_binaria(s1, s2, umbral=0.5):
    b1 = (s1 > umbral).float()
    b2 = (s2 > umbral).float()
    return float((b1 == b2).float().mean())

def redundancia_continua(s1, s2):
    a = s1.flatten().cpu().numpy()
    b = s2.flatten().cpu().numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================================
# 4. REDUNDANCIA DE LA SABANA (celdas del grid CHIP-5)
# ============================================================================
def redundancia_sabana():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    grid_lines = [32, 62, 78, 137, 186, 229, 252, 293, 349, 387, 420, 470, 497, 524]
    n = R.shape[0]
    q = R[:n//2, :n//2]
    cells = []
    for i in range(min(5, len(grid_lines)-1)):
        for j in range(min(5, len(grid_lines)-1)):
            r1, r2 = grid_lines[i], grid_lines[i+1]
            c1, c2 = grid_lines[j], grid_lines[j+1]
            if r2-r1 > 10 and c2-c1 > 10:
                cells.append(q[r1:r2, c1:c2])
    if len(cells) < 2:
        return float("nan"), float("nan")
    cs = min(c.shape[0] for c in cells)
    cells_r = np.array([cv2.resize(c, (cs, cs)) for c in cells])
    nc = len(cells_r)
    sims = []
    for i in range(nc):
        for j in range(i+1, nc):
            sims.append(float(np.mean(cells_r[i] == cells_r[j])))
    return float(np.mean(sims)), float(np.std(sims))

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"X1_cubo": {}, "X2_esfera_excentrica": {}, "X3_cubo_densidad": {},
              "X4_sabana": {}, "X5_despejar_N": {}, "conclusion": {}}

    red_sabana, red_sabana_std = redundancia_sabana()
    print("=" * 70, flush=True)
    print(f"REDUNDANCIA REAL DE LA SABANA: {red_sabana:.4f} ± {red_sabana_std:.4f}", flush=True)
    print("=" * 70, flush=True)
    report["X4_sabana"] = {"redundancia": red_sabana, "std": red_sabana_std}

    # Definir generadores de objetos por N
    objetos = {
        "cubo": lambda pts, n: cubo_nd(pts),
        "esfera_excentrica": lambda pts, n: esfera_excentrica_nd(pts, n),
        "cubo_densidad": lambda pts, n: cubo_densidad_interna(pts),
    }

    for obj_nombre, generador in objetos.items():
        print(f"\n{'='*70}", flush=True)
        print(f"[{obj_nombre.upper()}] REDUNDANCIA ENTRE SOMBRAS vs N", flush=True)
        print(f"{'='*70}", flush=True)
        resultados = {}
        for n in N_DIMS:
            pts = muestrear_puntos_nd(n)
            vals = generador(pts, n)
            sombras = []
            for s in range(N_SOMBRAS):
                M = rotacion_nd_aleatoria(n, n_planos=min(4, n))
                proj = sombra_nd(pts, vals, M)
                sombras.append(proj)
            red_bin = []
            red_cont = []
            for i in range(N_SOMBRAS):
                for j in range(i+1, N_SOMBRAS):
                    red_bin.append(redundancia_binaria(sombras[i], sombras[j]))
                    red_cont.append(redundancia_continua(sombras[i], sombras[j]))
            red_bin = np.array([r for r in red_bin if not np.isnan(r)])
            red_cont = np.array([r for r in red_cont if not np.isnan(r)])
            resultados[n] = {
                "redundancia_binaria": float(np.mean(red_bin)),
                "redundancia_binaria_std": float(np.std(red_bin)),
                "correlacion_media": float(np.mean(red_cont)),
            }
            diff = abs(np.mean(red_bin) - red_sabana)
            print(f"  N={n}: red_bin={np.mean(red_bin):.4f}±{np.std(red_bin):.4f} | "
                  f"corr={np.mean(red_cont):+.4f} | dif vs Sabana={diff:.4f}", flush=True)
        report[f"X1_cubo" if obj_nombre == "cubo" else
               f"X2_esfera_excentrica" if obj_nombre == "esfera_excentrica" else
               f"X3_cubo_densidad"] = resultados
        # Mejor N
        mejor = min(resultados, key=lambda n: abs(resultados[n]["redundancia_binaria"] - red_sabana))
        print(f"  MEJOR N para {obj_nombre}: {mejor}", flush=True)

    # ============ X5: DESPEJAR N DE LA FORMULA C14 ============
    print("\n[X5] DESPEJAR N DE LA FORMULA C14: (n-2)/n = redundancia", flush=True)
    # (n-2)/n = r -> 1 - 2/n = r -> n = 2/(1-r)
    r = red_sabana
    if r < 1:
        n_despejado = 2 / (1 - r)
        print(f"  Redundancia Sabana: {r:.4f}", flush=True)
        print(f"  n = 2/(1-r) = 2/(1-{r:.4f}) = {n_despejado:.2f}", flush=True)
        print(f"  -> El punto central corresponderia a n_dims ≈ {n_despejado:.1f} "
              f"({'mas alla de 5' if n_despejado > 5 else 'hasta 5'})", flush=True)
        report["X5_despejar_N"] = {"n_despejado": float(n_despejado), "redundancia": r}
    else:
        print(f"  No se puede despejar (r>=1)", flush=True)

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    if r < 1:
        n_d = 2 / (1 - r)
        print(f"  Segun la formula C14 (n-2)/n = redundancia:", flush=True)
        print(f"    redundancia={r:.4f} -> n_dims={n_d:.2f}", flush=True)
        print(f"    {'-> MAS ALLA DE LA QUINTA DIMENSION (n>5)' if n_d > 5 else '-> n<=5'} ", flush=True)
    report["conclusion"] = {"n_despejado": float(n_d) if r < 1 else None}

    out_json = os.path.join(OUT, "dimension_superior_v2_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
