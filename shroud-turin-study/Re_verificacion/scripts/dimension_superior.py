"""
VERIFICACION: ¿EL PUNTO CENTRAL LLEGA MAS ALLA DE LA QUINTA DIMENSION?
=====================================================================
El documento C14 establecio: n_dims=5 explica la redundancia 64.8%
(formula (n_dims-2)/n_dims = 3/5 = 60%). El usuario indica que el
punto central llega MAS ALLA de la quinta dimension.

Tests:
  W1. REDUNDANCIA ENTRE PROYECCIONES vs DIMENSION N
      - Objetos N-D (esferas con capas) para N=3,4,5,6,7
      - Proyeccion a 2D desde muchos angulos (rotaciones N-D)
      - Redundancia = fraccion de pixeles iguales entre sombras
        (misma metrica que el estudio CHIP-5: mean(cells==cells))
      - Comparar con el 64.8% real de la Sabana
      - ¿Que N reproduce 64.8%? ¿N>=5?

  W2. REDUNDANCIA CON BINARIZACION (como el estudio)
      - Las celdas de la Sabana son binarias (recurrencia 0/1)
      - Binarizar las sombras y medir redundancia binaria

  W3. DIMENSION EFECTIVA (D_eff = n_dims - 1 + (n_dims-1)/n_dims)
      - Verificar la formula del documento C14
      - D_eff(3)=2.67, D_eff(5)=4.80, D_eff(6)=5.83

  W4. LA CRUZ REAL: redundancia entre celdas vs prediccion N-D

Usa GPU (RTX 5060 Ti) con muestreo estocastico (no grid completo,
para N=7 el grid seria 48^7 = 587 billones de puntos - imposible).
NO modifica originales. Guarda en Re_verificacion/resultados/.
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
N_MUESTRAS = 20_000_000  # puntos muestreados por sombra
SIZE_2D = 96  # grid de la proyeccion 2D
N_SOMBRAS = 12  # sombras por dimension N
N_DIMS = [3, 4, 5, 6, 7]
torch.manual_seed(42)
rng = np.random.default_rng(42)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
print(f"Muestras por sombra: {N_MUESTRAS/1e6:.0f}M | Grid 2D: {SIZE_2D}x{SIZE_2D}", flush=True)

# ============================================================================
# 1. MUESTREO DE PUNTOS N-D (esfera con capas)
# ============================================================================
def muestrear_puntos_nd(n_dims, n_muestras=N_MUESTRAS):
    """Puntos aleatorios uniformes en [-1,1]^N (para la esfera N-D)."""
    pts = torch.rand(n_muestras, n_dims, device=DEVICE) * 2 - 1  # [-1,1]
    return pts

def valores_esfera_nd(pts, n_capas=5):
    """Esfera N-D con capas: rho(r_N)."""
    r = torch.sqrt((pts**2).sum(dim=1))
    vals = torch.zeros(len(pts), device=DEVICE)
    for rc, vc in zip(torch.linspace(1.0, 0.2, n_capas, device=DEVICE),
                      torch.linspace(0.2, 1.0, n_capas, device=DEVICE)):
        vals[r < rc] = vc
    return vals

# ============================================================================
# 2. ROTACIONES N-D (planos (i,j) para i<j)
# ============================================================================
def rotacion_nd_aleatoria(n_dims, n_planos=3):
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

# ============================================================================
# 3. PROYECCION N-D -> 2D (sombra)
# ============================================================================
def sombra_nd(pts, vals, M, size=SIZE_2D):
    """Rota puntos N-D y proyecta a 2D (usa las 2 primeras coordenadas)."""
    pts_rot = pts @ M.T  # (M, N)
    idx = ((pts_rot[:, :2] + 1) / 2 * (size - 1)).long().clamp(0, size-1)
    flat_idx = idx[:, 0] * size + idx[:, 1]
    proj = torch.zeros(size * size, device=DEVICE, dtype=torch.float32)
    proj.scatter_add_(0, flat_idx, vals)
    proj = proj.view(size, size)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

# ============================================================================
# 4. REDUNDANCIA ENTRE SOMBRAS (misma metrica que el estudio CHIP-5)
# ============================================================================
def redundancia_binaria(s1, s2, umbral=0.5):
    """Fraccion de pixeles iguales tras binarizar (como el estudio)."""
    b1 = (s1 > umbral).float()
    b2 = (s2 > umbral).float()
    return float((b1 == b2).float().mean())

def redundancia_continua(s1, s2):
    """Correlacion de Pearson entre sombras."""
    a = s1.flatten().cpu().numpy()
    b = s2.flatten().cpu().numpy()
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================================
# 5. REDUNDANCIA REAL DE LA SABANA (celdas del grid)
# ============================================================================
def redundancia_sabana():
    """Redundancia real: similitud entre celdas del grid (CHIP-5 = 0.648)."""
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    # Grid del estudio: lineas detectadas CHIP-4
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
    report = {"W1_redundancia_vs_N": {}, "W2_binaria": {}, "W3_D_eff": {},
              "W4_sabana": {}, "conclusion": {}}

    # Redundancia real de la Sabana
    red_sabana, red_sabana_std = redundancia_sabana()
    print("=" * 70, flush=True)
    print(f"REDUNDANCIA REAL DE LA SABANA (CHIP-5): {red_sabana:.4f} ± {red_sabana_std:.4f}", flush=True)
    print(f"(documento C14: 64.8% ~ (n_dims-2)/n_dims para n_dims=5 -> 60%)", flush=True)
    print("=" * 70, flush=True)
    report["W4_sabana"] = {"redundancia": red_sabana, "std": red_sabana_std}

    # W3: D_eff formula
    print("\n[W3] DIMENSION EFECTIVA (formula C14: D_eff = n-1 + (n-1)/n)", flush=True)
    deffs = {}
    for n in N_DIMS:
        deff = (n - 1) + (n - 1) / n
        deffs[n] = deff
        print(f"  n_dims={n}: D_eff={deff:.3f}", flush=True)
    report["W3_D_eff"] = deffs

    # W1/W2: redundancia entre sombras por dimension N
    print(f"\n[W1/W2] REDUNDANCIA ENTRE SOMBRAS vs DIMENSION N (proyeccion N-D -> 2D)", flush=True)
    resultados = {}
    for n in N_DIMS:
        print(f"\n  --- DIMENSION N={n} ---", flush=True)
        pts = muestrear_puntos_nd(n)
        vals = valores_esfera_nd(pts)
        sombras = []
        for s in range(N_SOMBRAS):
            M = rotacion_nd_aleatoria(n, n_planos=min(3, n))
            proj = sombra_nd(pts, vals, M)
            sombras.append(proj)
        # Redundancia binaria entre pares
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
            "n_pares": len(red_bin),
        }
        print(f"    Redundancia binaria entre sombras: {np.mean(red_bin):.4f} ± {np.std(red_bin):.4f}", flush=True)
        print(f"    Correlacion media entre sombras: {np.mean(red_cont):+.4f}", flush=True)
        # Comparar con la Sabana (64.8%)
        diff = abs(np.mean(red_bin) - red_sabana)
        print(f"    |diferencia vs Sabana ({red_sabana:.4f})| = {diff:.4f}", flush=True)
    report["W1_redundancia_vs_N"] = resultados

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿QUE N REPRODUCE LA REDUNDANCIA DE LA SABANA?", flush=True)
    print("=" * 70, flush=True)
    # Mejor N
    mejor_n = min(resultados, key=lambda n: abs(resultados[n]["redundancia_binaria"] - red_sabana))
    print(f"  Redundancia Sabana: {red_sabana:.4f} (64.8%)", flush=True)
    for n in N_DIMS:
        r = resultados[n]["redundancia_binaria"]
        print(f"  N={n}: redundancia={r:.4f} (diferencia {abs(r-red_sabana):.4f})", flush=True)
    print(f"  MEJOR N: {mejor_n}", flush=True)
    # Formula C14: (n-2)/n
    print(f"  Formula C14 (n-2)/n:", flush=True)
    for n in N_DIMS:
        print(f"    n={n}: (n-2)/n = {(n-2)/n:.3f} ({100*(n-2)/n:.0f}%)", flush=True)
    report["conclusion"] = {
        "mejor_n": mejor_n,
        "formula_C14": {n: (n-2)/n for n in N_DIMS},
        "redundancia_sabana": red_sabana,
        "resultados_por_N": resultados,
    }

    out_json = os.path.join(OUT, "dimension_superior_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
