"""
SOMBRAS DE OBJETOS 4D EN GPU (RTX 5060 Ti) - CAMPO COMPLETO
===========================================================
Idea del usuario: para distinguir un objeto 4D del azar, hay que
MOVERLO (rotaciones SO(4)) y capturar SUS SOMBRAS (proyecciones).
La familia de sombras de un objeto 4D es coherente; el azar no.

Implementacion con PyTorch CUDA:
  - Grid 4D completo 96^4 (85M puntos) en GPU
  - Rotaciones SO(4) aleatorias (planos xy,xz,xw,yz,yw,zw)
  - Proyeccion 4D->2D por scatter_add (sombra)
  - 30 sombras por familia: hiperesfera 4D (S3 capas), esfera 3D, azar
  - Comparar con la cruz real: correlacion perfil radial, ratio centro/perif

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
SIZE = 96
N_SOMBRAS = 30
torch.manual_seed(42)
rng = np.random.default_rng(42)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)
print(f"Grid: {SIZE}^4 = {SIZE**4} puntos", flush=True)

# ============================================================================
# 1. OBJETOS EN GPU (campo completo)
# ============================================================================
def grid_4d(size=SIZE):
    """Grid 4D completo como puntos (N,4) en GPU."""
    lin = torch.linspace(-1, 1, size, device=DEVICE)
    coords = torch.stack(torch.meshgrid(lin, lin, lin, lin, indexing='ij'), dim=-1)
    return coords.reshape(-1, 4)

def grid_3d(size=SIZE):
    lin = torch.linspace(-1, 1, size, device=DEVICE)
    coords = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1)
    return coords.reshape(-1, 3)

def objeto_4d_hiperesfera(pts4, n_capas=5):
    """Hiperesfera S3 con capas concentricas: rho(r4)."""
    r4 = torch.sqrt((pts4**2).sum(dim=1))
    vals = torch.zeros(len(pts4), device=DEVICE)
    capas_radio = torch.linspace(1.0, 0.2, n_capas, device=DEVICE)
    capas_valor = torch.linspace(0.2, 1.0, n_capas, device=DEVICE)
    for r_capa, v_capa in zip(capas_radio, capas_valor):
        vals[r4 < r_capa] = v_capa
    return vals

def objeto_3d_esfera(pts3, n_capas=5):
    r3 = torch.sqrt((pts3**2).sum(dim=1))
    vals = torch.zeros(len(pts3), device=DEVICE)
    capas_radio = torch.linspace(1.0, 0.2, n_capas, device=DEVICE)
    capas_valor = torch.linspace(0.2, 1.0, n_capas, device=DEVICE)
    for r_capa, v_capa in zip(capas_radio, capas_valor):
        vals[r3 < r_capa] = v_capa
    return vals

# ============================================================================
# 2. ROTACIONES SO(4) y SO(3) en GPU
# ============================================================================
def rotacion_so4_aleatoria(n_planos=3):
    """Composicion de rotaciones en planos aleatorios de SO(4)."""
    planos = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    M = torch.eye(4, device=DEVICE)
    idx = rng.choice(len(planos), n_planos, replace=False)
    for k in idx:
        i, j = planos[k]
        theta = rng.uniform(0, 2*np.pi)
        R = torch.eye(4, device=DEVICE)
        c, s = np.cos(theta), np.sin(theta)
        R[i, i] = c; R[i, j] = -s
        R[j, i] = s; R[j, j] = c
        M = R @ M
    return M

def rotacion_so3_aleatoria(n_planos=2):
    planos = [(0,1),(0,2),(1,2)]
    M = torch.eye(3, device=DEVICE)
    idx = rng.choice(len(planos), n_planos, replace=False)
    for k in idx:
        i, j = planos[k]
        theta = rng.uniform(0, 2*np.pi)
        R = torch.eye(3, device=DEVICE)
        c, s = np.cos(theta), np.sin(theta)
        R[i, i] = c; R[i, j] = -s
        R[j, i] = s; R[j, j] = c
        M = R @ M
    return M

# ============================================================================
# 3. PROYECCION (sombra) con scatter_add
# ============================================================================
def sombra_4d(pts4, vals, M, size=SIZE):
    """Rota y proyecta 4D->2D (suma sobre ejes z,w)."""
    pts_rot = pts4 @ M.T  # (N,4)
    # Indices 2D: x=dim0, y=dim1
    idx = ((pts_rot[:, :2] + 1) / 2 * (size - 1)).long().clamp(0, size-1)
    flat_idx = idx[:, 0] * size + idx[:, 1]
    proj = torch.zeros(size * size, device=DEVICE, dtype=torch.float32)
    proj.scatter_add_(0, flat_idx, vals)
    proj = proj.view(size, size)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj.cpu().numpy()

def sombra_3d(pts3, vals, M, size=SIZE):
    pts_rot = pts3 @ M.T
    idx = ((pts_rot[:, :2] + 1) / 2 * (size - 1)).long().clamp(0, size-1)
    flat_idx = idx[:, 0] * size + idx[:, 1]
    proj = torch.zeros(size * size, device=DEVICE, dtype=torch.float32)
    proj.scatter_add_(0, flat_idx, vals)
    proj = proj.view(size, size)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj.cpu().numpy()

# ============================================================================
# 4. FIRMA DE UNA SOMBRA
# ============================================================================
def perfil_radial_sombra(proj, n_anillos=20, ancho=2):
    h, w = proj.shape
    cx, cy = w//2, h//2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 0:
            res.append(float(proj[mask].mean()))
    return np.array(res)

def firma_sombra(proj):
    h, w = proj.shape
    cyy, cxx = h//2, w//2
    centro = proj[cyy-15:cyy+15, cxx-15:cxx+15]
    perif = proj[:20, :20]
    perfil = perfil_radial_sombra(proj)
    return {
        "dens_centro": float(centro.mean()),
        "dens_periferia": float(perif.mean()),
        "ratio_centro_perif": float(centro.mean() / (perif.mean() + 1e-9)),
        "perfil": perfil.tolist(),
    }

# ============================================================================
# 5. CRUZ REAL
# ============================================================================
def perfil_real_cruz():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    n = R.shape[0]
    cx, cy = 416, 416
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    # Mismo esquema de anillos que las sombras (24 anillos, ancho 2)
    n_anillos, ancho = 20, 2
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 100:
            res.append(float(R[mask].mean()))
    return np.array(res), R, cx, cy

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"U1_sombras_4d": {}, "U2_sombras_3d": {}, "U3_azar": {},
              "U4_cruz_real": {}, "U5_mejor_angulo": {}, "conclusion": {}}

    # Cruz real
    perfil_real, R_real, cx, cy = perfil_real_cruz()
    # Normalizar perfil real a 0-1 para comparar con sombras
    perfil_real_n = perfil_real / (perfil_real.max() + 1e-12)
    print("=" * 70, flush=True)
    print(f"CRUZ REAL: perfil radial {perfil_real_n.round(3)[:8]}...", flush=True)
    print("=" * 70, flush=True)

    # Generar grids en GPU
    print("\nGenerando grids en GPU...", flush=True)
    pts4 = grid_4d()
    pts3 = grid_3d()
    vals4 = objeto_4d_hiperesfera(pts4)
    vals3 = objeto_3d_esfera(pts3)
    print(f"  pts4: {pts4.shape} | pts3: {pts3.shape}", flush=True)

    # ============ U1: SOMBRAS 4D ============
    print(f"\n[U1] HIPERESFERA 4D (S3 capas): {N_SOMBRAS} sombras SO(4)", flush=True)
    sombras_4d = []
    for i in range(N_SOMBRAS):
        M = rotacion_so4_aleatoria(n_planos=3)
        proj = sombra_4d(pts4, vals4, M)
        firma = firma_sombra(proj)
        perfil = np.array(firma["perfil"])
        perfil_n = perfil / (perfil.max() + 1e-12)
        min_len = min(len(perfil_n), len(perfil_real_n))
        if min_len >= 5:
            corr = float(np.corrcoef(perfil_n[:min_len], perfil_real_n[:min_len])[0, 1])
        else:
            corr = float("nan")
        firma["corr_real"] = corr
        sombras_4d.append(firma)
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{N_SOMBRAS} sombras", flush=True)
    corrs_4d = [s["corr_real"] for s in sombras_4d if not np.isnan(s["corr_real"])]
    ratios_4d = [s["ratio_centro_perif"] for s in sombras_4d]
    print(f"  corr con real: media={np.mean(corrs_4d):+.3f} | max={np.max(corrs_4d):+.3f} | "
          f"p95={np.percentile(corrs_4d, 95):+.3f}", flush=True)
    print(f"  ratio centro/perif: media={np.mean(ratios_4d):.2f} | std={np.std(ratios_4d):.2f}", flush=True)
    report["U1_sombras_4d"] = {"corrs": corrs_4d, "ratios": ratios_4d,
                                "corr_mean": float(np.mean(corrs_4d)), "corr_max": float(np.max(corrs_4d)),
                                "ratio_mean": float(np.mean(ratios_4d)), "ratio_std": float(np.std(ratios_4d))}

    # ============ U2: SOMBRAS 3D ============
    print(f"\n[U2] ESFERA 3D (capas): {N_SOMBRAS} sombras SO(3)", flush=True)
    sombras_3d = []
    for i in range(N_SOMBRAS):
        M = rotacion_so3_aleatoria(n_planos=2)
        proj = sombra_3d(pts3, vals3, M)
        firma = firma_sombra(proj)
        perfil = np.array(firma["perfil"])
        perfil_n = perfil / (perfil.max() + 1e-12)
        min_len = min(len(perfil_n), len(perfil_real_n))
        if min_len >= 5:
            corr = float(np.corrcoef(perfil_n[:min_len], perfil_real_n[:min_len])[0, 1])
        else:
            corr = float("nan")
        firma["corr_real"] = corr
        sombras_3d.append(firma)
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{N_SOMBRAS} sombras", flush=True)
    corrs_3d = [s["corr_real"] for s in sombras_3d if not np.isnan(s["corr_real"])]
    ratios_3d = [s["ratio_centro_perif"] for s in sombras_3d]
    print(f"  corr con real: media={np.mean(corrs_3d):+.3f} | max={np.max(corrs_3d):+.3f} | "
          f"p95={np.percentile(corrs_3d, 95):+.3f}", flush=True)
    print(f"  ratio centro/perif: media={np.mean(ratios_3d):.2f} | std={np.std(ratios_3d):.2f}", flush=True)
    report["U2_sombras_3d"] = {"corrs": corrs_3d, "ratios": ratios_3d,
                                "corr_mean": float(np.mean(corrs_3d)), "corr_max": float(np.max(corrs_3d)),
                                "ratio_mean": float(np.mean(ratios_3d)), "ratio_std": float(np.std(ratios_3d))}

    # ============ U3: AZAR ============
    print(f"\n[U3] AZAR: {N_SOMBRAS} sombras aleatorias con gradiente radial", flush=True)
    sombras_azar = []
    for i in range(N_SOMBRAS):
        proj = rng.random((SIZE, SIZE))
        hh, ww = proj.shape
        cyy, cxx = hh//2, ww//2
        yy, xx = np.mgrid[0:hh, 0:ww]
        dist = np.sqrt((xx-cxx)**2 + (yy-cyy)**2)
        grad = 1 - dist / dist.max() * 0.3
        proj = proj * grad
        proj = proj / proj.max()
        firma = firma_sombra(proj)
        perfil = np.array(firma["perfil"])
        perfil_n = perfil / (perfil.max() + 1e-12)
        min_len = min(len(perfil_n), len(perfil_real_n))
        if min_len >= 5:
            corr = float(np.corrcoef(perfil_n[:min_len], perfil_real_n[:min_len])[0, 1])
        else:
            corr = float("nan")
        firma["corr_real"] = corr
        sombras_azar.append(firma)
    corrs_az = [s["corr_real"] for s in sombras_azar if not np.isnan(s["corr_real"])]
    ratios_az = [s["ratio_centro_perif"] for s in sombras_azar]
    print(f"  corr con real: media={np.mean(corrs_az):+.3f} | max={np.max(corrs_az):+.3f} | "
          f"p95={np.percentile(corrs_az, 95):+.3f}", flush=True)
    print(f"  ratio centro/perif: media={np.mean(ratios_az):.2f} | std={np.std(ratios_az):.2f}", flush=True)
    report["U3_azar"] = {"corrs": corrs_az, "ratios": ratios_az,
                          "corr_mean": float(np.mean(corrs_az)), "corr_max": float(np.max(corrs_az)),
                          "ratio_mean": float(np.mean(ratios_az)), "ratio_std": float(np.std(ratios_az))}

    # ============ U4: CRUZ REAL vs FAMILIAS ============
    print("\n[U4] LA CRUZ REAL DENTRO DE LA DISTRIBUCION DE SOMBRAS", flush=True)
    centro_real = R_real[cy-10:cy+10, cx-10:cx+10]
    perif_real = R_real[:15, :15]
    ratio_real = float(centro_real.mean() / (perif_real.mean() + 1e-9))
    print(f"  Cruz real: ratio centro/perif = {ratio_real:.2f}", flush=True)
    print(f"  Sombras 4D: ratio = {np.mean(ratios_4d):.2f}±{np.std(ratios_4d):.2f}", flush=True)
    print(f"  Sombras 3D: ratio = {np.mean(ratios_3d):.2f}±{np.std(ratios_3d):.2f}", flush=True)
    print(f"  Azar:       ratio = {np.mean(ratios_az):.2f}±{np.std(ratios_az):.2f}", flush=True)
    z_4d = (ratio_real - np.mean(ratios_4d)) / np.std(ratios_4d) if np.std(ratios_4d) > 0 else float("nan")
    z_3d = (ratio_real - np.mean(ratios_3d)) / np.std(ratios_3d) if np.std(ratios_3d) > 0 else float("nan")
    z_az = (ratio_real - np.mean(ratios_az)) / np.std(ratios_az) if np.std(ratios_az) > 0 else float("nan")
    print(f"  z-score ratio real vs 4D={z_4d:+.2f} | vs 3D={z_3d:+.2f} | vs azar={z_az:+.2f}", flush=True)
    report["U4_cruz_real"] = {"ratio_real": ratio_real,
                               "z_vs_4d": float(z_4d), "z_vs_3d": float(z_3d), "z_vs_azar": float(z_az)}

    # ============ U5: MEJOR ANGULO ============
    print("\n[U5] MEJOR SOMBRA DE CADA FAMILIA (mejor angulo de vision)", flush=True)
    mejor_4d = np.max(corrs_4d)
    mejor_3d = np.max(corrs_3d)
    mejor_az = np.max(corrs_az)
    print(f"  Mejor sombra 4D: corr={mejor_4d:+.3f}", flush=True)
    print(f"  Mejor sombra 3D: corr={mejor_3d:+.3f}", flush=True)
    print(f"  Mejor sombra azar: corr={mejor_az:+.3f}", flush=True)
    pct = float((np.array(corrs_az) < mejor_4d).mean() * 100)
    print(f"  La mejor sombra 4D supera al {pct:.1f}% de las sombras de azar", flush=True)
    # Y la mejor 4D vs la mejor azar
    print(f"  Mejor 4D ({mejor_4d:+.3f}) vs mejor azar ({mejor_az:+.3f}): "
          f"{'4D GANA' if mejor_4d > mejor_az else 'azar gana'}", flush=True)
    report["U5_mejor_angulo"] = {"mejor_4d": float(mejor_4d), "mejor_3d": float(mejor_3d),
                                  "mejor_azar": float(mejor_az), "pct_4d_vs_azar": pct}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: ¿LAS SOMBRAS 4D DISTINGUEN DEL AZAR?", flush=True)
    print("=" * 70, flush=True)
    print(f"  1. Correlacion media con la cruz real: 4D={np.mean(corrs_4d):+.3f} | "
          f"3D={np.mean(corrs_3d):+.3f} | azar={np.mean(corrs_az):+.3f}", flush=True)
    print(f"  2. Mejor sombra: 4D={mejor_4d:+.3f} | 3D={mejor_3d:+.3f} | azar={mejor_az:+.3f}", flush=True)
    print(f"  3. z-score ratio real: vs 4D={z_4d:+.2f} | vs azar={z_az:+.2f}", flush=True)
    report["conclusion"] = {
        "corr_media": {"4d": float(np.mean(corrs_4d)), "3d": float(np.mean(corrs_3d)), "azar": float(np.mean(corrs_az))},
        "mejor_sombra": {"4d": float(mejor_4d), "3d": float(mejor_3d), "azar": float(mejor_az)},
        "z_ratio": {"vs_4d": float(z_4d), "vs_3d": float(z_3d), "vs_azar": float(z_az)},
    }

    out_json = os.path.join(OUT, "sombras_4d_gpu_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
