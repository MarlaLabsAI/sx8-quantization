"""
SOMBRAS DE OBJETOS 4D: ROTACION SO(4) + PROYECCION MULTIPLE
===========================================================
Idea del usuario: para distinguir un objeto 4D del azar, hay que
MOVERLO de distintas maneras (rotaciones en los 6 planos de SO(4))
y capturar SUS SOMBRAS (proyecciones). La familia de sombras de un
objeto 4D es coherente (mismo objeto, distintos angulos); el azar
genera sombras independientes sin coherencia.

Tests:
  U1. OBJETO 4D: hiperesfera S3 con capas internas concéntricas
      - 30 rotaciones aleatorias SO(4) (planos xy,xz,xw,yz,yw,zw)
      - Proyeccion 4D->2D por suma (sombra)
      - Firma de cada sombra: perfil radial + D centro/periferia
      - Coherencia: correlacion entre sombras del mismo objeto

  U2. OBJETO 3D (control): esfera con capas, rotaciones SO(3)
      - Misma firma y coherencia

  U3. AZAR (control): 30 sombras aleatorias
      - Misma firma y coherencia (debe ser ~0)

  U4. LA CRUZ REAL: su perfil radial
      - ¿Cae dentro de la distribucion de sombras 4D?
      - ¿Es incompatible con azar?

  U5. MEJOR ANGULO: para cada familia (4D, 3D, azar), la sombra que
      mejor correlaciona con la cruz real. Un objeto 4D tendra una
      sombra muy cercana; el azar no.

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

# ============================================================================
# 1. ROTACIONES SO(4) y SO(3)
# ============================================================================
def matriz_rotacion_4d_plano(i, j, theta):
    """Matriz de rotacion 4x4 en el plano (i,j) con angulo theta."""
    M = np.eye(4)
    M[i, i] = np.cos(theta); M[i, j] = -np.sin(theta)
    M[j, i] = np.sin(theta); M[j, j] = np.cos(theta)
    return M

def rotacion_so4_aleatoria(n_planos=3):
    """Composicion de rotaciones en planos aleatorios de SO(4)."""
    planos = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    M = np.eye(4)
    idx = rng.choice(len(planos), n_planos, replace=False)
    for k in idx:
        i, j = planos[k]
        M = matriz_rotacion_4d_plano(i, j, rng.uniform(0, 2*np.pi)) @ M
    return M

def matriz_rotacion_3d_plano(i, j, theta):
    M = np.eye(3)
    M[i, i] = np.cos(theta); M[i, j] = -np.sin(theta)
    M[j, i] = np.sin(theta); M[j, j] = np.cos(theta)
    return M

def rotacion_so3_aleatoria(n_planos=2):
    planos = [(0,1),(0,2),(1,2)]
    M = np.eye(3)
    idx = rng.choice(len(planos), n_planos, replace=False)
    for k in idx:
        i, j = planos[k]
        M = matriz_rotacion_3d_plano(i, j, rng.uniform(0, 2*np.pi)) @ M
    return M

# ============================================================================
# 2. OBJETOS 4D Y 3D (con capas internas)
# ============================================================================
def objeto_4d_hiperesfera(size=48, n_capas=5):
    """Hiperesfera S3 con capas concentricas: rho(r4) por capa."""
    # Grid 4D
    lin = np.linspace(-1, 1, size)
    coords = np.stack(np.meshgrid(lin, lin, lin, lin, indexing='ij'), axis=-1)  # (s,s,s,s,4)
    pts = coords.reshape(-1, 4)  # (s^4, 4)
    r4 = np.sqrt((pts**2).sum(axis=1))
    # Capas: 1.0, 0.8, 0.6, 0.4, 0.2 desde fuera hacia dentro
    capas_radio = np.linspace(1.0, 0.2, n_capas)
    capas_valor = np.linspace(0.2, 1.0, n_capas)
    vals = np.zeros(len(pts))
    for r_capa, v_capa in zip(capas_radio, capas_valor):
        vals[r4 < r_capa] = v_capa
    return pts, vals

def objeto_3d_esfera(size=48, n_capas=5):
    """Esfera con capas concentricas (control 3D)."""
    lin = np.linspace(-1, 1, size)
    coords = np.stack(np.meshgrid(lin, lin, lin, indexing='ij'), axis=-1)
    pts = coords.reshape(-1, 3)
    r3 = np.sqrt((pts**2).sum(axis=1))
    capas_radio = np.linspace(1.0, 0.2, n_capas)
    capas_valor = np.linspace(0.2, 1.0, n_capas)
    vals = np.zeros(len(pts))
    for r_capa, v_capa in zip(capas_radio, capas_valor):
        vals[r3 < r_capa] = v_capa
    return pts, vals

# ============================================================================
# 3. PROYECCION (sombra)
# ============================================================================
def sombra_4d(pts, vals, M, size=48):
    """Rota los puntos 4D con M y proyecta 4D->2D (suma sobre 2 ejes)."""
    pts_rot = pts @ M.T  # (N,4)
    # Proyectar: sumar sobre ejes 2 y 3 -> grid 2D
    # Reorganizar a grid (size,size,size,size)
    grid = np.zeros((size, size, size, size))
    # Acumular con histograma 2D: sumar valores por celda (x,y)
    # Mas simple: reshape y sumar sobre z,w
    # pts_rot[:,0]=x, [:,1]=y, [:,2]=z, [:,3]=w
    # indices al grid original
    idx = np.clip(((pts_rot[:, :2] + 1) / 2 * (size - 1)).astype(int), 0, size-1)
    proj = np.zeros((size, size))
    np.add.at(proj, (idx[:, 0], idx[:, 1]), vals)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def sombra_3d(pts, vals, M, size=48):
    """Rota puntos 3D y proyecta 3D->2D (suma sobre z)."""
    pts_rot = pts @ M.T
    idx = np.clip(((pts_rot[:, :2] + 1) / 2 * (size - 1)).astype(int), 0, size-1)
    proj = np.zeros((size, size))
    np.add.at(proj, (idx[:, 0], idx[:, 1]), vals)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

# ============================================================================
# 4. FIRMA DE UNA SOMBRA
# ============================================================================
def perfil_radial_sombra(proj, n_anillos=20):
    """Perfil radial (densidad por anillo desde el centro)."""
    h, w = proj.shape
    cx, cy = w//2, h//2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*3, (i+1)*3
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 0:
            res.append(float(proj[mask].mean()))
    return np.array(res)

def firma_sombra(proj):
    """Firma: densidad centro/periferia + perfil radial."""
    h, w = proj.shape
    cyy, cxx = h//2, w//2
    centro = proj[cyy-10:cyy+10, cxx-10:cxx+10]
    perif = proj[:15, :15]
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
    res = []
    for i in range(20):
        r0, r1 = i*3, (i+1)*3
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

    SIZE = 48
    N_SOMBRAS = 30

    # Cruz real
    perfil_real, R_real, cx, cy = perfil_real_cruz()
    print("=" * 70, flush=True)
    print(f"CRUZ REAL: perfil radial {perfil_real.round(3)[:8]}...", flush=True)
    print("=" * 70, flush=True)

    # ============ U1: SOMBRAS 4D ============
    print(f"\n[U1] HIPERESFERA 4D (S3 con capas): {N_SOMBRAS} sombras por rotacion SO(4)", flush=True)
    pts4, vals4 = objeto_4d_hiperesfera(SIZE)
    sombras_4d = []
    for i in range(N_SOMBRAS):
        M = rotacion_so4_aleatoria(n_planos=3)
        proj = sombra_4d(pts4, vals4, M, SIZE)
        firma = firma_sombra(proj)
        perfil = np.array(firma["perfil"])
        # Correlacion con el perfil real
        if len(perfil) == len(perfil_real):
            corr = float(np.corrcoef(perfil, perfil_real)[0, 1])
        else:
            corr = float("nan")
        firma["corr_real"] = corr
        sombras_4d.append(firma)
    corrs_4d = [s["corr_real"] for s in sombras_4d if not np.isnan(s["corr_real"])]
    ratios_4d = [s["ratio_centro_perif"] for s in sombras_4d]
    print(f"  corr con real: media={np.mean(corrs_4d):+.3f} | max={np.max(corrs_4d):+.3f} | "
          f"p95={np.percentile(corrs_4d, 95):+.3f}", flush=True)
    print(f"  ratio centro/perif: media={np.mean(ratios_4d):.2f} | std={np.std(ratios_4d):.2f}", flush=True)
    report["U1_sombras_4d"] = {"corrs": corrs_4d, "ratios": ratios_4d,
                                "corr_mean": float(np.mean(corrs_4d)), "corr_max": float(np.max(corrs_4d)),
                                "ratio_mean": float(np.mean(ratios_4d)), "ratio_std": float(np.std(ratios_4d))}

    # ============ U2: SOMBRAS 3D ============
    print(f"\n[U2] ESFERA 3D (con capas): {N_SOMBRAS} sombras por rotacion SO(3)", flush=True)
    pts3, vals3 = objeto_3d_esfera(SIZE)
    sombras_3d = []
    for i in range(N_SOMBRAS):
        M = rotacion_so3_aleatoria(n_planos=2)
        proj = sombra_3d(pts3, vals3, M, SIZE)
        firma = firma_sombra(proj)
        perfil = np.array(firma["perfil"])
        if len(perfil) == len(perfil_real):
            corr = float(np.corrcoef(perfil, perfil_real)[0, 1])
        else:
            corr = float("nan")
        firma["corr_real"] = corr
        sombras_3d.append(firma)
    corrs_3d = [s["corr_real"] for s in sombras_3d if not np.isnan(s["corr_real"])]
    ratios_3d = [s["ratio_centro_perif"] for s in sombras_3d]
    print(f"  corr con real: media={np.mean(corrs_3d):+.3f} | max={np.max(corrs_3d):+.3f} | "
          f"p95={np.percentile(corrs_3d, 95):+.3f}", flush=True)
    print(f"  ratio centro/perif: media={np.mean(ratios_3d):.2f} | std={np.std(ratios_3d):.2f}", flush=True)
    report["U2_sombras_3d"] = {"corrs": corrs_3d, "ratios": ratios_3d,
                                "corr_mean": float(np.mean(corrs_3d)), "corr_max": float(np.max(corrs_3d)),
                                "ratio_mean": float(np.mean(ratios_3d)), "ratio_std": float(np.std(ratios_3d))}

    # ============ U3: AZAR ============
    print(f"\n[U3] AZAR: {N_SOMBRAS} sombras aleatorias", flush=True)
    sombras_azar = []
    for i in range(N_SOMBRAS):
        # Sombra aleatoria: mismo rango de densidad que las proyecciones
        proj = rng.random((SIZE, SIZE))
        # Con gradiente radial suave para ser mas realista (ruido + estructura debil)
        hh, ww = proj.shape
        cyy, cxx = hh//2, ww//2
        yy, xx = np.mgrid[0:hh, 0:ww]
        dist = np.sqrt((xx-cxx)**2 + (yy-cyy)**2)
        grad = 1 - dist / dist.max() * 0.3  # gradiente radial debil
        proj = proj * grad
        proj = proj / proj.max()
        firma = firma_sombra(proj)
        perfil = np.array(firma["perfil"])
        if len(perfil) == len(perfil_real):
            corr = float(np.corrcoef(perfil, perfil_real)[0, 1])
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
    # Ratio centro/perif de la cruz real
    centro_real = R_real[cy-10:cy+10, cx-10:cx+10]
    perif_real = R_real[:15, :15]
    ratio_real = float(centro_real.mean() / (perif_real.mean() + 1e-9))
    print(f"  Cruz real: ratio centro/perif = {ratio_real:.2f}", flush=True)
    print(f"  Sombras 4D: ratio = {np.mean(ratios_4d):.2f}±{np.std(ratios_4d):.2f}", flush=True)
    print(f"  Sombras 3D: ratio = {np.mean(ratios_3d):.2f}±{np.std(ratios_3d):.2f}", flush=True)
    print(f"  Azar:       ratio = {np.mean(ratios_az):.2f}±{np.std(ratios_az):.2f}", flush=True)
    # z-scores del ratio real vs cada familia
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
    # Percentil de la mejor 4D dentro de la distribucion de azar
    pct = float((np.array(corrs_az) < mejor_4d).mean() * 100)
    print(f"  La mejor sombra 4D supera al {pct:.1f}% de las sombras de azar", flush=True)
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

    out_json = os.path.join(OUT, "sombras_4d_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
