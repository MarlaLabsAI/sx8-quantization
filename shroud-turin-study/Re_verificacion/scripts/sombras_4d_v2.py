"""
SOMBRAS 4D v2: MITIGACION DEL PROBLEMA DEL MEJOR ANGULO + BANDA DIAGONAL
=======================================================================
Problema detectado: el "mejor angulo" de azar imitaba el perfil real
porque ambos son curvas monotonas decrecientes. La correlacion de
Pearson sobre la tendencia global no distingue estructura de ruido.

Soluciones implementadas:
  M1. CORRELACION DETRENDED: restar la tendencia suave del perfil y
      correlacionar solo las FLUCTUACIONES. Un objeto 4D con capas
      tiene fluctuaciones estructuradas (escalones); el azar tiene
      fluctuaciones aleatorias.
  M2. COHERENCIA INTERNA: correlacion media entre sombras de la MISMA
      familia. Las sombras de un objeto real (con simetrias) comparten
      estructura entre si; las de azar son independientes. Este es el
      discriminador mas fuerte.
  M3. TEST DE HIPOTESIS sobre la media de correlaciones (no el maximo):
      la media 4D vs media azar con distribucion empirica.
  M4. MOMENTOS Y ESPECTRO del perfil: los escalones de capas tienen
      firma distinta del gradiente suave del azar.

  V1. RESTAR LA BANDA DIAGONAL de la matriz real antes de medir el
      ratio centro/periferia (la banda diagonal infla la densidad).

  V2. Variante: perfil real SIN banda diagonal vs sombras.

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
N_SOMBRAS = 40
torch.manual_seed(42)
rng = np.random.default_rng(42)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# ============================================================================
# OBJETOS EN GPU
# ============================================================================
def grid_4d(size=SIZE):
    lin = torch.linspace(-1, 1, size, device=DEVICE)
    coords = torch.stack(torch.meshgrid(lin, lin, lin, lin, indexing='ij'), dim=-1)
    return coords.reshape(-1, 4)

def grid_3d(size=SIZE):
    lin = torch.linspace(-1, 1, size, device=DEVICE)
    coords = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1)
    return coords.reshape(-1, 3)

def objeto_4d_hiperesfera(pts4, n_capas=5):
    r4 = torch.sqrt((pts4**2).sum(dim=1))
    vals = torch.zeros(len(pts4), device=DEVICE)
    for rc, vc in zip(torch.linspace(1.0, 0.2, n_capas, device=DEVICE),
                      torch.linspace(0.2, 1.0, n_capas, device=DEVICE)):
        vals[r4 < rc] = vc
    return vals

def objeto_3d_esfera(pts3, n_capas=5):
    r3 = torch.sqrt((pts3**2).sum(dim=1))
    vals = torch.zeros(len(pts3), device=DEVICE)
    for rc, vc in zip(torch.linspace(1.0, 0.2, n_capas, device=DEVICE),
                      torch.linspace(0.2, 1.0, n_capas, device=DEVICE)):
        vals[r3 < rc] = vc
    return vals

def rotacion_so4_aleatoria(n_planos=3):
    planos = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    M = torch.eye(4, device=DEVICE)
    for k in rng.choice(len(planos), n_planos, replace=False):
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
    for k in rng.choice(len(planos), n_planos, replace=False):
        i, j = planos[k]
        theta = rng.uniform(0, 2*np.pi)
        R = torch.eye(3, device=DEVICE)
        c, s = np.cos(theta), np.sin(theta)
        R[i, i] = c; R[i, j] = -s
        R[j, i] = s; R[j, j] = c
        M = R @ M
    return M

def sombra_4d(pts4, vals, M, size=SIZE):
    pts_rot = pts4 @ M.T
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
# PERFIL RADIAL
# ============================================================================
def perfil_radial(proj, n_anillos=20, ancho=2):
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

def detrend(perfil):
    """Resta la tendencia suave (lowess/gaussiano) y devuelve fluctuaciones."""
    p = np.asarray(perfil, dtype=np.float64)
    tendencia = ndimage.gaussian_filter1d(p, sigma=3)
    return p - tendencia

def momentos_perfil(perfil):
    p = np.asarray(perfil, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    x = np.arange(len(p))
    mu = (x * p).sum()
    var = ((x - mu)**2 * p).sum()
    skew = (((x - mu)**3 * p).sum()) / (var**1.5 + 1e-12)
    kurt = (((x - mu)**4 * p).sum()) / (var**2 + 1e-12)
    return {"media": float(mu), "var": float(var), "skew": float(skew), "kurt": float(kurt)}

# ============================================================================
# CRUZ REAL
# ============================================================================
def matriz_real_con_banda():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R = (np.abs(profile[:, None] - profile[None, :]) < 10.0).astype(np.float32)
    n = R.shape[0]
    cx, cy = 416, 416
    # Banda diagonal: |i-j| < 15
    yy, xx = np.mgrid[0:n, 0:n]
    banda = np.abs(yy - xx) < 15
    # Matriz SIN banda diagonal
    R_sin_banda = R.copy()
    R_sin_banda[banda] = 0.0
    return R, R_sin_banda, profile, cx, cy, banda

def perfil_radial_matriz(R, cx, cy, n_anillos=20, ancho=2):
    n = R.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    res = []
    for i in range(n_anillos):
        r0, r1 = i*ancho, (i+1)*ancho
        mask = (dist >= r0) & (dist < r1)
        if mask.sum() > 100:
            res.append(float(R[mask].mean()))
    return np.array(res)

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"familias": {}, "cruz_real": {}, "M1_detrended": {}, "M2_coherencia": {},
              "M3_test_media": {}, "M4_momentos": {}, "V1_banda_diagonal": {},
              "conclusion": {}}

    # ============ CRUZ REAL ============
    R_full, R_sin_banda, profile, cx, cy, banda = matriz_real_con_banda()
    n = R_full.shape[0]
    print("=" * 70, flush=True)
    print(f"CRUZ REAL: ({cx},{cy}) | banda diagonal |i-j|<15 eliminada", flush=True)
    print("=" * 70, flush=True)

    # Perfil radial con y sin banda
    perfil_real_full = perfil_radial_matriz(R_full, cx, cy)
    # Perfil SIN banda: anillos excluyendo puntos de la banda
    n = R_full.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    dist_m = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    perfil_sb = []
    for i in range(20):
        r0, r1 = i*2, (i+1)*2
        mask = (dist_m >= r0) & (dist_m < r1) & (~banda)
        if mask.sum() > 100:
            perfil_sb.append(float(R_full[mask].mean()))
        else:
            perfil_sb.append(0.0)
    perfil_real_sb = np.array(perfil_sb)
    # Normalizar a 0-1
    def norm01(p):
        p = np.asarray(p, dtype=np.float64)
        return p / (p.max() + 1e-12)
    perfil_real_full_n = norm01(perfil_real_full)
    perfil_real_sb_n = norm01(perfil_real_sb)
    print(f"  Perfil CON banda:    {perfil_real_full_n.round(3)[:6]}...", flush=True)
    print(f"  Perfil SIN banda:    {perfil_real_sb_n.round(3)[:6]}...", flush=True)

    # Ratio centro/periferia con y sin banda
    centro_full = R_full[cy-10:cy+10, cx-10:cx+10].mean()
    perif_full = R_full[:15, :15].mean()
    # Centro SIN banda: region 40x40 de la cruz excluyendo |i-j|<15
    centro_sb_region = R_sin_banda[cy-20:cy+20, cx-20:cx+20]
    centro_sb = centro_sb_region.mean()
    # Periferia lejana SIN banda: anillo 150-250 (fuera de la banda)
    yy, xx = np.mgrid[0:n, 0:n]
    dist_m = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    perif_sb_mask = (dist_m >= 150) & (dist_m < 250) & (~banda)
    perif_sb = float(R_sin_banda[perif_sb_mask].mean())
    # Periferia lejana CON banda (contexto)
    perif_full_mask = (dist_m >= 150) & (dist_m < 250)
    perif_full_lejana = float(R_full[perif_full_mask].mean())
    print(f"  Ratio CON banda (perif cercana): {centro_full/(perif_full+1e-9):.3f}", flush=True)
    print(f"  Periferia lejana CON banda: {perif_full_lejana:.4f}", flush=True)
    print(f"  Centro SIN banda (40x40): {centro_sb:.4f}", flush=True)
    print(f"  Periferia lejana SIN banda: {perif_sb:.4f}", flush=True)
    print(f"  Ratio SIN banda (centro/perif lejana): {centro_sb/(perif_sb+1e-9):.3f}", flush=True)
    report["V1_banda_diagonal"] = {
        "ratio_con_banda": float(centro_full/(perif_full+1e-9)),
        "ratio_sin_banda": float(centro_sb/(perif_sb+1e-9)),
        "centro_con": float(centro_full), "perif_con": float(perif_full),
        "centro_sin": float(centro_sb), "perif_sin": float(perif_sb),
        "perif_lejana_con_banda": float(perif_full_lejana),
    }

    # ============ GENERAR SOMBRAS ============
    print("\nGenerando sombras en GPU...", flush=True)
    pts4 = grid_4d()
    pts3 = grid_3d()
    vals4 = objeto_4d_hiperesfera(pts4)
    vals3 = objeto_3d_esfera(pts3)

    familias = {"4d": [], "3d": [], "azar": []}
    for i in range(N_SOMBRAS):
        # 4D
        M4 = rotacion_so4_aleatoria(n_planos=3)
        proj4 = sombra_4d(pts4, vals4, M4)
        familias["4d"].append(proj4)
        # 3D
        M3 = rotacion_so3_aleatoria(n_planos=2)
        proj3 = sombra_3d(pts3, vals3, M3)
        familias["3d"].append(proj3)
        # Azar (con gradiente radial debil)
        proj_a = rng.random((SIZE, SIZE))
        hh, ww = proj_a.shape
        cyy, cxx = hh//2, ww//2
        yy, xx = np.mgrid[0:hh, 0:ww]
        dist = np.sqrt((xx-cxx)**2 + (yy-cyy)**2)
        grad = 1 - dist / dist.max() * 0.3
        proj_a = proj_a * grad
        proj_a = proj_a / proj_a.max()
        familias["azar"].append(proj_a)
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{N_SOMBRAS}", flush=True)

    # Perfiles de todas las sombras
    perfiles = {}
    for nombre, lista in familias.items():
        perfiles[nombre] = [perfil_radial(p) for p in lista]

    # ============ M1: CORRELACION DETRENDED ============
    print("\n[M1] CORRELACION DETRENDED (fluctuaciones, no tendencia)", flush=True)
    # Detrend del perfil real (sin banda)
    real_dt_full = detrend(perfil_real_full_n)
    real_dt_sb = detrend(perfil_real_sb_n)
    resultados_dt = {}
    for nombre, lista_perfiles in perfiles.items():
        corrs = []
        for p in lista_perfiles:
            p_n = norm01(p)
            p_dt = detrend(p_n)
            min_len = min(len(p_dt), len(real_dt_sb))
            if min_len >= 5:
                corrs.append(float(np.corrcoef(p_dt[:min_len], real_dt_sb[:min_len])[0, 1]))
        corrs = np.array([c for c in corrs if not np.isnan(c)])
        resultados_dt[nombre] = {
            "media": float(np.mean(corrs)), "std": float(np.std(corrs)),
            "max": float(np.max(corrs)), "p95": float(np.percentile(corrs, 95)),
        }
        print(f"  {nombre}: media={np.mean(corrs):+.3f} | max={np.max(corrs):+.3f} | p95={np.percentile(corrs,95):+.3f}", flush=True)
    report["M1_detrended"] = resultados_dt

    # ============ M2: COHERENCIA INTERNA ============
    print("\n[M2] COHERENCIA INTERNA (correlacion sombra-sombra de la misma familia)", flush=True)
    coherencias = {}
    for nombre, lista_perfiles in perfiles.items():
        corrs_int = []
        for i in range(len(lista_perfiles)):
            for j in range(i+1, len(lista_perfiles)):
                pi = norm01(lista_perfiles[i])
                pj = norm01(lista_perfiles[j])
                min_len = min(len(pi), len(pj))
                if min_len >= 5:
                    corrs_int.append(float(np.corrcoef(pi[:min_len], pj[:min_len])[0, 1]))
        corrs_int = np.array([c for c in corrs_int if not np.isnan(c)])
        coherencias[nombre] = {
            "media": float(np.mean(corrs_int)), "std": float(np.std(corrs_int)),
        }
        print(f"  {nombre}: coherencia media = {np.mean(corrs_int):+.3f} ± {np.std(corrs_int):.3f}", flush=True)
    report["M2_coherencia"] = coherencias

    # ============ M3: TEST DE HIPOTESIS (media) ============
    print("\n[M3] TEST DE HIPOTESIS: media de correlaciones 4D vs azar", flush=True)
    # Distribucion empirica: permutar etiquetas
    corrs_4d = []
    for p in perfiles["4d"]:
        p_n = norm01(p)
        min_len = min(len(detrend(p_n)), len(real_dt_sb))
        if min_len >= 5:
            corrs_4d.append(float(np.corrcoef(detrend(p_n)[:min_len], real_dt_sb[:min_len])[0, 1]))
    corrs_az = []
    for p in perfiles["azar"]:
        p_n = norm01(p)
        min_len = min(len(detrend(p_n)), len(real_dt_sb))
        if min_len >= 5:
            corrs_az.append(float(np.corrcoef(detrend(p_n)[:min_len], real_dt_sb[:min_len])[0, 1]))
    corrs_4d = np.array([c for c in corrs_4d if not np.isnan(c)])
    corrs_az = np.array([c for c in corrs_az if not np.isnan(c)])
    # Mann-Whitney U (no parametrico)
    from scipy.stats import mannwhitneyu
    if len(corrs_4d) > 3 and len(corrs_az) > 3:
        u_stat, p_val = mannwhitneyu(corrs_4d, corrs_az, alternative='greater')
        print(f"  Media 4D={np.mean(corrs_4d):+.3f} vs Azar={np.mean(corrs_az):+.3f}", flush=True)
        print(f"  Mann-Whitney U={u_stat:.0f}, p={p_val:.4f} {'(SIGNIFICATIVO)' if p_val < 0.05 else '(NO significativo)'}", flush=True)
        report["M3_test_media"] = {
            "media_4d": float(np.mean(corrs_4d)), "media_azar": float(np.mean(corrs_az)),
            "U": float(u_stat), "p": float(p_val),
        }
    # Tambien para coherencia interna: 4D vs azar
    corrs_int_4d = []
    for i in range(len(perfiles["4d"])):
        for j in range(i+1, len(perfiles["4d"])):
            pi = norm01(perfiles["4d"][i]); pj = norm01(perfiles["4d"][j])
            ml = min(len(pi), len(pj))
            if ml >= 5:
                corrs_int_4d.append(float(np.corrcoef(pi[:ml], pj[:ml])[0, 1]))
    corrs_int_az = []
    for i in range(len(perfiles["azar"])):
        for j in range(i+1, len(perfiles["azar"])):
            pi = norm01(perfiles["azar"][i]); pj = norm01(perfiles["azar"][j])
            ml = min(len(pi), len(pj))
            if ml >= 5:
                corrs_int_az.append(float(np.corrcoef(pi[:ml], pj[:ml])[0, 1]))
    u2, p2 = mannwhitneyu(corrs_int_4d, corrs_int_az, alternative='greater')
    print(f"  Coherencia interna: 4D={np.mean(corrs_int_4d):+.3f} vs Azar={np.mean(corrs_int_az):+.3f}", flush=True)
    print(f"  Mann-Whitney U={u2:.0f}, p={p2:.4f} {'(SIGNIFICATIVO)' if p2 < 0.05 else '(NO significativo)'}", flush=True)
    report["M3_test_media"]["coherencia_4d"] = float(np.mean(corrs_int_4d))
    report["M3_test_media"]["coherencia_azar"] = float(np.mean(corrs_int_az))
    report["M3_test_media"]["U_coherencia"] = float(u2)
    report["M3_test_media"]["p_coherencia"] = float(p2)

    # ============ M4: MOMENTOS DEL PERFIL ============
    print("\n[M4] MOMENTOS DEL PERFIL (forma de la curva)", flush=True)
    mom_real = momentos_perfil(perfil_real_sb_n)
    print(f"  Cruz real: media={mom_real['media']:.2f} | skew={mom_real['skew']:+.3f} | kurt={mom_real['kurt']:.2f}", flush=True)
    mom_familias = {}
    for nombre, lista_perfiles in perfiles.items():
        sk = []; ku = []
        for p in lista_perfiles:
            m = momentos_perfil(norm01(p))
            sk.append(m["skew"]); ku.append(m["kurt"])
        mom_familias[nombre] = {"skew_mean": float(np.mean(sk)), "skew_std": float(np.std(sk)),
                                "kurt_mean": float(np.mean(ku)), "kurt_std": float(np.std(ku))}
        print(f"  {nombre}: skew={np.mean(sk):+.3f}±{np.std(sk):.3f} | kurt={np.mean(ku):.2f}±{np.std(ku):.2f}", flush=True)
    report["M4_momentos"] = {"real": mom_real, "familias": mom_familias}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: MITIGACION DEL PROBLEMA DEL MEJOR ANGULO", flush=True)
    print("=" * 70, flush=True)
    print(f"  V1: ratio con banda={centro_full/(perif_full+1e-9):.3f} vs sin banda={centro_sb/(perif_sb+1e-9):.3f}", flush=True)
    print(f"  M1 detrended: 4D={resultados_dt['4d']['media']:+.3f} vs azar={resultados_dt['azar']['media']:+.3f}", flush=True)
    print(f"  M2 coherencia: 4D={coherencias['4d']['media']:+.3f} vs azar={coherencias['azar']['media']:+.3f}", flush=True)
    report["conclusion"] = {
        "V1": report["V1_banda_diagonal"],
        "M1": resultados_dt,
        "M2": coherencias,
        "M3": report["M3_test_media"],
        "M4": report["M4_momentos"],
    }

    out_json = os.path.join(OUT, "sombras_4d_v2_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
