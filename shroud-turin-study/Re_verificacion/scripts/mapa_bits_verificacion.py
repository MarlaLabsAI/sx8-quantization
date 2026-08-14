"""
VERIFICACION DEL ANALISIS DE MAPA DE BITS (varianza binomial + percolacion)
===========================================================================
La otra IA analizo el ROSTRO (imagen1, crop 1000x1000) y concluyo:
  1. Varianza binomial: sigma^2 = C*mu*(1-mu) con C=0.0026
     -> "efecto todo-o-nada" (pixeles binarios, no tono continuo)
  2. Percolacion: a p=0.137 el cluster gigante ya es 76%
     -> "correlacion espacial de largo alcance, autoorganizacion"

Este script:
  A. Replica EXACTAMENTE ambos tests sobre el rostro real
  B. Ejecuta los mismos tests sobre CONTROLES:
     - permutacion espacial (misma distribucion, cero correlacion)
     - ruido gaussiano (misma media/std)
     - gaussiano suavizado (misma correlacion aproximada)
     - textura de tela simulada (periodica + ruido)
  C. Compara: si los controles tambien dan parabola binomial y
     percolacion temprana, los hallazgos NO son especificos.

NO modifica archivos originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy.ndimage import label
from multiprocessing import Pool

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)
N_CONTROLS = 50
N_WORKERS = 12

# ============================================================================
# 1. TESTS EXACTOS DE LA OTRA IA
# ============================================================================
def test_varianza_binomial(img_norm, patch_size=5):
    """Parches patch_size x patch_size sin solapamiento: medias y varianzas.
    Ajusta C en sigma^2 = C * mu * (1-mu)."""
    h, w = img_norm.shape
    means, variances = [], []
    for r in range(0, h - patch_size, patch_size):
        for c in range(0, w - patch_size, patch_size):
            patch = img_norm[r:r+patch_size, c:c+patch_size]
            means.append(np.mean(patch))
            variances.append(np.var(patch))
    means = np.array(means)
    variances = np.array(variances)
    mu_term = means * (1.0 - means)
    valid = mu_term > 0.01
    C, _, _, _ = np.linalg.lstsq(mu_term[valid, np.newaxis], variances[valid], rcond=None)
    C_val = float(C[0])
    # R^2 del ajuste
    pred = C_val * mu_term[valid]
    ss_res = np.sum((variances[valid] - pred) ** 2)
    ss_tot = np.sum((variances[valid] - variances[valid].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # Varianza media en la zona central (mu ~ 0.4-0.6)
    mid = (means > 0.4) & (means < 0.6)
    var_mid = float(variances[mid].mean()) if mid.sum() > 0 else float("nan")
    return {"C": C_val, "r2": float(r2), "n_patches": int(len(means)),
            "var_mid": var_mid, "mu_mid_esperado": float(0.25 * C_val)}

def test_percolacion(img_norm, size=250):
    """Umbrales 0.05-0.95: fraccion activa vs fraccion del cluster gigante."""
    small = cv2.resize(img_norm, (size, size))
    thresholds = np.linspace(0.05, 0.95, 30)
    active_fractions = []
    giant_fractions = []
    structure = np.ones((3, 3), dtype=np.int32)
    for T in thresholds:
        binary = (small > T).astype(np.int32)
        af = np.mean(binary)
        active_fractions.append(float(af))
        if af == 0:
            giant_fractions.append(0.0)
            continue
        labeled, nf = label(binary, structure=structure)
        if nf > 0:
            unique, counts = np.unique(labeled, return_counts=True)
            sizes = counts[unique > 0]
            if len(sizes) > 0:
                giant_fractions.append(float(sizes.max() / np.sum(binary)))
            else:
                giant_fractions.append(0.0)
        else:
            giant_fractions.append(0.0)
    # Metricas clave: cluster gigante a p~0.1, p~0.2, p~0.5
    def giant_at(p_target):
        idx = np.argmin(np.abs(np.array(active_fractions) - p_target))
        return giant_fractions[idx]
    return {"active_fractions": active_fractions, "giant_fractions": giant_fractions,
            "giant_at_0_1": giant_at(0.1), "giant_at_0_2": giant_at(0.2),
            "giant_at_0_5": giant_at(0.5)}

# ============================================================================
# 2. CONTROLES
# ============================================================================
def control_permutacion(img_norm):
    """Permuta espacialmente: misma distribucion, cero correlacion."""
    flat = img_norm.flatten()
    rng.shuffle(flat)
    return flat.reshape(img_norm.shape)

def control_gaussiano(img_norm):
    """Ruido blanco con misma media y std."""
    return rng.normal(img_norm.mean(), img_norm.std(), size=img_norm.shape)

def control_gaussiano_suavizado(img_norm, sigma=3.0):
    """Ruido gaussiano suavizado (correlacion espacial similar a imagen real)."""
    noise = rng.normal(0, img_norm.std(), size=img_norm.shape)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigma)
    smooth = smooth / smooth.std() * img_norm.std() + img_norm.mean()
    return np.clip(smooth, 0, 1)

def control_tela(img_norm, freq=8.0):
    """Textura de tela simulada: patron periodico + ruido."""
    h, w = img_norm.shape
    yy, xx = np.mgrid[0:h, 0:w]
    tela = 0.5 + 0.15 * np.sin(2*np.pi*yy/freq) + 0.15 * np.sin(2*np.pi*xx/freq)
    tela = tela + rng.normal(0, 0.05, size=(h, w))
    tela = (tela - tela.min()) / (tela.max() - tela.min())
    # Ajustar media/std a la imagen real
    tela = (tela - tela.mean()) / tela.std() * img_norm.std() + img_norm.mean()
    return np.clip(tela, 0, 1)

def worker_analisis(args):
    img_norm, nombre = args
    vb = test_varianza_binomial(img_norm)
    pc = test_percolacion(img_norm)
    return {"nombre": nombre, "varianza_binomial": vb, "percolacion": pc}

# ============================================================================
# 3. MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"real": {}, "controles": {}, "conclusion": {}}

    # Cargar rostro (imagen1 crop 1000x1000, como la otra IA)
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    face = img1[100:1100, 1000:2000]
    face_norm = face.astype(float) / 255.0
    print("=" * 70, flush=True)
    print(f"ROSTRO REAL: {face.shape} (crop de imagen1)", flush=True)
    print("=" * 70, flush=True)

    # A. Tests exactos sobre el rostro real
    vb_real = test_varianza_binomial(face_norm)
    pc_real = test_percolacion(face_norm)
    report["real"]["varianza_binomial"] = vb_real
    report["real"]["percolacion"] = pc_real
    print(f"\n[VARIANZA BINOMIAL] C={vb_real['C']:.4f} | R2={vb_real['r2']:.4f} | "
          f"var_mid={vb_real['var_mid']:.4f} (esperado 0.25*C={vb_real['mu_mid_esperado']:.4f})", flush=True)
    print(f"  (la otra IA reporto C=0.0026)", flush=True)
    print(f"[PERCOLACION] giant@0.1={pc_real['giant_at_0_1']:.3f} | giant@0.2={pc_real['giant_at_0_2']:.3f} | "
          f"giant@0.5={pc_real['giant_at_0_5']:.3f}", flush=True)
    print(f"  (la otra IA reporto 0.76 a p=0.137)", flush=True)

    # B. Controles
    print(f"\nCONTROLES ({N_CONTROLS} x 4 tipos):", flush=True)
    tipos = ["permutacion", "gaussiano", "gaussiano_suavizado", "tela"]
    tasks = []
    for tipo in tipos:
        for i in range(N_CONTROLS):
            if tipo == "permutacion":
                img_c = control_permutacion(face_norm)
            elif tipo == "gaussiano":
                img_c = control_gaussiano(face_norm)
            elif tipo == "gaussiano_suavizado":
                img_c = control_gaussiano_suavizado(face_norm)
            else:
                img_c = control_tela(face_norm)
            tasks.append((img_c, f"{tipo}_{i}"))
    with Pool(N_WORKERS) as pool:
        res = list(pool.imap_unordered(worker_analisis, tasks))
    # Agrupar por tipo
    for tipo in tipos:
        grupo = [r for r in res if r["nombre"].startswith(tipo)]
        Cs = [r["varianza_binomial"]["C"] for r in grupo]
        r2s = [r["varianza_binomial"]["r2"] for r in grupo]
        g01 = [r["percolacion"]["giant_at_0_1"] for r in grupo]
        g02 = [r["percolacion"]["giant_at_0_2"] for r in grupo]
        g05 = [r["percolacion"]["giant_at_0_5"] for r in grupo]
        print(f"  [{tipo}] C={np.mean(Cs):.4f}±{np.std(Cs):.4f} | R2={np.mean(r2s):.3f} | "
              f"giant@0.1={np.mean(g01):.3f}±{np.std(g01):.3f} | giant@0.2={np.mean(g02):.3f}±{np.std(g02):.3f} | "
              f"giant@0.5={np.mean(g05):.3f}±{np.std(g05):.3f}", flush=True)
        report["controles"][tipo] = {
            "C_mean": float(np.mean(Cs)), "C_std": float(np.std(Cs)),
            "r2_mean": float(np.mean(r2s)),
            "giant_0_1_mean": float(np.mean(g01)), "giant_0_1_std": float(np.std(g01)),
            "giant_0_2_mean": float(np.mean(g02)), "giant_0_2_std": float(np.std(g02)),
            "giant_0_5_mean": float(np.mean(g05)), "giant_0_5_std": float(np.std(g05)),
        }

    # C. Conclusion
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    # Comparar C del real vs controles
    Cs_perm = [r["varianza_binomial"]["C"] for r in res if r["nombre"].startswith("permutacion")]
    z_C = (vb_real["C"] - np.mean(Cs_perm)) / np.std(Cs_perm) if np.std(Cs_perm) > 0 else float("nan")
    print(f"  C real={vb_real['C']:.4f} vs permutaciones={np.mean(Cs_perm):.4f}±{np.std(Cs_perm):.4f} (z={z_C:+.2f})", flush=True)
    # Comparar percolacion
    g01_perm = [r["percolacion"]["giant_at_0_1"] for r in res if r["nombre"].startswith("permutacion")]
    z_g = (pc_real["giant_at_0_1"] - np.mean(g01_perm)) / np.std(g01_perm) if np.std(g01_perm) > 0 else float("nan")
    print(f"  giant@0.1 real={pc_real['giant_at_0_1']:.3f} vs permutaciones={np.mean(g01_perm):.3f}±{np.std(g01_perm):.3f} (z={z_g:+.2f})", flush=True)
    report["conclusion"] = {
        "C_real": vb_real["C"], "C_perm_mean": float(np.mean(Cs_perm)), "z_C": float(z_C),
        "giant01_real": pc_real["giant_at_0_1"], "giant01_perm_mean": float(np.mean(g01_perm)),
        "z_giant01": float(z_g),
        "interpretacion": "Si z es pequeno, los hallazgos no son especificos de la Sabana"
    }

    out_json = os.path.join(OUT, "mapa_bits_verificacion_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
