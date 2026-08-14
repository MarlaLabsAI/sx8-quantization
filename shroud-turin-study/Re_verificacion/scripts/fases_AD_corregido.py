"""
FASES A-D CORREGIDAS: PERFIL HORIZONTAL (ANATOMIA CABEZA->PIES)
===============================================================
CORRECCION: el cuerpo en imagen3 (1920x1080) esta HORIZONTAL.
El perfil correcto es la FILA central y=540. La matriz de recurrencia
horizontal tiene 13 bloques anatomicos.

Fase A: ¿Proceso deliberado? (proporciones anatomicas + control)
Fase B: ¿Conexion tipo agujero de gusano? (bidireccionalidad, MI-dist)
Fase C: ¿Consistente con proyeccion de cuerpo 3D?
Fase D: Curvatura + topologia

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import pearsonr
from collections import Counter

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

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

def proporcion_anatomica(rel):
    """Region anatomica canonica (cuerpo acostado, cabeza->pies, izq->der)."""
    if rel < 0.125: return "cabeza"
    if rel < 0.16: return "cuello"
    if rel < 0.48: return "torso"
    return "piernas"

def main():
    t0 = time.time()
    report = {}

    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    y_perfil = h // 2
    perfil_h = ndimage.gaussian_filter1d(img3[y_perfil, :].astype(np.float32), sigma=15)
    R = (np.abs(perfil_h[:, None] - perfil_h[None, :]) < 10.0).astype(np.float32)
    n = R.shape[0]
    cx, cy = n // 2, n // 2
    fila = R[cy, :]
    bloques = bloques_recurrencia(fila)
    print("=" * 70, flush=True)
    print(f"PERFIL HORIZONTAL: y={y_perfil} | matriz {n}x{n} | punto central ({cx},{cy}) | {len(bloques)} bloques", flush=True)
    print("=" * 70, flush=True)

    # ============ FASE A: DELIBERADO ============
    print("\n[FASE A] ¿Proceso deliberado? (anatomia cabeza->pies)", flush=True)
    regiones = []
    for b0, b1 in bloques:
        rel = (b0 + b1) / 2 / n
        regiones.append(proporcion_anatomica(rel))
    obs = Counter(regiones)
    n_b = len(bloques)
    esperado = {"cabeza": 0.125, "cuello": 0.035, "torso": 0.32, "piernas": 0.52}
    chi = sum((obs[r]/n_b - esperado[r])**2 / esperado[r] for r in esperado)
    print(f"  Bloques: cabeza={obs['cabeza']}, cuello={obs['cuello']}, torso={obs['torso']}, piernas={obs['piernas']}", flush=True)
    print(f"  Chi2 vs proporcion canonica: {chi:.3f}", flush=True)
    # Control aleatorio
    chis_ctrl = []
    for _ in range(100):
        py = rng.integers(0, n-1)
        pfila = R[py, :]
        pbloques = bloques_recurrencia(pfila)
        if len(pbloques) < 3:
            continue
        pregiones = [proporcion_anatomica((b0+b1)/2/n) for b0, b1 in pbloques]
        pobs = Counter(pregiones)
        pchi = sum((pobs[r]/len(pbloques) - esperado[r])**2 / esperado[r] for r in esperado)
        chis_ctrl.append(pchi)
    chis_ctrl = np.array(chis_ctrl)
    z = (chi - chis_ctrl.mean()) / chis_ctrl.std() if chis_ctrl.std() > 0 else float("nan")
    print(f"  Control aleatorio chi2: {chis_ctrl.mean():.3f}±{chis_ctrl.std():.3f} | z={z:+.2f}", flush=True)
    report["fase_A"] = {"chi2": float(chi), "obs": dict(obs), "ctrl_mean": float(chis_ctrl.mean()),
                        "ctrl_std": float(chis_ctrl.std()), "z": float(z)}

    # ============ FASE B: NO-LOCALIDAD ============
    print("\n[FASE B] ¿Conexion tipo agujero de gusano?", flush=True)
    # B1: Bidireccionalidad
    bidir = [float(R[(b0+b1)//2, cx]) for b0, b1 in bloques]
    frac_bidir = float(np.mean(bidir))
    print(f"  B1 bidireccionalidad: {frac_bidir*100:.0f}% (simetria trivial de la matriz)", flush=True)
    # B3: MI vs distancia
    def mi_2d(a, b):
        a_b = (a > 0).astype(np.uint8)
        b_b = (b > 0).astype(np.uint8)
        if a_b.shape != b_b.shape:
            b_b = cv2.resize(b_b, (a_b.shape[1], a_b.shape[0]), interpolation=cv2.INTER_NEAREST)
        c = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                c[i,j] = np.mean((a_b == i) & (b_b == j))
        c /= c.sum()
        pa, pb = c.sum(axis=1), c.sum(axis=0)
        m = 0.0
        for i in range(2):
            for j in range(2):
                if c[i,j] > 0 and pa[i] > 0 and pb[j] > 0:
                    m += c[i,j] * np.log2(c[i,j] / (pa[i]*pb[j]))
        return m
    centro = R[cy-15:cy+15, cx-15:cx+15]
    mis, dists = [], []
    for b0, b1 in bloques:
        xc = (b0 + b1) // 2
        bloque = R[cy-10:cy+10, max(0,xc-10):xc+10]
        if bloque.size == 0:
            continue
        mis.append(mi_2d(centro, bloque))
        dists.append(abs(xc - cx))
    corr_mi = pearsonr(dists, mis)[0] if len(mis) >= 5 else float("nan")
    print(f"  B3 corr(MI, distancia) = {corr_mi:+.3f} "
          f"-> {'NO-LOCAL' if abs(corr_mi) < 0.3 else 'LOCAL'}", flush=True)
    report["fase_B"] = {"bidireccional": frac_bidir, "mis": mis, "dists": dists,
                        "corr_mi": float(corr_mi)}

    # ============ FASE C: PROYECCION CUERPO 3D ============
    print("\n[FASE C] ¿Consistente con proyeccion de cuerpo 3D?", flush=True)
    # Cuerpo humanoide 3D: perfil a lo largo del cuerpo (horizontal en imagen)
    # Simular perfil de densidad de un cuerpo acostado: cabeza (pico), cuello (valle),
    # torso (meseta), piernas (decae)
    size = n
    x = np.arange(size) / size  # 0..1
    # Perfil sintetico de cuerpo acostado (izq=cabeza, der=pies)
    cuerpo = np.zeros(size)
    # Cabeza (x<0.15): pico
    cabeza_mask = x < 0.15
    cuerpo[cabeza_mask] = 0.5 + 0.5 * np.sin(np.pi * x[cabeza_mask] / 0.15)
    # Cuello (0.15-0.2): valle
    cuello_mask = (x >= 0.15) & (x < 0.2)
    cuerpo[cuello_mask] = 0.3
    # Torso (0.2-0.5): meseta alta
    torso_mask = (x >= 0.2) & (x < 0.5)
    cuerpo[torso_mask] = 0.7
    # Piernas (0.5-1.0): decae
    piernas_mask = x >= 0.5
    cuerpo[piernas_mask] = 0.6 * (1 - (x[piernas_mask] - 0.5) * 0.8)
    cuerpo = cuerpo * 255
    # Suavizar y matriz de recurrencia
    cuerpo_s = ndimage.gaussian_filter1d(cuerpo, sigma=15)
    R_sim = (np.abs(cuerpo_s[:, None] - cuerpo_s[None, :]) < 10.0).astype(np.float32)
    fila_sim = R_sim[size//2, :]
    bloques_sim = bloques_recurrencia(fila_sim)
    print(f"  Cuerpo 3D simulado (perfil horizontal): {len(bloques_sim)} bloques", flush=True)
    for b in bloques_sim:
        print(f"    x={b[0]}-{b[1]} (rel={((b[0]+b[1])/2)/size:.2f})", flush=True)
    report["fase_C"] = {"n_bloques_sim": len(bloques_sim),
                        "bloques": [(int(b[0]), int(b[1])) for b in bloques_sim]}

    # ============ FASE D: CURVATURA ============
    print("\n[FASE D] Curvatura gaussiana alrededor del punto central", flush=True)
    region = R[cx-50:cx+50, cy-50:cy+50].astype(np.float64)
    region_s = cv2.GaussianBlur(region, (5, 5), 0)
    gy, gx = np.gradient(region_s)
    gyy, gxy = np.gradient(gy)
    gxy2, gxx = np.gradient(gx)
    K = (gxx * gyy - gxy**2) / (1 + gx**2 + gy**2)**2
    K_neg_frac = float((K < 0).mean())
    print(f"  Fraccion de curvatura negativa: {K_neg_frac*100:.1f}%", flush=True)
    report["fase_D"] = {"K_neg_frac": K_neg_frac}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION (perfil horizontal, anatomia correcta)", flush=True)
    print("=" * 70, flush=True)
    print(f"  A: chi2={chi:.3f} (z={z:+.2f}) {'ANATOMICO' if z < -1.5 else 'no distinto de azar'}", flush=True)
    print(f"  B: corr(MI,dist)={corr_mi:+.3f} {'NO-LOCAL' if abs(corr_mi) < 0.3 else 'LOCAL'}", flush=True)
    print(f"  C: cuerpo 3D simulado da {len(bloques_sim)} bloques (real: {n_b})", flush=True)
    print(f"  D: curvatura negativa {K_neg_frac*100:.1f}%", flush=True)
    report["conclusion"] = {
        "A_z": float(z), "B_corr": float(corr_mi),
        "C_sim": len(bloques_sim), "C_real": n_b,
        "D_neg": K_neg_frac,
    }

    out_json = os.path.join(OUT, "fases_AD_corregido_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
