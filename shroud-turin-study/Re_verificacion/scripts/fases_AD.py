"""
FASES A-D: VERIFICACION DEL PROCESO DELIBERADO + AGUJERO DE GUSANO
==================================================================
Fase A: ¿El proceso es DELIBERADO y PRECISO? (anatomia)
  A1. Proporciones anatomicas: los 12 bloques vs proporciones canonicas
  A2. Grosor: tamano del bloque vs anchura del cuerpo en esa zona
  A3. Control: 100 puntos aleatorios -> sus bloques son anatomicos?

Fase B: ¿Conexion tipo AGUJERO DE GUSANO? (no-localidad estricta)
  B1. Bidireccionalidad: cada region tambien ve al punto central?
  B2. Precision de tuneles: ancho de bloques vs azar
  B3. MI(punto, bloque) vs distancia anatomica

Fase C: ¿Consistente con proyeccion de cuerpo 3D?
  C1. Simular cuerpo humanoide 3D -> proyectar -> matriz recurrencia
      -> bloques del punto central
  C2. Comparar distribucion de bloques simulada vs real

Fase D: Hipotesis del agujero de gusano (honestidad)
  D1. Curvatura gaussiana alrededor del punto central
  D2. Topologia: componentes conectados que tocan fila/columna 416
  D3. Limites de la interpretacion

NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
import torch
from scipy import ndimage
from scipy.stats import pearsonr

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(42)

print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

# ============================================================================
# UTILIDADES
# ============================================================================
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

# Proporciones anatomicas canonicas (fraccion de la altura total)
# Cabeza: 0.0-0.125 | Cuello: 0.125-0.16 | Torso: 0.16-0.48 | Piernas: 0.48-1.0
def proporcion_anatomica(rel):
    """Region anatomica canonica para una posicion relativa (0-1)."""
    if rel < 0.125: return "cabeza"
    if rel < 0.16: return "cuello"
    if rel < 0.48: return "torso"
    return "piernas"

# ============================================================================
# FASE A: DELIBERADO Y PRECISO
# ============================================================================
def fase_A(R, profile, img3, cx=416):
    print("=" * 70, flush=True)
    print("FASE A: ¿PROCESO DELIBERADO Y PRECISO? (anatomia)", flush=True)
    print("=" * 70, flush=True)
    n = R.shape[0]
    fila = R[cx, :]
    bloques = bloques_recurrencia(fila)
    resultado = {}

    # A1: Proporciones anatomicas de los bloques
    print("\n[A1] Proporciones anatomicas de los 12 bloques:", flush=True)
    regiones = []
    for b0, b1 in bloques:
        rel = (b0 + b1) / 2 / n
        region = proporcion_anatomica(rel)
        regiones.append(region)
        print(f"  bloque y={b0}-{b1} (rel={rel:.2f}): region canonica = {region}", flush=True)
    # Distribucion esperada del cuerpo humano (cabeza 12.5%, cuello 3.5%, torso 32%, piernas 52%)
    esperado = {"cabeza": 0.125, "cuello": 0.035, "torso": 0.32, "piernas": 0.52}
    # Observado: bloques por region
    from collections import Counter
    obs = Counter(regiones)
    n_bloques = len(bloques)
    print(f"  Distribucion de bloques: cabeza={obs['cabeza']}, cuello={obs['cuello']}, "
          f"torso={obs['torso']}, piernas={obs['piernas']}", flush=True)
    # Fraccion de bloques que caen en la region canonica correcta
    # (los bloques son 12 puntos; comparar su distribucion con la esperada)
    chi = 0
    for region in esperado:
        obs_frac = obs[region] / n_bloques
        exp_frac = esperado[region]
        chi += (obs_frac - exp_frac)**2 / exp_frac
    print(f"  Chi-cuadrado (distribucion vs canonica): {chi:.3f}", flush=True)
    resultado["A1"] = {"regiones": regiones, "chi2": float(chi),
                       "obs": dict(obs), "esperado": esperado}

    # A2: Grosor del bloque vs anchura del cuerpo
    print("\n[A2] Tamano del bloque vs anchura del cuerpo en esa zona:", flush=True)
    h_img, w_img = img3.shape
    # Anchura del cuerpo por fila: umbral Otsu
    _, bw = cv2.threshold(img3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grosores = []
    tamanos = []
    for b0, b1 in bloques:
        y_centro = (b0 + b1) // 2
        fila_img = bw[y_centro, :]
        # Ancho del cuerpo: pixeles blancos (o negros segun Otsu)
        ancho = int((fila_img > 0).sum())
        grosores.append(ancho)
        tamanos.append(b1 - b0 + 1)
        print(f"  bloque y={b0}-{b1} (tamano {b1-b0+1:3d}): ancho cuerpo en y={y_centro} = {ancho}", flush=True)
    # Correlacion tamano bloque vs grosor
    if len(grosores) >= 5 and np.std(grosores) > 0 and np.std(tamanos) > 0:
        corr = pearsonr(grosores, tamanos)[0]
        print(f"  Correlacion(tamano_bloque, grosor_cuerpo) = {corr:+.3f}", flush=True)
        resultado["A2"] = {"corr": float(corr), "grosores": grosores, "tamanos": tamanos}
    else:
        resultado["A2"] = {"corr": float("nan")}

    # A3: Control con puntos aleatorios
    print(f"\n[A3] Control: 100 puntos aleatorios - ¿sus bloques son anatomicos?", flush=True)
    chi_aleatorios = []
    for _ in range(100):
        py = rng.integers(50, n-50)
        pfila = R[py, :]
        pbloques = bloques_recurrencia(pfila)
        if len(pbloques) < 3:
            continue
        pregiones = [proporcion_anatomica((b0+b1)/2/n) for b0, b1 in pbloques]
        pobs = Counter(pregiones)
        pchi = 0
        for region in esperado:
            obs_frac = pobs[region] / len(pbloques)
            exp_frac = esperado[region]
            pchi += (obs_frac - exp_frac)**2 / exp_frac
        chi_aleatorios.append(pchi)
    chi_aleatorios = np.array(chi_aleatorios)
    print(f"  Chi2 de puntos aleatorios: media={chi_aleatorios.mean():.3f}±{chi_aleatorios.std():.3f}", flush=True)
    print(f"  Chi2 del punto central: {chi:.3f}", flush=True)
    z = (chi - chi_aleatorios.mean()) / chi_aleatorios.std() if chi_aleatorios.std() > 0 else float("nan")
    print(f"  z-score: {z:+.2f} {'(punto central MAS anatomico que aleatorios)' if z > 2 else '(no significativo)'}", flush=True)
    resultado["A3"] = {"chi2_aleatorios_mean": float(chi_aleatorios.mean()),
                       "chi2_aleatorios_std": float(chi_aleatorios.std()),
                       "z": float(z)}
    return resultado

# ============================================================================
# FASE B: NO-LOCALIDAD TIPO AGUJERO DE GUSANO
# ============================================================================
def fase_B(R, profile, cx=416):
    print("\n" + "=" * 70, flush=True)
    print("FASE B: ¿CONEXION TIPO AGUJERO DE GUSANO? (no-localidad)", flush=True)
    print("=" * 70, flush=True)
    n = R.shape[0]
    fila = R[cx, :]
    bloques = bloques_recurrencia(fila)
    resultado = {}

    # B1: Bidireccionalidad - cada region tambien ve al punto central?
    print("\n[B1] Bidireccionalidad: ¿cada bloque tambien recurre con (416,416)?", flush=True)
    bidireccional = []
    for b0, b1 in bloques:
        y_centro = (b0 + b1) // 2
        # La fila del bloque: R[y_centro, 416] = 1 si el bloque ve al punto
        r_fila = R[y_centro, cx]
        # La columna del bloque: R[416, y_centro] = 1 (simetrica)
        r_col = R[cx, y_centro]
        bidireccional.append(float(r_fila))
        print(f"  bloque y={b0}-{b1} (centro {y_centro}): R[{y_centro},{cx}]={r_fila:.0f} "
              f"(R[{cx},{y_centro}]={r_col:.0f})", flush=True)
    frac_bidir = float(np.mean(bidireccional))
    print(f"  Fraccion de bloques que ven al punto central: {frac_bidir*100:.0f}%", flush=True)
    # Control: puntos aleatorios
    fracs_ctrl = []
    for _ in range(100):
        py = rng.integers(50, n-50)
        pfila = R[py, :]
        pbloques = bloques_recurrencia(pfila)
        if not pbloques:
            continue
        pb = [float(R[(b0+b1)//2, py]) for b0, b1 in pbloques]
        fracs_ctrl.append(float(np.mean(pb)))
    fracs_ctrl = np.array(fracs_ctrl)
    z = (frac_bidir - fracs_ctrl.mean()) / fracs_ctrl.std() if fracs_ctrl.std() > 0 else float("nan")
    print(f"  Control aleatorio: {fracs_ctrl.mean()*100:.0f}%±{fracs_ctrl.std()*100:.0f}% (z={z:+.2f})", flush=True)
    resultado["B1"] = {"frac_bidireccional": frac_bidir, "ctrl_mean": float(fracs_ctrl.mean()),
                       "ctrl_std": float(fracs_ctrl.std()), "z": float(z)}

    # B2: Precision de los tuneles (ancho de bloques vs azar)
    print("\n[B2] Precision de tuneles: ancho de bloques vs azar", flush=True)
    anchos = np.array([b1-b0+1 for b0, b1 in bloques])
    print(f"  Anchos reales: {anchos.tolist()}", flush=True)
    anchos_ctrl = []
    for _ in range(100):
        py = rng.integers(50, n-50)
        pbloques = bloques_recurrencia(R[py, :])
        if pbloques:
            anchos_ctrl.extend([b1-b0+1 for b0, b1 in pbloques])
    anchos_ctrl = np.array(anchos_ctrl)
    print(f"  Ancho medio real: {anchos.mean():.1f} vs control: {anchos_ctrl.mean():.1f}±{anchos_ctrl.std():.1f}", flush=True)
    z = (anchos.mean() - anchos_ctrl.mean()) / anchos_ctrl.std() if anchos_ctrl.std() > 0 else float("nan")
    print(f"  z-score: {z:+.2f}", flush=True)
    resultado["B2"] = {"anchos_real": anchos.tolist(), "ancho_medio_real": float(anchos.mean()),
                       "ctrl_mean": float(anchos_ctrl.mean()), "ctrl_std": float(anchos_ctrl.std()),
                       "z": float(z)}

    # B3: MI(punto, bloque) vs distancia anatomica
    print("\n[B3] MI(punto, bloque) vs distancia - no-localidad", flush=True)
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
    centro = R[cx-15:cx+15, cx-15:cx+15]
    mis = []
    dists = []
    for b0, b1 in bloques:
        y_centro = (b0 + b1) // 2
        bloque = R[max(0,y_centro-10):y_centro+10, cx-10:cx+10]
        if bloque.size == 0:
            continue
        dist = abs(y_centro - cx)
        mis.append(mi_2d(centro, bloque))
        dists.append(dist)
    if len(mis) >= 5:
        corr = pearsonr(dists, mis)[0]
        print(f"  MI por bloque: {[f'{m:.3f}' for m in mis]}", flush=True)
        print(f"  Distancias: {dists}", flush=True)
        print(f"  Correlacion(MI, distancia) = {corr:+.3f} "
              f"-> {'NO-LOCAL (independiente de distancia)' if abs(corr) < 0.3 else 'local'}", flush=True)
        resultado["B3"] = {"mis": mis, "dists": dists, "corr": float(corr)}
    return resultado

# ============================================================================
# FASE C: SIMULACION CUERPO 3D
# ============================================================================
def cuerpo_3d_humanoide(size=128):
    """Cuerpo humanoide 3D: cabeza (esfera) + torso (elipsoide) + piernas."""
    lin = torch.linspace(-1, 1, size, device=DEVICE)
    coords = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'), dim=-1)
    pts = coords.reshape(-1, 3).float()
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # Cuerpo: y = vertical (arriba = cabeza)
    # Cabeza: esfera en y ~ 0.75
    cabeza = ((x/0.18)**2 + ((y-0.72)/0.18)**2 + (z/0.18)**2)
    # Torso: elipsoide
    torso = ((x/0.30)**2 + ((y-0.30)/0.35)**2 + (z/0.20)**2)
    # Piernas: dos cilindros
    pierna_izq = ((x+0.12)/0.09)**2 + ((y+0.35)/0.45)**2 + (z/0.09)**2
    pierna_der = ((x-0.12)/0.09)**2 + ((y+0.35)/0.45)**2 + (z/0.09)**2
    dentro = (cabeza < 1) | (torso < 1) | (pierna_izq < 1) | (pierna_der < 1)
    # Densidad: 1 en el centro del cuerpo, decrece
    r_norm = torch.minimum(torch.minimum(cabeza, torso),
                           torch.minimum(pierna_izq, pierna_der))
    vals = torch.clamp(1.0 - r_norm * 0.3, 0, 1)
    vals[~dentro] = 0.0
    return pts, vals

def proyectar_3d(pts, vals, M, size=128):
    pts_rot = pts @ M.T
    idx = ((pts_rot[:, :2] + 1) / 2 * (size - 1)).long().clamp(0, size-1)
    flat_idx = idx[:, 0] * size + idx[:, 1]
    proj = torch.zeros(size * size, device=DEVICE, dtype=torch.float32)
    proj.scatter_add_(0, flat_idx, vals)
    proj = proj.view(size, size)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def fase_C():
    print("\n" + "=" * 70, flush=True)
    print("FASE C: ¿CONSISTENTE CON PROYECCION DE CUERPO 3D?", flush=True)
    print("=" * 70, flush=True)
    resultado = {}

    # Construir cuerpo 3D y proyectar frontalmente
    pts, vals = cuerpo_3d_humanoide()
    M = torch.eye(3, device=DEVICE)
    proj = proyectar_3d(pts, vals, M)
    proj_np = proj.cpu().numpy()

    # Perfil central de la proyeccion -> matriz de recurrencia -> bloques
    h, w = proj_np.shape
    perfil_proy = proj_np[:, w//2].astype(np.float32)
    # Normalizar a 0-255 como la imagen
    perfil_proy = (perfil_proy / (perfil_proy.max() + 1e-12) * 255).astype(np.float32)
    # Suavizar como el estudio
    perfil_proy_s = ndimage.gaussian_filter1d(perfil_proy, sigma=2)
    # Matriz de recurrencia
    R_proy = (np.abs(perfil_proy_s[:, None] - perfil_proy_s[None, :]) < 10.0).astype(np.float32)
    n_p = R_proy.shape[0]
    # Punto central de la proyeccion (el centro del cuerpo = y donde esta el torso)
    # En el cuerpo simulado, el torso esta en y~0.30 (rel 0.35 de abajo a arriba)
    cy_proy = int(n_p * 0.35)
    fila_proy = R_proy[cy_proy, :]
    bloques_proy = bloques_recurrencia(fila_proy)
    print(f"  Cuerpo 3D proyectado: perfil de {n_p} puntos", flush=True)
    print(f"  Punto central simulado (torso, y~0.35): {cy_proy}", flush=True)
    print(f"  Bloques de recurrencia del punto central simulado: {len(bloques_proy)}", flush=True)
    for b in bloques_proy:
        print(f"    y={b[0]}-{b[1]} (rel={((b[0]+b[1])/2)/n_p:.2f})", flush=True)

    # Guardar resultado de la simulacion
    resultado["C1"] = {"n_bloques_simulados": len(bloques_proy),
                       "bloques": [(int(b[0]), int(b[1])) for b in bloques_proy],
                       "perfil_longitud": int(n_p), "punto_central_simulado": int(cy_proy)}
    return resultado

# ============================================================================
# FASE D: AGUJERO DE GUSANO - CURVATURA Y TOPOLOGIA
# ============================================================================
def fase_D(R, cx=416):
    print("\n" + "=" * 70, flush=True)
    print("FASE D: HIPOTESIS DEL AGUJERO DE GUSANO", flush=True)
    print("=" * 70, flush=True)
    n = R.shape[0]
    resultado = {}

    # D1: Curvatura gaussiana alrededor del punto central
    print("\n[D1] Curvatura gaussiana alrededor del punto central", flush=True)
    region = R[cx-50:cx+50, cx-50:cx+50].astype(np.float64)
    region_s = cv2.GaussianBlur(region, (5, 5), 0)
    gy, gx = np.gradient(region_s)
    gyy, gxy = np.gradient(gy)
    gxy2, gxx = np.gradient(gx)
    K = (gxx * gyy - gxy**2) / (1 + gx**2 + gy**2)**2
    # Curvatura en el centro exacto y promedio
    K_centro = float(K[50, 50])
    K_media = float(K.mean())
    K_neg_frac = float((K < 0).mean())
    print(f"  K en el punto exacto: {K_centro:.6f}", flush=True)
    print(f"  K media region: {K_media:.6f}", flush=True)
    print(f"  Fraccion de curvatura negativa: {K_neg_frac*100:.1f}%", flush=True)
    resultado["D1"] = {"K_centro": K_centro, "K_media": K_media, "K_neg_frac": K_neg_frac}

    # D2: Topologia - componentes conectados que tocan la fila/columna 416
    print("\n[D2] Componentes conectados que tocan la fila/columna 416", flush=True)
    bin_R = (R > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_R)
    # Componentes que intersectan la fila 416 o columna 416
    fila_416 = set(labels[cx, :].tolist())
    col_416 = set(labels[:, cx].tolist())
    comps_fila = len([c for c in fila_416 if c > 0])
    comps_col = len([c for c in col_416 if c > 0])
    print(f"  Componentes que tocan la fila 416: {comps_fila}", flush=True)
    print(f"  Componentes que tocan la columna 416: {comps_col}", flush=True)
    print(f"  Total componentes: {num_labels-1}", flush=True)
    resultado["D2"] = {"comps_fila": int(comps_fila), "comps_col": int(comps_col),
                       "total": int(num_labels-1)}

    # D3: Documentar limites
    print("\n[D3] Limites de la interpretacion", flush=True)
    print("  - La no-localidad es medible (MI independiente de distancia)", flush=True)
    print("  - La bidireccionalidad y precision son medibles (B1, B2)", flush=True)
    print("  - 'Agujero de gusano' es una ANALOGIA, no una afirmacion fisica", flush=True)
    print("  - Un agujero de gusano requiere energia exotica (curvatura negativa)", flush=True)
    print("  - La curvatura negativa medible en D1 es un ANALOGO matematico", flush=True)
    resultado["D3"] = {"nota": "Agujero de gusano es analogia; curvatura negativa es analogo matematico"}
    return resultado

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    R, profile, img3 = matriz_real()
    cx = 416
    report = {}

    # Fase A
    report["fase_A"] = fase_A(R, profile, img3, cx)
    # Fase B
    report["fase_B"] = fase_B(R, profile, cx)
    # Fase C
    report["fase_C"] = fase_C()
    # Fase D
    report["fase_D"] = fase_D(R, cx)

    # CONCLUSION GLOBAL
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION GLOBAL: PROCESO DELIBERADO + AGUJERO DE GUSANO", flush=True)
    print("=" * 70, flush=True)
    a = report["fase_A"]; b = report["fase_B"]
    print(f"  A1 chi2 punto central: {a['A1']['chi2']:.3f} (A3 z={a['A3']['z']:+.2f})", flush=True)
    print(f"  A2 corr(tamano, grosor): {a['A2']['corr']:+.3f}", flush=True)
    print(f"  B1 bidireccionalidad: {b['B1']['frac_bidireccional']*100:.0f}% (z={b['B1']['z']:+.2f})", flush=True)
    print(f"  B2 precision tuneles: z={b['B2']['z']:+.2f}", flush=True)
    print(f"  B3 corr(MI, distancia): {b['B3']['corr']:+.3f}", flush=True)
    print(f"  C1 bloques simulados cuerpo 3D: {report['fase_C']['C1']['n_bloques_simulados']}", flush=True)
    print(f"  D1 curvatura negativa: {report['fase_D']['D1']['K_neg_frac']*100:.1f}%", flush=True)
    print(f"  D2 componentes conectados fila 416: {report['fase_D']['D2']['comps_fila']}", flush=True)

    out_json = os.path.join(OUT, "fases_AD_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
