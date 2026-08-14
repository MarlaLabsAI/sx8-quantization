"""
TESTS T6-T10: FRACTALIDAD DE LA CRUZ + HIPOTESIS 4D
====================================================
El usuario: "la cruz tiene fractalidad, fijate bien" y "el punto de
conveccion multidimensional esta en el centro de la cruz".

T6. FRACTALIDAD MULTIESCALA DE LA CRUZ (test C del estudio)
    - D fractal de la cruz a escalas 100/50/25/10 (como test C)
    - Auto-similitud: CV de D entre escalas
    - Comparar con cruces genericas (gaussianos) y con controles

T7. HIPERESFERA 4D vs OBJETOS 3D (firma completa)
    - No solo perfil radial: D2 centro/periferia, da, densidad
    - La S3 proyectada tiene firma distinta a la esfera 3D

T8. LOCALIZACION PRECISA DEL PUNTO DE PROYECCION
    - Barrido de centros candidatos (416,416) vs otros
    - El punto de proyeccion debe maximizar densidad/D/da del centro
    - Si (416,416) es el maximo -> punto de proyeccion confirmado

T9. SIMETRIA CUATERNARIA (90 grados) COMO FIRMA 4D
    - La S3 tiene simetria SO(4) -> proyeccion con simetria 90 grados
    - Medir simetria rotacional de la cruz real vs genericas

T10. CONSISTENCIA TOMOGRAFICA (teorema de proyeccion central)
    - FFT de la proyeccion = corte central de la FFT 3D
    - Verificar consistencia interna de la cruz como proyeccion

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
N_CONTROLES = 30
N_WORKERS = 12

# ============================================================================
# METODOS (D12 exacto)
# ============================================================================
def box_counting_simple(matrix):
    sizes = [2, 4, 8]
    counts = []
    for size in sizes:
        h, w = matrix.shape
        n_boxes = 0
        for i in range(0, h, size):
            for j in range(0, w, size):
                box = matrix[i:i+size, j:j+size]
                if box.sum() > 0:
                    n_boxes += 1
        counts.append(n_boxes)
    sizes = np.array(sizes)
    counts = np.array(counts)
    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(coeffs[0])

def simple_multifractal_width(matrix):
    measure = matrix.flatten()
    measure = measure[measure > 0]
    if len(measure) == 0:
        return 0.0
    measure = measure / measure.sum()
    tau_q = []
    q_values = np.linspace(-3, 3, 13)
    for q in q_values:
        if q == 0:
            tau = 0
        else:
            tau = np.log(np.sum(measure ** q)) / np.log(10)
        tau_q.append(tau)
    tau_q = np.array(tau_q)
    alpha = np.gradient(tau_q, q_values)
    return float(alpha.max() - alpha.min())

def matriz_real():
    img3 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen3_sepia.jpeg"), cv2.IMREAD_GRAYSCALE)
    h, w = img3.shape
    profile = ndimage.gaussian_filter1d(img3[:, w//2].astype(np.float32), sigma=15)
    R_cont = np.abs(profile[:, None] - profile[None, :])
    R_bin = (R_cont < 10.0).astype(np.float32)
    return R_bin, profile

# ============================================================================
# T6: FRACTALIDAD MULTIESCALA (como test C del estudio)
# ============================================================================
def fractalidad_multiescala(R, cx, cy, escalas=(100, 50, 25, 10)):
    """D fractal de la cruz a multiples escalas (test C)."""
    n = R.shape[0]
    res = []
    for escala in escalas:
        region_size = escala * 2
        x0 = max(0, cx - escala)
        x1 = min(n, cx + escala)
        y0 = max(0, cy - escala)
        y1 = min(n, cy + escala)
        region = R[y0:y1, x0:x1]
        if region.size == 0:
            continue
        D = box_counting_simple(region)
        da = simple_multifractal_width(region)
        res.append({"escala": escala, "D": D, "da": da,
                    "densidad": float(region.mean())})
    return res

# ============================================================================
# T7: FIRMA COMPLETA DE OBJETOS (S3 vs 3D)
# ============================================================================
def objeto_3d(tipo, size=100):
    x, y, z = np.mgrid[-1:1:size*1j, -1:1:size*1j, -1:1:size*1j]
    r = np.sqrt(x**2 + y**2 + z**2)
    if tipo == "esfera":
        obj = (r < 0.9).astype(np.float64)
        obj[r < 0.7] = 0.8
        obj[r < 0.5] = 0.6
        obj[r < 0.3] = 0.4
    elif tipo == "elipsoide":
        obj = ((x/0.9)**2 + (y/0.6)**2 + (z/0.4)**2 < 1).astype(np.float64)
    elif tipo == "cruz3d":
        obj = ((np.abs(x) < 0.3) | (np.abs(y) < 0.3) | (np.abs(z) < 0.3)).astype(np.float64)
        obj = obj * (r < 1.0).astype(np.float64)
    elif tipo == "hiperesfera4d":
        w_max = np.sqrt(np.maximum(0.9**2 - r**2, 0))
        obj = 2.0 * w_max
    else:
        obj = (r < 0.9).astype(np.float64)
    return obj

def proyectar(obj, modo="suma"):
    if modo == "suma":
        proj = obj.sum(axis=2)
    else:
        proj = obj.max(axis=2)
    if proj.max() > 0:
        proj = proj / proj.max()
    return proj

def firma_objeto(proj):
    """Firma completa: D2 centro/periferia, da, densidad (metodo D12)."""
    h, w = proj.shape
    cy, cx = h//2, w//2
    centro = proj[cy-10:cy+10, cx-10:cx+10]
    perif = proj[:20, :20]
    return {
        "D_centro": box_counting_simple(centro),
        "D_periferia": box_counting_simple(perif),
        "da_centro": simple_multifractal_width(centro),
        "da_periferia": simple_multifractal_width(perif),
        "densidad_centro": float(centro.mean()),
        "densidad_periferia": float(perif.mean()),
    }

# ============================================================================
# T8: LOCALIZACION DEL PUNTO DE PROYECCION
# ============================================================================
def barrido_centros(R, candidatos):
    """Firma del centro para cada candidato."""
    res = {}
    for nombre, (cx, cy) in candidatos.items():
        region = R[max(0,cy-15):cy+15, max(0,cx-15):cx+15]
        res[nombre] = {
            "pos": (cx, cy),
            "densidad": float(region.mean()),
            "D": box_counting_simple(region),
            "da": simple_multifractal_width(region),
        }
    return res

# ============================================================================
# T9: SIMETRIA CUATERNARIA (90 grados)
# ============================================================================
def simetria_rotacional(R, cx, cy, radio=60):
    """Simetria rotacional 90 grados: comparar region con rotaciones."""
    region = R[max(0,cy-radio):cy+radio, max(0,cx-radio):cx+radio]
    if region.shape[0] != region.shape[1]:
        return float("nan")
    r90 = np.rot90(region)
    r180 = np.rot90(region, 2)
    r270 = np.rot90(region, 3)
    # Simetria: correlacion con rotaciones
    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return float("nan")
        return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])
    s90 = corr(region, r90)
    s180 = corr(region, r180)
    s270 = corr(region, r270)
    return {"sim_90": s90, "sim_180": s180, "sim_270": s270,
            "sim_90_promedio": float(np.nanmean([s90, s270]))}

# ============================================================================
# T10: CONSISTENCIA TOMOGRAFICA
# ============================================================================
def consistencia_tomografica(proj):
    """Teorema de proyeccion central: FFT(proyeccion) = corte central FFT 3D.
    Verificacion indirecta: la proyeccion debe tener simetria de Fourier
    consistente con un objeto 3D (espectro radialmente suave)."""
    f = np.fft.fft2(proj)
    f_shift = np.fft.fftshift(f)
    mag = np.abs(f_shift)
    h, w = mag.shape
    cy, cx = h//2, w//2
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2).astype(int)
    # Perfil radial del espectro
    max_r = min(h, w)//2
    radial = []
    for r in range(0, max_r, 2):
        mask = (dist >= r) & (dist < r+2)
        if mask.sum() > 0:
            radial.append(float(mag[mask].mean()))
    radial = np.array(radial)
    # Suavidad del espectro radial (objeto 3D -> decaimiento suave)
    if len(radial) < 10:
        return {"suavidad": float("nan")}
    log_r = np.log(np.arange(1, len(radial)+1))
    log_m = np.log(radial + 1e-12)
    # Ajuste de ley de potencia: mag ~ r^-alpha
    coeffs = np.polyfit(log_r, log_m, 1)
    alpha = -coeffs[0]
    # Residuo del ajuste (suavidad)
    pred = coeffs[0]*log_r + coeffs[1]
    resid = np.std(log_m - pred)
    return {"alpha": float(alpha), "residuo": float(resid)}

# ============================================================================
# MAIN
# ============================================================================
def main():
    t0 = time.time()
    report = {"T6_fractalidad": {}, "T7_firma_objetos": {}, "T8_punto_proyeccion": {},
              "T9_simetria": {}, "T10_tomografia": {}, "conclusion": {}}

    R_real, profile = matriz_real()
    n = R_real.shape[0]
    cx, cy = 416, 416
    print("=" * 70, flush=True)
    print(f"MATRIZ REAL: {n}x{n} | cruz=({cx},{cy})", flush=True)
    print("=" * 70, flush=True)

    # ============ T6: FRACTALIDAD MULTIESCALA ============
    print("\n[T6] FRACTALIDAD MULTIESCALA DE LA CRUZ (test C)", flush=True)
    fr = fractalidad_multiescala(R_real, cx, cy)
    print(f"  Escala | D     | da    | densidad", flush=True)
    for f in fr:
        print(f"  {f['escala']:6d} | {f['D']:.4f} | {f['da']:.4f} | {f['densidad']:.4f}", flush=True)
    Ds = np.array([f["D"] for f in fr])
    CV = Ds.std() / Ds.mean() if Ds.mean() > 0 else float("nan")
    print(f"  D media={Ds.mean():.4f} | CV={CV:.4f} (auto-similitud si CV<0.2)", flush=True)
    report["T6_fractalidad"]["real"] = fr
    report["T6_fractalidad"]["D_media"] = float(Ds.mean())
    report["T6_fractalidad"]["CV"] = float(CV)

    # Cruces genericas
    print(f"  Cruces genericas (gaussianos):", flush=True)
    gen_CVs = []
    for sigma_px in [50, 100, 200]:
        x = np.arange(n)
        p = np.exp(-0.5 * ((x - n//2) / sigma_px) ** 2)
        p = p + rng.normal(0, 0.05 * p.std(), size=n)
        Rc = (np.abs(p[:, None] - p[None, :]) < 10.0).astype(np.float32)
        frc = fractalidad_multiescala(Rc, n//2, n//2)
        Ds_c = np.array([f["D"] for f in frc])
        CV_c = Ds_c.std() / Ds_c.mean() if Ds_c.mean() > 0 else float("nan")
        gen_CVs.append(CV_c)
        print(f"    gauss sigma={sigma_px}: D_media={Ds_c.mean():.4f} | CV={CV_c:.4f}", flush=True)
    report["T6_fractalidad"]["genericas_CV"] = gen_CVs
    print(f"  CV real={CV:.4f} vs genericas={np.mean(gen_CVs):.4f}", flush=True)

    # ============ T7: FIRMA DE OBJETOS ============
    print("\n[T7] FIRMA COMPLETA DE OBJETOS (S3 vs 3D)", flush=True)
    firmas = {}
    for tipo in ["esfera", "elipsoide", "cruz3d", "hiperesfera4d"]:
        obj = objeto_3d(tipo)
        proj = proyectar(obj, "suma")
        firma = firma_objeto(proj)
        firmas[tipo] = firma
        print(f"  {tipo}: D_c={firma['D_centro']:.3f} D_p={firma['D_periferia']:.3f} | "
              f"da_c={firma['da_centro']:.3f} da_p={firma['da_periferia']:.3f} | "
              f"dens_c={firma['densidad_centro']:.3f} dens_p={firma['densidad_periferia']:.3f}", flush=True)
    report["T7_firma_objetos"] = firmas

    # ============ T8: PUNTO DE PROYECCION ============
    print("\n[T8] LOCALIZACION DEL PUNTO DE PROYECCION (barrido de centros)", flush=True)
    candidatos = {
        "cruz_416": (416, 416),
        "centro_540": (540, 540),
        "centro_geometrico": (n//2, n//2),
        "max_densidad": None,
    }
    # Encontrar el punto de maxima densidad local
    yy, xx = np.mgrid[0:n, 0:n]
    dens_map = ndimage.gaussian_filter(R_real, sigma=10)
    max_idx = np.unravel_index(np.argmax(dens_map), dens_map.shape)
    candidatos["max_densidad"] = (max_idx[1], max_idx[0])
    barrido = barrido_centros(R_real, candidatos)
    for nombre, b in barrido.items():
        print(f"  {nombre} {b['pos']}: densidad={b['densidad']:.4f} | D={b['D']:.4f} | da={b['da']:.4f}", flush=True)
    report["T8_punto_proyeccion"] = barrido
    # ¿Es (416,416) el maximo en densidad?
    mejor = max(barrido, key=lambda k: barrido[k]["densidad"])
    print(f"  Mejor por densidad: {mejor} ({barrido[mejor]['pos']})", flush=True)
    report["T8_mejor"] = mejor

    # ============ T9: SIMETRIA CUATERNARIA ============
    print("\n[T9] SIMETRIA ROTACIONAL 90 GRADOS (firma 4D)", flush=True)
    sim_real = simetria_rotacional(R_real, cx, cy)
    print(f"  REAL: sim_90={sim_real['sim_90']:+.3f} | sim_180={sim_real['sim_180']:+.3f} | sim_90_prom={sim_real['sim_90_promedio']:+.3f}", flush=True)
    # Genericas
    sims_gen = []
    for sigma_px in [50, 100, 200]:
        x = np.arange(n)
        p = np.exp(-0.5 * ((x - n//2) / sigma_px) ** 2)
        Rc = (np.abs(p[:, None] - p[None, :]) < 10.0).astype(np.float32)
        s = simetria_rotacional(Rc, n//2, n//2)
        sims_gen.append(s["sim_90_promedio"])
        print(f"    gauss sigma={sigma_px}: sim_90_prom={s['sim_90_promedio']:+.3f}", flush=True)
    print(f"  sim_90 real={sim_real['sim_90_promedio']:+.3f} vs genericas={np.mean(sims_gen):+.3f}", flush=True)
    report["T9_simetria"] = {"real": sim_real, "genericas": sims_gen}

    # ============ T10: CONSISTENCIA TOMOGRAFICA ============
    print("\n[T10] CONSISTENCIA TOMOGRAFICA (espectro radial)", flush=True)
    # Proyeccion de la esfera 3D y de la S3
    for tipo in ["esfera", "hiperesfera4d"]:
        obj = objeto_3d(tipo)
        proj = proyectar(obj, "suma")
        ct = consistencia_tomografica(proj)
        print(f"  {tipo}: alpha={ct['alpha']:.3f} | residuo={ct['residuo']:.4f}", flush=True)
        report["T10_tomografia"][tipo] = ct
    # La cruz real: usar la region alrededor de la cruz como 'proyeccion'
    region_cruz = R_real[cy-100:cy+100, cx-100:cx+100]
    ct_real = consistencia_tomografica(region_cruz)
    print(f"  CRUZ REAL: alpha={ct_real['alpha']:.3f} | residuo={ct_real['residuo']:.4f}", flush=True)
    report["T10_tomografia"]["cruz_real"] = ct_real

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION", flush=True)
    print("=" * 70, flush=True)
    report["conclusion"] = {
        "T6_CV_real": float(CV), "T6_CV_genericas": float(np.mean(gen_CVs)),
        "T8_mejor_centro": mejor,
        "T9_sim90_real": sim_real["sim_90_promedio"], "T9_sim90_genericas": float(np.mean(sims_gen)),
        "T10_alpha_cruz": ct_real["alpha"],
    }
    print(f"  T6: CV real={CV:.4f} vs genericas={np.mean(gen_CVs):.4f}", flush=True)
    print(f"  T8: mejor centro = {mejor}", flush=True)
    print(f"  T9: sim_90 real={sim_real['sim_90_promedio']:+.3f} vs genericas={np.mean(sims_gen):+.3f}", flush=True)
    print(f"  T10: alpha cruz={ct_real['alpha']:.3f}", flush=True)

    out_json = os.path.join(OUT, "tests_T6_T10_fractalidad_4d_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
