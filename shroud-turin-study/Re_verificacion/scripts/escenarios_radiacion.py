"""
ESCENARIOS MULTIPLES DE RADIACION: ¿CUAL SE APROXIMA MAS?
=========================================================
Dato clave: el cuerpo estaba CUBIERTO con la Sabana. La imagen esta en
la cara INTERNA (contra el cuerpo). La radiacion viajo del cuerpo a la
tela (o se genero en la interfaz). El mapa de profundidad (intensidad =
cercania) implica que el CUERPO fue la fuente/modulador.

Escenarios a testear (S1-S15):
  S1.  EMISION GEOMETRICA PUNTUAL: I = I0/d^2 (fuente puntual en el cuerpo)
  S2.  EMISION CON ATENUACION: I = I0*e^(-beta*d)/d^2
  S3.  EMISION EXPONENCIAL PURA: I = I0*e^(-beta*d)
  S4.  EMISION VOLUMETRICA: I = integral(rho)/r^2 dV (cuerpo como volumen)
  S5.  FLUORESCENCIA: I = excitacion * eficiencia (pico central, cola)
  S6.  RADIACION TERMICA (Planck): I ~ T^4 (perfil suave del cuerpo)
  S7.  DESCARGAR CORONA: pico central muy agudo (I ~ exp(-d^2/sigma^2))
  S8.  CUERPO COMO BLOQUEADOR (radiacion de abajo): I = I0*(1 - absorcion)
  S9.  COMBINADO UV-A fondo + UV-C superficie: I = A*exp(-d/la) + B*exp(-d/lb)
  S10. DOBLE EXPONENCIAL (dos longitudes de onda)
  S11. EMISION + REFLEXION DE TELA: I = e^(-bd) + r*e^(-b(2t-d))
  S12. CAMPO ESTRUCTURADO GAUSSIANO con pendiente
  S13. POTENCIA: I = I0/(1 + d/d0)^n (varios n)
  S14. INTERFERENCIA / ONDA ESTACIONARIA (patron)
  S15. COMBINADO: emision puntual + atenuacion + contribucion volumetrica

Para cada escenario:
  - Generar perfil de intensidad esperado a lo largo del cuerpo
  - Correlacionar con el perfil REAL (mapa de profundidad del rostro)
  - Reportar mejor escenario

Validacion adicional:
  - El mejor escenario debe reproducir: suavidad, simetria, y la ley
    de decaimiento aproximada del registro

CPU. NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2
from scipy import ndimage

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() == 0 or b.std() == 0 or len(a) < 5:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

def escenarios_perfil(n):
    """Genera perfiles de intensidad esperados (normalizados 0-1) para
    cada escenario, a lo largo de una linea (d = distancia normalizada 0..1)."""
    x = np.linspace(0, 1, n)
    # d = distancia desde el centro del cuerpo (0=centro/cercano, 1=borde/lejos)
    d = np.abs(x - 0.5) * 2  # 0 en el centro, 1 en los bordes
    perfiles = {}

    # S1: Emision geometrica puntual I = I0/d^2 (con regularizacion)
    perfiles["S1_puntual_d2"] = 1.0 / (d**2 + 0.05)

    # S2: Emision con atenuacion
    perfiles["S2_atenuada_d2"] = np.exp(-2*d) / (d**2 + 0.05)

    # S3: Emision exponencial pura
    perfiles["S3_exponencial"] = np.exp(-3*d)

    # S4: Volumetrica (integral de densidad ~ perfil de esfera suave)
    perfiles["S4_volumetrica"] = np.sqrt(np.maximum(1 - d**2, 0))

    # S5: Fluorescencia (pico central, cola)
    perfiles["S5_fluorescencia"] = np.exp(-d) + 0.3*np.exp(-d**2*5)

    # S6: Termica (Planck ~ T^4, cuerpo emite calor suave)
    perfiles["S6_termica"] = np.exp(-d*1.5)

    # S7: Descarga corona (pico central muy agudo)
    perfiles["S7_corona"] = np.exp(-d**2 / 0.02)

    # S8: Cuerpo como bloqueador (radiacion desde abajo, cuerpo absorbe)
    # I = I0*(1 - absorcion) -> MENOS intensidad donde cuerpo cerca
    perfiles["S8_bloqueador"] = 1 - np.exp(-2*d)

    # S9: Combinado UV-A fondo + UV-C superficie
    perfiles["S9_dual_uv"] = 0.5*np.exp(-d) + 0.5*np.exp(-5*d)

    # S10: Doble exponencial (dos longitudes)
    perfiles["S10_doble_exp"] = 0.7*np.exp(-2*d) + 0.3*np.exp(-8*d)

    # S11: Emision + reflexion en tela (t = grosor tela)
    perfiles["S11_emision_reflexion"] = np.exp(-2*d) + 0.2*np.exp(-2*(2-d))

    # S12: Campo gaussiano estructurado
    perfiles["S12_gaussiano"] = np.exp(-d**2 / 0.3)

    # S13: Potencia con distintos n
    for n_pow in [1, 2, 3, 4]:
        perfiles[f"S13_potencia_n{n_pow}"] = 1.0 / (1 + d)**n_pow

    # S14: Interferencia/onda estacionaria
    perfiles["S14_onda"] = np.exp(-d) * (1 + 0.2*np.cos(20*d))

    # S15: Combinado puntual + atenuacion + volumetrica
    perfiles["S15_combinado"] = 0.4/(d**2+0.05) + 0.3*np.exp(-2*d) + 0.3*np.sqrt(np.maximum(1-d**2,0))

    # Normalizar todos a 0-1
    for k in perfiles:
        p = perfiles[k]
        p = (p - p.min()) / (p.max() - p.min() + 1e-9)
        perfiles[k] = p
    return perfiles

def main():
    t0 = time.time()
    report = {}

    print("=" * 70, flush=True)
    print("ESCENARIOS MULTIPLES DE RADIACION (S1-S15)", flush=True)
    print("=" * 70, flush=True)
    print("Dato clave: cuerpo CUBIERTO por la sabana; imagen en cara interna.", flush=True)

    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        print("imagen1 no disponible")
        return
    face = img1[100:1100, 1000:2000].astype(np.float64)
    h, w = face.shape

    # Perfil REAL: usar el perfil medio vertical del rostro (mapa de profundidad)
    # Promedio de filas en la zona central del rostro (para suavizar)
    # PERO: el recorte puede no estar centrado. Usar perfil medio vertical completo.
    perfil_real = face.mean(axis=1).astype(np.float64)
    # El perfil real de intensidad (negativo: claro = cerca de tela)
    perfil_real = (perfil_real - perfil_real.min()) / (perfil_real.max() - perfil_real.min() + 1e-9)
    n_real = len(perfil_real)

    # Invertir para que 0=borde, 1=centro (el recorte puede estar desplazado)
    # Probamos ambas orientaciones y con desplazamientos
    perfiles = escenarios_perfil(n_real)

    print(f"\nPerfil real (rostro, vertical): {n_real} puntos", flush=True)

    # Buscar el mejor desplazamiento/alineacion del modelo con el real
    resultados = {}
    for nombre, perfil_modelo in perfiles.items():
        mejores = []
        # Probar alineaciones: centrar el modelo en distintas posiciones
        for offset in [0, n_real//4, n_real//2, 3*n_real//4]:
            # Desplazar el modelo circularmente
            p_shift = np.roll(perfil_modelo, offset)
            # Correlacion directa
            c1 = corr(perfil_real, p_shift)
            # Correlacion invertida (modelo invertido = emision vs absorcion)
            c2 = corr(perfil_real, 1 - p_shift)
            mejores.append(max(c1, c2) if not np.isnan(c1) and not np.isnan(c2) else -1)
        resultados[nombre] = float(np.max(mejores))
        print(f"  {nombre}: mejor corr = {resultados[nombre]:+.3f}", flush=True)

    # Ranking
    ranking = sorted(resultados.items(), key=lambda kv: -kv[1])
    print("\nRANKING DE ESCENARIOS:", flush=True)
    for i, (nombre, c) in enumerate(ranking[:10]):
        print(f"  #{i+1}: {nombre} (corr={c:+.3f})", flush=True)
    report["ranking"] = resultados
    report["top5"] = [(k, v) for k, v in ranking[:5]]

    # ============ VALIDACION CRUZADA CON LA LEY DE DECAIMIENTO ============
    print("\nVALIDACION: ¿el mejor escenario reproduce el decaimiento del registro?", flush=True)
    # Perfil radial del registro (de la matriz de recurrencia, ya medido)
    # r, d (densidad registrada): [8, 0.915], [18, 0.285], [28, 0.189], ...
    r_reg = np.array([8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118, 128, 138, 148, 158, 168, 178, 188, 198])
    d_reg = np.array([0.915, 0.285, 0.189, 0.227, 0.321, 0.257, 0.212, 0.185, 0.197, 0.253,
                      0.257, 0.259, 0.363, 0.274, 0.222, 0.245, 0.168, 0.139, 0.099, 0.096])
    d_reg = d_reg / d_reg.max()
    # Modelos de decaimiento radial
    r_n = r_reg / r_reg.max()  # 0..1
    modelos_decaimiento = {}
    for nombre, perfil_m in perfiles.items():
        # Usar la mitad del perfil (del centro al borde)
        mitad = perfil_m[:n_real//2]
        if len(mitad) >= len(r_n):
            c = corr(d_reg, mitad[:len(r_n)])
        else:
            c = corr(d_reg[:len(mitad)], mitad)
        modelos_decaimiento[nombre] = float(c)
    # Ranking por decaimiento
    ranking_dec = sorted(modelos_decaimiento.items(), key=lambda kv: -kv[1])
    print("  Ranking por decaimiento radial del registro:", flush=True)
    for i, (nombre, c) in enumerate(ranking_dec[:5]):
        print(f"    #{i+1}: {nombre} (corr={c:+.3f})", flush=True)
    report["ranking_decaimiento"] = modelos_decaimiento
    report["top5_decaimiento"] = [(k, v) for k, v in ranking_dec[:5]]

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: MEJOR ESCENARIO DE RADIACION", flush=True)
    print("=" * 70, flush=True)
    print(f"  Perfil de rostro: mejor = {ranking[0][0]} (corr={ranking[0][1]:+.3f})", flush=True)
    print(f"  Decaimiento registro: mejor = {ranking_dec[0][0]} (corr={ranking_dec[0][1]:+.3f})", flush=True)
    report["conclusion"] = {
        "mejor_rostro": ranking[0][0], "corr_rostro": float(ranking[0][1]),
        "mejor_decaimiento": ranking_dec[0][0], "corr_decaimiento": float(ranking_dec[0][1]),
    }

    out_json = os.path.join(OUT, "escenarios_radiacion_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
