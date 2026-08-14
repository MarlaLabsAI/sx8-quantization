"""
BATERIA DE CORRELACIONES FISICAS: TODAS LAS PRUEBAS
====================================================
Correlacionar el evento con la fisica de materiales y procesos:

  X1. DENSIDADES DE MATERIALES:
      - Cuerpo humano (~1060 kg/m3), lino (~1500), hueso (~1900),
        aire (~1.2), celulosa (~1500)
      - ¿Como afecta la densidad a la atenuacion de radiacion?
      - Coeficientes de atenuacion de masa (mu/rho) para UV, X, gamma

  X2. LA PARADOJA DE LA SUPERFICIALIDAD:
      - La imagen esta SOLO en las primeras 40-60 micras de las fibras
      - Si fue una radiacion que "atraviesa" (como TAC), ¿por que no
        penetro la tela?
      - Penetracion de distintos tipos de radiacion en celulosa:
        UV-A (~mm), UV-B (~micras), VUV (<200nm: ~10-100nm), X (mm-cm)
      - ¿Que radiacion produce oxidacion superficial SOLO?

  X3. LA PARADOJA DE LA ATENUACION EN AIRE:
      - beta=0.005 medido. ¿Es compatible con atenuacion en AIRE?
      - Aire: mu ~0.0001-0.01 /m para UV; cuerpo-tela ~1-3cm
      - Si el aire atenua tan poco, ¿por que el gradiente de intensidad
        con la distancia es tan marcado?

  X4. TRES MODELOS DE PROYECCION:
      - Modelo TAC: I ∝ espesor integrado del cuerpo (radiografia)
      - Modelo EMISION: I ∝ e^(-beta*d) desde el cuerpo (campo emitido)
      - Modelo CONTACTO: I uniforme donde el cuerpo toca, 0 fuera
      - ¿Cual reproduce el mapa de profundidad real?

  X5. ENERGIA DEL PULSO (estimacion):
      - Oxidacion de celulosa: ~100-300 kJ/mol (enlace C-C/C-H)
      - Densidad de celulosa, espesor afectado (40-60 micras)
      - Energia por cm2 necesaria -> comparar con fuentes conocidas

  X6. SIMETRIA vs ROSTRO SINTETICO:
      - Simetria del mapa de profundidad real (0.745) vs un rostro
        humanoide sintetico (simetria esperada ~0.9+)
      - ¿La simetria real es compatible con un rostro?

  X7. COHERENCIA DE LA DIRECCION 45°:
      - Si el cuerpo estaba en la tumba (orientacion E-W?), la proyeccion
        tendria una direccion relacionada con la posicion
      - Analizar si la direccion preferente es consistente con un
        cuerpo acostado

CPU/GPU. NO modifica originales. Guarda en Re_verificacion/resultados/.
"""

import os
import json
import time
import numpy as np
import cv2

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def main():
    t0 = time.time()
    report = {}

    print("=" * 70, flush=True)
    print("BATERIA DE CORRELACIONES FISICAS DEL EVENTO", flush=True)
    print("=" * 70, flush=True)

    # ============ X1: DENSIDADES Y ATENUACION ============
    print("\n[X1] DENSIDADES DE MATERIALES Y ATENUACION DE RADIACION", flush=True)
    densidades = {
        "aire": 1.2, "cuerpo_tejido": 1060, "lino": 1500,
        "celulosa": 1500, "hueso": 1900, "agua": 1000,
    }
    for k, v in densidades.items():
        print(f"  {k}: {v} kg/m3", flush=True)
    report["X1_densidades"] = densidades

    # Coeficientes de atenuacion de masa (mu/rho) cm2/g aproximados
    # para tejido (agua) y lino (celulosa)
    print("\n  Coeficientes de atenuacion de masa (cm2/g):", flush=True)
    print("  (cuanto MAS alto, MENOS penetra la radiacion)", flush=True)
    radiaciones = {
        "UV-B (280nm)": {"tejido": 100.0, "celulosa": 90.0, "penetracion": "micras"},
        "UV-C (254nm)": {"tejido": 300.0, "celulosa": 280.0, "penetracion": "sub-micras"},
        "VUV (150nm)": {"tejido": 2000.0, "celulosa": 1900.0, "penetracion": "10-100 nm"},
        "X suave (10keV)": {"tejido": 4.5, "celulosa": 3.0, "penetracion": "mm"},
        "X dura (100keV)": {"tejido": 0.17, "celulosa": 0.15, "penetracion": "cm"},
        "gamma (1MeV)": {"tejido": 0.07, "celulosa": 0.06, "penetracion": "decenas cm"},
    }
    for nombre, coefs in radiaciones.items():
        print(f"  {nombre}: tejido={coefs['tejido']} | celulosa={coefs['celulosa']} | "
              f"penetracion={coefs['penetracion']}", flush=True)
    report["X1_atenuacion"] = radiaciones

    # ============ X2: PARADOJA DE LA SUPERFICIALIDAD ============
    print("\n[X2] LA PARADOJA DE LA SUPERFICIALIDAD", flush=True)
    print("  La imagen esta SOLO en las primeras 40-60 micras de las fibras.", flush=True)
    print("  ¿Que radiacion produce oxidacion superficial SOLO, sin penetrar?", flush=True)
    print("  Penetracion (1/mu) en celulosa:", flush=True)
    for nombre, coefs in radiaciones.items():
        # mu = coef * densidad (densidad celulosa ~1.5 g/cm3)
        mu = coefs["celulosa"] * 1.5  # 1/cm
        penetracion_cm = 1.0 / mu
        penetracion_um = penetracion_cm * 10000
        compat = "COMPATIBLE con 40-60 um" if 1 <= penetracion_um <= 200 else "no"
        print(f"  {nombre}: 1/mu = {penetracion_um:.1f} um -> {compat}", flush=True)
        coefs["penetracion_um"] = float(penetracion_um)
    report["X2_superficialidad"] = radiaciones

    # ============ X3: PARADOJA DE LA ATENUACION EN AIRE ============
    print("\n[X3] LA PARADOJA DE LA ATENUACION EN AIRE", flush=True)
    print("  beta=0.005 medido en el registro (ley exponencial del perfil radial).", flush=True)
    print("  Si la atenuacion es en AIRE (cuerpo->tela, ~1-3 cm):", flush=True)
    # Aire: mu_aire ~ 0.001-0.01 cm^-1 para UV (despreciable)
    for beta_cm in [0.001, 0.005, 0.01, 0.05]:
        distancia_1e = 1.0 / beta_cm  # distancia para atenuar 1/e
        print(f"  beta={beta_cm} cm^-1: distancia 1/e = {distancia_1e:.0f} cm", flush=True)
    print("  Aire real para UV: mu ~0.001-0.01 cm^-1 -> 1/e a 100-1000 cm", flush=True)
    print("  -> Un gradiente marcado en 1-3 cm NO es atenuacion en aire.", flush=True)
    print("  -> El gradiente de intensidad con la distancia NO puede ser", flush=True)
    print("     atenuacion atmosferica. Debe ser otra cosa:", flush=True)
    print("     (a) Intensidad geometrica: I ~ 1/d^2 (fuente puntual)", flush=True)
    print("     (b) Campo de intensidad espacial del evento (estructurado)", flush=True)
    print("     (c) Efecto de angulo: la radiacion incide con angulo", flush=True)
    report["X3_paradoja_aire"] = {"beta": 0.005, "analisis": "gradiente no es atenuacion en aire"}

    # ============ X4: TRES MODELOS DE PROYECCION ============
    print("\n[X4] TRES MODELOS DE PROYECCION vs MAPA DE PROFUNDIDAD REAL", flush=True)
    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    if img1 is not None:
        face = img1[100:1100, 1000:2000].astype(np.float64)
        h, w = face.shape
        # Perfil de intensidad real a lo largo del eje vertical central
        perfil_real = face[:, w//2].astype(np.float64)
        perfil_real = (perfil_real - perfil_real.min()) / (perfil_real.max() - perfil_real.min() + 1e-9)

        # Modelo 1: TAC (I ∝ espesor) - el centro del cuerpo mas denso
        # Perfil de espesor de un objeto convexo: 2*sqrt(R^2 - x^2) -> parabola
        x = np.linspace(-1, 1, len(perfil_real))
        perfil_tac = np.sqrt(np.maximum(1 - x**2, 0))  # semicirculo (espesor)
        perfil_tac = perfil_tac / perfil_tac.max()

        # Modelo 2: EMISION con atenuacion (I ∝ e^(-beta*d), d = distancia)
        # d crece del centro (cercano a tela) hacia bordes (lejos)
        d = np.abs(x)  # distancia normalizada desde el centro
        perfil_emis = np.exp(-d * 2.0)  # atenuacion desde el centro
        perfil_emis = perfil_emis / perfil_emis.max()

        # Modelo 3: CONTACTO (uniforme donde toca, 0 fuera)
        perfil_cont = (np.abs(x) < 0.7).astype(np.float64)

        # Correlaciones con el real
        def corr(a, b):
            if a.std() == 0 or b.std() == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        c_tac = corr(perfil_real, perfil_tac)
        c_emis = corr(perfil_real, perfil_emis)
        c_cont = corr(perfil_real, perfil_cont)
        print(f"  Correlacion con perfil real:", flush=True)
        print(f"    Modelo TAC (espesor): {c_tac:+.3f}", flush=True)
        print(f"    Modelo EMISION (e^-beta*d): {c_emis:+.3f}", flush=True)
        print(f"    Modelo CONTACTO: {c_cont:+.3f}", flush=True)
        mejor = max([("TAC", c_tac), ("EMISION", c_emis), ("CONTACTO", c_cont)], key=lambda m: m[1])
        print(f"  MEJOR MODELO: {mejor[0]} ({mejor[1]:+.3f})", flush=True)
        report["X4_modelos"] = {"tac": c_tac, "emision": c_emis, "contacto": c_cont,
                                "mejor": mejor[0]}
    else:
        print("  imagen1 no disponible", flush=True)

    # ============ X5: ENERGIA DEL PULSO ============
    print("\n[X5] ESTIMACION DE LA ENERGIA DEL PULSO", flush=True)
    # Oxidacion de celulosa: energia de activacion ~100-200 kJ/mol
    # Celulosa: densidad 1.5 g/cm3, peso molecular monomero ~162 g/mol
    # Espesor afectado: 40-60 micras = 0.004-0.006 cm
    energia_activacion = 150  # kJ/mol
    peso_molar = 162  # g/mol
    densidad_cel = 1.5  # g/cm3
    espesor_cm = 0.005  # 50 micras
    # Moles de celulosa por cm2 afectadas
    moles_cm2 = (densidad_cel * espesor_cm) / peso_molar  # mol/cm2
    energia_cm2 = moles_cm2 * energia_activacion * 1000  # J/cm2 (kJ->J)
    print(f"  Celulosa afectada: {espesor_cm*10000:.0f} um", flush=True)
    print(f"  Moles/cm2: {moles_cm2:.6f}", flush=True)
    print(f"  Energia minima por cm2 (activacion): {energia_cm2:.3f} J/cm2", flush=True)
    # Total del cuerpo: ~1 m2 = 10000 cm2
    energia_total = energia_cm2 * 10000
    print(f"  Energia total (1 m2 de cuerpo): {energia_total:.0f} J = {energia_total/1000:.1f} kJ", flush=True)
    # Comparar con fuentes
    print(f"  Comparacion:", flush=True)
    print(f"    Flash de camara profesional: ~100-1000 J", flush=True)
    print(f"    Lampara UV industrial (1s): ~10-100 J/cm2", flush=True)
    print(f"    Relampago: ~10^9 J (pero difuso)", flush=True)
    print(f"    Necesario (estimado): {energia_cm2:.1f} J/cm2 -> {'COMPARABLE a lampara UV' if 1 < energia_cm2 < 100 else 'muy alto'}", flush=True)
    report["X5_energia"] = {"energia_cm2": float(energia_cm2), "energia_total_kJ": float(energia_total/1000)}

    # ============ X6: SIMETRIA vs ROSTRO SINTETICO ============
    print("\n[X6] SIMETRIA: mapa real vs rostro sintetico", flush=True)
    if img1 is not None:
        face = img1[100:1100, 1000:2000].astype(np.float64)
        h, w = face.shape
        # Simetria del real
        izq = face[:, :w//2]
        der = np.fliplr(face[:, w//2:])
        mw = min(izq.shape[1], der.shape[1])
        sim_real = float(np.corrcoef(izq[:, :mw].flatten(), der[:, :mw].flatten())[0, 1])
        print(f"  Simetria mapa real: {sim_real:+.3f}", flush=True)

        # Rostro sintetico: elipsoide suave + rasgos (nariz, ojos)
        rng = np.random.default_rng(42)
        yy, xx = np.mgrid[0:h, 0:w]
        # Cara ovalada
        cx, cy = w//2, h//2
        r = np.sqrt(((xx-cx)/(w*0.35))**2 + ((yy-cy)/(h*0.45))**2)
        cara = np.clip(1 - r, 0, 1)
        # Nariz (pico central)
        nariz = 0.3 * np.exp(-((xx-cx)**2 + (yy-cy*0.8)**2) / (2*15**2))
        # Ojos (valles)
        ojo_izq = -0.2 * np.exp(-((xx-(cx-70))**2 + (yy-cy*0.65)**2) / (2*12**2))
        ojo_der = -0.2 * np.exp(-((xx-(cx+70))**2 + (yy-cy*0.65)**2) / (2*12**2))
        sintetico = cara + nariz + ojo_izq + ojo_der
        sintetico = (sintetico - sintetico.min()) / (sintetico.max() - sintetico.min() + 1e-9)
        sintetico = sintetico * 255
        # Simetria del sintetico (debe ser ~0.95+)
        s_izq = sintetico[:, :w//2]
        s_der = np.fliplr(sintetico[:, w//2:])
        sim_sint = float(np.corrcoef(s_izq[:, :mw].flatten(), s_der[:, :mw].flatten())[0, 1])
        print(f"  Simetria rostro sintetico: {sim_sint:+.3f}", flush=True)
        # ¿El real es compatible con un rostro? (simetria real/sintetica)
        ratio_sim = sim_real / (sim_sint + 1e-9)
        print(f"  Ratio simetria real/sintetica: {ratio_sim:.2f}", flush=True)
        print(f"  -> {'COMPATIBLE (el mapa real tiene simetria de rostro)' if ratio_sim > 0.7 else 'no compatible'}", flush=True)
        report["X6_simetria"] = {"real": sim_real, "sintetico": sim_sint, "ratio": float(ratio_sim)}

    # ============ X7: DIRECCION 45° vs GEOMETRIA ============
    print("\n[X7] LA DIRECCION 45° - consistencia geometrica", flush=True)
    print("  La direccion preferente del registro fue ~45° (SE) en la matriz.", flush=True)
    print("  La matriz de recurrencia es del PERFIL VERTICAL (columna central).", flush=True)
    print("  La direccion 45° en la matriz corresponde a la banda diagonal.", flush=True)
    print("  La banda diagonal es la geometria OBLIGATORIA de toda matriz", flush=True)
    print("  de recurrencia (R(i,i)=1 siempre).", flush=True)
    print("  -> La 'direccion preferente' a 45° es en gran parte la banda", flush=True)
    print("     diagonal (artefacto de la matriz), NO la direccion del evento.", flush=True)
    print("  -> La direccion REAL del evento debe medirse en la IMAGEN,", flush=True)
    print("     no en la matriz de recurrencia.", flush=True)
    report["X7_direccion"] = {"nota": "45° es en gran parte la banda diagonal (artefacto)"}

    # ============ CONCLUSION ============
    print("\n" + "=" * 70, flush=True)
    print("CONCLUSION: CORRELACIONES FISICAS", flush=True)
    print("=" * 70, flush=True)
    print(f"  X2: radiacion compatible con superficialidad: VUV/UV-C (penetracion <100um)", flush=True)
    print(f"  X3: el gradiente NO es atenuacion en aire -> intensidad geometrica o campo estructurado", flush=True)
    if img1 is not None:
        print(f"  X4: mejor modelo de proyeccion: {mejor[0]} ({mejor[1]:+.3f})", flush=True)
        print(f"  X6: simetria real={sim_real:+.3f} vs sintetico={sim_sint:+.3f} (ratio {ratio_sim:.2f})", flush=True)
    print(f"  X5: energia estimada del pulso: {energia_cm2:.1f} J/cm2", flush=True)
    report["conclusion"] = {
        "radiacion_superficial": "VUV/UV-C",
        "paradoja_aire": "gradiente no es atenuacion atmosferica",
        "mejor_modelo": mejor[0] if img1 is not None else None,
        "simetria_real": float(sim_real) if img1 is not None else None,
        "energia_cm2": float(energia_cm2),
    }

    out_json = os.path.join(OUT, "correlaciones_fisicas_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"\nGuardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
