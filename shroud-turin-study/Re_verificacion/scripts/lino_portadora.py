"""
EL LINO COMO PORTADORA MODULADA POR EL EVENTO
==============================================
El usuario tiene razon: el lino NO es ruido a eliminar — es una COPIA
del proceso. La tela fue el soporte del registro, asi que el patron del
tejido quedo MODULADO por el evento (oxidacion UV diferencial de las
fibras). La textura del lino amplificada contiene informacion del mapa
de profundidad.

Las QUEMADURAS de 1532 SI son un suceso posterior independiente -> se
excluyen.

Proceso:
  L1. Excluir quemaduras (1532): outliers extremos de intensidad
  L2. Demodulacion del lino: extraer la AMPLITUD LOCAL de la frecuencia
      del tejido en cada punto (portadora = lino, modulacion = evento)
      - FFT local en ventanas -> magnitud del pico del lino por region
  L3. Comparar amplitud del lino en FIGURA vs FONDO:
      - Si el evento modulo el lino, la amplitud del patron cambia
        donde el cuerpo estuvo cerca
  L4. Reconstruir mapa de profundidad desde la modulacion del lino
      (la amplitud local del tejido = intensidad del evento)
  L5. Validar: simetria del mapa reconstruido desde la portadora
"""
import os, json, time
import numpy as np
import cv2
from scipy import ndimage

BASE = "/mnt/Data_3TB/Estudios_Sabana_Santa_Turin"
OUT = os.path.join(BASE, "Re_verificacion", "resultados")
os.makedirs(OUT, exist_ok=True)

def amplitud_lino_local(img, ventana=64, paso=32, frec_y=10):
    """Demodulacion: amplitud local de la frecuencia del lino.
    Usa FFT en ventanas deslizantes; toma la magnitud del pico del lino
    (que en imagen1/3 esta en el eje vertical, ~51-171 px/ciclo)."""
    h, w = img.shape
    mapa_amp = np.zeros((h, w), dtype=np.float64)
    mapa_freq = np.zeros((h, w), dtype=np.float64)
    # Para cada ventana, FFT y medir magnitud en banda de lino
    for y0 in range(0, h-ventana, paso):
        for x0 in range(0, w-ventana, paso):
            patch = img[y0:y0+ventana, x0:x0+ventana].astype(np.float64)
            patch = patch - patch.mean()
            F = np.fft.fft2(patch)
            F = np.fft.fftshift(F)
            mag = np.abs(F)
            cy, cx = ventana//2, ventana//2
            # Banda del lino: periodos 40-180 px -> frecuencias
            # En ventana de 64px: periodo 64/k -> k entre 0.35 y 1.6
            # Medir energia en anillo de frecuencias del lino (excluyendo DC)
            yy, xx = np.mgrid[0:ventana, 0:ventana]
            dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)
            # Frecuencia del lino ~ 1/periodo en px
            # periodos 51-171 -> frecuencias 0.006-0.02 (ciclos/px)
            # en unidades de la ventana: dist = freq * ventana
            f_min, f_max = ventana/171, ventana/51
            mascara_lino = (dist >= f_min) & (dist <= f_max)
            # Excluir ejes? el lino es vertical (solo dy), incluir todo
            amp = mag[mascara_lino].mean() if mascara_lino.sum() > 0 else 0
            # Rellenar la ventana
            mapa_amp[y0:y0+ventana, x0:x0+ventana] = amp
    # Interpolar regiones no cubiertas
    return mapa_amp

def main():
    t0 = time.time()
    report = {}
    print("="*70, flush=True)
    print("EL LINO COMO PORTADORA MODULADA POR EL EVENTO", flush=True)
    print("="*70, flush=True)

    img1 = cv2.imread(os.path.join(BASE, "04_IMAGENES_ORIGINALES", "imagen1_negativo.jpeg"), cv2.IMREAD_GRAYSCALE)
    face = img1[100:1100, 1000:2000].astype(np.float64)
    h, w = face.shape
    print(f"Rostro: {w}x{h}", flush=True)

    # L1: Excluir quemaduras (1532) - outliers extremos
    print("\n[L1] EXCLUSION DE QUEMADURAS (1532)", flush=True)
    media, std = face.mean(), face.std()
    quemadura = face > media + 2.5*std
    print(f"  Quemaduras (>2.5std): {quemadura.mean()*100:.1f}% del rostro", flush=True)
    # Tambien las zonas muy oscuras podrian ser quemaduras en positivo
    quemadura2 = face < media - 2.5*std
    print(f"  Zonas muy oscuras (<-2.5std): {quemadura2.mean()*100:.1f}%", flush=True)
    mask_quemadura = quemadura | quemadura2
    report["L1"] = {"frac_quemadura": float(mask_quemadura.mean())}

    # L2: Demodulacion del lino (amplitud local de la portadora)
    print("\n[L2] DEMODULACION DEL LINO (amplitud local del patron)", flush=True)
    mapa_amp = amplitud_lino_local(face, ventana=64, paso=32)
    # Guardar mapa de amplitud del lino
    amp_norm = (mapa_amp - mapa_amp.min()) / (mapa_amp.max() - mapa_amp.min() + 1e-9)
    amp_img = (amp_norm * 255).astype(np.uint8)
    p_amp = os.path.join(OUT, "lino_amplitud_local.png")
    cv2.imwrite(p_amp, amp_img)
    print(f"  Mapa de amplitud del lino guardado: {p_amp}", flush=True)
    report["L2_path"] = p_amp

    # Estadisticas: amplitud en figura vs fondo
    # La 'figura' en el negativo: el cuerpo es la zona de medio tono
    # (ni muy clara = quemadura, ni muy oscura = fondo)
    cuerpo_mask = ~mask_quemadura
    amp_figura = mapa_amp[cuerpo_mask].mean()
    amp_total = mapa_amp.mean()
    print(f"  Amplitud media del lino (todo): {amp_total:.0f}", flush=True)
    print(f"  Amplitud media del lino (sin quemaduras): {amp_figura:.0f}", flush=True)

    # L3: Comparar amplitud del lino en zonas de alto relieve vs bajo relieve
    print("\n[L3] AMPLITUD DEL LINO vs RELIEVE (intensidad facial)", flush=True)
    # En el negativo: intensidad alta = cerca de tela = mas oxidacion
    # Si el evento modula el lino: la amplitud del patron deberia
    # correlacionar con la intensidad (relieve)
    # Dividir en cuartiles de intensidad y medir amplitud del lino en cada uno
    intensidad = face[~mask_quemadura]
    amp_lino = mapa_amp[~mask_quemadura]
    cuartiles = np.percentile(intensidad, [25, 50, 75])
    print("  Cuartil intensidad | amplitud media del lino")
    for i, (q0, q1) in enumerate([(intensidad.min(), cuartiles[0]),
                                   (cuartiles[0], cuartiles[1]),
                                   (cuartiles[1], cuartiles[2]),
                                   (cuartiles[2], intensidad.max())]):
        mask = (intensidad >= q0) & (intensidad <= q1)
        if mask.sum() > 100:
            amp_q = amp_lino[mask].mean()
            print(f"    [{q0:5.0f}-{q1:5.0f}]: {amp_q:.0f}")
    # Correlacion amplitud-lino vs intensidad
    corr_lino = float(np.corrcoef(intensidad, amp_lino)[0, 1]) if len(intensidad) > 100 else float("nan")
    print(f"  Correlacion(amplitud lino, intensidad facial): {corr_lino:+.3f}", flush=True)
    print(f"  -> {'EL LINO ESTA MODULADO POR EL EVENTO (amplitud correlaciona con relieve)' if abs(corr_lino) > 0.3 else 'sin correlacion clara'}", flush=True)
    report["L3"] = {"corr_amp_intensidad": float(corr_lino)}

    # L4: Reconstruir mapa de profundidad desde la modulacion del lino
    print("\n[L4] MAPA DE PROFUNDIDAD DESDE LA MODULACION DEL LINO", flush=True)
    # Si el evento modula la amplitud del lino, el mapa de profundidad
    # = amplitud del lino normalizada (con signo segun correlacion)
    Z_lino = amp_norm.copy()
    Z_lino[mask_quemadura] = 0
    # Invertir si correlacion negativa (mas oxidacion = menos patron visible)
    if corr_lino < 0:
        Z_lino = 1 - Z_lino
        Z_lino[mask_quemadura] = 0
    Z_img = (Z_lino * 255).astype(np.uint8)
    p_z = os.path.join(OUT, "relieve_desde_lino.png")
    cv2.imwrite(p_z, Z_img)
    print(f"  Relieve desde lino guardado: {p_z}", flush=True)
    report["L4_path"] = p_z

    # L5: Validacion - simetria del mapa desde lino
    print("\n[L5] VALIDACION: SIMETRIA DEL MAPA DESDE EL LINO", flush=True)
    izq = Z_lino[:, :w//2]; der = np.fliplr(Z_lino[:, w//2:])
    mw = min(izq.shape[1], der.shape[1])
    m_i = (izq[:, :mw] > 0) & (der[:, :mw] > 0)
    sim_lino = float(np.corrcoef(izq[:, :mw][m_i], der[:, :mw][m_i])[0, 1]) if m_i.sum() > 100 else float("nan")
    print(f"  Simetria del mapa desde el lino: {sim_lino:+.3f}", flush=True)
    # Comparar con la simetria del relieve directo (0.777)
    print(f"  (relieve directo: +0.777)", flush=True)
    report["L5"] = {"simetria_lino": float(sim_lino)}

    # Conclusion
    print("\n" + "="*70, flush=True)
    print("CONCLUSION: EL LINO COMO PORTADORA DEL EVENTO", flush=True)
    print("="*70, flush=True)
    print(f"  Correlacion amplitud-lino vs intensidad: {corr_lino:+.3f}", flush=True)
    print(f"  Simetria del mapa desde el lino: {sim_lino:+.3f}", flush=True)
    report["conclusion"] = {"corr": float(corr_lino), "simetria_lino": float(sim_lino)}

    out_json = os.path.join(OUT, "lino_portadora_resultados.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_, bool))
                  else float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    print(f"Guardado: {out_json} | Tiempo: {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
