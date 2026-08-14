"""
TESTS DIMENSIONALES DE LA CRUZ CENTRAL (D1-D10)
================================================
Investiga si la cruz central opera en una dimensión/topología distinta al resto.

Tests:
D1: Dimensión fractal local (centro vs resto)
D2: Análisis multifractal local (espectro de singularidades)
D3: Curvatura del campo de densidad (tensor de curvatura)
D4: Topología local (números de Betti, agujeros)
D5: Análisis espectral local (FFT centrado en cruz)
D6: Transformada wavelet 2D (resolución multiescala centrada)
D7: Análisis de homología persistente
D8: Tensor de tensión/información en el centro
D9: Proyección dimensional (¿sombra de objeto de mayor dimensión?)
D10: Análisis de flujo topológico (tipo de singularidad)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import torch
from scipy import ndimage, signal
from scipy.spatial.distance import pdist, squareform
import json
import time
import os
import cv2
from PIL import Image

# Configuración
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE3_PATH = r"C:\turin\Image June 06, 2026 - 12_22PM(2).jpeg"
OUTPUT_DIR = r"C:\turin\resultados\analisis_chip"
CROSS_CENTER = (416, 416)  # Centro de la cruz en imagen 3
REGION_SIZE = 100  # Tamaño de región central a analizar

print("=" * 70)
print("TESTS DIMENSIONALES DE LA CRUZ CENTRAL (D1-D10)")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Centro de cruz: {CROSS_CENTER}")
print(f"Región de análisis: {REGION_SIZE}x{REGION_SIZE}px centrada en la cruz\n")

# ============================================================================
# FASE 0: CARGAR Y PREPARAR MATRIZ DE RECURRENCIA
# ============================================================================

def load_and_prepare_recurrence_matrix():
    """Cargar imagen y generar matriz de recurrencia."""
    print("FASE 0: Cargando imagen y generando matriz de recurrencia...")
    t0 = time.time()
    
    # Cargar imagen
    img = cv2.imread(IMAGE3_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    print(f"   Imagen: {w}x{h}px")
    
    # Extraer perfil vertical del eje central
    profile = img[:, w//2].astype(np.float32)
    
    # Suavizado Gaussiano
    profile_smooth = ndimage.gaussian_filter1d(profile, sigma=15)
    
    # Generar matriz de recurrencia (binaria)
    threshold = 10.0
    n = len(profile_smooth)
    
    # Usar GPU para acelerar
    profile_tensor = torch.from_numpy(profile_smooth).to(DEVICE)
    
    # Calcular matriz de diferencias
    diff_matrix = torch.abs(profile_tensor.unsqueeze(0) - profile_tensor.unsqueeze(1))
    recurrence = (diff_matrix < threshold).float()
    
    R = recurrence.cpu().numpy()
    print(f"   Matriz de recurrencia: {n}x{n}")
    print(f"   Densidad: {R.mean():.4f}")
    print(f"   Tiempo: {time.time()-t0:.1f}s\n")
    
    return R, n

# ============================================================================
# D1: DIMENSIÓN FRACTAL LOCAL
# ============================================================================

def test_D1_local_fractal_dimension(R, n):
    """
    D1: Calcular dimensión fractal en región central vs resto.
    Si el centro tiene D diferente, opera en dimensión distinta.
    """
    print("=" * 70)
    print("D1: DIMENSIÓN FRACTAL LOCAL (centro vs resto)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # Extraer región periférica (esquinas)
    corner_size = half
    corner1 = R[:corner_size, :corner_size]
    corner2 = R[-corner_size:, -corner_size:]
    corner3 = R[:corner_size, -corner_size:]
    corner4 = R[-corner_size:, :corner_size]
    peripheral_region = np.concatenate([
        corner1.flatten(), corner2.flatten(), 
        corner3.flatten(), corner4.flatten()
    ]).reshape(2*corner_size, 2*corner_size)
    
    # Box-counting para región central
    def box_counting(matrix, min_size=2, max_size=None):
        if max_size is None:
            max_size = min(matrix.shape) // 2
        
        sizes = []
        counts = []
        
        size = min_size
        while size <= max_size:
            # Dividir matriz en cajas de tamaño 'size'
            h, w = matrix.shape
            n_boxes = 0
            
            for i in range(0, h, size):
                for j in range(0, w, size):
                    box = matrix[i:i+size, j:j+size]
                    if box.sum() > 0:
                        n_boxes += 1
            
            sizes.append(size)
            counts.append(n_boxes)
            size *= 2
        
        sizes = np.array(sizes)
        counts = np.array(counts)
        
        # Ajustar línea: log(N) vs log(1/size)
        if len(sizes) > 2:
            log_sizes = np.log(1.0 / sizes)
            log_counts = np.log(counts)
            
            # Regresión lineal
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            D = coeffs[0]
        else:
            D = 0
        
        return D, sizes, counts
    
    D_center, sizes_c, counts_c = box_counting(center_region)
    D_peripheral, sizes_p, counts_p = box_counting(peripheral_region)
    D_global, sizes_g, counts_g = box_counting(R[:512, :512])  # Muestra global
    
    print(f"   Región central ({REGION_SIZE}x{REGION_SIZE}):")
    print(f"      D = {D_center:.4f}")
    print(f"      Densidad = {center_region.mean():.4f}")
    print()
    print(f"   Región periférica (esquinas):")
    print(f"      D = {D_peripheral:.4f}")
    print(f"      Densidad = {peripheral_region.mean():.4f}")
    print()
    print(f"   Global (muestra 512x512):")
    print(f"      D = {D_global:.4f}")
    print()
    
    # Interpretación
    diff = abs(D_center - D_peripheral)
    if diff > 0.1:
        print(f"   [!] DIFERENCIA SIGNIFICATIVA: ΔD = {diff:.4f}")
        if D_center > D_peripheral:
            print(f"   -> El centro tiene MAYOR dimensión fractal (más complejo)")
        else:
            print(f"   -> El centro tiene MENOR dimensión fractal (más simple/colapsado)")
    else:
        print(f"   [OK] Diferencia pequeña: ΔD = {diff:.4f}")
        print(f"   -> Centro y periferia tienen dimensión similar")
    
    print()
    
    return {
        'D_center': float(D_center),
        'D_peripheral': float(D_peripheral),
        'D_global': float(D_global),
        'delta_D': float(diff),
        'center_density': float(center_region.mean()),
        'peripheral_density': float(peripheral_region.mean()),
    }

# ============================================================================
# D2: ANÁLISIS MULTIFRACTAL LOCAL
# ============================================================================

def test_D2_multifractal_analysis(R, n):
    """
    D2: Espectro multifractal en centro vs resto.
    El espectro f(α) revela si hay singularidades de diferente fuerza.
    """
    print("=" * 70)
    print("D2: ANÁLISIS MULTIFRACTAL LOCAL (espectro de singularidades)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer regiones
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # Región anular alrededor del centro
    annulus = np.zeros((REGION_SIZE*2, REGION_SIZE*2))
    annulus[half-20:half+20, half-20:half+20] = 0  # Excluir centro
    annulus[:REGION_SIZE, :REGION_SIZE] = R[x1-half:x1+half, y1-half:y1+half]
    
    # Método de momentos (partition function)
    def multifractal_spectrum(matrix, q_values=None):
        if q_values is None:
            q_values = np.linspace(-5, 5, 21)
        
        # Dividir en cajas
        size = 8
        h, w = matrix.shape
        n_h = h // size
        n_w = w // size
        
        # Calcular medida en cada caja
        measure = np.zeros((n_h, n_w))
        for i in range(n_h):
            for j in range(n_w):
                box = matrix[i*size:(i+1)*size, j*size:(j+1)*size]
                measure[i, j] = box.sum() / (size * size)
        
        # Normalizar
        measure = measure[measure > 0]
        if len(measure) == 0:
            return None, None, None
        
        measure = measure / measure.sum()
        
        # Calcular τ(q)
        tau_q = []
        for q in q_values:
            if q == 0:
                tau = 0
            else:
                tau = np.log(np.sum(measure ** q)) / np.log(1.0/size)
            tau_q.append(tau)
        
        tau_q = np.array(tau_q)
        
        # Calcular α y f(α) por derivada numérica
        alpha = np.gradient(tau_q, q_values)
        f_alpha = q_values * alpha - tau_q
        
        return q_values, alpha, f_alpha
    
    q, alpha_center, f_center = multifractal_spectrum(center_region)
    
    if alpha_center is not None:
        # Ancho del espectro (medida de multifractalidad)
        alpha_min = alpha_center.min()
        alpha_max = alpha_center.max()
        delta_alpha = alpha_max - alpha_min
        
        # Asimetría del espectro
        alpha_peak = alpha_center[np.argmax(f_center)]
        skewness = (alpha_max - alpha_peak) - (alpha_peak - alpha_min)
        
        print(f"   Espectro multifractal del centro:")
        print(f"      α_min = {alpha_min:.4f} (singularidad más fuerte)")
        print(f"      α_max = {alpha_max:.4f} (singularidad más débil)")
        print(f"      Δα = {delta_alpha:.4f} (ancho del espectro)")
        print(f"      α_pico = {alpha_peak:.4f}")
        print(f"      Asimetría = {skewness:.4f}")
        print()
        
        if delta_alpha > 0.5:
            print(f"   ⚠️  Espectro ANCHO: estructura multifractal compleja")
            print(f"   → Múltiples tipos de singularidades coexisten")
        elif delta_alpha < 0.2:
            print(f"   ✓ Espectro ESTRECHO: estructura casi monofractal")
            print(f"   → Singularidad uniforme")
        else:
            print(f"   Espectro MODERADO: multifractalidad intermedia")
        
        print()
        
        return {
            'alpha_min': float(alpha_min),
            'alpha_max': float(alpha_max),
            'delta_alpha': float(delta_alpha),
            'alpha_peak': float(alpha_peak),
            'skewness': float(skewness),
        }
    else:
        print(f"   No se pudo calcular espectro multifractal\n")
        return None

# ============================================================================
# D3: CURVATURA DEL CAMPO DE DENSIDAD
# ============================================================================

def test_D3_curvature_analysis(R, n):
    """
    D3: Calcular curvatura del campo de densidad.
    Si hay curvatura infinita en el centro → singularidad.
    """
    print("=" * 70)
    print("D3: CURVATURA DEL CAMPO DE DENSIDAD (tensor de curvatura)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2].astype(np.float64)
    
    # Calcular gradiente
    gy, gx = np.gradient(center_region)
    
    # Calcular curvatura gaussiana (K) y curvatura media (H)
    # Para superficie z = f(x,y):
    # K = (f_xx * f_yy - f_xy^2) / (1 + f_x^2 + f_y^2)^2
    # H = ((1+f_y^2)*f_xx - 2*f_x*f_y*f_xy + (1+f_x^2)*f_yy) / (2*(1+f_x^2+f_y^2)^(3/2))
    
    gyy, gxy = np.gradient(gy)
    gxy2, gxx = np.gradient(gx)
    
    # Curvatura gaussiana
    K = (gxx * gyy - gxy**2) / (1 + gx**2 + gy**2)**2
    
    # Curvatura media
    H = ((1 + gy**2) * gxx - 2 * gx * gy * gxy + (1 + gx**2) * gyy) / (2 * (1 + gx**2 + gy**2)**1.5)
    
    # Analizar en el centro exacto
    center_x = half
    center_y = half
    
    K_center = K[center_x, center_y]
    H_center = H[center_x, center_y]
    
    # Promedio en región central (radio 20px)
    radius = 20
    mask = np.zeros_like(K, dtype=bool)
    mask[center_x-radius:center_x+radius, center_y-radius:center_y+radius] = True
    K_center_avg = K[mask].mean()
    H_center_avg = H[mask].mean()
    
    # Promedio en periferia
    K_peripheral = K[~mask].mean()
    H_peripheral = H[~mask].mean()
    
    print(f"   En el centro exacto ({center_x}, {center_y}):")
    print(f"      Curvatura Gaussiana K = {K_center:.6f}")
    print(f"      Curvatura Media H = {H_center:.6f}")
    print()
    print(f"   Promedio región central (radio {radius}px):")
    print(f"      K = {K_center_avg:.6f}")
    print(f"      H = {H_center_avg:.6f}")
    print()
    print(f"   Promedio periferia:")
    print(f"      K = {K_peripheral:.6f}")
    print(f"      H = {H_peripheral:.6f}")
    print()
    
    # Ratio centro/periferia
    if abs(K_peripheral) > 1e-10:
        K_ratio = K_center_avg / K_peripheral
        print(f"   Ratio K_centro / K_periferia = {K_ratio:.2f}x")
        
        if abs(K_ratio) > 10:
            print(f"   ⚠️  CURVATURA MUY ELEVADA en el centro")
            print(f"   → Posible SINGULARIDAD (curvatura → ∞)")
        elif abs(K_ratio) > 2:
            print(f"   Curvatura moderadamente elevada en el centro")
        else:
            print(f"   Curvatura similar en centro y periferia")
    else:
        print(f"   Curvatura periférica ≈ 0 (plano)")
        if abs(K_center_avg) > 1e-6:
            print(f"   ️  Centro tiene curvatura no-nula en campo plano")
            print(f"   → Singularidad aislada")
    
    print()
    
    # Clasificar tipo de singularidad
    if K_center > 0.1:
        print(f"   Tipo: SINGULARIDAD ELÍPTICA (pico/valle)")
    elif K_center < -0.1:
        print(f"   Tipo: SINGULARIDAD HIPERBÓLICA (silla de montar)")
    else:
        print(f"   Tipo: SINGULARIDAD PARABÓLICA (cilindro/cono)")
    
    print()
    
    return {
        'K_center': float(K_center),
        'H_center': float(H_center),
        'K_center_avg': float(K_center_avg),
        'H_center_avg': float(H_center_avg),
        'K_peripheral': float(K_peripheral),
        'H_peripheral': float(H_peripheral),
        'K_ratio': float(K_center_avg / K_peripheral) if abs(K_peripheral) > 1e-10 else float('inf'),
    }

# ============================================================================
# D4: TOPOLOGÍA LOCAL (NÚMEROS DE BETTI)
# ============================================================================

def test_D4_topology_analysis(R, n):
    """
    D4: Análisis topológico - números de Betti (agujeros, componentes).
    Si el centro tiene topología distinta → opera en espacio diferente.
    """
    print("=" * 70)
    print("D4: TOPOLOGÍA LOCAL (números de Betti, agujeros)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # Calcular números de Betti usando componentes conectados
    # β0 = componentes conectados (islas)
    # β1 = agujeros 1D (túneles, anillos)
    
    from scipy import ndimage
    
    # β0: Componentes conectados (foreground)
    labeled_foreground, n_components_fg = ndimage.label(center_region)
    beta0_fg = n_components_fg
    
    # β0: Componentes conectados (background)
    labeled_background, n_components_bg = ndimage.label(1 - center_region)
    beta0_bg = n_components_bg
    
    # β1: Agujeros (componentes de background rodeados por foreground)
    # Usar fill_holes para detectar agujeros
    binary_center = center_region.astype(bool)
    filled = ndimage.binary_fill_holes(binary_center)
    holes = filled & ~binary_center
    labeled_holes, n_holes = ndimage.label(holes)
    beta1 = n_holes
    
    # Característica de Euler: χ = β0 - β1
    euler_characteristic = beta0_fg - beta1
    
    print(f"   Región central ({REGION_SIZE}x{REGION_SIZE}):")
    print(f"      β0 (componentes conectados) = {beta0_fg}")
    print(f"      β1 (agujeros 1D) = {beta1}")
    print(f"      Característica de Euler χ = β0 - β1 = {euler_characteristic}")
    print()
    
    # Comparar con región periférica
    corner_size = half // 2
    corner = R[:corner_size, :corner_size].astype(bool)
    labeled_c, n_c = ndimage.label(corner)
    filled_c = ndimage.binary_fill_holes(corner)
    holes_c = filled_c & ~corner
    _, n_holes_c = ndimage.label(holes_c)
    
    print(f"   Región periférica (esquina {corner_size}x{corner_size}):")
    print(f"      β0 = {n_c}")
    print(f"      β1 = {n_holes_c}")
    print(f"      χ = {n_c - n_holes_c}")
    print()
    
    # Interpretación
    if beta1 > 5:
        print(f"   ️  MÚLTIPLES AGUJEROS (β1={beta1})")
        print(f"   → Topología compleja con túneles/anillos")
        print(f"   → Posible estructura de 'esponja' o 'tejido'")
    elif beta1 > 0:
        print(f"   Algunos agujeros detectados (β1={beta1})")
    else:
        print(f"   Sin agujeros (β1=0)")
        print(f"   → Topología simple (islas sólidas)")
    
    print()
    
    # Relación β1/β0 (densidad de agujeros)
    if beta0_fg > 0:
        hole_density = beta1 / beta0_fg
        print(f"   Densidad de agujeros: β1/β0 = {hole_density:.4f}")
        
        if hole_density > 0.5:
            print(f"   ⚠️  ALTA densidad de agujeros")
            print(f"   → Estructura porosa/tamiz")
        elif hole_density > 0.1:
            print(f"   Densidad moderada de agujeros")
        else:
            print(f"   Baja densidad de agujeros")
    
    print()
    
    return {
        'beta0_center': int(beta0_fg),
        'beta1_center': int(beta1),
        'euler_center': int(euler_characteristic),
        'beta0_peripheral': int(n_c),
        'beta1_peripheral': int(n_holes_c),
        'hole_density': float(beta1 / beta0_fg) if beta0_fg > 0 else 0,
    }

# ============================================================================
# D5: ANÁLISIS ESPECTRAL LOCAL (FFT CENTRADO)
# ============================================================================

def test_D5_spectral_analysis(R, n):
    """
    D5: FFT 2D centrado en la cruz.
    Si el centro tiene frecuencias distintas → opera en régimen diferente.
    """
    print("=" * 70)
    print("D5: ANÁLISIS ESPECTRAL LOCAL (FFT centrado en cruz)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # FFT 2D
    fft_center = np.fft.fft2(center_region)
    fft_shifted = np.fft.fftshift(fft_center)
    magnitude = np.abs(fft_shifted)
    
    # Normalizar
    magnitude = magnitude / magnitude.max()
    
    # Analizar distribución de energía en bandas de frecuencia
    h, w = magnitude.shape
    cy_fft, cx_fft = h // 2, w // 2
    
    # Bandas: baja, media, alta frecuencia
    low_freq = magnitude[cy_fft-10:cy_fft+10, cx_fft-10:cx_fft+10].mean()
    mid_freq = magnitude[cy_fft-30:cy_fft-10, cx_fft-30:cx_fft+30].mean() + \
               magnitude[cy_fft+10:cy_fft+30, cx_fft-30:cx_fft+30].mean()
    high_freq = magnitude[:cy_fft-30, :].mean() + magnitude[cy_fft+30:, :].mean()
    
    total_energy = magnitude.sum()
    low_energy = magnitude[cy_fft-10:cy_fft+10, cx_fft-10:cx_fft+10].sum()
    mid_energy = magnitude[cy_fft-30:cy_fft-10, cx_fft-30:cx_fft+30].sum() + \
                 magnitude[cy_fft+10:cy_fft+30, cx_fft-30:cx_fft+30].sum()
    high_energy = total_energy - low_energy - mid_energy
    
    low_pct = low_energy / total_energy * 100
    mid_pct = mid_energy / total_energy * 100
    high_pct = high_energy / total_energy * 100
    
    print(f"   Distribución de energía espectral:")
    print(f"      Baja frecuencia (centro FFT): {low_pct:.2f}%")
    print(f"      Media frecuencia: {mid_pct:.2f}%")
    print(f"      Alta frecuencia: {high_pct:.2f}%")
    print()
    
    # Detectar picos espectrales
    # Umbral: magnitud > 0.1 del máximo
    threshold = 0.1
    peaks = magnitude > threshold
    n_peaks = peaks.sum()
    
    print(f"   Picos espectrales (magnitud > {threshold}):")
    print(f"      Número de picos: {n_peaks}")
    print()
    
    # Comparar con región periférica
    corner_size = half // 2
    corner = R[:corner_size, :corner_size]
    fft_corner = np.fft.fft2(corner)
    fft_corner_shifted = np.fft.fftshift(fft_corner)
    mag_corner = np.abs(fft_corner_shifted)
    mag_corner = mag_corner / mag_corner.max()
    
    h_c, w_c = mag_corner.shape
    cy_c, cx_c = h_c // 2, w_c // 2
    
    low_c = mag_corner[cy_c-5:cy_c+5, cx_c-5:cx_c+5].sum()
    total_c = mag_corner.sum()
    low_pct_c = low_c / total_c * 100
    
    print(f"   Comparación con periferia:")
    print(f"      Centro: {low_pct:.2f}% energía en baja frecuencia")
    print(f"      Periferia: {low_pct_c:.2f}% energía en baja frecuencia")
    print()
    
    # Interpretación
    if low_pct > 50:
        print(f"   ⚠️  Energía concentrada en BAJA frecuencia")
        print(f"   → Estructura suave, sin detalles finos")
        print(f"   → Posible 'sumidero' de información")
    elif high_pct > 50:
        print(f"   ⚠️  Energía concentrada en ALTA frecuencia")
        print(f"   → Estructura con muchos detalles finos")
        print(f"   → Posible 'fuente' de información")
    else:
        print(f"   Energía distribuida en múltiples bandas")
        print(f"   → Estructura multi-escala")
    
    print()
    
    return {
        'low_freq_pct': float(low_pct),
        'mid_freq_pct': float(mid_pct),
        'high_freq_pct': float(high_pct),
        'n_peaks': int(n_peaks),
        'low_freq_pct_peripheral': float(low_pct_c),
    }

# ============================================================================
# D6: TRANSFORMADA WAVELET 2D
# ============================================================================

def test_D6_wavelet_analysis(R, n):
    """
    D6: Wavelet 2D centrado en la cruz.
    Revela estructura a múltiples escalas con localización espacial.
    """
    print("=" * 70)
    print("D6: TRANSFORMADA WAVELET 2D (resolución multiescala centrada)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # Usar wavelet Haar (simple y efectiva)
    # Descomposición a 3 niveles
    def haar_wavelet_2d(matrix, levels=3):
        coefficients = []
        current = matrix.copy()
        
        for level in range(levels):
            h, w = current.shape
            # Asegurar dimensiones pares
            h = h - h % 2
            w = w - w % 2
            current = current[:h, :w]
            
            # Descomposición Haar
            # LL (aproximación), LH (horizontal), HL (vertical), HH (diagonal)
            LL = (current[0::2, 0::2] + current[0::2, 1::2] + 
                  current[1::2, 0::2] + current[1::2, 1::2]) / 2
            LH = (current[0::2, 0::2] - current[0::2, 1::2] + 
                  current[1::2, 0::2] - current[1::2, 1::2]) / 2
            HL = (current[0::2, 0::2] + current[0::2, 1::2] - 
                  current[1::2, 0::2] - current[1::2, 1::2]) / 2
            HH = (current[0::2, 0::2] - current[0::2, 1::2] - 
                  current[1::2, 0::2] + current[1::2, 1::2]) / 2
            
            coefficients.append({
                'level': level + 1,
                'LL': LL,
                'LH': LH,
                'HL': HL,
                'HH': HH,
                'LL_energy': (LL**2).sum(),
                'LH_energy': (LH**2).sum(),
                'HL_energy': (HL**2).sum(),
                'HH_energy': (HH**2).sum(),
            })
            
            current = LL
        
        return coefficients
    
    coeffs = haar_wavelet_2d(center_region, levels=3)
    
    print(f"   Descomposición wavelet (3 niveles):")
    print()
    
    total_energy = sum(c['LL_energy'] + c['LH_energy'] + c['HL_energy'] + c['HH_energy'] 
                       for c in coeffs)
    
    for c in coeffs:
        level = c['level']
        LL_pct = c['LL_energy'] / total_energy * 100
        LH_pct = c['LH_energy'] / total_energy * 100
        HL_pct = c['HL_energy'] / total_energy * 100
        HH_pct = c['HH_energy'] / total_energy * 100
        
        print(f"      Nivel {level}:")
        print(f"         LL (aproximación): {LL_pct:.2f}%")
        print(f"         LH (horizontal): {LH_pct:.2f}%")
        print(f"         HL (vertical): {HL_pct:.2f}%")
        print(f"         HH (diagonal): {HH_pct:.2f}%")
        print()
    
    # Interpretación
    # Si LH >> HL o HL >> LH → anisotropía direccional
    # Si HH es alto → estructura diagonal/cruz
    last_level = coeffs[-1]
    if last_level['HH_energy'] > last_level['LH_energy'] and last_level['HH_energy'] > last_level['HL_energy']:
        print(f"   ️  Energía DIAGONAL dominante en nivel fino")
        print(f"   → Confirma estructura de CRUZ (patrón X)")
    elif last_level['LH_energy'] > last_level['HL_energy']:
        print(f"   Anisotropía HORIZONTAL dominante")
        print(f"   → Estructura alargada horizontalmente")
    elif last_level['HL_energy'] > last_level['LH_energy']:
        print(f"   Anisotropía VERTICAL dominante")
        print(f"   → Estructura alargada verticalmente")
    else:
        print(f"   Energía balanceada en todas las direcciones")
    
    print()
    
    return {
        'wavelet_levels': len(coeffs),
        'level_details': [{
            'level': c['level'],
            'LL_pct': float(c['LL_energy'] / total_energy * 100),
            'LH_pct': float(c['LH_energy'] / total_energy * 100),
            'HL_pct': float(c['HL_energy'] / total_energy * 100),
            'HH_pct': float(c['HH_energy'] / total_energy * 100),
        } for c in coeffs],
    }

# ============================================================================
# D7: HOMOLOGÍA PERSISTENTE
# ============================================================================

def test_D7_persistent_homology(R, n):
    """
    D7: Homología persistente - detecta características topológicas
    que persisten a través de múltiples umbrales.
    """
    print("=" * 70)
    print("D7: HOMOLOGÍA PERSISTENTE (características topológicas estables)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # Calcular homología a múltiples umbrales
    thresholds = np.linspace(0.1, 0.9, 9)
    
    betti_0_history = []
    betti_1_history = []
    
    for thresh in thresholds:
        binary = (center_region > thresh).astype(bool)
        
        # β0: componentes conectados
        labeled, n_comp = ndimage.label(binary)
        betti_0_history.append(n_comp)
        
        # β1: agujeros
        filled = ndimage.binary_fill_holes(binary)
        holes = filled & ~binary
        _, n_holes = ndimage.label(holes)
        betti_1_history.append(n_holes)
    
    betti_0_history = np.array(betti_0_history)
    betti_1_history = np.array(betti_1_history)
    
    print(f"   Evolución de números de Betti con umbral:")
    print(f"   {'Umbral':<10} {'β0':<10} {'β1':<10}")
    print(f"   {'-'*30}")
    for i, thresh in enumerate(thresholds):
        print(f"   {thresh:<10.2f} {betti_0_history[i]:<10} {betti_1_history[i]:<10}")
    print()
    
    # Calcular persistencia (cuánto dura cada característica)
    # β0 persistente: máximo de β0
    beta0_max = betti_0_history.max()
    beta0_persistent = betti_0_history[betti_0_history == beta0_max].sum() / len(thresholds) * 100
    
    # β1 persistente
    beta1_max = betti_1_history.max()
    beta1_persistent = betti_1_history[betti_1_history == beta1_max].sum() / len(thresholds) * 100
    
    print(f"   Análisis de persistencia:")
    print(f"      β0 máximo: {beta0_max} (persiste en {beta0_persistent:.1f}% de umbrales)")
    print(f"      β1 máximo: {beta1_max} (persiste en {beta1_persistent:.1f}% de umbrales)")
    print()
    
    # Interpretación
    if beta1_max > 3 and beta1_persistent > 30:
        print(f"   ⚠️  AGUJEROS TOPOLOGICOS PERSISTENTES")
        print(f"   → Estructura con túneles/anillos estables")
        print(f"   → Topología no-trivial (como toro, esponja)")
    elif beta1_max > 0:
        print(f"   Agujeros detectados pero no muy persistentes")
    else:
        print(f"   Sin agujeros topológicos significativos")
    
    print()
    
    # Diagrama de persistencia simplificado
    print(f"   Diagrama de persistencia (simplificado):")
    print(f"      β0: {'█' * int(beta0_persistent/10)} ({beta0_persistent:.0f}%)")
    print(f"      β1: {'█' * int(beta1_persistent/10)} ({beta1_persistent:.0f}%)")
    print()
    
    return {
        'thresholds': thresholds.tolist(),
        'betti_0_history': betti_0_history.tolist(),
        'betti_1_history': betti_1_history.tolist(),
        'beta0_max': int(beta0_max),
        'beta1_max': int(beta1_max),
        'beta0_persistence_pct': float(beta0_persistent),
        'beta1_persistence_pct': float(beta1_persistent),
    }

# ============================================================================
# D8: TENSOR DE TENSIÓN/INFORMACIÓN
# ============================================================================

def test_D8_information_tensor(R, n):
    """
    D8: Tensor de información en el centro.
    Analogía con tensor de tensión en física: mide cómo fluye la información.
    """
    print("=" * 70)
    print("D8: TENSOR DE TENSIÓN/INFORMACIÓN EN EL CENTRO")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2].astype(np.float64)
    
    # Calcular gradiente
    gy, gx = np.gradient(center_region)
    
    # Tensor de información (analogía con tensor de tensión)
    # T_ij = ∂f/∂x_i * ∂f/∂x_j
    T_xx = gx * gx
    T_xy = gx * gy
    T_yx = gy * gx
    T_yy = gy * gy
    
    # Tensor en el centro exacto
    center_x, center_y = half, half
    T_center = np.array([
        [T_xx[center_x, center_y], T_xy[center_x, center_y]],
        [T_yx[center_x, center_y], T_yy[center_x, center_y]]
    ])
    
    # Autovalores del tensor (direcciones principales)
    eigenvalues, eigenvectors = np.linalg.eig(T_center)
    
    # Ordenar autovalores
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Invariantes del tensor
    trace = T_center[0, 0] + T_center[1, 1]  # T_xx + T_yy
    determinant = T_center[0, 0] * T_center[1, 1] - T_center[0, 1] * T_center[1, 0]
    
    # Presión (traza/2) y cizalla (diferencia de autovalores)
    pressure = trace / 2
    shear = (eigenvalues[0] - eigenvalues[1]) / 2
    
    print(f"   Tensor de información en el centro:")
    print(f"      T = [[{T_center[0,0]:.6f}, {T_center[0,1]:.6f}],")
    print(f"           [{T_center[1,0]:.6f}, {T_center[1,1]:.6f}]]")
    print()
    print(f"   Autovalores (direcciones principales):")
    print(f"      λ1 = {eigenvalues[0]:.6f} (dirección: {eigenvectors[:, 0]})")
    print(f"      λ2 = {eigenvalues[1]:.6f} (dirección: {eigenvectors[:, 1]})")
    print()
    print(f"   Invariantes:")
    print(f"      Traza (T_xx + T_yy) = {trace:.6f}")
    print(f"      Determinante = {determinant:.6f}")
    print()
    print(f"   Descomposición:")
    print(f"      Presión (isotrópica) = {pressure:.6f}")
    print(f"      Cizalla (anisotrópica) = {shear:.6f}")
    print()
    
    # Ratio cizalla/presión
    if abs(pressure) > 1e-10:
        shear_pressure_ratio = shear / pressure
        print(f"   Ratio cizalla/presión = {shear_pressure_ratio:.4f}")
        
        if abs(shear_pressure_ratio) > 1:
            print(f"   ️  CIZALLA DOMINANTE")
            print(f"   → Flujo de información ANISOTRÓPICO")
            print(f"   → Direcciones privilegiadas (como en cruz)")
        elif abs(shear_pressure_ratio) > 0.3:
            print(f"   Cizalla moderada")
            print(f"   → Algo de anisotropía direccional")
        else:
            print(f"   Presión dominante")
            print(f"   → Flujo de información ISOTRÓPICO (igual en todas direcciones)")
    else:
        print(f"   Presión ≈ 0 (campo plano)")
    
    print()
    
    # Clasificar tipo de flujo
    if eigenvalues[0] > 0 and eigenvalues[1] > 0:
        print(f"   Tipo: FUENTE (información fluye hacia afuera)")
    elif eigenvalues[0] < 0 and eigenvalues[1] < 0:
        print(f"   Tipo: SUMIDERO (información fluye hacia adentro)")
    elif eigenvalues[0] * eigenvalues[1] < 0:
        print(f"   Tipo: SILLA DE MONTAR (flujo mixto)")
    else:
        print(f"   Tipo: CASO DEGENERADO")
    
    print()
    
    return {
        'T_center': T_center.tolist(),
        'eigenvalues': eigenvalues.tolist(),
        'eigenvectors': eigenvectors.tolist(),
        'trace': float(trace),
        'determinant': float(determinant),
        'pressure': float(pressure),
        'shear': float(shear),
        'shear_pressure_ratio': float(shear / pressure) if abs(pressure) > 1e-10 else 0,
    }

# ============================================================================
# D9: PROYECCIÓN DIMENSIONAL
# ============================================================================

def test_D9_dimensional_projection(R, n):
    """
    D9: Analizar si el centro es proyección de objeto de mayor dimensión.
    Usar análisis de correlación no-lineal.
    """
    print("=" * 70)
    print("D9: PROYECCIÓN DIMENSIONAL (¿sombra de objeto de mayor dimensión?)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central
    x1, x2 = max(0, cx-half), min(n, cx+half)
    y1, y2 = max(0, cy-half), min(n, cy+half)
    center_region = R[x1:x2, y1:y2]
    
    # Método: Correlación integral (Grassberger-Procaccia)
    # Estima dimensión de correlación D2
    # Si D2 > dimensión embebida (2) → sugiere proyección de mayor dimensión
    
    def correlation_dimension(matrix, max_radius=None):
        # Aplanar matriz
        data = matrix.flatten()
        N = len(data)
        
        if max_radius is None:
            max_radius = data.max() * 0.5
        
        # Calcular matriz de distancias
        distances = pdist(data.reshape(-1, 1))
        
        # Contar pares dentro de radio r
        radii = np.linspace(0.01, max_radius, 20)
        counts = []
        
        for r in radii:
            count = np.sum(distances < r)
            counts.append(count)
        
        counts = np.array(counts)
        
        # Ajustar línea: log(C(r)) vs log(r)
        valid = counts > 0
        if valid.sum() > 5:
            log_r = np.log(radii[valid])
            log_C = np.log(counts[valid])
            
            coeffs = np.polyfit(log_r, log_C, 1)
            D2 = coeffs[0]
        else:
            D2 = 0
        
        return D2, radii, counts
    
    D2_center, radii_c, counts_c = correlation_dimension(center_region)
    
    # Comparar con región periférica
    corner_size = half // 2
    corner = R[:corner_size, :corner_size]
    D2_peripheral, radii_p, counts_p = correlation_dimension(corner)
    
    print(f"   Dimensión de correlación D2:")
    print(f"      Centro: D2 = {D2_center:.4f}")
    print(f"      Periferia: D2 = {D2_peripheral:.4f}")
    print()
    
    # Interpretación
    if D2_center > 2.5:
        print(f"   ⚠️  D2 > 2.5 en el centro")
        print(f"   → Sugiere PROYECCIÓN de objeto de dimensión ≥ 3")
        print(f"   → El centro sería 'sombra' de estructura de mayor dimensión")
    elif D2_center > 2.0:
        print(f"   D2 > 2.0 en el centro")
        print(f"   → Posible proyección de objeto 3D")
    elif D2_center < 1.5:
        print(f"   D2 < 1.5 en el centro")
        print(f"   → Estructura casi unidimensional (línea/curva)")
    else:
        print(f"   D2 ≈ 2.0 en el centro")
        print(f"   → Estructura bidimensional estándar")
    
    print()
    
    # Ratio centro/periferia
    if D2_peripheral > 0:
        ratio = D2_center / D2_peripheral
        print(f"   Ratio D2_centro / D2_periferia = {ratio:.4f}")
        
        if ratio > 1.3:
            print(f"   ⚠️  Centro tiene MAYOR dimensión de correlación")
            print(f"   → Centro es más 'complejo' que periferia")
        elif ratio < 0.7:
            print(f"   ⚠️  Centro tiene MENOR dimensión de correlación")
            print(f"   → Centro es más 'simple' que periferia (colapso dimensional)")
        else:
            print(f"   Dimensiones similares")
    
    print()
    
    return {
        'D2_center': float(D2_center),
        'D2_peripheral': float(D2_peripheral),
        'ratio': float(D2_center / D2_peripheral) if D2_peripheral > 0 else 0,
    }

# ============================================================================
# D10: ANÁLISIS DE FLUJO TOPOLÓGICO
# ============================================================================

def test_D10_topological_flow(R, n):
    """
    D10: Analizar tipo de singularidad de flujo en el centro.
    Clasificar: fuente, sumidero, silla, centro, foco.
    """
    print("=" * 70)
    print("D10: ANÁLISIS DE FLUJO TOPOLÓGICO (tipo de singularidad)")
    print("=" * 70)
    
    cx, cy = CROSS_CENTER
    half = REGION_SIZE // 2
    
    # Extraer región central más grande para mejor análisis
    x1, x2 = max(0, cx-half*2), min(n, cx+half*2)
    y1, y2 = max(0, cy-half*2), min(n, cy+half*2)
    center_region = R[x1:x2, y1:y2].astype(np.float64)
    
    # Calcular gradiente (campo vectorial)
    gy, gx = np.gradient(center_region)
    
    # Centro de la región
    cy_r, cx_r = center_region.shape[0] // 2, center_region.shape[1] // 2
    
    # Calcular divergencia y rotacional en el centro
    div_y, div_x = np.gradient(gy)
    rot_y, rot_x = np.gradient(gx)
    
    divergence = div_x + div_y
    curl = rot_y - rot_x
    
    div_center = divergence[cy_r, cx_r]
    curl_center = curl[cy_r, cx_r]
    
    # Jacobiano del campo vectorial en el centro
    # J = [[∂gx/∂x, ∂gx/∂y], [∂gy/∂x, ∂gy/∂y]]
    J = np.array([
        [rot_x[cy_r, cx_r], div_x[cy_r, cx_r]],
        [rot_y[cy_r, cx_r], div_y[cy_r, cx_r]]
    ])
    
    # Autovalores del Jacobiano
    eigenvalues_J = np.linalg.eigvals(J)
    
    # Clasificar tipo de punto crítico
    trace_J = np.trace(J)
    det_J = np.linalg.det(J)
    discriminant = trace_J**2 - 4 * det_J
    
    print(f"   Campo vectorial en el centro:")
    print(f"      Divergencia = {div_center:.6f}")
    print(f"      Rotacional (curl) = {curl_center:.6f}")
    print()
    print(f"   Jacobiano en el centro:")
    print(f"      J = [[{J[0,0]:.6f}, {J[0,1]:.6f}],")
    print(f"           [{J[1,0]:.6f}, {J[1,1]:.6f}]]")
    print()
    print(f"   Autovalores del Jacobiano:")
    print(f"      λ1 = {eigenvalues_J[0]}")
    print(f"      λ2 = {eigenvalues_J[1]}")
    print()
    print(f"   Invariantes:")
    print(f"      Traza = {trace_J:.6f}")
    print(f"      Determinante = {det_J:.6f}")
    print(f"      Discriminante = {discriminant:.6f}")
    print()
    
    # Clasificar tipo de singularidad
    if det_J < 0:
        print(f"   ⚠️  Tipo: SILLA DE MONTAR (saddle point)")
        print(f"   → Flujo entra por una dirección y sale por otra")
        print(f"   → Punto de decisión/bifurcación")
    elif det_J > 0 and trace_J < 0:
        if discriminant < 0:
            print(f"   Tipo: FOCO ESTABLE (spiral sink)")
            print(f"   → Flujo espiral hacia el centro")
            print(f"   → Atractor con rotación")
        else:
            print(f"   Tipo: NODO ESTABLE")
            print(f"   → Flujo converge directamente al centro")
            print(f"   → Sumidero puro")
    elif det_J > 0 and trace_J > 0:
        if discriminant < 0:
            print(f"   Tipo: FOCO INESTABLE (spiral source)")
            print(f"   → Flujo espiral desde el centro")
            print(f"   → Fuente con rotación")
        else:
            print(f"   Tipo: NODO INESTABLE")
            print(f"   → Flujo diverge desde el centro")
            print(f"   → Fuente pura")
    elif det_J > 0 and trace_J == 0:
        print(f"   Tipo: CENTRO")
        print(f"   → Flujo circular alrededor del centro")
        print(f"   → Vórtice sin convergencia/divergencia")
    else:
        print(f"   Tipo: CASO DEGENERADO")
    
    print()
    
    # Interpretación física
    if abs(div_center) > abs(curl_center):
        print(f"   Dominio: DIVERGENCIA (expansión/contracción)")
        if div_center < 0:
            print(f"   → SUMIDERO (información converge al centro)")
        else:
            print(f"   → FUENTE (información emana del centro)")
    else:
        print(f"   Dominio: ROTACIONAL (vórtice)")
        print(f"   → Información circula alrededor del centro")
    
    print()
    
    return {
        'divergence': float(div_center),
        'curl': float(curl_center),
        'Jacobian': J.tolist(),
        'eigenvalues': [str(e) for e in eigenvalues_J],
        'trace': float(trace_J),
        'determinant': float(det_J),
        'discriminant': float(discriminant),
    }

# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    t_start = time.time()
    
    # Fase 0: Cargar datos
    R, n = load_and_prepare_recurrence_matrix()
    
    # Ejecutar tests D1-D10
    results = {}
    
    results['D1_fractal_dimension'] = test_D1_local_fractal_dimension(R, n)
    results['D2_multifractal'] = test_D2_multifractal_analysis(R, n)
    results['D3_curvature'] = test_D3_curvature_analysis(R, n)
    results['D4_topology'] = test_D4_topology_analysis(R, n)
    results['D5_spectral'] = test_D5_spectral_analysis(R, n)
    results['D6_wavelet'] = test_D6_wavelet_analysis(R, n)
    results['D7_homology'] = test_D7_persistent_homology(R, n)
    results['D8_information_tensor'] = test_D8_information_tensor(R, n)
    results['D9_dimensional_projection'] = test_D9_dimensional_projection(R, n)
    results['D10_topological_flow'] = test_D10_topological_flow(R, n)
    
    # Guardar resultados
    output_file = os.path.join(OUTPUT_DIR, 'TESTS_D1_D10_resultados.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("=" * 70)
    print("RESUMEN EJECUTIVO")
    print("=" * 70)
    print()
    
    # Resumen de hallazgos clave
    print("HALLAZGOS CLAVE:")
    print()
    
    # D1: Dimensión fractal
    D1 = results['D1_fractal_dimension']
    print(f"D1: Dimensión fractal local")
    print(f"    Centro D={D1['D_center']:.4f} vs Periferia D={D1['D_peripheral']:.4f} (Δ={D1['delta_D']:.4f})")
    print()
    
    # D3: Curvatura
    D3 = results['D3_curvature']
    print(f"D3: Curvatura Gaussiana")
    print(f"    Centro K={D3['K_center']:.6f} vs Periferia K={D3['K_peripheral']:.6f} (ratio={D3['K_ratio']:.2f}x)")
    print()
    
    # D4: Topología
    D4 = results['D4_topology']
    print(f"D4: Topología (agujeros)")
    print(f"    Centro β1={D4['beta1_center']} agujeros, densidad={D4['hole_density']:.4f}")
    print()
    
    # D8: Tensor
    D8 = results['D8_information_tensor']
    print(f"D8: Tensor de información")
    print(f"    Presión={D8['pressure']:.6f}, Cizalla={D8['shear']:.6f} (ratio={D8['shear_pressure_ratio']:.4f})")
    print()
    
    # D9: Proyección dimensional
    D9 = results['D9_dimensional_projection']
    print(f"D9: Dimensión de correlación")
    print(f"    Centro D2={D9['D2_center']:.4f} vs Periferia D2={D9['D2_peripheral']:.4f} (ratio={D9['ratio']:.4f})")
    print()
    
    # D10: Flujo topológico
    D10 = results['D10_topological_flow']
    print(f"D10: Tipo de singularidad")
    print(f"     Divergencia={D10['divergence']:.6f}, Det(J)={D10['determinant']:.6f}")
    print()
    
    print(f"Resultados guardados en: {output_file}")
    print(f"Tiempo total: {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    main()
