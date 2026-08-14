"""
NIVEL 3: ANÁLISIS DEL PROCESO DE FORMACIÓN
============================================
Enfocado en:
1. El "entorno" del ASIC en la matriz de recurrencia
2. Residuo después de filtrar la estructura ASIC
3. Patrones de correlación no-local
4. Firma espectral del proceso
5. Simetría del proceso (no del objeto)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2 as torch_fft2, fftshift as torch_fftshift
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import entropy as shannon_entropy
import json
import time
import os

# Configuración
OUTPUT_DIR = r"C:\turin\resultados\analisis_chip\sindonologia_16_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE3_PATH = r"C:\turin\Image June 06, 2026 - 12_22PM(2).jpeg"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    VRAM_TOTAL = props.total_memory / (1024**3)
    print(f"GPU: {props.name} | VRAM: {VRAM_TOTAL:.2f} GB")

def check_vram(required_mb):
    if not torch.cuda.is_available():
        return True
    allocated = torch.cuda.memory_allocated() / (1024**2)
    free = VRAM_TOTAL * 1024 - allocated
    return free >= required_mb * 1.5

def to_gpu_tensor(img_np, dtype=torch.float32):
    return torch.from_numpy(img_np).to(DEVICE).to(dtype)

# ============================================================================
# GENERAR MATRIZ DE RECURRENCIA
# ============================================================================

def generate_recurrence_matrix(img_path):
    print("Generando matriz de recurrencia...")
    t0 = time.time()
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    profile = img[:, w//2].astype(np.float32)
    profile_smooth = ndimage.gaussian_filter1d(profile, sigma=15)
    
    threshold = 10.0
    n = len(profile_smooth)
    
    if check_vram(500):
        profile_tensor = torch.from_numpy(profile_smooth).to(DEVICE)
        diff_matrix = torch.abs(profile_tensor.unsqueeze(0) - profile_tensor.unsqueeze(1))
        R = (diff_matrix < threshold).float()
        R = R.cpu().numpy()
    else:
        R = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if abs(profile_smooth[i] - profile_smooth[j]) < threshold:
                    R[i, j] = 1.0
    
    print(f"   Matriz: {n}x{n}, Densidad: {R.mean():.4f}, Tiempo: {time.time()-t0:.1f}s")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return R

# ============================================================================
# NIVEL 3: PRUEBAS DEL PROCESO DE FORMACIÓN
# ============================================================================

def test_process_1_residual_analysis(R, name):
    """Test P1: Análisis del residuo después de filtrar el ASIC"""
    print(f"   Test P1: Análisis de Residuo ({name})")
    
    # El ASIC está en el cuadrante superior izquierdo (primeros ~540px)
    # Analizar qué hay fuera del ASIC
    h, w = R.shape
    asic_size = min(h, w) // 2
    
    # Región ASIC (cuadrante superior izquierdo)
    asic_region = R[:asic_size, :asic_size]
    
    # Región "entorno" (resto de la matriz)
    environment_regions = [
        R[:asic_size, asic_size:],  # Superior derecho
        R[asic_size:, :asic_size],  # Inferior izquierdo
        R[asic_size:, asic_size:]   # Inferior derecho
    ]
    
    # Estadísticas
    asic_density = float(asic_region.mean())
    env_densities = [float(region.mean()) for region in environment_regions]
    env_mean_density = float(np.mean(env_densities))
    
    # Ratio ASIC vs entorno
    density_ratio = asic_density / env_mean_density if env_mean_density > 0 else 0
    
    # Simetría entre regiones del entorno (varianza de densidades)
    env_variance = float(np.var(env_densities))
    env_symmetry = 1 - (env_variance / (env_mean_density + 1e-10)) if env_mean_density > 0 else 0
    
    return {
        'asic_density': asic_density,
        'environment_density': env_mean_density,
        'density_ratio': density_ratio,
        'environment_symmetry': env_symmetry,
        'asic_dominance': asic_density > env_mean_density * 2
    }

def test_process_2_nonlocal_correlations(R, name):
    """Test P2: Patrones de correlación no-local"""
    print(f"   Test P2: Correlaciones No-Locales ({name})")
    
    h, w = R.shape
    
    # Dividir en 4 cuadrantes
    quadrant_size = h // 2
    Q1 = R[:quadrant_size, :quadrant_size]
    Q2 = R[:quadrant_size, quadrant_size:]
    Q3 = R[quadrant_size:, :quadrant_size]
    Q4 = R[quadrant_size:, quadrant_size:]
    
    # Calcular correlación entre cuadrantes
    def flatten_quadrant(Q):
        return Q.flatten()
    
    Q1_flat = flatten_quadrant(Q1)
    Q2_flat = flatten_quadrant(Q2)
    Q3_flat = flatten_quadrant(Q3)
    Q4_flat = flatten_quadrant(Q4)
    
    # Correlaciones cruzadas
    corr_12 = float(np.corrcoef(Q1_flat, Q2_flat)[0, 1])
    corr_13 = float(np.corrcoef(Q1_flat, Q3_flat)[0, 1])
    corr_14 = float(np.corrcoef(Q1_flat, Q4_flat)[0, 1])
    corr_23 = float(np.corrcoef(Q2_flat, Q3_flat)[0, 1])
    corr_24 = float(np.corrcoef(Q2_flat, Q4_flat)[0, 1])
    corr_34 = float(np.corrcoef(Q3_flat, Q4_flat)[0, 1])
    
    # Información mutua aproximada (correlación al cuadrado)
    mi_14 = corr_14 ** 2
    
    return {
        'correlations': {
            'Q1-Q2': corr_12,
            'Q1-Q3': corr_13,
            'Q1-Q4': corr_14,
            'Q2-Q3': corr_23,
            'Q2-Q4': corr_24,
            'Q3-Q4': corr_34
        },
        'max_correlation': max([corr_12, corr_13, corr_14, corr_23, corr_24, corr_34]),
        'min_correlation': min([corr_12, corr_13, corr_14, corr_23, corr_24, corr_34]),
        'diagonal_correlation': corr_14,  # Q1-Q4 son diagonales opuestos
        'has_nonlocal_correlation': corr_14 > 0.3
    }

def test_process_3_spectral_signature(R, name):
    """Test P3: Firma espectral del proceso"""
    print(f"   Test P3: Firma Espectral ({name})")
    
    if check_vram(200):
        R_tensor = to_gpu_tensor(R)
        f_transform = torch_fft2(R_tensor)
        f_shift = torch_fftshift(f_transform)
        magnitude = torch.abs(f_shift).cpu().numpy()
    else:
        from scipy.fftpack import fft2, fftshift
        f_transform = fft2(R.astype(float))
        f_shift = fftshift(f_transform)
        magnitude = np.abs(f_shift)
    
    # Analizar distribución de energía en bandas
    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2
    
    # Bandas concéntricas
    bands = []
    for radius_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
        radius = int(min(h, w) * radius_ratio)
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        band_energy = float(magnitude[mask].sum())
        bands.append(band_energy)
    
    # Normalizar
    total_energy = bands[-1] if bands[-1] > 0 else 1
    bands_normalized = [b / total_energy for b in bands]
    
    # Detectar picos en alta frecuencia (firma del proceso)
    high_freq_region = magnitude[h//4:3*h//4, w//4:3*w//4]
    mean_hf = float(high_freq_region.mean())
    std_hf = float(high_freq_region.std())
    
    # Picos significativos
    threshold_hf = mean_hf + 3 * std_hf
    n_peaks_hf = int(np.sum(high_freq_region > threshold_hf))
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'energy_bands': bands_normalized,
        'low_freq_concentration': bands_normalized[0],
        'high_freq_peaks': n_peaks_hf,
        'has_spectral_signature': n_peaks_hf > 100
    }

def test_process_4_process_symmetry(R, name):
    """Test P4: Simetría del proceso (no del objeto)"""
    print(f"   Test P4: Simetría del Proceso ({name})")
    
    # La matriz de recurrencia es simétrica por definición: R(i,j) = R(j,i)
    # Pero analizamos la simetría de la DENSIDAD en diferentes regiones
    
    h, w = R.shape
    
    # Simetría diagonal (debería ser perfecta)
    diagonal_symmetry = float(np.corrcoef(R.flatten(), R.T.flatten())[0, 1])
    
    # Simetría horizontal (reflejo vertical)
    horizontal_symmetry = float(np.corrcoef(R.flatten(), np.flipud(R).flatten())[0, 1])
    
    # Simetría vertical (reflejo horizontal)
    vertical_symmetry = float(np.corrcoef(R.flatten(), np.fliplr(R).flatten())[0, 1])
    
    # Simetría rotacional 180°
    rotation_180 = float(np.corrcoef(R.flatten(), np.flipud(np.fliplr(R)).flatten())[0, 1])
    
    return {
        'diagonal_symmetry': diagonal_symmetry,
        'horizontal_symmetry': horizontal_symmetry,
        'vertical_symmetry': vertical_symmetry,
        'rotation_180_symmetry': rotation_180,
        'is_diagonal_symmetric': diagonal_symmetry > 0.99,
        'process_symmetry_type': 'diagonal' if diagonal_symmetry > 0.99 else 'other'
    }

def test_process_5_grid_structure_detection(R, name):
    """Test P5: Detección de estructura de grid en el ASIC"""
    print(f"   Test P5: Detección de Grid ({name})")
    
    # Analizar solo el cuadrante superior izquierdo (donde está el ASIC)
    h, w = R.shape
    asic_region = R[:h//2, :w//2]
    
    # Proyecciones horizontal y vertical
    horizontal_projection = asic_region.sum(axis=1)
    vertical_projection = asic_region.sum(axis=0)
    
    # Detectar picos en las proyecciones (líneas del grid)
    from scipy.signal import find_peaks
    
    # Suavizar proyecciones
    h_smooth = ndimage.gaussian_filter1d(horizontal_projection, sigma=5)
    v_smooth = ndimage.gaussian_filter1d(vertical_projection, sigma=5)
    
    # Encontrar picos
    h_peaks, _ = find_peaks(h_smooth, distance=20, prominence=h_smooth.std() * 0.5)
    v_peaks, _ = find_peaks(v_smooth, distance=20, prominence=v_smooth.std() * 0.5)
    
    # Analizar espaciados entre picos
    h_spacings = np.diff(h_peaks) if len(h_peaks) > 1 else []
    v_spacings = np.diff(v_peaks) if len(v_peaks) > 1 else []
    
    # Verificar si los espaciados son regulares (grid)
    h_spacing_std = float(np.std(h_spacings)) if len(h_spacings) > 0 else 0
    v_spacing_std = float(np.std(v_spacings)) if len(v_spacings) > 0 else 0
    h_spacing_mean = float(np.mean(h_spacings)) if len(h_spacings) > 0 else 0
    v_spacing_mean = float(np.mean(v_spacings)) if len(v_spacings) > 0 else 0
    
    # Regularidad del grid (menor std = más regular)
    h_regularidad = 1 - (h_spacing_std / h_spacing_mean) if h_spacing_mean > 0 else 0
    v_regularidad = 1 - (v_spacing_std / v_spacing_mean) if v_spacing_mean > 0 else 0
    
    return {
        'n_horizontal_lines': len(h_peaks),
        'n_vertical_lines': len(v_peaks),
        'horizontal_spacing_mean': h_spacing_mean,
        'vertical_spacing_mean': v_spacing_mean,
        'horizontal_regularidad': h_regularidad,
        'vertical_regularidad': v_regularidad,
        'has_regular_grid': h_regularidad > 0.7 and v_regularidad > 0.7,
        'grid_size_estimate': f"{len(h_peaks)}x{len(v_peaks)}"
    }

def test_process_6_cross_central_analysis(R, name):
    """Test P6: Análisis de la cruz central"""
    print(f"   Test P6: Análisis de Cruz Central ({name})")
    
    h, w = R.shape
    
    # La cruz está en el centro del ASIC (aproximadamente en 1/4 de la matriz)
    cross_center_y = h // 4
    cross_center_x = w // 4
    
    # Extraer región alrededor de la cruz
    region_size = 50
    y1 = max(0, cross_center_y - region_size)
    y2 = min(h, cross_center_y + region_size)
    x1 = max(0, cross_center_x - region_size)
    x2 = min(w, cross_center_x + region_size)
    
    cross_region = R[y1:y2, x1:x2]
    
    # Analizar densidad radial desde el centro
    cy, cx = region_size, region_size
    distances = []
    densities = []
    
    for r in range(5, region_size, 5):
        y, x = np.ogrid[:cross_region.shape[0], :cross_region.shape[1]]
        mask = (x - cx)**2 + (y - cy)**2 <= r**2
        ring_mask = mask & ~((x - cx)**2 + (y - cy)**2 <= (r-5)**2)
        if ring_mask.sum() > 0:
            distances.append(r)
            densities.append(float(cross_region[ring_mask].mean()))
    
    # Ajustar decaimiento exponencial
    if len(distances) > 2:
        log_densities = np.log(np.array(densities) + 1e-10)
        coeffs = np.polyfit(distances, log_densities, 1)
        decay_rate = -coeffs[0]
    else:
        decay_rate = 0
    
    # Densidad en el centro vs periferia
    center_density = float(cross_region[cy-10:cy+10, cx-10:cx+10].mean())
    peripheral_density = float(cross_region.mean())
    
    return {
        'cross_center_position': [int(cross_center_y), int(cross_center_x)],
        'center_density': center_density,
        'peripheral_density': peripheral_density,
        'density_ratio': center_density / peripheral_density if peripheral_density > 0 else 0,
        'decay_rate': decay_rate,
        'has_central_peak': center_density > peripheral_density * 1.5
    }

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    t_start = time.time()
    
    print("\n" + "="*70)
    print("NIVEL 3: ANÁLISIS DEL PROCESO DE FORMACIÓN")
    print("="*70)
    
    # Generar matriz de recurrencia
    R = generate_recurrence_matrix(IMAGE3_PATH)
    
    results = {
        'analysis_type': 'process_formation',
        'source_image': 'imagen3_sepia',
        'matrix_dimensions': [R.shape[0], R.shape[1]],
        'matrix_density': float(R.mean()),
        'tests': {}
    }
    
    print("\nAnálisis del Proceso de Formación:")
    results['tests']['P1_residual_analysis'] = test_process_1_residual_analysis(R, 'recurrence_matrix')
    results['tests']['P2_nonlocal_correlations'] = test_process_2_nonlocal_correlations(R, 'recurrence_matrix')
    results['tests']['P3_spectral_signature'] = test_process_3_spectral_signature(R, 'recurrence_matrix')
    results['tests']['P4_process_symmetry'] = test_process_4_process_symmetry(R, 'recurrence_matrix')
    results['tests']['P5_grid_structure'] = test_process_5_grid_structure_detection(R, 'recurrence_matrix')
    results['tests']['P6_cross_central'] = test_process_6_cross_central_analysis(R, 'recurrence_matrix')
    
    # Guardar resultados
    output_file = os.path.join(OUTPUT_DIR, 'sindonologia_nivel3_process_formation.json')
    
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    results_serializable = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n{'='*70}")
    print("RESULTADOS NIVEL 3 GUARDADOS")
    print(f"{'='*70}")
    print(f"Archivo: {output_file}")
    print(f"Tiempo total: {time.time()-t_start:.1f}s")
    print()
    
    # Resumen
    print("RESUMEN NIVEL 3 (Proceso de Formación):")
    tests = results['tests']
    print(f"   P1 - ASIC vs Entorno: ratio={tests['P1_residual_analysis']['density_ratio']:.2f}x")
    print(f"   P2 - Correlación no-local Q1-Q4: {tests['P2_nonlocal_correlations']['diagonal_correlation']:.3f}")
    print(f"   P3 - Picos alta frecuencia: {tests['P3_spectral_signature']['high_freq_peaks']}")
    print(f"   P4 - Simetría diagonal: {tests['P4_process_symmetry']['diagonal_symmetry']:.3f}")
    print(f"   P5 - Grid detectado: {tests['P5_grid_structure']['grid_size_estimate']}")
    print(f"   P6 - Cruz central peak: {tests['P6_cross_central']['has_central_peak']}")

if __name__ == '__main__':
    main()
