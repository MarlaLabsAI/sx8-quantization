"""
TEST A2: BÚSQUEDA DE INFORMACIÓN DIRECCIONAL EN MATRIZ CONTINUA
=================================================================
El Test A no encontró matriz antisimétrica porque usamos |profile[i] - profile[j]|
(que es simétrica por definición).

Pero la matriz continua ORIGINAL (sin valor absoluto) podría tener
información direccional:

Si profile[i] > profile[j] → dirección positiva
Si profile[i] < profile[j] → dirección negativa

Esto codificaría información ASIMÉTRICA que complementa la simétrica.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import torch
import scipy.ndimage as ndimage
import json
import time
import os

# Configuración
OUTPUT_DIR = r"C:\turin\resultados\analisis_chip\sindonologia_16_tests"
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

def test_A2_directional_information():
    """
    Test A2: Buscar información direccional en matriz continua
    
    Matriz direccional: D(i,j) = profile[i] - profile[j]
    - Si D(i,j) > 0: i es más brillante que j
    - Si D(i,j) < 0: i es más oscuro que j
    - D(i,j) = -D(j,i) (antisimétrica por definición)
    
    Analizamos si esta información direccional tiene estructura
    o es ruido aleatorio.
    """
    print(f"\n{'='*70}")
    print(f"TEST A2: Información Direccional en Matriz Continua")
    print(f"{'='*70}")
    
    print("   Generando matriz direccional D(i,j) = profile[i] - profile[j]...")
    t0 = time.time()
    
    img = cv2.imread(IMAGE3_PATH, cv2.IMREAD_GRAYSCALE)
    h_img, w_img = img.shape
    profile = img[:, w_img//2].astype(np.float32)
    profile_smooth = ndimage.gaussian_filter1d(profile, sigma=15)
    
    n = len(profile_smooth)
    
    # Matriz direccional (antisimétrica por definición)
    if check_vram(500):
        profile_tensor = torch.from_numpy(profile_smooth).to(DEVICE)
        # D(i,j) = profile[i] - profile[j]
        D = profile_tensor.unsqueeze(0) - profile_tensor.unsqueeze(1)
        D = D.cpu().numpy()
    else:
        D = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                D[i, j] = profile_smooth[i] - profile_smooth[j]
    
    print(f"   Matriz direccional: {n}x{n}, Tiempo: {time.time()-t0:.1f}s")
    
    # Verificar antisimetría: D(i,j) = -D(j,i)
    print("\n   Verificando antisimetría...")
    antisymmetry_error = float(np.abs(D + D.T).mean())
    is_antisymmetric = antisymmetry_error < 1e-6
    print(f"   Error de antisimetría: {antisymmetry_error:.10f}")
    print(f"   Es antisimétrica: {is_antisymmetric}")
    
    # Analizar magnitud de información direccional
    print("\n   Analizando magnitud de información direccional...")
    D_magnitude = float(np.abs(D).mean())
    D_max = float(np.abs(D).max())
    D_std = float(D.std())
    
    print(f"   Magnitud media |D|: {D_magnitude:.4f}")
    print(f"   Magnitud máxima |D|: {D_max:.4f}")
    print(f"   Desviación estándar: {D_std:.4f}")
    
    # Comparar con matriz simétrica (valores absolutos)
    print("\n   Comparando con matriz simétrica...")
    S = np.abs(D)  # Matriz simétrica
    S_magnitude = float(S.mean())
    
    ratio_directional = D_magnitude / S_magnitude if S_magnitude > 0 else 0
    print(f"   Magnitud simétrica (|D|): {S_magnitude:.4f}")
    print(f"   Ratio direccional/simétrica: {ratio_directional:.4f}")
    
    # Analizar estructura de D
    # ¿D tiene patrones o es ruido aleatorio?
    print("\n   Analizando estructura de D...")
    
    # Autocorrelación de D
    if check_vram(200):
        D_tensor = to_gpu_tensor(D)
        D_norm = D_tensor - D_tensor.mean()
        from torch.fft import fft2, ifft2, fftshift
        f_D = fft2(D_norm)
        f_D_conj = torch.conj(f_D)
        autocorr_D = fftshift(ifft2(f_D * f_D_conj).real)
        autocorr_D = autocorr_D / autocorr_D.max()
        autocorr_D = autocorr_D.cpu().numpy()
    else:
        from scipy.fftpack import fft2, fftshift
        D_norm = D - D.mean()
        f_D = fft2(D_norm)
        f_D_conj = np.conj(f_D)
        autocorr_D = np.fft.ifft2(f_D * f_D_conj).real
        autocorr_D = np.fft.fftshift(autocorr_D)
        autocorr_D = autocorr_D / autocorr_D.max()
    
    # Pico central de autocorrelación
    center_y, center_x = autocorr_D.shape[0] // 2, autocorr_D.shape[1] // 2
    central_peak = float(autocorr_D[center_y, center_x])
    
    # Energía en frecuencias bajas vs altas
    h, w = autocorr_D.shape
    low_freq_energy = float(np.sum(autocorr_D[h//2-50:h//2+50, w//2-50:w//2+50]**2))
    total_energy = float(np.sum(autocorr_D**2))
    low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
    
    print(f"   Pico central autocorrelación: {central_peak:.4f}")
    print(f"   Energía baja frecuencia: {low_freq_ratio:.4f}")
    
    # ¿D tiene estructura significativa?
    has_structure = central_peak > 0.3 and low_freq_ratio > 0.2
    
    # Analizar distribución de signos
    print("\n   Analizando distribución de signos...")
    n_positive = int(np.sum(D > 0))
    n_negative = int(np.sum(D < 0))
    n_zero = int(np.sum(D == 0))
    total = n * n
    
    ratio_positive = n_positive / total
    ratio_negative = n_negative / total
    ratio_zero = n_zero / total
    
    print(f"   Elementos positivos: {n_positive} ({ratio_positive*100:.2f}%)")
    print(f"   Elementos negativos: {n_negative} ({ratio_negative*100:.2f}%)")
    print(f"   Elementos cero: {n_zero} ({ratio_zero*100:.2f}%)")
    
    # Si hay estructura direccional, debería haber asimetría en la distribución
    # (más positivos que negativos o viceversa en ciertas regiones)
    
    # Analizar regiones: ¿hay direcciones preferentes?
    print("\n   Analizando direcciones preferentes...")
    
    # Dividir D en 4 cuadrantes
    quadrant_size = n // 2
    Q1 = D[:quadrant_size, :quadrant_size]
    Q2 = D[:quadrant_size, quadrant_size:]
    Q3 = D[quadrant_size:, :quadrant_size]
    Q4 = D[quadrant_size:, quadrant_size:]
    
    # Media de cada cuadrante
    Q1_mean = float(Q1.mean())
    Q2_mean = float(Q2.mean())
    Q3_mean = float(Q3.mean())
    Q4_mean = float(Q4.mean())
    
    print(f"   Media Q1 (superior-izq): {Q1_mean:.4f}")
    print(f"   Media Q2 (superior-der): {Q2_mean:.4f}")
    print(f"   Media Q3 (inferior-izq): {Q3_mean:.4f}")
    print(f"   Media Q4 (inferior-der): {Q4_mean:.4f}")
    
    # Si hay dirección preferente, algunos cuadrantes tendrán media positiva y otros negativa
    n_positive_quadrants = sum(1 for q in [Q1_mean, Q2_mean, Q3_mean, Q4_mean] if q > 0)
    n_negative_quadrants = sum(1 for q in [Q1_mean, Q2_mean, Q3_mean, Q4_mean] if q < 0)
    
    has_directional_preference = (n_positive_quadrants > 0 and n_negative_quadrants > 0)
    
    print(f"   Cuadrantes con media positiva: {n_positive_quadrants}")
    print(f"   Cuadrantes con media negativa: {n_negative_quadrants}")
    print(f"   Tiene dirección preferente: {has_directional_preference}")
    
    # Conclusión
    print(f"\n{'='*70}")
    print(f"CONCLUSIÓN TEST A2")
    print(f"{'='*70}")
    
    is_significant = has_structure and has_directional_preference
    
    if is_significant:
        print(f"   [HALLAZGO] Existe información direccional estructurada")
        print(f"   [IMPLICACIÓN] El sistema codifica información ASIMÉTRICA")
        print(f"   [ADAPTACIÓN FGN v2] Usar doble matriz: simétrica + direccional")
    else:
        print(f"   [SIN HALLAZGO] No hay información direccional significativa")
        print(f"   [IMPLICACIÓN] El sistema solo codifica información SIMÉTRICA")
        print(f"   [ADAPTACIÓN FGN v2] Usar solo matriz simétrica")
    
    print()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'antisymmetry_error': antisymmetry_error,
        'is_antisymmetric': is_antisymmetric,
        'D_magnitude': D_magnitude,
        'D_max': D_max,
        'D_std': D_std,
        'S_magnitude': S_magnitude,
        'ratio_directional': ratio_directional,
        'central_peak_autocorr': central_peak,
        'low_freq_ratio': low_freq_ratio,
        'has_structure': has_structure,
        'ratio_positive': ratio_positive,
        'ratio_negative': ratio_negative,
        'ratio_zero': ratio_zero,
        'quadrant_means': {
            'Q1': Q1_mean,
            'Q2': Q2_mean,
            'Q3': Q3_mean,
            'Q4': Q4_mean
        },
        'has_directional_preference': has_directional_preference,
        'is_significant': is_significant
    }

def main():
    t_start = time.time()
    
    print("\n" + "="*70)
    print("TEST A2: INFORMACIÓN DIRECCIONAL")
    print("="*70)
    
    results = test_A2_directional_information()
    
    # Guardar resultados
    output_file = os.path.join(OUTPUT_DIR, 'test_A2_directional_information.json')
    
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
    
    print(f"Archivo: {output_file}")
    print(f"Tiempo total: {time.time()-t_start:.1f}s")

if __name__ == '__main__':
    main()
