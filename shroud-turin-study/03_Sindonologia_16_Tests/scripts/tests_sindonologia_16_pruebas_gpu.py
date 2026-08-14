"""
16 PRUEBAS SINDONOLÓGICAS DIGITALES - GPU OPTIMIZED
====================================================
Usa PyTorch CUDA para cálculos pesados con verificación de VRAM
Falls back to CPU multiprocessing para operaciones iterativas
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2 as torch_fft2, ifft2 as torch_ifft2, fftshift as torch_fftshift
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import entropy as shannon_entropy
from scipy.fftpack import fft2 as scipy_fft2, fftshift as scipy_fftshift
import json
import time
import os
from multiprocessing import Pool, cpu_count

# Configuración
OUTPUT_DIR = r"C:\turin\resultados\analisis_chip\sindonologia_16_tests"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGES = {
    'imagen1_negativo': r"C:\turin\Image June 06, 2026 - 12_22PM.jpeg",
    'imagen2_dos_caras': r"C:\turin\Image June 06, 2026 - 12_22PM(1).jpeg",
    'imagen3_sepia': r"C:\turin\Image June 06, 2026 - 12_22PM(2).jpeg"
}

# Configurar GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    VRAM_TOTAL = props.total_memory / (1024**3)  # GB
    print(f"GPU: {props.name} | VRAM: {VRAM_TOTAL:.2f} GB")
else:
    VRAM_TOTAL = 0
    print("CPU mode (no GPU disponible)")

def check_vram(required_mb):
    """Verificar si hay suficiente VRAM"""
    if not torch.cuda.is_available():
        return True
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    free = VRAM_TOTAL * 1024 - allocated
    if free < required_mb * 1.5:  # 50% margen de seguridad
        print(f"   [WARNING] VRAM insuficiente ({free:.0f}MB libre, {required_mb}MB necesario). Usando CPU.")
        return False
    return True

def to_gpu_tensor(img_np, dtype=torch.float32):
    """Convertir numpy array a tensor GPU"""
    return torch.from_numpy(img_np).to(DEVICE).to(dtype)

# ============================================================================
# BLOQUE A: PROYECCIÓN Y FRECUENCIAS
# ============================================================================

def test_1_z_relief(img, name):
    """Test 1: Relieve 3D de Intensidad-Distancia (Z-Relief) - GPU"""
    print(f"   Test 1: Z-Relief ({name})")
    
    if check_vram(50):
        img_tensor = to_gpu_tensor(img)
        # Gaussiano via convolución
        kernel_size = 5
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=DEVICE) / (kernel_size**2)
        img_4d = img_tensor.unsqueeze(0).unsqueeze(0)
        Z = F.conv2d(img_4d, kernel, padding=kernel_size//2).squeeze() / 255.0
        Z = Z.cpu().numpy()
    else:
        Z = cv2.GaussianBlur(img, (5, 5), 0) / 255.0
    
    return {
        'z_min': float(Z.min()),
        'z_max': float(Z.max()),
        'z_mean': float(Z.mean()),
        'z_std': float(Z.std()),
        'continuous_encoding': (Z.max() - Z.min()) > 0.3
    }

def test_2_fft_2d(img, name):
    """Test 2: Transformada de Fourier 2D (FFT) - GPU"""
    print(f"   Test 2: FFT 2D ({name})")
    
    if check_vram(100):
        img_tensor = to_gpu_tensor(img)
        f_transform = torch_fft2(img_tensor)
        f_shift = torch_fftshift(f_transform)
        magnitude = 20 * torch.log(torch.abs(f_shift) + 1e-5)
        magnitude = magnitude.cpu().numpy()
    else:
        f_transform = scipy_fft2(img.astype(float))
        f_shift = scipy_fftshift(f_transform)
        magnitude = 20 * np.log(np.abs(f_shift) + 1e-5)
    
    max_spectral = float(magnitude.max())
    mean_spectral = float(magnitude.mean())
    
    h, w = magnitude.shape
    high_freq_quadrant = magnitude[h//4:3*h//4, w//4:3*w//4]
    threshold = mean_spectral + 2 * magnitude.std()
    peaks_count = int(np.sum(high_freq_quadrant > threshold))
    
    low_freq_energy = float(np.sum(magnitude[h//2-50:h//2+50, w//2-50:w//2+50]))
    total_energy = float(np.sum(magnitude))
    low_freq_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'max_spectral': max_spectral,
        'mean_spectral': mean_spectral,
        'periodic_peaks': peaks_count,
        'low_freq_ratio': low_freq_ratio,
        'has_periodic_pattern': peaks_count > 10
    }

def test_3_gradient_isotropy(img, name):
    """Test 3: Isotropía del Gradiente - GPU"""
    print(f"   Test 3: Isotropía del Gradiente ({name})")
    
    if check_vram(50):
        img_tensor = to_gpu_tensor(img)
        img_4d = img_tensor.unsqueeze(0).unsqueeze(0)
        
        # Sobel kernels
        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        
        sobel_x = F.conv2d(img_4d, sobel_x_kernel, padding=1).squeeze()
        sobel_y = F.conv2d(img_4d, sobel_y_kernel, padding=1).squeeze()
        
        magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
        angle = torch.atan2(sobel_y, sobel_x) * (180 / np.pi) % 360
        
        magnitude = magnitude.cpu().numpy()
        angle = angle.cpu().numpy()
    else:
        sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        angle = np.arctan2(sobel_y, sobel_x) * (180 / np.pi) % 360
    
    threshold = np.percentile(magnitude, 70)
    significant_angles = angle[magnitude > threshold]
    
    mean_angle = float(significant_angles.mean())
    std_angle = float(significant_angles.std())
    
    hist, _ = np.histogram(significant_angles, bins=8, range=(0, 360))
    hist_normalized = hist / hist.sum()
    
    isotropy_score = float(1 - np.std(hist_normalized) / (np.mean(hist_normalized) + 1e-10))
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'mean_angle': mean_angle,
        'std_angle': std_angle,
        'isotropy_score': isotropy_score,
        'is_isotropic': isotropy_score > 0.7,
        'angle_distribution': hist_normalized.tolist()
    }

def test_4_spatial_autocorrelation(img, name):
    """Test 4: Autocorrelación Espacial 2D - GPU"""
    print(f"   Test 4: Autocorrelación Espacial 2D ({name})")
    
    if check_vram(100):
        img_tensor = to_gpu_tensor(img)
        img_norm = img_tensor - img_tensor.mean()
        
        f_img = torch_fft2(img_norm)
        f_conj = torch.conj(f_img)
        autocorr = torch_ifft2(f_img * f_conj).real
        autocorr = torch_fftshift(autocorr)
        autocorr = autocorr / autocorr.max()
        autocorr = autocorr.cpu().numpy()
    else:
        img_norm = img.astype(float) - img.mean()
        f_img = scipy_fft2(img_norm)
        f_conj = np.conj(f_img)
        autocorr = np.fft.ifft2(f_img * f_conj).real
        autocorr = scipy_fftshift(autocorr)
        autocorr /= autocorr.max()
    
    h, w = autocorr.shape
    center_y, center_x = h // 2, w // 2
    
    max_radius = min(h, w) // 4
    radial_profile = []
    for r in range(0, max_radius, 5):
        y, x = np.ogrid[:h, :w]
        mask = (np.abs(x - center_x)**2 + np.abs(y - center_y)**2 >= (r-2)**2) & \
               (np.abs(x - center_x)**2 + np.abs(y - center_y)**2 <= (r+2)**2)
        if mask.sum() > 0:
            radial_profile.append(float(autocorr[mask].mean()))
    
    min_len = min(center_x - 1, w - center_x - 1)
    left_right_corr = float(np.corrcoef(
        autocorr[center_y, center_x-min_len:center_x],
        autocorr[center_y, center_x+1:center_x+1+min_len][::-1]
    )[0, 1])
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'radial_decay': radial_profile[:10],
        'left_right_symmetry': left_right_corr,
        'is_symmetric': left_right_corr > 0.8,
        'correlation_length': len([v for v in radial_profile if v > 0.1])
    }

# Función global para multiprocessing
def box_count_single(args):
    thresh, img = args
    binary = (img / 255.0 > thresh).astype(np.uint8)
    sizes = [2, 4, 8, 16, 32]
    counts = []
    h, w = binary.shape
    for size in sizes:
        n_boxes = 0
        for i in range(0, h - size + 1, size):
            for j in range(0, w - size + 1, size):
                if binary[i:i+size, j:j+size].sum() > 0:
                    n_boxes += 1
        counts.append(n_boxes)
    
    sizes = np.array(sizes)
    counts = np.array(counts)
    if len(sizes) > 2 and counts[0] > 0:
        log_sizes = np.log(1.0 / sizes)
        log_counts = np.log(np.maximum(counts, 1))
        coeffs = np.polyfit(log_sizes, log_counts, 1)
        return float(coeffs[0])
    return 0.0

# ============================================================================
# BLOQUE B: FRACTAL Y COLOR
# ============================================================================

def test_5_multifractal_spectrum(img, name):
    """Test 5: Espectro Multifractal - CPU con multiprocessing"""
    print(f"   Test 5: Espectro Multifractal ({name})")
    
    thresholds = np.linspace(0.1, 0.9, 20)
    tasks = [(t, img) for t in thresholds]
    
    # Usar multiprocessing si hay muchos cores
    if cpu_count() > 2:
        with Pool(min(cpu_count(), 8)) as pool:
            dimensions = pool.map(box_count_single, tasks)
    else:
        dimensions = [box_count_single((t, img)) for t in thresholds]
    
    dimensions = np.array(dimensions)
    
    return {
        'D_min': float(dimensions.min()),
        'D_max': float(dimensions.max()),
        'D_mean': float(dimensions.mean()),
        'D_range': float(dimensions.max() - dimensions.min()),
        'is_multifractal': (dimensions.max() - dimensions.min()) > 0.1,
        'dimension_curve': dimensions.tolist()
    }

def test_6_laplacian_pyramid(img, name):
    """Test 6: Pirámide Laplaciana - GPU"""
    print(f"   Test 6: Pirámide Laplaciana ({name})")
    
    if check_vram(100):
        img_tensor = to_gpu_tensor(img)
        img_4d = img_tensor.unsqueeze(0).unsqueeze(0)
        
        levels = 4
        pyramid = [img_4d]
        
        for i in range(levels - 1):
            down = F.interpolate(pyramid[-1], scale_factor=0.5, mode='bilinear', align_corners=False)
            up = F.interpolate(down, size=(pyramid[-1].shape[2], pyramid[-1].shape[3]), mode='bilinear', align_corners=False)
            laplacian = pyramid[-1] - up
            pyramid.append(laplacian)
        
        energies = []
        for level in pyramid:
            energy = float((level ** 2).sum().cpu().item())
            energies.append(energy)
    else:
        levels = 4
        pyramid = [img.astype(float)]
        
        for i in range(levels - 1):
            down = cv2.pyrDown(pyramid[-1])
            up = cv2.pyrUp(down, dstsize=(pyramid[-1].shape[1], pyramid[-1].shape[0]))
            laplacian = pyramid[-1] - up
            pyramid.append(laplacian)
        
        energies = [float(np.sum(level ** 2)) for level in pyramid]
    
    total_energy = sum(energies)
    energy_ratios = [e / total_energy if total_energy > 0 else 0 for e in energies]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'n_levels': levels,
        'energy_ratios': energy_ratios,
        'low_freq_dominance': energy_ratios[0] > 0.5,
        'high_freq_content': sum(energy_ratios[2:]) > 0.2
    }

def test_7_hurst_exponent(img, name):
    """Test 7: Exponente de Hurst R/S - CPU (perfil 1D rápido)"""
    print(f"   Test 7: Exponente de Hurst ({name})")
    
    h, w = img.shape
    profile = img[h // 2, :].astype(float)
    
    n = len(profile)
    max_k = n // 2
    
    log_n = []
    log_rs = []
    
    for k in range(10, max_k, max_k // 20):
        n_segments = n // k
        rs_values = []
        
        for i in range(n_segments):
            segment = profile[i*k:(i+1)*k]
            mean_seg = segment.mean()
            cum_dev = np.cumsum(segment - mean_seg)
            R = cum_dev.max() - cum_dev.min()
            S = segment.std()
            if S > 0:
                rs_values.append(R / S)
        
        if rs_values:
            log_n.append(np.log(k))
            log_rs.append(np.log(np.mean(rs_values)))
    
    if len(log_n) > 2:
        coeffs = np.polyfit(log_n, log_rs, 1)
        H = float(coeffs[0])
    else:
        H = 0.5
    
    return {
        'H': H,
        'is_persistent': H > 0.5,
        'memory_type': 'larga' if H > 0.7 else ('corta' if H < 0.3 else 'media')
    }

# ============================================================================
# BLOQUE C: TOMOGRÁFICO Y VOLUMÉTRICO
# ============================================================================

def test_8_radon_transform(img, name):
    """Test 8: Transformada de Radon - GPU con batches"""
    print(f"   Test 8: Transformada de Radon ({name})")
    
    angles = np.arange(0, 180, 10)
    h, w = img.shape
    
    if check_vram(200):
        img_tensor = to_gpu_tensor(img)
        img_4d = img_tensor.unsqueeze(0).unsqueeze(0)
        
        sinogram = np.zeros((len(angles), w))
        
        for i, angle in enumerate(angles):
            rotated = ndimage.rotate(img, angle, reshape=False)
            projection = rotated.sum(axis=0)
            sinogram[i, :] = projection
    else:
        sinogram = np.zeros((len(angles), w))
        for i, angle in enumerate(angles):
            rotated = ndimage.rotate(img, angle, reshape=False)
            projection = rotated.sum(axis=0)
            sinogram[i, :] = projection
    
    sinogram_variance = float(sinogram.var())
    sinogram_mean = float(sinogram.mean())
    projection_similarity = float(np.corrcoef(sinogram)[0, 1])
    
    return {
        'sinogram_variance': sinogram_variance,
        'sinogram_mean': sinogram_mean,
        'projection_similarity': projection_similarity,
        'is_cylindrical': projection_similarity > 0.7,
        'n_projections': len(angles)
    }

def test_9_tomographic_cuts(img, name):
    """Test 9: Cortes Tomográficos - GPU"""
    print(f"   Test 9: Cortes Tomográficos ({name})")
    
    if check_vram(50):
        img_tensor = to_gpu_tensor(img)
        high_depth = (img_tensor > 180).float()
        mid_depth = ((img_tensor >= 110) & (img_tensor <= 180)).float()
        low_depth = (img_tensor < 110).float()
        
        n_high = int(high_depth.sum().cpu().item())
        n_mid = int(mid_depth.sum().cpu().item())
        n_low = int(low_depth.sum().cpu().item())
    else:
        high_depth = (img > 180).astype(np.uint8)
        mid_depth = ((img >= 110) & (img <= 180)).astype(np.uint8)
        low_depth = (img < 110).astype(np.uint8)
        
        n_high = int(high_depth.sum())
        n_mid = int(mid_depth.sum())
        n_low = int(low_depth.sum())
    
    total = n_high + n_mid + n_low
    ratio_high = n_high / total if total > 0 else 0
    ratio_mid = n_mid / total if total > 0 else 0
    ratio_low = n_low / total if total > 0 else 0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'n_high_depth': n_high,
        'n_mid_depth': n_mid,
        'n_low_depth': n_low,
        'ratio_high': ratio_high,
        'ratio_mid': ratio_mid,
        'ratio_low': ratio_low,
        'is_pyramidal': (ratio_low > ratio_mid > ratio_high)
    }

# ============================================================================
# BLOQUE D: GEOMÉTRICO Y DE RIDGES
# ============================================================================

def test_10_topological_skeleton(img, name):
    """Test 10: Esqueleto Topológico - GPU"""
    print(f"   Test 10: Esqueleto Topológico ({name})")
    
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    thresh_val, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if check_vram(100):
        bin_tensor = to_gpu_tensor(bin_img)
        # Distance transform aproximado via convolución iterativa
        dist = torch.zeros_like(bin_tensor)
        dist[bin_tensor > 0] = 1
        
        for _ in range(10):
            dist_4d = dist.unsqueeze(0).unsqueeze(0)
            kernel = torch.ones(1, 1, 3, 3, device=DEVICE) / 9
            dist_smooth = F.conv2d(dist_4d, kernel, padding=1).squeeze()
            dist = torch.maximum(dist, dist_smooth * (bin_tensor / 255.0))
        
        dist = dist.cpu().numpy()
        skeleton = ((dist > 0.5) & (bin_img > 0)).astype(np.uint8)
    else:
        dist_transform = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
        local_max = ndimage.maximum_filter(dist_transform, size=5) == dist_transform
        skeleton = (local_max & (dist_transform > 1)).astype(np.uint8)
    
    skeleton_pixels = int(skeleton.sum())
    total_pixels = img.shape[0] * img.shape[1]
    skeleton_density = skeleton_pixels / total_pixels
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'skeleton_pixels': skeleton_pixels,
        'skeleton_density': skeleton_density,
        'has_clear_skeleton': skeleton_density > 0.001,
        'structural_complexity': 'alta' if skeleton_density > 0.01 else ('media' if skeleton_density > 0.005 else 'baja')
    }

# Función global para multiprocessing de entropía
def compute_entropy_row(args):
    r, half_w, img_small = args
    ws = 9
    row_entropies = []
    for c in range(half_w, img_small.shape[1] - half_w):
        patch = img_small[r-half_w:r+half_w+1, c-half_w:c+half_w+1]
        hist, _ = np.histogram(patch.flatten(), bins=16, range=(0, 256))
        hist = hist / hist.sum()
        row_entropies.append(shannon_entropy(hist))
    return r, row_entropies

def test_11_local_shannon_entropy(img, name):
    """Test 11: Entropía de Shannon Local - CPU con multiprocessing"""
    print(f"   Test 11: Entropía Local ({name})")
    
    img_small = cv2.resize(img, (200, 200))
    
    ws = 9
    half_w = ws // 2
    entropy_map = np.zeros(img_small.shape)
    
    tasks = [(r, half_w, img_small) for r in range(half_w, img_small.shape[0] - half_w)]
    
    if cpu_count() > 2:
        with Pool(min(cpu_count(), 8)) as pool:
            results = pool.map(compute_entropy_row, tasks)
        for r, row_entropies in results:
            entropy_map[r, half_w:img_small.shape[1]-half_w] = row_entropies
    else:
        for r, half_w, img_small in tasks:
            _, row_entropies = compute_entropy_row((r, half_w, img_small))
            entropy_map[r, half_w:img_small.shape[1]-half_w] = row_entropies
    
    return {
        'entropy_mean': float(entropy_map.mean()),
        'entropy_max': float(entropy_map.max()),
        'entropy_min': float(entropy_map.min()),
        'entropy_std': float(entropy_map.std()),
        'high_entropy_regions': int((entropy_map > entropy_map.mean() + entropy_map.std()).sum()),
        'information_density': 'alta' if entropy_map.mean() > 3 else ('media' if entropy_map.mean() > 2 else 'baja')
    }

def test_12_recurrence_plot(img, name):
    """Test 12: Recurrence Plot - GPU"""
    print(f"   Test 12: Recurrence Plot ({name})")
    
    h, w = img.shape
    profile = img[h // 2, :].astype(float)
    
    if check_vram(50):
        profile_tensor = torch.from_numpy(profile).to(DEVICE)
        diff_matrix = torch.abs(profile_tensor.unsqueeze(0) - profile_tensor.unsqueeze(1))
        R = (diff_matrix < 10.0).float()
        R = R.cpu().numpy()
    else:
        threshold = 10.0
        n = len(profile)
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if abs(profile[i] - profile[j]) < threshold:
                    R[i, j] = 1
    
    recurrence_rate = float(R.mean())
    
    min_line = 4
    n_lines = 0
    total_line_points = 0
    
    for k in range(-R.shape[0]+1, R.shape[0]):
        diagonal = np.diag(R, k)
        in_line = False
        line_length = 0
        for val in diagonal:
            if val == 1:
                line_length += 1
            else:
                if line_length >= min_line:
                    n_lines += 1
                    total_line_points += line_length
                line_length = 0
        if line_length >= min_line:
            n_lines += 1
            total_line_points += line_length
    
    determinism = total_line_points / R.sum() if R.sum() > 0 else 0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'recurrence_rate': recurrence_rate,
        'n_diagonal_lines': n_lines,
        'determinism': determinism,
        'is_deterministic': determinism > 0.3
    }

def test_13_bilateral_symmetry(img, name):
    """Test 13: Simetría Bilateral NCC - GPU"""
    print(f"   Test 13: Simetría Bilateral ({name})")
    
    h, w = img.shape
    center = w // 2
    
    if check_vram(50):
        img_tensor = to_gpu_tensor(img)
        left = img_tensor[:, :center]
        right = img_tensor[:, center:]
        
        min_width = min(left.shape[1], right.shape[1])
        left = left[:, :min_width]
        right = right[:, :min_width]
        
        left_norm = (left - left.mean()) / (left.std() + 1e-10)
        right_flipped = torch.fliplr(right)
        right_norm = (right_flipped - right_flipped.mean()) / (right_flipped.std() + 1e-10)
        
        ncc = float((left_norm * right_norm).mean().cpu().item())
    else:
        left = img[:, :center].astype(float)
        right = img[:, center:].astype(float)
        min_width = min(left.shape[1], right.shape[1])
        left = left[:, :min_width]
        right = right[:, :min_width]
        left_norm = (left - left.mean()) / (left.std() + 1e-10)
        right_flipped = np.fliplr(right)
        right_norm = (right_flipped - right_flipped.mean()) / (right_flipped.std() + 1e-10)
        ncc = float((left_norm * right_norm).mean())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'ncc': ncc,
        'is_symmetric': ncc > 0.7,
        'symmetry_strength': 'fuerte' if ncc > 0.8 else ('moderada' if ncc > 0.5 else 'débil')
    }

# ============================================================================
# BLOQUE E: ALGEBRAICO Y SISTEMAS COMPLEJOS
# ============================================================================

def test_14_svd_analysis(img, name):
    """Test 14: SVD - GPU"""
    print(f"   Test 14: SVD ({name})")
    
    if check_vram(200):
        img_tensor = to_gpu_tensor(img)
        U, S, Vt = torch.linalg.svd(img_tensor, full_matrices=False)
        S = S.cpu().numpy()
    else:
        U, S, Vt = np.linalg.svd(img.astype(float), full_matrices=False)
    
    total_variance = np.sum(S ** 2)
    top_10_variance = np.sum(S[:10] ** 2)
    variance_explained = top_10_variance / total_variance if total_variance > 0 else 0
    
    S_normalized = S / S.sum()
    spectral_entropy = float(shannon_entropy(S_normalized))
    
    threshold = S.max() * 0.01
    effective_rank = int(np.sum(S > threshold))
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'variance_explained_top10': variance_explained,
        'spectral_entropy': spectral_entropy,
        'effective_rank': effective_rank,
        'total_singular_values': len(S),
        'is_low_rank': variance_explained > 0.9
    }

def test_15_game_of_life(img, name):
    """Test 15: Game of Life - CPU (iterativo)"""
    print(f"   Test 15: Game of Life ({name})")
    
    thresh_val, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grid = (bin_img > 0).astype(int)
    
    h, w = grid.shape
    patch = grid[h//2-50:h//2+50, w//2-50:w//2+50]
    
    densities = [float(patch.mean())]
    
    for gen in range(6):
        new_patch = np.zeros_like(patch)
        for i in range(1, patch.shape[0]-1):
            for j in range(1, patch.shape[1]-1):
                neighbors = patch[i-1:i+2, j-1:j+2].sum() - patch[i, j]
                if patch[i, j] == 1:
                    new_patch[i, j] = 1 if neighbors in [2, 3] else 0
                else:
                    new_patch[i, j] = 1 if neighbors == 3 else 0
        patch = new_patch
        densities.append(float(patch.mean()))
    
    return {
        'initial_density': densities[0],
        'final_density': densities[-1],
        'density_sequence': densities,
        'convergence_type': 'estable' if abs(densities[-1] - densities[-2]) < 0.01 else 'oscilante',
        'is_stable': densities[-1] < 0.1
    }

def test_16_eigenvector_centrality(img, name):
    """Test 16: Centralidad de Autovector - GPU"""
    print(f"   Test 16: Centralidad de Autovector ({name})")
    
    img_small = cv2.resize(img, (40, 40))
    h, w = img_small.shape
    n_nodes = h * w
    img_flat = img_small.flatten().astype(float)
    
    if check_vram(50):
        img_tensor = torch.from_numpy(img_flat).to(DEVICE)
        adjacency = torch.zeros((n_nodes, n_nodes), device=DEVICE)
        
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                if j < w - 1:
                    idx_right = i * w + (j + 1)
                    similarity = 1 - torch.abs(img_tensor[idx] - img_tensor[idx_right]) / 255
                    adjacency[idx, idx_right] = similarity
                    adjacency[idx_right, idx] = similarity
                if i < h - 1:
                    idx_down = (i + 1) * w + j
                    similarity = 1 - torch.abs(img_tensor[idx] - img_tensor[idx_down]) / 255
                    adjacency[idx, idx_down] = similarity
                    adjacency[idx_down, idx] = similarity
        
        # Power iteration
        x = torch.rand(n_nodes, device=DEVICE)
        for _ in range(100):
            x_new = adjacency @ x
            x_new = x_new / torch.norm(x_new)
            if torch.norm(x_new - x) < 1e-6:
                break
            x = x_new
        
        x = x.cpu().numpy()
    else:
        adjacency = np.zeros((n_nodes, n_nodes))
        for i in range(h):
            for j in range(w):
                idx = i * w + j
                if j < w - 1:
                    idx_right = i * w + (j + 1)
                    similarity = 1 - abs(img_flat[idx] - img_flat[idx_right]) / 255
                    adjacency[idx, idx_right] = similarity
                    adjacency[idx_right, idx] = similarity
                if i < h - 1:
                    idx_down = (i + 1) * w + j
                    similarity = 1 - abs(img_flat[idx] - img_flat[idx_down]) / 255
                    adjacency[idx, idx_down] = similarity
                    adjacency[idx_down, idx] = similarity
        
        x = np.random.rand(n_nodes)
        for _ in range(100):
            x_new = adjacency @ x
            x_new = x_new / np.linalg.norm(x_new)
            if np.linalg.norm(x_new - x) < 1e-6:
                break
            x = x_new
    
    max_centrality_idx = np.argmax(x)
    max_centrality_row = max_centrality_idx // w
    max_centrality_col = max_centrality_idx % w
    
    mean_centrality = float(x.mean())
    max_centrality = float(x.max())
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        'max_centrality_position': [int(max_centrality_row), int(max_centrality_col)],
        'mean_centrality': mean_centrality,
        'max_centrality': max_centrality,
        'centrality_ratio': max_centrality / mean_centrality if mean_centrality > 0 else 0,
        'has_central_hub': (max_centrality / mean_centrality) > 2
    }

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def run_all_tests_on_image(img_path, image_name):
    """Ejecutar las 16 pruebas sobre una imagen"""
    print(f"\n{'='*70}")
    print(f"ANALIZANDO: {image_name}")
    print(f"{'='*70}")
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"   ERROR: No se pudo cargar {img_path}")
        return None
    
    h, w = img.shape
    print(f"   Dimensiones: {w}x{h}px")
    print()
    
    results = {
        'image_name': image_name,
        'dimensions': [h, w],
        'tests': {}
    }
    
    print("BLOQUE A: Proyección y Frecuencias")
    results['tests']['test_1_z_relief'] = test_1_z_relief(img, image_name)
    results['tests']['test_2_fft_2d'] = test_2_fft_2d(img, image_name)
    results['tests']['test_3_gradient_isotropy'] = test_3_gradient_isotropy(img, image_name)
    results['tests']['test_4_spatial_autocorrelation'] = test_4_spatial_autocorrelation(img, image_name)
    
    print("\nBLOQUE B: Fractal y Color")
    results['tests']['test_5_multifractal_spectrum'] = test_5_multifractal_spectrum(img, image_name)
    results['tests']['test_6_laplacian_pyramid'] = test_6_laplacian_pyramid(img, image_name)
    results['tests']['test_7_hurst_exponent'] = test_7_hurst_exponent(img, image_name)
    
    print("\nBLOQUE C: Tomográfico y Volumétrico")
    results['tests']['test_8_radon_transform'] = test_8_radon_transform(img, image_name)
    results['tests']['test_9_tomographic_cuts'] = test_9_tomographic_cuts(img, image_name)
    
    print("\nBLOQUE D: Geométrico y de Ridges")
    results['tests']['test_10_topological_skeleton'] = test_10_topological_skeleton(img, image_name)
    results['tests']['test_11_local_shannon_entropy'] = test_11_local_shannon_entropy(img, image_name)
    results['tests']['test_12_recurrence_plot'] = test_12_recurrence_plot(img, image_name)
    results['tests']['test_13_bilateral_symmetry'] = test_13_bilateral_symmetry(img, image_name)
    
    print("\nBLOQUE E: Algebraico y Sistemas Complejos")
    results['tests']['test_14_svd_analysis'] = test_14_svd_analysis(img, image_name)
    results['tests']['test_15_game_of_life'] = test_15_game_of_life(img, image_name)
    results['tests']['test_16_eigenvector_centrality'] = test_16_eigenvector_centrality(img, image_name)
    
    print()
    return results

def main():
    t_start = time.time()
    all_results = {}
    
    print("\n" + "="*70)
    print("NIVEL 1: ANÁLISIS DE IMÁGENES ORIGINALES (GPU OPTIMIZED)")
    print("="*70)
    
    for name, path in IMAGES.items():
        results = run_all_tests_on_image(path, name)
        if results:
            all_results[name] = results
    
    output_file = os.path.join(OUTPUT_DIR, 'sindonologia_16_tests_gpu_results.json')
    
    # Convertir tipos numpy a Python nativo para JSON
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
    
    all_results_serializable = convert_to_serializable(all_results)
    
    with open(output_file, 'w') as f:
        json.dump(all_results_serializable, f, indent=2)
    
    print(f"\n{'='*70}")
    print("RESULTADOS GUARDADOS")
    print(f"{'='*70}")
    print(f"Archivo: {output_file}")
    print(f"Tiempo total: {time.time()-t_start:.1f}s")
    print()
    
    print("RESUMEN EJECUTIVO:")
    for name, results in all_results.items():
        print(f"\n{name}:")
        tests = results['tests']
        print(f"   Z-Relief: [{tests['test_1_z_relief']['z_min']:.3f}, {tests['test_1_z_relief']['z_max']:.3f}]")
        print(f"   FFT: max={tests['test_2_fft_2d']['max_spectral']:.2f}, picos={tests['test_2_fft_2d']['periodic_peaks']}")
        print(f"   Isotropía: {tests['test_3_gradient_isotropy']['isotropy_score']:.3f}")
        print(f"   Hurst H: {tests['test_7_hurst_exponent']['H']:.3f}")
        print(f"   Simetría NCC: {tests['test_13_bilateral_symmetry']['ncc']:.3f}")
        print(f"   SVD varianza: {tests['test_14_svd_analysis']['variance_explained_top10']:.3f}")

if __name__ == '__main__':
    main()
