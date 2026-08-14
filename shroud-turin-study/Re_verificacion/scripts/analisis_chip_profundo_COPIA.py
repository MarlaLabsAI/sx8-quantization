import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage, signal
from scipy.stats import entropy as scipy_entropy
from scipy.optimize import curve_fit
import torch
import torch.nn.functional as F
import os
import json
import time

OUTPUT_DIR = r"/mnt/Data_3TB/Estudios_Sabana_Santa_Turin/Re_verificacion/resultados/ejecucion_original"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGES = {
    "imagen1": r"/mnt/Data_3TB/Estudios_Sabana_Santa_Turin/04_IMAGENES_ORIGINALES/imagen1_negativo.jpeg",
    "imagen2": r"/mnt/Data_3TB/Estudios_Sabana_Santa_Turin/04_IMAGENES_ORIGINALES/imagen2_dos_caras.jpeg",
    "imagen3": r"/mnt/Data_3TB/Estudios_Sabana_Santa_Turin/04_IMAGENES_ORIGINALES/imagen3_sepia.jpeg",
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS = {}

def save_result(name, data):
    RESULTS[name] = data
    with open(os.path.join(OUTPUT_DIR, "analisis_chip.json"), 'w') as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)

print("="*80)
print("ANÁLISIS PROFUNDO: ESTRUCTURA TIPO CHIP/ASIC EN MATRICES DE RECURRENCIA")
print("="*80)

for img_name, img_path in IMAGES.items():
    print(f"\n{'#'*80}")
    print(f"  ANALIZANDO: {img_name}")
    print(f"{'#'*80}")
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.float64)
    img_u8 = img_norm.astype(np.uint8)
    
    # Generar matriz de recurrencia del eje central
    perfil = cv2.GaussianBlur(img_norm[:, w//2].astype(float).reshape(-1,1), (15,1), 0).flatten()
    recurrence = np.abs(perfil[:, None] - perfil[None, :]) < 10.0
    recurrence_float = recurrence.astype(float)
    rec_h, rec_w = recurrence.shape  # Debería ser (h, h)
    
    print(f"\n  [CHIP-1] Análisis de Simetría de la Matriz de Recurrencia")
    # Simetría horizontal y vertical
    sym_h = np.mean(np.abs(recurrence_float - np.flipud(recurrence_float)))
    sym_v = np.mean(np.abs(recurrence_float - np.fliplr(recurrence_float)))
    sym_diag = np.mean(np.abs(recurrence_float - recurrence_float.T))
    print(f"    Asimetría H: {sym_h:.4f}, V: {sym_v:.4f}, Diagonal: {sym_diag:.4f}")
    save_result(f"{img_name}_simetria", {"H": sym_h, "V": sym_v, "diag": sym_diag})
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(recurrence_float - np.flipud(recurrence_float), cmap='RdBu')
    axes[0].set_title(f"Asimetría H: {sym_h:.4f}")
    axes[1].imshow(recurrence_float - np.fliplr(recurrence_float), cmap='RdBu')
    axes[1].set_title(f"Asimetría V: {sym_v:.4f}")
    axes[2].imshow(recurrence_float - recurrence_float.T, cmap='RdBu')
    axes[2].set_title(f"Asimetría Diagonal: {sym_diag:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip1_simetria.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-2] Autocorrelación 2D de la Matriz de Recurrencia")
    rec_mean = recurrence_float - np.mean(recurrence_float)
    fft_rec = np.fft.fft2(rec_mean)
    autocorr_rec = np.fft.ifft2(fft_rec * np.conj(fft_rec)).real
    autocorr_rec = np.fft.fftshift(autocorr_rec)
    autocorr_rec /= np.max(autocorr_rec)
    
    # Buscar picos en autocorrelación (indica periodicidad)
    from scipy.signal import find_peaks
    ac_center_row = autocorr_rec[rec_h//2, :]
    ac_center_col = autocorr_rec[:, rec_w//2]
    
    peaks_row, _ = find_peaks(ac_center_row[:len(ac_center_row)//2], distance=20, prominence=0.1)
    peaks_col, _ = find_peaks(ac_center_col[:len(ac_center_col)//2], distance=20, prominence=0.1)
    
    periodicity_h = np.diff(peaks_row) if len(peaks_row) > 1 else np.array([0])
    periodicity_v = np.diff(peaks_col) if len(peaks_col) > 1 else np.array([0])
    
    print(f"    Periodicidad H: {periodicity_h.mean():.1f} px (std={periodicity_h.std():.1f})")
    print(f"    Periodicidad V: {periodicity_v.mean():.1f} px (std={periodicity_v.std():.1f})")
    save_result(f"{img_name}_periodicidad", {"H_mean": periodicity_h.mean(), "V_mean": periodicity_v.mean()})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(autocorr_rec, cmap='hot')
    axes[0,0].set_title("Autocorrelación 2D de Recurrencia")
    axes[0,1].plot(ac_center_row[:len(ac_center_row)//2])
    axes[0,1].plot(peaks_row, ac_center_row[peaks_row], 'rx')
    axes[0,1].set_title(f"Perfil H con picos (periodicidad={periodicity_h.mean():.0f}px)")
    axes[1,0].plot(ac_center_col[:len(ac_center_col)//2])
    axes[1,0].plot(peaks_col, ac_center_col[peaks_col], 'rx')
    axes[1,0].set_title(f"Perfil V con picos (periodicidad={periodicity_v.mean():.0f}px)")
    axes[1,1].hist(periodicity_h, bins=20, alpha=0.5, label='H')
    axes[1,1].hist(periodicity_v, bins=20, alpha=0.5, label='V')
    axes[1,1].legend()
    axes[1,1].set_title("Distribución de periodicidades")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip2_autocorr.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-3] Análisis Espectral (FFT) de la Matriz de Recurrencia")
    fft2_rec = np.fft.fft2(recurrence_float)
    fft2_shift = np.fft.fftshift(fft2_rec)
    magnitude = np.log(np.abs(fft2_shift) + 1)
    
    # Buscar picos en el espectro (patrones repetitivos)
    mag_center_row = magnitude[rec_h//2, :]
    mag_center_col = magnitude[:, rec_w//2]
    
    peaks_fft_row, _ = find_peaks(mag_center_row, distance=10, prominence=0.5)
    peaks_fft_col, _ = find_peaks(mag_center_col, distance=10, prominence=0.5)
    
    print(f"    Picos espectrales H: {len(peaks_fft_row)}, V: {len(peaks_fft_col)}")
    save_result(f"{img_name}_espectral", {"peaks_H": len(peaks_fft_row), "peaks_V": len(peaks_fft_col)})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(magnitude, cmap='gray')
    axes[0,0].set_title("Espectro FFT de Recurrencia")
    axes[0,1].plot(mag_center_row)
    axes[0,1].plot(peaks_fft_row, mag_center_row[peaks_fft_row], 'rx')
    axes[0,1].set_title(f"Perfil espectral H ({len(peaks_fft_row)} picos)")
    axes[1,0].plot(mag_center_col)
    axes[1,0].plot(peaks_fft_col, mag_center_col[peaks_fft_col], 'rx')
    axes[1,0].set_title(f"Perfil espectral V ({len(peaks_fft_col)} picos)")
    
    # Análisis radial del espectro
    yi, xi = np.ogrid[:rec_h, :rec_w]
    dist = np.sqrt((xi - rec_w//2)**2 + (yi - rec_h//2)**2).astype(int)
    max_r = min(rec_h, rec_w) // 2
    radial_mag = np.zeros(max_r)
    for r in range(max_r):
        mask = dist == r
        if np.any(mask):
            radial_mag[r] = np.mean(magnitude[mask])
    axes[1,1].plot(radial_mag[:max_r//2])
    axes[1,1].set_title("Perfil radial del espectro")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip3_espectral.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-4] Detección de Estructura tipo Cruz/Grid")
    # Buscar la cruz en la parte superior izquierda
    # Analizar quadrant superior izquierdo
    quadrant_size = min(rec_h, rec_w) // 2
    quadrant_tl = recurrence_float[:quadrant_size, :quadrant_size]
    
    # Detectar líneas horizontales y verticales dominantes
    row_density = np.mean(quadrant_tl, axis=1)
    col_density = np.mean(quadrant_tl, axis=0)
    
    # Buscar líneas fuertes (alta densidad)
    threshold_row = np.mean(row_density) + np.std(row_density)
    threshold_col = np.mean(col_density) + np.std(col_density)
    
    strong_rows = np.where(row_density > threshold_row)[0]
    strong_cols = np.where(col_density > threshold_col)[0]
    
    # Agrupar líneas cercanas
    def group_lines(lines, gap=10):
        if len(lines) == 0:
            return []
        groups = []
        current_group = [lines[0]]
        for line in lines[1:]:
            if line - current_group[-1] <= gap:
                current_group.append(line)
            else:
                groups.append(int(np.mean(current_group)))
                current_group = [line]
        groups.append(int(np.mean(current_group)))
        return groups
    
    grid_rows = group_lines(strong_rows)
    grid_cols = group_lines(strong_cols)
    
    print(f"    Líneas de grid detectadas - Filas: {len(grid_rows)}, Columnas: {len(grid_cols)}")
    print(f"    Posiciones filas: {grid_rows[:10]}")
    print(f"    Posiciones columnas: {grid_cols[:10]}")
    save_result(f"{img_name}_grid", {"rows": grid_rows, "cols": grid_cols})
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(quadrant_tl, cmap='gray_r')
    for r in grid_rows:
        axes[0,0].axhline(y=r, color='red', linewidth=0.5, alpha=0.7)
    for c in grid_cols:
        axes[0,0].axvline(x=c, color='blue', linewidth=0.5, alpha=0.7)
    axes[0,0].set_title(f"Grid detectado: {len(grid_rows)}x{len(grid_cols)} celdas")
    
    axes[0,1].plot(row_density)
    axes[0,1].axhline(y=threshold_row, color='r', linestyle='--')
    axes[0,1].set_title("Densidad por filas")
    axes[1,0].plot(col_density)
    axes[1,0].axhline(y=threshold_col, color='r', linestyle='--')
    axes[1,0].set_title("Densidad por columnas")
    
    # Histograma de espaciados
    if len(grid_rows) > 1:
        spacings_rows = np.diff(grid_rows)
        axes[1,1].hist(spacings_rows, bins=20, alpha=0.5, label='Filas')
    if len(grid_cols) > 1:
        spacings_cols = np.diff(grid_cols)
        axes[1,1].hist(spacings_cols, bins=20, alpha=0.5, label='Columnas')
    axes[1,1].legend()
    axes[1,1].set_title("Distribución de espaciados del grid")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip4_grid.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-5] Análisis de Celdas/Bloques Individuales")
    if len(grid_rows) > 1 and len(grid_cols) > 1:
        # Extraer celdas individuales
        cells = []
        for i in range(min(5, len(grid_rows)-1)):
            for j in range(min(5, len(grid_cols)-1)):
                r1, r2 = grid_rows[i], grid_rows[i+1]
                c1, c2 = grid_cols[j], grid_cols[j+1]
                if r2-r1 > 10 and c2-c1 > 10:
                    cell = recurrence_float[r1:r2, c1:c2]
                    cells.append(cell)
        
        if len(cells) > 0:
            # Calcular similitud entre celdas
            cell_size = min(c.shape[0] for c in cells)
            cells_resized = [cv2.resize(c, (cell_size, cell_size)) for c in cells]
            cells_array = np.array(cells_resized)
            
            # Matriz de similitud entre celdas
            n_cells = len(cells_array)
            similarity_matrix = np.zeros((n_cells, n_cells))
            for i in range(n_cells):
                for j in range(n_cells):
                    similarity_matrix[i,j] = np.mean(cells_array[i] == cells_array[j])
            
            mean_similarity = np.mean(similarity_matrix[np.triu_indices(n_cells, 1)])
            print(f"    Celdas analizadas: {n_cells}")
            print(f"    Similitud media entre celdas: {mean_similarity:.4f}")
            save_result(f"{img_name}_celdas", {"n_cells": n_cells, "mean_similarity": mean_similarity})
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0,0].imshow(cells_array[0], cmap='gray_r')
            axes[0,0].set_title("Celda 0 (referencia)")
            if n_cells > 1:
                axes[0,1].imshow(cells_array[1], cmap='gray_r')
                axes[0,1].set_title(f"Celda 1 (sim={similarity_matrix[0,1]:.3f})")
            if n_cells > 2:
                axes[1,0].imshow(cells_array[2], cmap='gray_r')
                axes[1,0].set_title(f"Celda 2 (sim={similarity_matrix[0,2]:.3f})")
            axes[1,1].imshow(similarity_matrix, cmap='hot')
            axes[1,1].set_title(f"Matriz de similitud (media={mean_similarity:.3f})")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip5_celdas.png"), dpi=100)
            plt.close()
    
    print(f"\n  [CHIP-6] Análisis Fractal de la Estructura de Recurrencia")
    # Box-counting en la matriz de recurrencia
    def box_counting_2d(binary_img):
        p = max(binary_img.shape)
        n = 2**int(np.ceil(np.log2(p)))
        padded = np.pad(binary_img, ((0, n-binary_img.shape[0]), (0, n-binary_img.shape[1])), 'constant')
        sizes = 2**np.arange(int(np.log2(n)), 1, -1)
        counts = []
        for s in sizes:
            reshaped = padded.reshape(n//s, s, n//s, s)
            c = np.sum(np.any(reshaped, axis=(1,3)))
            counts.append(c)
        counts = np.array(counts)
        valid = counts > 0
        if np.sum(valid) < 2:
            return 0.0
        return -np.polyfit(np.log(sizes[valid]), np.log(counts[valid]), 1)[0]
    
    D_rec = box_counting_2d(recurrence)
    print(f"    Dimensión fractal de matriz de recurrencia: {D_rec:.4f}")
    save_result(f"{img_name}_fractal_rec", {"D": D_rec})
    
    # Análisis multifractal
    thresholds = np.linspace(0.1, 0.9, 10)
    mfa_dims = []
    for T in thresholds:
        binary = recurrence_float > T
        d = box_counting_2d(binary)
        mfa_dims.append(d)
    
    print(f"    Rango multifractal: [{min(mfa_dims):.3f}, {max(mfa_dims):.3f}]")
    save_result(f"{img_name}_multifractal_rec", {"dims": mfa_dims})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(recurrence, cmap='gray_r')
    axes[0].set_title(f"Matriz de recurrencia (D={D_rec:.3f})")
    axes[1].plot(thresholds, mfa_dims, 'b-o')
    axes[1].set_title(f"Espectro multifractal")
    axes[1].set_xlabel("Umbral")
    axes[1].set_ylabel("Dimensión D")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip6_fractal.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-7] Análisis de Información Mutua entre Regiones")
    # Dividir la matriz en 4 cuadrantes y calcular información mutua
    h_mid, w_mid = rec_h//2, rec_w//2
    q1 = recurrence_float[:h_mid, :w_mid]
    q2 = recurrence_float[:h_mid, w_mid:]
    q3 = recurrence_float[h_mid:, :w_mid]
    q4 = recurrence_float[h_mid:, w_mid:]
    
    def mutual_information_2d(img1, img2, bins=16):
        # Reducir resolución para cálculo rápido
        img1_small = cv2.resize(img1, (64, 64))
        img2_small = cv2.resize(img2, (64, 64))
        
        hist_2d, _, _ = np.histogram2d(
            img1_small.flatten(), img2_small.flatten(),
            bins=bins, range=[[0,1],[0,1]]
        )
        hist_2d = hist_2d / np.sum(hist_2d) + 1e-10
        
        px = np.sum(hist_2d, axis=1)
        py = np.sum(hist_2d, axis=0)
        
        mi = np.sum(hist_2d * np.log(hist_2d / np.outer(px, py)))
        return mi
    
    mi_12 = mutual_information_2d(q1, q2)
    mi_13 = mutual_information_2d(q1, q3)
    mi_14 = mutual_information_2d(q1, q4)
    mi_23 = mutual_information_2d(q2, q3)
    mi_24 = mutual_information_2d(q2, q4)
    mi_34 = mutual_information_2d(q3, q4)
    
    print(f"    MI Q1-Q2: {mi_12:.4f}, Q1-Q3: {mi_13:.4f}, Q1-Q4: {mi_14:.4f}")
    print(f"    MI Q2-Q3: {mi_23:.4f}, Q2-Q4: {mi_24:.4f}, Q3-Q4: {mi_34:.4f}")
    save_result(f"{img_name}_informacion_mutua", {
        "Q1-Q2": mi_12, "Q1-Q3": mi_13, "Q1-Q4": mi_14,
        "Q2-Q3": mi_23, "Q2-Q4": mi_24, "Q3-Q4": mi_34
    })
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].imshow(q1, cmap='gray_r')
    axes[0,0].set_title("Q1 (superior-izq)")
    axes[0,1].imshow(q2, cmap='gray_r')
    axes[0,1].set_title("Q2 (superior-der)")
    axes[0,2].imshow(q3, cmap='gray_r')
    axes[0,2].set_title("Q3 (inferior-izq)")
    axes[1,0].imshow(q4, cmap='gray_r')
    axes[1,0].set_title("Q4 (inferior-der)")
    
    mi_matrix = np.array([[0, mi_12, mi_13, mi_14],
                          [mi_12, 0, mi_23, mi_24],
                          [mi_13, mi_23, 0, mi_34],
                          [mi_14, mi_24, mi_34, 0]])
    im = axes[1,1].imshow(mi_matrix, cmap='hot')
    axes[1,1].set_title("Matriz Información Mutua")
    plt.colorbar(im, ax=axes[1,1])
    
    axes[1,2].bar(['Q1-Q2', 'Q1-Q3', 'Q1-Q4', 'Q2-Q3', 'Q2-Q4', 'Q3-Q4'],
                  [mi_12, mi_13, mi_14, mi_23, mi_24, mi_34])
    axes[1,2].set_title("MI entre cuadrantes")
    axes[1,2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip7_info_mutua.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-8] Análisis de la Estructura tipo Cruz (Patrón ASIC)")
    # Buscar el centro de la cruz en el cuadrante superior izquierdo
    quadrant_tl = recurrence_float[:quadrant_size, :quadrant_size]
    
    # Proyección horizontal y vertical para encontrar el centro de la cruz
    proj_h = np.mean(quadrant_tl, axis=0)
    proj_v = np.mean(quadrant_tl, axis=1)
    
    # El centro de la cruz debería ser donde ambas proyecciones tienen máximo
    center_x = np.argmax(proj_h)
    center_y = np.argmax(proj_v)
    
    print(f"    Centro de cruz detectado en: ({center_x}, {center_y})")
    print(f"    Posición relativa: ({center_x/quadrant_size:.2f}, {center_y/quadrant_size:.2f})")
    save_result(f"{img_name}_cruz_centro", {"x": int(center_x), "y": int(center_y), "rel_x": float(center_x/quadrant_size), "rel_y": float(center_y/quadrant_size)})
    
    # Analizar simetría radial alrededor del centro
    yi, xi = np.ogrid[:quadrant_size, :quadrant_size]
    dist_from_center = np.sqrt((xi - center_x)**2 + (yi - center_y)**2)
    
    # Perfil radial
    max_dist = min(center_x, center_y, quadrant_size - center_x, quadrant_size - center_y)
    radial_profile = np.zeros(int(max_dist))
    for r in range(int(max_dist)):
        mask = np.abs(dist_from_center - r) < 0.5
        if np.any(mask):
            radial_profile[r] = np.mean(quadrant_tl[mask])
    
    print(f"    Perfil radial calculado (longitud={len(radial_profile)})")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0,0].imshow(quadrant_tl, cmap='gray_r')
    axes[0,0].plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=3)
    axes[0,0].set_title(f"Centro de cruz: ({center_x}, {center_y})")
    
    axes[0,1].plot(proj_h)
    axes[0,1].axvline(x=center_x, color='r', linestyle='--')
    axes[0,1].set_title("Proyección horizontal")
    
    axes[1,0].plot(proj_v)
    axes[1,0].axvline(x=center_y, color='r', linestyle='--')
    axes[1,0].set_title("Proyección vertical")
    
    axes[1,1].plot(radial_profile[:min(200, len(radial_profile))])
    axes[1,1].set_title("Perfil radial desde centro")
    axes[1,1].set_xlabel("Distancia (px)")
    axes[1,1].set_ylabel("Densidad")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip8_cruz.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-9] Análisis de Patrones Jerárquicos/Anidados")
    # Buscar estructuras a diferentes escalas
    scales = [8, 16, 32, 64, 128]
    scale_features = []
    
    for scale in scales:
        # Reducir resolución
        small = cv2.resize(recurrence_float, (w//scale, h//scale))
        # Calcular entropía local
        small_u8 = (small * 255).astype(np.uint8)
        entropy_local = ndimage.generic_filter(small_u8, lambda x: scipy_entropy(np.histogram(x, bins=8)[0] + 1e-10), size=3)
        scale_features.append(np.mean(entropy_local))
    
    print(f"    Entropía media por escala: {[f'{s:.3f}' for s in scale_features]}")
    save_result(f"{img_name}_jerarquico", {"scales": scales, "entropy": scale_features})
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(scales, scale_features, 'b-o')
    axes[0].set_title("Entropía vs Escala")
    axes[0].set_xlabel("Factor de reducción")
    axes[0].set_ylabel("Entropía media")
    axes[0].grid(True)
    
    # Mostrar la matriz a diferentes escalas
    axes[1].imshow(cv2.resize(recurrence_float, (w//32, h//32)), cmap='gray_r')
    axes[1].set_title("Matriz a escala 1/32")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip9_jerarquico.png"), dpi=100)
    plt.close()
    
    print(f"\n  [CHIP-10] Análisis de Conectividad y Componentes")
    # Análisis de componentes conectados en la matriz de recurrencia
    recurrence_binary = (recurrence_float > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(recurrence_binary)
    
    # Filtrar componentes pequeños
    min_area = 100
    large_components = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] > min_area]
    large_components.sort(key=lambda x: x[1], reverse=True)
    
    print(f"    Componentes totales: {num_labels-1}")
    print(f"    Componentes grandes (> {min_area} px): {len(large_components)}")
    if len(large_components) > 0:
        print(f"    Top 5 componentes: {[f'{c[0]}({c[1]}px)' for c in large_components[:5]]}")
    save_result(f"{img_name}_conectividad", {"total": num_labels-1, "large": len(large_components)})
    
    # Visualizar componentes grandes
    viz = np.zeros((rec_h, rec_w, 3), dtype=np.uint8)
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(large_components))))
    for idx, (label, area) in enumerate(large_components[:20]):
        mask = labels == label
        color = (colors[idx][:3] * 255).astype(np.uint8)
        viz[mask] = color
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(recurrence_binary, cmap='gray_r')
    axes[0].set_title(f"Binaria ({num_labels-1} componentes)")
    axes[1].imshow(viz)
    axes[1].set_title(f"Top {min(20, len(large_components))} componentes grandes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{img_name}_chip10_conectividad.png"), dpi=100)
    plt.close()

print(f"\n{'='*80}")
print("ANÁLISIS COMPLETADO")
print(f"Resultados guardados en: {OUTPUT_DIR}")
print(f"{'='*80}")

# Generar resumen
with open(os.path.join(OUTPUT_DIR, "RESUMEN_ANALISIS.txt"), 'w') as f:
    f.write("ANÁLISIS PROFUNDO: ESTRUCTURA TIPO CHIP/ASIC\n")
    f.write("="*80 + "\n\n")
    for img_name in IMAGES.keys():
        f.write(f"\n{img_name.upper()}\n")
        f.write("-"*40 + "\n")
        for key, value in RESULTS.items():
            if key.startswith(img_name):
                f.write(f"  {key}: {value}\n")
