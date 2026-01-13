import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--num_figures', type=int, default=0, help='Number of figures to generate')
pars = parser.parse_args()

# Paths
path_figure = "/home/infres/eguez-22/EquivariantPnP/figures"
base_path = "/tsi/data_education/Ladjal/Tancrede_Eliot_MVA_2025/Result_ERED/deblurring/CBSD10"

# Image indices to process

if pars.num_figures == 0:

    # Load dictionaries
    path_GS = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_1500/stepsize_1.9/sigma_denoiser_5.0/annealing_number_16/dict_2_results.npy"
    path_PROX = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_1500/stepsize_0.95/sigma_denoiser_4.0/annealing_number_16/dict_2_results.npy"


    dic_GS = np.load(path_GS, allow_pickle=True).item()
    dic_PROX = np.load(path_PROX, allow_pickle=True).item()
    
    # Extract data
    gt = dic_GS["GT"]
    blur = dic_GS["Blur"]
    kernel = dic_GS["kernel"]
    psnr_blur = dic_GS["PSNR_blur"]
    lpips_blur = dic_GS["LPIPS_blur"]
    
    deblur_GS = dic_GS["Deblur"]
    psnr_GS = dic_GS["PSNR_output"]
    lpips_GS = dic_GS["LPIPS_output"]
    psnr_tab_GS = dic_GS["psnr_tab"]
    
    deblur_PROX = dic_PROX["Deblur"]
    psnr_PROX = dic_PROX["PSNR_output"]
    lpips_PROX = dic_PROX["LPIPS_output"]
    psnr_tab_PROX = dic_PROX["psnr_tab"]
    
    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    gs.update(wspace=0.15, hspace=0)
    
    text_size = 20
    
    c = 140  # zoom patch size
    wid, hei = 70, 70  # original patch size
    x_c, y_c = 60, 40

    height_rect = 35
    width_rect = 230
    
    # ============== ROW 1: Original and Blurred ==============
    # Original Image (GT)
    ax1 = plt.subplot(gs[0, 0])
    gt_display = gt.copy()

    patch_c_gt = cv2.resize(gt[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    gt_display[-patch_c_gt.shape[0]:, -patch_c_gt.shape[1]:] = patch_c_gt

    ax1.imshow(gt_display.astype(np.float32))
    rect_metrics = patches.Rectangle((0, 0), 220, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax1.add_patch(rect_metrics)
    ax1.annotate(r"PSNR$\uparrow$ / LPIPS$\downarrow$", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')

    rect_z_gt = patches.Rectangle((gt_display.shape[1]-c-1, gt_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect_z_gt)
    rect_src_gt = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect_src_gt)
    
    ax1.axis('off')
    ax1.set_title("Ground Truth", fontsize=text_size, pad=10)
    
    # Blurred Image
    ax2 = plt.subplot(gs[0, 1])
    blur_display = blur.copy()
    
    k_size = 100  
    k_resize = cv2.resize(kernel, dsize=(k_size, k_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    blur_display[-k_resize.shape[0]:, :k_resize.shape[1]] = k_resize[:, :, None] * np.ones(3)[None, None, :] / np.max(k_resize)
    
    ax2.imshow(blur_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax2.add_patch(rect_psnr)
    ax2.annotate(f"{psnr_blur:.2f} dB / {lpips_blur:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    ax2.axis('off')
    ax2.set_title("Blurred", fontsize=text_size, pad=10) 
    
    # ============== ROW 2: GS DRUNET ==============
    # Deblurred with GS DRUNET
    ax3 = plt.subplot(gs[1, 0])
    deblur_GS_display = deblur_GS.copy()
    

    patch_c = cv2.resize(deblur_GS[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_GS_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c 
    
    ax3.imshow(deblur_GS_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax3.add_patch(rect_psnr)
    ax3.annotate(f"{psnr_GS:.2f} dB / {lpips_GS:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    

    rect_z = patches.Rectangle((deblur_GS_display.shape[1]-c-1, deblur_GS_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none') 
    ax3.add_patch(rect_z)
    
    # Add source rectangle (shifted 20 pixels to the right)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect_src)
    
    ax3.axis('off')
    ax3.set_title("GS DRUNet", fontsize=text_size, pad=10)  
    
    # PSNR Evolution for GS DRUNET
    ax4 = plt.subplot(gs[1, 1])
    iterations_GS = np.arange(len(psnr_tab_GS))
    ax4.plot(iterations_GS, psnr_tab_GS, linewidth=2, color='blue')  

    ax4.set_ylim(20, 35)
    ax4.set_yticks([20, 35])  
    ax4.set_xlim(0, 1500)  
    ax4.set_xticks([0, 1500])

    ax4.tick_params(axis='both', labelsize=text_size)
    ax4.set_aspect(aspect=66)

    # ax4.set_xlabel('Iterations', fontsize=text_size-2, labelpad=-10)
    # ax4.set_ylabel('PSNR (dB)', fontsize=text_size-2, labelpad=-15)

    ax4.set_xlabel('Iterations', fontsize=text_size, labelpad=-15)
    ax4.set_ylabel('PSNR (dB)', fontsize=text_size, labelpad=-30)
    # ============== ROW 3: PROX-DRUNET ==============
    # Deblurred with PROX-DRUNET
    ax5 = plt.subplot(gs[2, 0])
    deblur_PROX_display = deblur_PROX.copy()
    

    patch_c = cv2.resize(deblur_PROX[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_PROX_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax5.imshow(deblur_PROX_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax5.add_patch(rect_psnr)
    ax5.annotate(f"{psnr_PROX:.2f} dB / {lpips_PROX:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    
    rect_z = patches.Rectangle((deblur_PROX_display.shape[1]-c-1, deblur_PROX_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')  
    ax5.add_patch(rect_z)

    # Add source rectangle (shifted 20 pixels to the right)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax5.add_patch(rect_src)

    ax5.axis('off')
    ax5.set_title("PROX-DRUNet", fontsize=text_size, pad=10)  
    
    # PSNR Evolution for PROX-DRUNET
    ax6 = plt.subplot(gs[2, 1])
    iterations_PROX = np.arange(len(psnr_tab_PROX))
    ax6.plot(iterations_PROX, psnr_tab_PROX, linewidth=2, color='green')

    ax6.set_ylim(20, 35)  
    ax6.set_yticks([20, 35])  
    ax6.set_xlim(0, 1500) 
    ax6.set_xticks([0, 1500])

    ax6.tick_params(axis='both', labelsize=text_size)
    ax6.set_aspect(aspect=66)

    ax6.set_xlabel('Iterations', fontsize=text_size, labelpad=-15)
    ax6.set_ylabel('PSNR (dB)', fontsize=text_size, labelpad=-30)

    
    # Save figure
    output_path = f"{path_figure}/Deblurring_Comparison_GS_vs_PROX_image_2.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure for image 2 to: {output_path}")
    plt.close()
    print("Figures 0 generated successfully.")

if pars.num_figures == 1:    
    path_ered_flip = '/home/infres/eguez-22/EquivariantPnP/dict_0_results_ERED_flip_kernel_3'
    path_ered = '/home/infres/eguez-22/EquivariantPnP/dict_0_results_RED_kernel_3'
    path_PnP = f"{base_path}/PnP_PGD_k_3/noise_5.0/maxitr_400/stepsize_1.9/sigma_denoiser_4.0/annealing_number_16/dict_8_results.npy"
    path_snoPnP = f"{base_path}/SPnP_PGD_k_3/noise_5.0/maxitr_100/stepsize_1.9/sigma_denoiser_4.5/annealing_number_16/dict_8_results.npy"
    # path_snoPnP = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_100/stepsize_1.9/sigma_denoiser_8.0/annealing_number_16/dict_8_results.npy"
    # Load dictionaries
    dic_ered_flip = np.load(path_ered_flip, allow_pickle=True).item()
    dic_ered = np.load(path_ered, allow_pickle=True).item()
    dic_PnP = np.load(path_PnP, allow_pickle=True).item()
    dic_snoPnP = np.load(path_snoPnP, allow_pickle=True).item()
    
    gt = dic_PnP["GT"]
    blur = dic_PnP["Blur"]
    kernel = dic_PnP["kernel"]
    psnr_blur = dic_PnP["PSNR_blur"]
    lpips_blur = dic_PnP["LPIPS_blur"]
    
    deblur_ered = dic_ered["Deblur"]
    psnr_ered = dic_ered["PSNR_output"]
    lpips_ered = dic_ered["LPIPS_output"]
    
    deblur_ered_flip = dic_ered_flip["Deblur"]
    psnr_ered_flip = dic_ered_flip["PSNR_output"]
    lpips_ered_flip = dic_ered_flip["LPIPS_output"]
    
    deblur_PnP = dic_PnP["Deblur"]
    psnr_PnP = dic_PnP["PSNR_output"]
    lpips_PnP = dic_PnP["LPIPS_output"]
    
    deblur_snoPnP = dic_snoPnP["Deblur"]
    psnr_snoPnP = dic_snoPnP["PSNR_output"]
    lpips_snoPnP = dic_snoPnP["LPIPS_output"]
    
    # Create figure with 1 row and 6 columns
    fig = plt.figure(figsize=(22, 5))
    gs = gridspec.GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 1, 1])
    gs.update(wspace=0, hspace=0.1)
    
    text_size = 18
    
    c = 125  # zoom patch size (smaller for compact layout)
    wid, hei = 50, 50  # original patch size
    x_c, y_c = 250, 40

    height_rect = 30
    width_rect = 240
    
    # ============== Column 1: Blurred ==============
    ax1 = plt.subplot(gs[0, 0])
    blur_display = blur.copy()

    
    ax1.imshow(blur_display.astype(np.float32))
    
    # Add PSNR/LPIPS text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax1.add_patch(rect_psnr)
    ax1.annotate(f"{psnr_blur:.2f} dB / {lpips_blur:.3f}", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')

    ax1.axis('off')
    ax1.set_title("Blurred", fontsize=text_size, pad=10)
    
    # ============== Column 2: RED ==============
    ax2 = plt.subplot(gs[0, 1])
    deblur_ered_display = deblur_ered.copy()
    
    # Add zoom patch
    patch_c = cv2.resize(deblur_ered[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_ered_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax2.imshow(deblur_ered_display.astype(np.float32))
    
    # Add PSNR/LPIPS text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax2.add_patch(rect_psnr)
    ax2.annotate(f"{psnr_ered:.2f} dB / {lpips_ered:.3f}", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')
    
    # Add zoom rectangles
    rect_z = patches.Rectangle((deblur_ered_display.shape[1]-c-1, deblur_ered_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect_z)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect_src)

    ax2.axis('off')
    ax2.set_title("RED", fontsize=text_size, pad=10)
    
    # ============== Column 3: ERED flip ==============
    ax3 = plt.subplot(gs[0, 2])
    deblur_ered_flip_display = deblur_ered_flip.copy()
    
    # Add zoom patch
    patch_c = cv2.resize(deblur_ered_flip[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_ered_flip_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax3.imshow(deblur_ered_flip_display.astype(np.float32))
    
    # Add PSNR/LPIPS text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax3.add_patch(rect_psnr)
    ax3.annotate(f"{psnr_ered_flip:.2f} dB / {lpips_ered_flip:.3f}", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')
    
    # Add zoom rectangles
    rect_z = patches.Rectangle((deblur_ered_flip_display.shape[1]-c-1, deblur_ered_flip_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect_z)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect_src)
    
    ax3.axis('off')
    ax3.set_title("ERED flip", fontsize=text_size, pad=10)
    
    # ============== Column 4: PnP ==============
    ax4 = plt.subplot(gs[0, 3])
    deblur_PnP_display = deblur_PnP.copy()
    
    # Add zoom patch
    patch_c = cv2.resize(deblur_PnP[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_PnP_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax4.imshow(deblur_PnP_display.astype(np.float32))
    
    # Add PSNR/LPIPS text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax4.add_patch(rect_psnr)
    ax4.annotate(f"{psnr_PnP:.2f} dB / {lpips_PnP:.3f}", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')
    
    # Add zoom rectangles
    rect_z = patches.Rectangle((deblur_PnP_display.shape[1]-c-1, deblur_PnP_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax4.add_patch(rect_z)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax4.add_patch(rect_src)
    
    ax4.axis('off')
    ax4.set_title("PnP", fontsize=text_size, pad=10)
    
    # ============== Column 5: snoPnP ==============
    ax5 = plt.subplot(gs[0, 4])
    deblur_snoPnP_display = deblur_snoPnP.copy()
    
    # Add zoom patch
    patch_c = cv2.resize(deblur_snoPnP[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_snoPnP_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax5.imshow(deblur_snoPnP_display.astype(np.float32))
    
    # Add PSNR/LPIPS text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax5.add_patch(rect_psnr)
    ax5.annotate(f"{psnr_snoPnP:.2f} dB / {lpips_snoPnP:.3f}", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')
    
    # Add zoom rectangles
    rect_z = patches.Rectangle((deblur_snoPnP_display.shape[1]-c-1, deblur_snoPnP_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax5.add_patch(rect_z)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax5.add_patch(rect_src)
    
    ax5.axis('off')
    ax5.set_title("SnoPnP", fontsize=text_size, pad=10)
    
    # ============== Column 6: Ground Truth (Clean) ==============
    ax6 = plt.subplot(gs[0, 5])
    gt_display = gt.copy()
    
    # Add zoom patch
    patch_c_gt = cv2.resize(gt[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    gt_display[-patch_c_gt.shape[0]:, -patch_c_gt.shape[1]:] = patch_c_gt
    
    ax6.imshow(gt_display.astype(np.float32))
    
    # Add label (no PSNR/LPIPS for ground truth)
    rect_label = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax6.add_patch(rect_label)
    ax6.annotate(r"PSNR$\uparrow$ / LPIPS$\downarrow$", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')
    
    # Add zoom rectangles
    rect_z = patches.Rectangle((gt_display.shape[1]-c-1, gt_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax6.add_patch(rect_z)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax6.add_patch(rect_src)
    
    ax6.axis('off')
    ax6.set_title("Ground Truth", fontsize=text_size, pad=10)
    
    # Save figure
    output_path = f"{path_figure}/Deblurring_Comparison_All_Methods_image_8.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure for image 8 to: {output_path}")
    plt.close()
    print("Figure 1 generated successfully!")



if pars.num_figures == 2:

    # Load dictionaries
    path_GS = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_1500/stepsize_1.9/sigma_denoiser_5.0/annealing_number_16/dict_2_results.npy"
    path_PROX = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_1500/stepsize_0.95/sigma_denoiser_4.0/annealing_number_16/dict_2_results.npy"


    dic_GS = np.load(path_GS, allow_pickle=True).item()
    dic_PROX = np.load(path_PROX, allow_pickle=True).item()
    
    # Extract data
    gt, blur, kernel = dic_GS["GT"], dic_GS["Blur"], dic_GS["kernel"]
    psnr_blur, lpips_blur = dic_GS["PSNR_blur"], dic_GS["LPIPS_blur"]
    
    deblur_GS = dic_GS["Deblur"]
    psnr_GS = dic_GS["PSNR_output"]
    lpips_GS = dic_GS["LPIPS_output"]
    residual_tab_GS = dic_GS["residual_list"]
    
    deblur_PROX = dic_PROX["Deblur"]
    psnr_PROX = dic_PROX["PSNR_output"]
    lpips_PROX = dic_PROX["LPIPS_output"]
    residual_tab_PROX = dic_PROX["residual_list"]
    
    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    gs.update(wspace=0.2, hspace=0)
    
    text_size = 20
    
    c = 140  # zoom patch size
    wid, hei = 70, 70  # original patch size
    x_c, y_c = 60, 40

    height_rect = 35
    width_rect = 230
    
    # ============== ROW 1: Original and Blurred ==============
    # Original Image (GT)
    ax1 = plt.subplot(gs[0, 0])
    gt_display = gt.copy()

    patch_c_gt = cv2.resize(gt[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    gt_display[-patch_c_gt.shape[0]:, -patch_c_gt.shape[1]:] = patch_c_gt

    ax1.imshow(gt_display.astype(np.float32))
    rect_metrics = patches.Rectangle((0, 0), 220, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax1.add_patch(rect_metrics)
    ax1.annotate(r"PSNR$\uparrow$ / LPIPS$\downarrow$", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')

    rect_z_gt = patches.Rectangle((gt_display.shape[1]-c-1, gt_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect_z_gt)
    rect_src_gt = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect_src_gt)
    
    ax1.axis('off')
    ax1.set_title("Ground Truth", fontsize=text_size, pad=10)
    
    # Blurred Image
    ax2 = plt.subplot(gs[0, 1])
    blur_display = blur.copy()
    
    k_size = 100  
    k_resize = cv2.resize(kernel, dsize=(k_size, k_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    blur_display[-k_resize.shape[0]:, :k_resize.shape[1]] = k_resize[:, :, None] * np.ones(3)[None, None, :] / np.max(k_resize)
    
    ax2.imshow(blur_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax2.add_patch(rect_psnr)
    ax2.annotate(f"{psnr_blur:.2f} dB / {lpips_blur:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    ax2.axis('off')
    ax2.set_title("Blurred", fontsize=text_size, pad=10) 
    
    # ============== ROW 2: GS DRUNET ==============
    # Deblurred with GS DRUNET
    ax3 = plt.subplot(gs[1, 0])
    deblur_GS_display = deblur_GS.copy()
    

    patch_c = cv2.resize(deblur_GS[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_GS_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c 
    
    ax3.imshow(deblur_GS_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax3.add_patch(rect_psnr)
    ax3.annotate(f"{psnr_GS:.2f} dB / {lpips_GS:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    

    rect_z = patches.Rectangle((deblur_GS_display.shape[1]-c-1, deblur_GS_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none') 
    ax3.add_patch(rect_z)
    
    # Add source rectangle (shifted 20 pixels to the right)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect_src)
    
    ax3.axis('off')
    ax3.set_title("GS DRUNet", fontsize=text_size, pad=10)  
    
    # PSNR Evolution for GS DRUNET
    ax4 = plt.subplot(gs[1, 1])

    res_GS = np.asarray(dic_GS["residual_list"], dtype= np.float64)
    it = np.arange(res_GS.size)

    eps= 1e-16
    res_GS = np.maximum(res_GS,eps)
    ax4.plot(it, res_GS, linewidth=2, color='blue')  
    ax4.set_yscale("log")

    ax4.set_xlim(0, 1500)  
    ax4.set_xticks([0, 1500])

    ax4.tick_params(axis='both', labelsize=text_size)
    ax4.set_xlabel('Iterations', fontsize=text_size, labelpad=-10)
    ax4.set_ylabel('Residual', fontsize=text_size, labelpad=-3)

    # ============== ROW 3: PROX-DRUNET ==============
    # Deblurred with PROX-DRUNET
    ax5 = plt.subplot(gs[2, 0])
    deblur_PROX_display = deblur_PROX.copy()
    

    patch_c = cv2.resize(deblur_PROX[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_PROX_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax5.imshow(deblur_PROX_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax5.add_patch(rect_psnr)
    ax5.annotate(f"{psnr_PROX:.2f} dB / {lpips_PROX:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    
    rect_z = patches.Rectangle((deblur_PROX_display.shape[1]-c-1, deblur_PROX_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')  
    ax5.add_patch(rect_z)

    # Add source rectangle (shifted 20 pixels to the right)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax5.add_patch(rect_src)

    ax5.axis('off')
    ax5.set_title("PROX-DRUNet", fontsize=text_size, pad=10)  
    
    # PSNR Evolution for PROX-DRUNET
    ax6 = plt.subplot(gs[2, 1])


    res_PROX = np.asarray(dic_PROX["residual_list"], dtype=np.float64)
    it_PROX = np.arange(res_PROX.size)

    res_PROX = np.maximum(res_PROX, eps)

    ax6.plot(it_PROX, res_PROX, linewidth=2)
    ax6.set_yscale("log")

    ax6.set_xlim(0, 1500)
    ax6.set_xticks([0, 1500])

    ymin = max(res_PROX.min(), 1e-12)
    ymax = res_PROX.max()
    ax6.set_ylim(ymin, ymax)

    ax6.tick_params(axis='both', labelsize=text_size)
    ax6.set_xlabel('Iterations', fontsize=text_size, labelpad=-10)
    ax6.set_ylabel('Residual', fontsize=text_size, labelpad=-3)
    for ax in [ax4, ax6]:
        pos = ax.get_position()
        ax.set_position([
            pos.x0,
            pos.y0 + 0.02,     # remonte légèrement
            pos.width,
            pos.height * 0.84  # réduit la hauteur
        ])
    # Save figure
    output_path = f"{path_figure}/Deblurring_Comparison_GS_vs_PROX_image_2_residual.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure for image 2 to: {output_path}")
    plt.close()
    print("Figures 2 generated successfully.")
    
if pars.num_figures == 3:

    # Load dictionaries
    path_GS = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_1500/stepsize_1.9/sigma_denoiser_5.0/annealing_number_16/dict_2_results.npy"
    path_PROX = f"{base_path}/SPnP_PGD_k_0/noise_5.0/maxitr_1500/stepsize_0.95/sigma_denoiser_4.0/annealing_number_16/dict_2_results.npy"


    dic_GS = np.load(path_GS, allow_pickle=True).item()
    dic_PROX = np.load(path_PROX, allow_pickle=True).item()
    
    # Extract data
    gt, blur, kernel = dic_GS["GT"], dic_GS["Blur"], dic_GS["kernel"]
    psnr_blur, lpips_blur = dic_GS["PSNR_blur"], dic_GS["LPIPS_blur"]
    
    deblur_GS = dic_GS["Deblur"]
    psnr_GS = dic_GS["PSNR_output"]
    lpips_GS = dic_GS["LPIPS_output"]
    F_tab_GS = dic_GS["F_tab"]
    
    deblur_PROX = dic_PROX["Deblur"]
    psnr_PROX = dic_PROX["PSNR_output"]
    lpips_PROX = dic_PROX["LPIPS_output"]
    F_tab_PROX = dic_PROX["F_tab"]
    
    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    gs.update(wspace=0.15, hspace=0.)
    
    text_size = 20
    
    c = 140  # zoom patch size
    wid, hei = 70, 70  # original patch size
    x_c, y_c = 60, 40

    height_rect = 35
    width_rect = 230
    
    # ============== ROW 1: Original and Blurred ==============
    # Original Image (GT)
    ax1 = plt.subplot(gs[0, 0])
    gt_display = gt.copy()

    patch_c_gt = cv2.resize(gt[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    gt_display[-patch_c_gt.shape[0]:, -patch_c_gt.shape[1]:] = patch_c_gt

    ax1.imshow(gt_display.astype(np.float32))
    rect_metrics = patches.Rectangle((0, 0), 220, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax1.add_patch(rect_metrics)
    ax1.annotate(r"PSNR$\uparrow$ / LPIPS$\downarrow$", xy=(5, 5), color='white', fontsize=text_size-2, va='top', ha='left')

    rect_z_gt = patches.Rectangle((gt_display.shape[1]-c-1, gt_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect_z_gt)
    rect_src_gt = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect_src_gt)
    
    ax1.axis('off')
    ax1.set_title("Ground Truth", fontsize=text_size, pad=10)
    
    # Blurred Image
    ax2 = plt.subplot(gs[0, 1])
    blur_display = blur.copy()
    
    k_size = 100  
    k_resize = cv2.resize(kernel, dsize=(k_size, k_size), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    blur_display[-k_resize.shape[0]:, :k_resize.shape[1]] = k_resize[:, :, None] * np.ones(3)[None, None, :] / np.max(k_resize)
    
    ax2.imshow(blur_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax2.add_patch(rect_psnr)
    ax2.annotate(f"{psnr_blur:.2f} dB / {lpips_blur:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    ax2.axis('off')
    ax2.set_title("Blurred", fontsize=text_size, pad=10) 
    
    # ============== ROW 2: GS DRUNET ==============
    # Deblurred with GS DRUNET
    ax3 = plt.subplot(gs[1, 0])
    deblur_GS_display = deblur_GS.copy()
    

    patch_c = cv2.resize(deblur_GS[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_GS_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c 
    
    ax3.imshow(deblur_GS_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax3.add_patch(rect_psnr)
    ax3.annotate(f"{psnr_GS:.2f} dB / {lpips_GS:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    

    rect_z = patches.Rectangle((deblur_GS_display.shape[1]-c-1, deblur_GS_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none') 
    ax3.add_patch(rect_z)
    
    # Add source rectangle (shifted 20 pixels to the right)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax3.add_patch(rect_src)
    
    ax3.axis('off')
    ax3.set_title("GS DRUNet", fontsize=text_size, pad=10)  
    
    # Functionnal Evolution for GS DRUNET
    ax4 = plt.subplot(gs[1, 1])
    it = np.arange(len(F_tab_GS))
    ax4.plot(it, F_tab_GS, linewidth=2, color='blue')
    ax4.set_xlim(0, 1500)
    ax4.set_xticks([0, 1500])
    ax4.tick_params(axis='both', labelsize=text_size-10)
    ax4.set_xlabel('Iterations', fontsize=text_size-5, labelpad=-10)
    ax4.set_ylabel('Functional', fontsize=text_size-5)

    # ============== ROW 3: PROX-DRUNET ==============
    # Deblurred with PROX-DRUNET
    ax5 = plt.subplot(gs[2, 0])
    deblur_PROX_display = deblur_PROX.copy()
    

    patch_c = cv2.resize(deblur_PROX[y_c:y_c+hei, x_c:x_c+wid], dsize=(c, c), interpolation=cv2.INTER_CUBIC)
    deblur_PROX_display[-patch_c.shape[0]:, -patch_c.shape[1]:] = patch_c
    
    ax5.imshow(deblur_PROX_display.astype(np.float32))
    
    # Add PSNR text
    rect_psnr = patches.Rectangle((0, 0), width_rect, height_rect, linewidth=1, edgecolor='black', facecolor='black')
    ax5.add_patch(rect_psnr)
    ax5.annotate(f"{psnr_PROX:.2f} dB / {lpips_PROX:.3f}", xy=(5, 5), color='white', fontsize=text_size, va='top', ha='left')
    
    rect_z = patches.Rectangle((deblur_PROX_display.shape[1]-c-1, deblur_PROX_display.shape[0]-c-1), c, c, linewidth=2, edgecolor='red', facecolor='none')  
    ax5.add_patch(rect_z)

    # Add source rectangle (shifted 20 pixels to the right)
    rect_src = patches.Rectangle((x_c, y_c), wid, hei, linewidth=2, edgecolor='red', facecolor='none')
    ax5.add_patch(rect_src)

    ax5.axis('off')
    ax5.set_title("PROX-DRUNet", fontsize=text_size, pad=10)  
    
    # Functionnal Evolution for PROX-DRUNET
    ax6 = plt.subplot(gs[2, 1])
    it = np.arange(len(F_tab_PROX))
    ax6.plot(it, F_tab_PROX, linewidth=2, color='green')
    ax6.set_xlim(0, 1500)
    ax6.set_xticks([0, 1500])
    ax6.tick_params(axis='both', labelsize=text_size-10)
    ax6.set_xlabel('Iterations', fontsize=text_size-5, labelpad=-10)
    ax6.set_ylabel('Functional', fontsize=text_size-5)
    
    for ax in [ax4, ax6]:
        pos = ax.get_position()
        ax.set_position([
            pos.x0,
            pos.y0 + 0.02,     # remonte légèrement
            pos.width,
            pos.height * 0.8  # réduit la hauteur
        ])
    # Save figure
    output_path = f"{path_figure}/Deblurring_Comparison_GS_vs_PROX_image_2_functional.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure for image 2 to: {output_path}")
    plt.close()
    print("Figures 3 generated successfully.")
    