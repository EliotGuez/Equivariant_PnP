#!/bin/bash
#SBATCH --job-name=all_test_2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00

# exec 1> >(tee -a "SNORE_ERED_${SLURM_JOB_ID}.out")
# exec 2> >(tee -a "SNORE_ERED_${SLURM_JOB_ID}.err" >&2)


echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "=========================================="

echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eguez_env

echo "Conda environment activated: $CONDA_DEFAULT_ENV"

echo "Starting experiments..."

echo "Running SNORE"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SNORE" --noise_level_img 5. --maxitr 300 --stepsize 1.5 --lamb 0.3 --sigma_denoiser 7

echo "Running SNORE"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SNORE" --noise_level_img 5. --maxitr 300 --stepsize 1.5 --lamb 0.5 --sigma_denoiser 5 

echo "Running RED"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "RED" --noise_level_img 5. --maxitr 100 --stepsize 1.5 --lamb 0.15 --sigma_denoiser 7

echo "Running PnP_PGD (baseline)"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100 --stepsize 1.9 --sigma_denoiser 5

echo "Running SPnP_PGD 200 iter"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 200 --sigma_denoiser 3. --stepsize 1.0

echo "Running SPnP_PGD 200 iter"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 200 --sigma_denoiser 5. --stepsize 1.0

echo "Running SPnP_PGD 200 iter"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 200 --sigma_denoiser 3. --stepsize 0.8

echo "Running SPnP_PGD 200 iter"
echo "=========================================="
python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 200 --sigma_denoiser 6. --stepsize 1.8

# echo "Running SPnP_PGD 1000 iter"
# echo "=========================================="
# python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 1000 --sigma_denoiser --stepsize 1.0







#####################################################################

# for i in {0..4}; do
#     echo ""
#     echo "=========================================="
#     echo "Time: $(date)"
#     echo "=========================================="
    
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SNORE" --noise_level_img 5. --maxitr 100 --stepsize 1.5 --lamb 0.5 --sigma_denoiser 5 --extract_images --kernel_indexes 0
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "RED" --noise_level_img 5. --maxitr 100 --stepsize 1.5 --lamb 0.15 --sigma_denoiser 7 --extract_images --kernel_indexes 0
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "ERED" --noise_level_img 5. --maxitr 100 --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8 --extract_images --kernel_indexes 0 --transformation "all_transformations"
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "ERED" --noise_level_img 5. --maxitr 100 --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8 --extract_images --kernel_indexes 0 --transformation "flip"
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "ERED" --noise_level_img 5. --maxitr 100 --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8 --extract_images --kernel_indexes 0 --transformation "rotation"
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  --sigma_denoiser 8 --extract_images --kernel_indexes 0 
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  --sigma_denoiser 8 --extract_images --kernel_indexes 0 --transformation "all_transformation"
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  --sigma_denoiser 8 --extract_images --kernel_indexes 0 --transformation "rotation"
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  --sigma_denoiser 8 --extract_images --kernel_indexes 0 --transformation "flip"
#     python3 -u PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 100  --noise_level_SPnP 4.85 --stepsize 1.85 -- --extract_images --kernel_indexes 0 

#     exit_code=$?
#     echo "Iteration $i completed with exit code: $exit_code at $(date)"
# done

# echo ""
# echo "=========================================="
# echo "Job finished at: $(date)"
# echo "=========================================="


# echo "Starting job on node: $(hostname)"
# echo "Job started at: $(date)"
# echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# python3 PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images  --extract_curves --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 100
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "flip" --noise_level_img 5. --maxitr 100
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "rotation" --noise_level_img 5. --maxitr 100
# python3 PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "subpixel_rotation" --noise_level_img 5. --maxitr 100
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "translation" --noise_level_img 5. --maxitr 100
#python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "all_transformations" --noise_level_img 5. --maxitr 100

# for i in {0..4}; do
#     # python3 PnP_restoration/deblur.py --dataset_name "CBSD10"   --gpu_number 0 --opt_alg "ERED" --transformation "flip" --noise_level_img 5. --maxitr 100 --seed  $i  --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8
#     # python3 PnP_restoration/deblur.py --dataset_name "CBSD10"   --gpu_number 0 --opt_alg "ERED" --transformation "subpixel_rotation" --noise_level_img 5. --maxitr 100 --seed  $i --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8
#     # python3 PnP_restoration/deblur.py --dataset_name "CBSD10"   --gpu_number 0 --opt_alg "ERED" --transformation "translation" --noise_level_img 5. --maxitr 100 --seed  $i --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8
#     srun python3 PnP_restoration/deblur.py --dataset_name "CBSD10"   --gpu_number 0 --opt_alg "ERED" --transformation "all_transformations" --noise_level_img 5. --maxitr 100 --seed  $i --stepsize 1.5 --lamb 0.17 --sigma_denoiser 8
# done

# echo "Job finished at: $(date)"


# python3 PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100 --kernel_indexes 0 --transformation flip
# python3 PnP_restoration/deblur.py --dataset_name "CBSD10"  --gpu_number 0 --opt_alg "SPnP_PGD" --pretrained_checkpoint 'GS_denoising/ckpts/Prox-DRUNet.pth' --noise_level_img 5. --maxitr 100 --kernel_indexes 0 --stepsize 0.99 --sigma_denoiser 0.5
# python3 PnP_restoration/deblur.py --dataset_name "CBSD10"  --gpu_number 0 --opt_alg "SPnP_PGD" --pretrained_checkpoint 'GS_denoising/ckpts/Prox-DRUNet.pth' --noise_level_img 5. --maxitr 100 --kernel_indexes 0 --stepsize 0.99 --sigma_denoiser 0.5 --act_mode_denoiser 's'
