#!/bin/bash
#SBATCH --job-name=subpixel_rotation
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=7:00:00

source ~/miniconda3/etc/profile.d/conda.sh  # or wherever your conda is installed
conda activate eguez_env

cd ~/EquivariantPnP

echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# python3 PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images  --extract_curves --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 100
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "flip" --noise_level_img 5. --maxitr 100
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "rotation" --noise_level_img 5. --maxitr 100
python3 PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "subpixel_rotation" --noise_level_img 5. --maxitr 100
# python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "translation" --noise_level_img 5. --maxitr 100
#python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "all_transformations" --noise_level_img 5. --maxitr 100

# for i in {0..4} do
#     python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 100 --seed  $i
#     python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "flip" --noise_level_img 5. --maxitr 100 --seed  $i
#     python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "rotation" --noise_level_img 5. --maxitr 100 --seed  $i
#     python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "subpixel_rotation" --noise_level_img 5. --maxitr 100 --seed  $i
#     python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "translation" --noise_level_img 5. --maxitr 100 --seed  $i
#     python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --transformation "all_transformations" --noise_level_img 5. --maxitr 100 --seed  $i
# done

echo "Job finished at: $(date)"


python3 PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --extract_curves --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100 --kernel_indexes 0 --transformation flip
