python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 500
python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 100
python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "flip" --noise_level_img 5. --maxitr 500
python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "rotation" --noise_level_img 5. --maxitr 500
python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "subpixel_rotation" --noise_level_img 5. --maxitr 500
python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "translation" --noise_level_img 5. --maxitr 500
python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "all_transformations" --noise_level_img 5. --maxitr 500

for i in range(10): 
    python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "SPnP_PGD" --noise_level_img 5. --maxitr 500 --seed  $i
    python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "flip" --noise_level_img 5. --maxitr 500 --seed  $i
    python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "rotation" --noise_level_img 5. --maxitr 500 --seed  $i
    python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "subpixel_rotation" --noise_level_img 5. --maxitr 500 --seed  $i
    python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "translation" --noise_level_img 5. --maxitr 500 --seed  $i
    python PnP_restoration/deblur.py --dataset_name "CBSD10" --extract_images --gpu_number 0 --opt_alg "PnP_PGD" --transformation "all_transformations" --noise_level_img 5. --maxitr 500 --seed  $i

# --kernel_indexes 

python PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100  --n_images 4 --kernel_indexes 0 8 --use_wandb --transformation translation
python PnP_restoration/deblur.py --dataset_name "CBSD10" --gpu_number 0 --opt_alg "PnP_PGD" --noise_level_img 5. --maxitr 100 --transformation all_transformations
