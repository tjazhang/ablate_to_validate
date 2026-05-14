# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: run.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
module load gcc/11.2.0
source ~/.bashrc

# python data/preprocessing.py

# python eval_generated_vs_gt.py \
#   --answers_dir "/path/to/hardblink/answers/run_name" \
#   --k 5 \
#   --datasets 3 \

# python eval_generated_vs_gt.py \
#   --answers_dir /path/to/hardblink/answers/run_name \
#   --datasets 3 \
#   --interp_mode bilinear \
#   --target_num_patches 16 \
#   --print_first_n 3

# python data/depth_point_generator.py --save-intermediates --output /path/to/ADE20K/mixed_depth/mixed_depth.json --answer-type long --convert-only
# python data/feature_interpolation.py --target_num_patches 16 --encoder_name openai_clip_vit_large_patch14_336
# python data/feature_interpolation.py --target_num_patches 64 --encoder_name openai_clip_vit_large_patch14_336
# python data/feature_interpolation.py --target_num_patches 4 --encoder_name openai_clip_vit_large_patch14_336
# python data/feature_interpolation.py --target_num_patches 16 --encoder_name google_siglip2_large_patch16_256
# python data/feature_interpolation.py --target_num_patches 64 --encoder_name google_siglip2_large_patch16_256
# python data/feature_interpolation.py --target_num_patches 4 --encoder_name google_siglip2_large_patch16_256

python data/feature_interpolation.py --target_num_patches 16 --encoder_name facebook_dinov2_base
python data/feature_interpolation.py --target_num_patches 64 --encoder_name facebook_dinov2_base
python data/feature_interpolation.py --target_num_patches 4 --encoder_name facebook_dinov2_base


# python eval_summary.py \
#   /path/to/hardblink/answers/aurora_depth_discrete \
#   /path/to/hardblink/answers/ADE20K_10_llava-v1.5-13b-task-lora \
#   --txt-out /path/to/hardblink/answers/eval_summary.txt \
#   --csv-out /path/to/hardblink/answers/eval_summary.csv
