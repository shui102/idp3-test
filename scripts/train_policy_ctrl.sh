# Examples:

#   bash scripts/train_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/train_policy.sh dp_224x224_r3m gr1_dex-image 0913_example
#   bash scripts/train_policy.sh idp3_6d rm_3d put_two_cup_into_basket_20251109_19-21
dataset_path=/home/shui/idp3_test/Improved-3D-Diffusion-Policy/data/dataset_11092057_processed_joint.zarr


DEBUG=False
wandb_mode=online


alg_name=${1}
config_name=${alg_name}
addition_info=${2}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    save_ckpt=False
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    save_ckpt=True
    echo -e "\033[33mTrain mode\033[0m"
fi


cd Improved-3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

python train.py --config-name=${config_name}.yaml \
                            hydra.run.dir=${run_dir} \




                                