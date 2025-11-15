# Examples:

#   bash scripts/deploy_policy.sh idp3 gr1_dex-3d 0913_example
#   bash scripts/deploy_policy.sh dp_224x224_r3m gr1_dex-image 0913_example
#   bash scripts/deploy_policy.sh idp3 rm_3d 20251108_put_cup_in_basket
#   bash scripts/deploy_policy.sh idp3_6d rm_3d put_two_cup_into_bakset_20251109_19-21
dataset_path=/home/shui/idp3_test/Improved-3D-Diffusion-Policy/data/put_cup_into_basket_dataset_1108.zarr


DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=0
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=0
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


cd Improved-3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# python deploy.py --config-name=${config_name}.yaml \
#                             task=${task_name} \
#                             hydra.run.dir=${run_dir} \
#                             training.debug=$DEBUG \
#                             training.seed=${seed} \
#                             training.device="cuda:0" \
#                             exp_name=${exp_name} \
#                             logging.mode=${wandb_mode} \
#                             checkpoint.save_ckpt=${save_ckpt} \
#                             task.dataset.zarr_path=$dataset_path 


python deploy_rm.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            task.dataset.zarr_path=$dataset_path 

                                