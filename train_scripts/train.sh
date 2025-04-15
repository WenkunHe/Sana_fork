#!/bin/bash
#SBATCH -A nvr_elm_llm      #account
#SBATCH -p interactive      #partition
#SBATCH -t 02:00:00         #wall time limit, hr:min:sec
#SBATCH -N 2                #number of nodes
#SBATCH -J sana_t2i_trainer #job name
#SBATCH --array=1-40%1
#SBATCH --output=exp_wenkunh/sana_reproduce/600M_512px_cw/slurm_out/%A_%a.out
#SBATCH --gpus-per-node 8
#SBATCH --exclusive

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export LOGLEVEL=INFO
export PATH="/lustre/fsw/portfolios/nvr/users/wenkunh/workspace/anaconda3/envs/dcae/bin:$PATH"
export PYTHONPATH="/lustre/fsw/portfolios/nvr/users/wenkunh/workspace/code/Sana_fork:$PYTHONPATH"

cd /lustre/fsw/portfolios/nvr/users/wenkunh/workspace/code/Sana_fork

read -r -d '' cmd <<EOF
torchrun --nnodes=2 --nproc_per_node=8 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    train_scripts/train.py \
    --config_path="configs/sana_config/512ms/Sana_600M_img512.yaml" \
    --work_dir="output/600M_512px" \
    --name="600M_512px" \
    --resume_from="latest" \
    --report_to="wandb" \
    --debug=false
EOF

srun bash -c "${cmd}"
