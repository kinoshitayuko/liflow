export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

python -m liflow.experiment.train \
    task=propagate \
    name=P_LGPS_20260626_1 \
    data.data_path=../data/data/LGPS \
    data.index_files="[train_150ps_20260514.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=67 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[10.0,10.0],[0.5,0.5]]"

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_LGPS_0.1_20260626 \
#     data.data_path=../data/data/LGPS \
#     data.index_files="[train_150ps_20260514.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=67 \
#     data.in_memory=True \
#     correct_noise.params.scale=0.1

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_LGPS_0.2_20260626 \
#     data.data_path=../data/data/LGPS \
#     data.index_files="[train_150ps_20260514.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=67 \
#     data.in_memory=True \
#     correct_noise.params.scale=0.2
