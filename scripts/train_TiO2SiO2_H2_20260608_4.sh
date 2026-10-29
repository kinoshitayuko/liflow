#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260608_6 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260608_1.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-04

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.5_20260608_7 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.params.scale=0.5 \
#     correct_noise.params.scale=2.0 \
#     optimizer.lr=3e-04

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.5_20260608_8 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.params.scale=0.5 \
#     correct_noise.params.scale=3.0 \
#     optimizer.lr=3e-04
