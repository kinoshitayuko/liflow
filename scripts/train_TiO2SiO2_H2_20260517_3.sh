#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260517_3 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260517.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[10.0,10.0],[0.01,0.01]]" \
    optimizer.lr=3e-04
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.1_20260517_3 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260517.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_noise.params.scale=0.1 \
    optimizer.lr=3e-04

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260517_3 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260517.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-04

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_1.0_20260517_3 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260517.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_noise.params.scale=1.0 \
    optimizer.lr=3e-04

