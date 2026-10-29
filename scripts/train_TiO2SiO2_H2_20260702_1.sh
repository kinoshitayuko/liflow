#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260702_1_default \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260513.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[24.0,24.0],[0.01,0.01]]" \
    optimizer.lr=3e-06 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    model.r_max=12.0 \

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260702_2_default \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260513.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[24.0,24.0],[0.01,0.01]]" \
    optimizer.lr=3e-06 \
    data.batch_size=64 \
    data.num_train_samples=40000 \
    model.r_max=12.0 \

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260702_3_default \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260513.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[24.0,24.0],[0.01,0.01]]" \
#     optimizer.lr=3e-04 \
#     data.batch_size=256 \
#     data.num_train_samples=40000

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260702_4_extract \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260518_3.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[24.0,24.0],[0.01,0.01]]" \
#     optimizer.lr=3e-06
#     #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
#     #trainer.callbacks.early_stopping.patience=5

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260610_6 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260518_3.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[24.0,24.0],[0.5,0.5]]" \
#     optimizer.lr=3e-06
#     #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
#     #trainer.callbacks.early_stopping.patience=5

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260610_7 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[24.0,24.0],[0.01,0.01]]" \
#     optimizer.lr=3e-06
#     #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
#     #trainer.callbacks.early_stopping.patience=5

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260610_8 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[24.0,24.0],[0.5,0.5]]" \
#     optimizer.lr=3e-06
#     #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
#     #trainer.callbacks.early_stopping.patience=5

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.5_20260610_1 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.class_name=NormalPrior \
#     correct_prior.params.scale=0.5 \
#     correct_noise.params.scale=0.5 \
#     optimizer.lr=3e-06

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.3_20260610_1 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.class_name=NormalPrior \
#     correct_prior.params.scale=0.3 \
#     correct_noise.params.scale=0.3 \
#     optimizer.lr=3e-06

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.5_20260610_2 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.class_name=NormalPrior \
#     correct_prior.params.scale=0.3 \
#     correct_noise.params.scale=0.5 \
#     optimizer.lr=3e-06

# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.3_20260610_2 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260608_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.class_name=NormalPrior \
#     correct_prior.params.scale=0.5 \
#     correct_noise.params.scale=0.3 \
#     optimizer.lr=3e-06