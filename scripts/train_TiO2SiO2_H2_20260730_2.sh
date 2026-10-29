#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260730_7 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-04 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260730_8 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-06 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260730_7 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-04 \

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260730_8 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-06 \

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260730_9 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_2.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-04 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260730_10 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_2.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-06 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260730_9 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_2.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-04 \

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260730_10 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_2.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-06 \

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260730_11 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_3.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-04 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260730_12 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_3.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-06 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260730_11 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_3.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-04 \

python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20260730_12 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1_weighted_3.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-06 \
