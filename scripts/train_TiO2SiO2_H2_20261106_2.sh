#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=2

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20261106_test_3 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=GaussianMixtureAdaptivePrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-07 \
    data.batch_size=128 \
    model.r_max=5.0 \
    data.cutoff=5.0 \
    +signed_log.enabled=True \
    +signed_log.parameters_path="$PWD/../data/data/small_H2_300K_nvt/positions_300K_all_species_lambda0p1_lag50f_parameters.json"
    # +active_learning_weight=False

    # data.num_train_samples=4000 \
    # data.num_valid_samples=400 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5


python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20261106_test_4 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=GaussianMixtureAdaptivePrior \
    propagate_prior.params.scale="[[36.0,36.0],[3.0,3.0]]" \
    optimizer.lr=3e-07 \
    data.batch_size=128 \
    model.r_max=5.0 \
    data.cutoff=5.0 \
    +signed_log.enabled=True \
    +signed_log.parameters_path="$PWD/../data/data/small_H2_300K_nvt/positions_300K_all_species_lambda0p1_lag50f_parameters.json"
    # +active_learning_weight=False

    # data.num_train_samples=4000 \
    # data.num_valid_samples=400 \
    #propagate_prior.params.scale="[[1.0,10.0],[0.5,0.5]]"
    #trainer.callbacks.early_stopping.patience=5



python -m liflow.experiment.train \
    task=correct \
    name=C_TiO2SiO2_H2_0.5_20261106_2 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    correct_prior.class_name=NormalPrior \
    correct_prior.params.scale=0.5 \
    correct_noise.params.scale=0.5 \
    optimizer.lr=3e-05 \
    data.batch_size=128 \
    model.r_max=5.0 \
    data.cutoff=5.0 \
    +signed_log.enabled=True \
    +signed_log.parameters_path="$PWD/../data/data/small_H2_300K_nvt/positions_300K_all_species_lambda0p1_lag50f_parameters.json"




# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20261029_2 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260728_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=GaussianMixtureAdaptivePrior \
#     propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
#     optimizer.lr=3e-06 \
#     data.batch_size=128 \
#     model.r_max=5.0 \
#     data.cutoff=5.0 \
#     +active_learning_weight=True





# python -m liflow.experiment.train \
#     task=correct \
#     name=C_TiO2SiO2_H2_0.5_20261029_1 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260728_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     correct_prior.class_name=NormalPrior \
#     correct_prior.params.scale=0.5 \
#     correct_noise.params.scale=0.5 \
#     optimizer.lr=3e-04 \
#     data.batch_size=128 \
#     model.r_max=5.0 \
#     data.cutoff=5.0


# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20261014_2 \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260728_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=GaussianMixtureAdaptivePrior \
#     propagate_prior.params.scale="[[36.0,36.0],[0.5,0.5]]" \
#     optimizer.lr=3e-06 \
#     data.batch_size=128 \
#     model.r_max=5.0 \
#     data.cutoff=5.0