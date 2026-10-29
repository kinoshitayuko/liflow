#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=3

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20260721_2_hex825 \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260513.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.01,0.01]]" \
    optimizer.lr=3e-06 \
    data.batch_size=16 \
    data.num_train_samples=40000 \
    data.num_valid_samples=4000 \
    # model.r_max=12.0 \
    # data.cutoff=12.0 \

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260716_2_default \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260513.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[36.0,36.0],[0.01,0.01]]" \
#     optimizer.lr=3e-06 \
#     data.batch_size=16 \
#     data.num_train_samples=4000 \
#     data.num_valid_samples=400 \

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260716_3_default \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260513.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[36.0,36.0],[0.01,0.01]]" \
#     optimizer.lr=3e-04 \
#     data.batch_size=16 \
#     data.num_train_samples=40000 \
#     data.num_valid_samples=4000 \

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260716_4_default \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260513.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[36.0,36.0],[0.01,0.01]]" \
#     optimizer.lr=3e-06 \
#     data.batch_size=16 \
#     data.num_train_samples=40000 \
#     data.num_valid_samples=4000 \