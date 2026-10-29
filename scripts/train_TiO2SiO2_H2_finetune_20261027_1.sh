#conda activate liflow
export OMP_NUM_THREADS=1

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
#export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=1

# python -m liflow.experiment.train \
#     task=propagate \
#     name=P_TiO2SiO2_H2_20260806_1_finetune_test \
#     data.data_path=../data/data/small_H2_300K_nvt \
#     data.index_files="[train_1000ps_20260728_1.csv]" \
#     data.train_valid_split=True \
#     data.sample_weight_comp=False \
#     data.time_delay_steps=50 \
#     data.in_memory=True \
#     propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
#     propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
#     data.batch_size=128 \
#     data.num_train_samples=4000 \
#     data.num_valid_samples=400 \
#     model.pretrained_ckpt="ckpt/P_TiO2SiO2_H2_20260804_2.ckpt" \
#     +active_learning_weight=True

python -m liflow.experiment.train \
    task=propagate \
    name=P_TiO2SiO2_H2_20261027_1_finetune \
    data.data_path=../data/data/small_H2_300K_nvt \
    data.index_files="[train_1000ps_20260728_1.csv]" \
    data.train_valid_split=True \
    data.sample_weight_comp=False \
    data.time_delay_steps=50 \
    data.in_memory=True \
    propagate_prior.class_name=AdaptiveMaxwellBoltzmannPrior \
    propagate_prior.params.scale="[[36.0,36.0],[0.001,0.001]]" \
    optimizer.lr=3e-06 \
    data.batch_size=128 \
    model.pretrained_ckpt="ckpt/P_TiO2SiO2_H2_20260804_2.ckpt" \
    #+active_learning_weight=True
    

