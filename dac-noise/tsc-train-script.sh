#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 01:48:36 2020

@author: daniyalusmani1
"""

#simple without experiment name
#python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-resnet --noise_percentage 0.3 --epochs 100 --loss_fn dac_loss --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0

# with experiment name
#python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 5 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-no-noise-epoch-500 --lr 0.9 --output_path results/crop/


python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-dac-0.3-epoch-300 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.5-epoch-300 --noise_percentage 0.5 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.75-epoch-300 --noise_percentage 0.75 --lr 0.9 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-0.3-epoch-300 --noise_percentage 0.3 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-0.5-epoch-300 --noise_percentage 0.5 --lr 0.9 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 3 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 3-lstm-simple-0.75-epoch-300 --noise_percentage 0.75 --lr 0.9 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.3-epoch-300-learning-100-dac --loss_fn dac_loss --noise_percentage 0.3 --learn_epochs 100 --lr 0.9 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-simple-0.3-epoch-300-learning-100-dac-lr- --loss_fn dac_loss --noise_percentage 0.3 --learn_epochs 100 --lr 0.9 --output_path results/crop/


# supports gpu sinteractive --gres=gpu:1 --time=6:00:00 --mem=20G -c 3
# srun --gres=gpu:1 --time=6:00:00 --mem=20G -c 3
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0 --loss_fn dac_loss --noise_percentage 0.0 --learn_epochs 100 --lr 0.9 --output_path results/crop/


python3 tsc-train.py --datadir data/ai_crop --dataset crop_tsc_balanced_imputed_2015.csv  --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name inception-simple-crop_tsc_balanced_imputed_2015-noise-0-lr-0.9 --loss_fn dac_loss --noise_percentage 0.0 --learn_epochs 100 --lr 0.9 --output_path results/ai_crop/


python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128  --save_train_scores --save_val_scores --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-no-dac --noise_percentage 0 --lr 0.9 --output_path results/crop/

#kfold-5 crop database lstm
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-no-dac-kfold-5-lr-0.1 --noise_percentage 0 --lr 0.1 --output_path results/crop/ --kfold 5

CUDA_LAUNCH_BLOCKING=1 python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-no-dac-kfold-5-lr-0.1 --noise_percentage 0 --lr 0.1 --output_path results/crop/ --kfold 5


python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-2000-no-dac-lr-0.2_pow_0.9_550_750_1500 --noise_percentage 0 --lr 0.2 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-2000-no-dac-lr-0.2_pow_0.9_300_550_750_1500 --noise_percentage 0 --lr 0.2 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-2000-no-dac-lr-0.2_pow_0.9_300_550_750_950_1200_1500_1700_1900 --noise_percentage 0 --lr 0.2 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-2000-no-dac-lr-0.001 --noise_percentage 0 --lr 0.001 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-2000-no-dac-lr-0.001_pow_0.1_550_750_950_1200_1500_1700_1900 --noise_percentage 0 --lr 0.001 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-3000-no-dac-lr-0.001_pow_0.1_550_750_950_1200_1500_1700_1900 --noise_percentage 0 --lr 0.001 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 1500 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0-epoch-1500-no-dac-lr-0.001_pow_0.2_550_750_950_1200_1500 --noise_percentage 0 --lr 0.001 --output_path results/crop/

# Crop LSTM with noise

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-3000-dac-learning-epoch-800-lr-0.001_pow_0.2_550_750_950_1200_1500_1700_1900 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 800 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 1500 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-1500-dac-learning-epoch-800-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 800 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 1500 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-1500-dac-learning-epoch-800-lr-0.001 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 800 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 1500 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-1500-dac-learning-epoch-800-lr-0.001-testing --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 800 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 800 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-800-dac-learning-epoch-400-lr-0.001-testing --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-2000-dac-learning-epoch-400-lr-0.001-testing --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-2000-dac-learning-epoch-400-lr-0.001_pow_0.2_950_1700 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 2000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-2000-dac-learning-epoch-400-lr-0.01_pow_0.2_950_1700 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.01 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-3000-dac-learning-epoch-400-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-3000-dac-learning-epoch-400-lr-0.001 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 800 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-800-dac-learning-epoch-200-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 200 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-3000-dac-learning-epoch-200-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 200 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-3000-dac-learning-epoch-200-lr-0.001 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 200 --output_path results/crop/
# have to plot after this
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-3000-dac-learning-epoch-40-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-3000-dac-learning-epoch-40-lr-0.001 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-3000-dac-learning-epoch-40-lr-0.001 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 40 --output_path results/crop/
# after noise dataset update
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 3000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-3000-dac-learning-epoch-200-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 200 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/

#iter no-noise
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-epoch-5000-dac-learning-epoch-40-lr-0.001 --loss_fn dac_loss --lr 0.001 --learn_epochs 40 --output_path results/crop/
#iter-1 0.3
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter3 --iteration 3 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter4 --iteration 4 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter5 --iteration 5 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/

# with noisy validation set
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --no_overwrite --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-noisy_val-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/


#iter-1 0.5
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter3 --iteration 3 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter4 --iteration 4 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter5 --iteration 5 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
#iter-1 0.75
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter3 --iteration 3 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter4 --iteration 4 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter5 --iteration 5 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/

#iter 0.5-1 updates best model after learning epochs and at second last epoch
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter1-1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 1000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-1000-dac-learning-epoch-100-lr-0.001-iter1-1-test --iteration 1 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/ --no_overwrite
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 1000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-1000-dac-learning-epoch-100-lr-0.001-iter1-1-test --iteration 1 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/ --no_overwrite --save_epoch_model 998

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-without-abstained-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.5 --lr 0.001 --learn_epochs 100 --output_path results/crop/ --no_overwrite --save_epoch_model 4998

# iter 0.3-1
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-without-abstained-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/ --no_overwrite --save_epoch_model 4998
srun --gres=gpu:1 --time=6:00:00 --mem=20G python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-without-abstained-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.3 --lr 0.001 --learn_epochs 40 --output_path results/crop/ --no_overwrite --save_epoch_model 4998

# iter 0.75-1
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-without-abstained-iter3 --iteration 3 --loss_fn dac_loss --noise_percentage 0.75 --lr 0.001 --learn_epochs 400 --output_path results/crop/ --no_overwrite --save_epoch_model 4998

#inceptionTime
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-5000-dac-learning-epoch-400-lr-0.001  --noise_percentage 0.3  --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-100-lr-0.001_pow_0.5_100-1_150-2  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 100 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-40-lr-0.001_pow_0.5_100-1_150-2  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 40 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-40-lr-0.001  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 40 --output_path results/crop/


# iter-0 noise

# iter-0.3
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-5-lr-0.001-iter1 --iteration 1  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 5 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-5-lr-0.001-iter2 --iteration 2  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 5 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-5-lr-0.001-iter3 --iteration 3  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 5 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-5-lr-0.001-iter4 --iteration 4  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 5 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-5-lr-0.001-iter5 --iteration 5  --loss_fn dac_loss --noise_percentage 0.3  --lr 0.001 --learn_epochs 5 --output_path results/crop/




# iter-0.5
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.5-epoch-300-dac-learning-epoch-10-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.5  --lr 0.1 --learn_epochs 50 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.5-epoch-300-dac-learning-epoch-10-lr-0.001-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.5  --lr 0.001 --learn_epochs 10 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.5-epoch-300-dac-learning-epoch-10-lr-0.001-iter3 --iteration 3 --loss_fn dac_loss --noise_percentage 0.5  --lr 0.001 --learn_epochs 10 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.5-epoch-300-dac-learning-epoch-10-lr-0.001-iter4 --iteration 4 --loss_fn dac_loss --noise_percentage 0.5  --lr 0.001 --learn_epochs 10 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.5-epoch-300-dac-learning-epoch-10-lr-0.001-iter5 --iteration 5 --loss_fn dac_loss --noise_percentage 0.5  --lr 0.001 --learn_epochs 10 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 500 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.5-epoch-500-dac-learning-epoch-150-lr-0.1-iter2 --iteration 2 --loss_fn dac_loss --noise_percentage 0.5  --lr 0.1 --learn_epochs 150 --output_path results/crop/

# iter-0.75
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.75-epoch-300-dac-learning-epoch-15-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.75  --lr 0.001 --learn_epochs 15 --output_path results/crop/
python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0.75-epoch-300-dac-learning-epoch-15-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --noise_percentage 0.75  --lr 0.1 --learn_epochs 50 --output_path results/crop/

# testing model
python3 test-tsc-dac-models.py --datadir data/UCRArchive_2018 --dataset ucr-archive --net_type tsc-lstm --depth 2 --batch_size 128 --test_batch_size 128 --checkpoint_path checkpoint/ucr-archive/tsc-lstm-2x10_expt_name_2-lstm-crop-noise-0.3-epoch-3000-dac-learning-epoch-40-lr-0.001-updated.t7
python3 test-tsc-dac-models.py --datadir data/UCRArchive_2018 --dataset ucr-archive --net_type tsc-lstm --depth 2 --batch_size 128 --test_batch_size 128 --checkpoint_path results/crop/2-lstm-crop-noise-0.5-epoch-1000-dac-learning-epoch-100-lr-0.001-iter1-1-test/checkpoint/tsc-lstm_expt_name_2-lstm-crop-noise-0.5-epoch-1000-dac-learning-epoch-100-lr-0.001-iter1-1-test_epoch_998.t7

python3 test-tsc-dac-models.py --datadir data/UCRArchive_2018 --dataset ucr-archive --net_type tsc-lstm --depth 2 --batch_size 128 --test_batch_size 128 --noise_percentage 0.3 --iteration 2 --checkpoint_path results/crop/2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter2/checkpoint/tsc-lstm_expt_name_2-lstm-crop-noise-0.3-epoch-5000-dac-learning-epoch-40-lr-0.001-iter2.t7
python3 test-tsc-dac-models.py --datadir data/UCRArchive_2018 --dataset ucr-archive --net_type tsc-lstm --depth 2 --batch_size 128 --test_batch_size 128 --noise_percentage 0.5 --iteration 2 --checkpoint_path results/crop/2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter2/checkpoint/tsc-lstm_expt_name_2-lstm-crop-noise-0.5-epoch-5000-dac-learning-epoch-100-lr-0.001-iter2.t7
python3 test-tsc-dac-models.py --datadir data/UCRArchive_2018 --dataset ucr-archive --net_type tsc-lstm --depth 2 --batch_size 128 --test_batch_size 128 --noise_percentage 0.75 --iteration 3 --checkpoint_path results/crop/2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter3/checkpoint/tsc-lstm_expt_name_2-lstm-crop-noise-0.75-epoch-5000-dac-learning-epoch-400-lr-0.001-iter3.t7

python3 test-tsc-dac-models.py --datadir data/UCRArchive_2018 --dataset ucr-archive --net_type inception_simple --depth 2 --batch_size 64 --test_batch_size 64 --checkpoint_path results/crop/inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-50-lr-0.1-iter2/checkpoint/inception-simple_expt_name_inception-simple-crop-noise-0.3-epoch-300-dac-learning-epoch-50-lr-0.1-iter2.t7



python3 tsc-train.py --datadir data/ai_crop --dataset crop_tsc_balanced_imputed_2015.csv --nesterov --net_type inception_simple --depth 2 --epochs 100 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name inception-simple-noise-0-epoch-500-no-dac-lr-0.001_pow_0.1 --noise_percentage 0 --lr 0.001 --output_path results/ai_crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name inception-simple-crop-noise-0  --noise_percentage 0.0  --lr 0.001 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 200 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name inception-simple-noise-0-epoch-200-no-dac-lr-0.001_pow_0.1_80-1_100-2_120-3_150-4-filter-change  --noise_percentage 0.0  --lr 0.001 --output_path results/crop/

python3 tsc-train.py --datadir data/UCRArchive_2018 --dataset ucr-archive --nesterov --net_type inception_simple --depth 2 --epochs 200 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name inception-simple-noise-0-epoch-200-no-dac-lr-0.001_pow_0.5_100-1_150-2-filter-change --noise_percentage 0.0  --lr 0.001 --output_path results/crop/

# ai_crop iterations
#LSTM
#iter-1
python3 tsc-train.py --datadir data/ai_crop --dataset ai_crop --nesterov --net_type tsc-lstm --depth 2 --epochs 5000 --batch_size 128 --test_batch_size 128 --save_best_model --seed 0 --expt_name 2-lstm-ai_crop-epoch-5000-dac-learning-epoch-40-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --lr 0.001 --learn_epochs 40 --output_path results/ai_crop/

# Inception Time
python3 tsc-train.py --datadir data/ai_crop --dataset ai_crop --nesterov --net_type inception_simple --depth 2 --epochs 300 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-ai_crop-epoch-300-dac-learning-epoch-15-lr-0.001-iter1 --iteration 1 --loss_fn dac_loss --lr 0.001 --learn_epochs 15 --output_path results/ai_crop/

python3 tsc-train.py --datadir data/ai_crop --dataset ai_crop --nesterov --net_type inception_simple --depth 2 --epochs 800 --batch_size 64 --test_batch_size 64 --save_best_model --seed 0 --expt_name inception-simple-ai_crop-epoch-800-dac-learning-epoch-10-lr-0.0001-iter3 --iteration 3 --loss_fn dac_loss --lr 0.0001 --learn_epochs 10 --output_path results/ai_crop/



