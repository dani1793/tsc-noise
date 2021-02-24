python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Crop/Crop_TRAIN.tsv --dataset Crop --noisy_dataset data/UCRArchive_2018/Crop/Crop --noisy_percentage 0.3
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Crop/Crop_TRAIN.tsv --dataset Crop --noisy_dataset data/UCRArchive_2018/Crop/Crop --noisy_percentage 0.5
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Crop/Crop_TRAIN.tsv --dataset Crop --noisy_dataset data/UCRArchive_2018/Crop/Crop --noisy_percentage 0.75

python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace_TRAIN.tsv --dataset SmoothSubspace --noisy_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace --noisy_percentage 0.3
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace_TRAIN.tsv --dataset SmoothSubspace --noisy_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace --noisy_percentage 0.5
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace_TRAIN.tsv --dataset SmoothSubspace --noisy_dataset data/UCRArchive_2018/SmoothSubspace/SmoothSubspace --noisy_percentage 0.75

python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Chinatown/Chinatown_TRAIN.tsv --dataset Chinatown --noisy_dataset data/UCRArchive_2018/Chinatown/Chinatown --noisy_percentage 0.3
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Chinatown/Chinatown_TRAIN.tsv --dataset Chinatown --noisy_dataset data/UCRArchive_2018/Chinatown/Chinatown --noisy_percentage 0.5
python3 tsc-dac-plots.py --orig_dataset data/UCRArchive_2018/Chinatown/Chinatown_TRAIN.tsv --dataset Chinatown --noisy_dataset data/UCRArchive_2018/Chinatown/Chinatown --noisy_percentage 0.75


python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-simple-no-noise

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-simple-0.3-epoch-300-learning-100

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-crop-noise-0-epoch-1000-no-dac-lr-0.1_300_550_750

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-crop-noise-0-epoch-1000-no-dac-lr-0.1_300_550_750

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-crop-noise-0-epoch-1000-no-dac-lr-0.2_pow_0.9_550_750

python3 tsc-dac-plots.py --dataset crop --exp_name 2-lstm-crop-noise-0-epoch-2000-no-dac-lr-0.2_pow_0.9_550_750_1500

python3 tsc-dac-plots.py --dataset crop --exp_name inception-simple-noise-0-epoch-200-no-dac-lr-0.001_pow_0.1_80-1_100-2_120-3_150-4-filter-change

python3 tsc-dac-plots.py --dataset ai_crop --exp_name inception-simple-ai_crop-epoch-300-dac-learning-epoch-10-lr-0.0001-iter1
python3 tsc-dac-plots.py --dataset ai_crop --exp_name inception-simple-ai_crop-epoch-300-dac-learning-epoch-10-lr-0.0001-iter2
python3 tsc-dac-plots.py --dataset ai_crop --exp_name inception-simple-ai_crop-epoch-300-dac-learning-epoch-10-lr-0.0001-iter3
python3 tsc-dac-plots.py --dataset ai_crop --exp_name inception-simple-ai_crop-epoch-300-dac-learning-epoch-10-lr-0.0001-iter4
python3 tsc-dac-plots.py --dataset ai_crop --exp_name inception-simple-ai_crop-epoch-300-dac-learning-epoch-10-lr-0.0001-iter5