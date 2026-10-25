# TF-RecSys
Training-Free Recommendations


[ Instructions for Running Base Model ]

1. Create Environment:
conda env create -f ~/TF-RecSys/lgn.yml
conda activate <environment-name>

2. Go to directory:
cd ~/TF-RecSys/AggregationModel/code

3. Run base model:
python main.py --ensemble_models EASE SGFCF --M 4 --dataset amazon-electronics --seed 2020 --gpu_ID 0 --params_filename anonymous

4. Find results:
The run saves its results in ~/TF-RecSys/AggregationModel/code/etc/anonymous.txt