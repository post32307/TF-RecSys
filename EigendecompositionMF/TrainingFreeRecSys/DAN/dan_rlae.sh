cd code &&
python main.py --dataset="gowalla" --model="RLAE_DAN" --reg_p=20 --alpha=0.2 --beta=0.3 --drop_p=0.6 --xi=0.1 &&
python main.py --dataset="yelp2018" --model="RLAE_DAN" --reg_p=20 --alpha=0.3 --beta=0.3 --drop_p=0.7 --xi=0.1 &&
python main.py --dataset="abook" --model="RLAE_DAN" --reg_p=10 --alpha=0.2 --beta=0.2 --drop_p=0.4--xi=0.3
