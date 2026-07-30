cd code &&
python main.py --dataset="gowalla" --model="LAE_DAN" --reg_p=20 --alpha=0.2 --beta=0.3 --drop_p=0.5 &&
python main.py --dataset="yelp2018" --model="LAE_DAN" --reg_p=50 --alpha=0.3 --beta=0.2 --drop_p=0.7 &&
python main.py --dataset="abook" --model="LAE_DAN" --reg_p=10 --alpha=0.2 --beta=0.2 --drop_p=0.2
