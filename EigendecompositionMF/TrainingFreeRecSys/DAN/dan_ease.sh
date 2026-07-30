cd code &&
python main.py --dataset="gowalla" --model="EASE_DAN" --reg_p=20 --alpha=0.2 --beta=0.3 --drop_p=0.6 &&
python main.py --dataset="yelp2018" --model="EASE_DAN" --reg_p=20 --alpha=0.3 --beta=0.3 --drop_p=0.7 &&
python main.py --dataset="abook" --model="EASE_DAN" --reg_p=10 --alpha=0.2 --beta=0.2 --drop_p=0.4
