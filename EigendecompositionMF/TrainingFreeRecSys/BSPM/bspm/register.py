import os
import world
import dataloader
import model
import utils
from pathlib import Path
from pprint import pprint

DATA_ROOT = Path(os.environ.get("TFRECSYS_DATA_ROOT", Path(__file__).resolve().parents[3] / "data"))

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'dummy_small', 'dummy_large', 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions','yelp']:
    dataset = dataloader.Loader(path=str(DATA_ROOT / world.dataset))
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM(path=str(DATA_ROOT / "lastfm"))

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}
