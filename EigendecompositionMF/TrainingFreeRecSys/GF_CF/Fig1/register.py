import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'dummy_small', 'dummy_large', 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions','yelp']:
    dataset = dataloader.Loader(path="/home/hakbeom/MainCodes/TF-RecSys/EigendecompositionMF/data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM(path="/home/hakbeom/MainCodes/TF-RecSys/EigendecompositionMF/data/lastfm")

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