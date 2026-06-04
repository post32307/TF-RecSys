import world
import dataloader
import model
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'dummy_small', 'dummy_large', 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions','yelp']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("Test Topks:", world.topks)
print('===========end===================')

MODELS = {
    'pse-model': model.PSEmodel,
}