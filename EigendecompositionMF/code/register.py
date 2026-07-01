import world
import dataloader
import model
import os
from pathlib import Path
from pprint import pprint

DATA_ROOT = Path(os.environ.get(
    "TFRECSYS_DATA_ROOT",
    Path(__file__).resolve().parents[1] / "data" / f"00_dataset_seed_{world.seed}",
))

if world.dataset in ['lastfm', 'gowalla', 'yelp2018', 'amazon-book', 'dummy_small', 'dummy_large', 'amazon-CD','amazon-electronics','citeulike','movielens', 'news', 'douban-book','epinions','yelp']:
    dataset = dataloader.Loader(path=str(DATA_ROOT / world.dataset))

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("Test Topks:", world.topks)
print('===========end===================')

MODELS = {
    'pse-model': model.PSEmodel,
}
