import world
import dataloader
import model
from pprint import pprint


dataset = dataloader.Loader(path="../data/" + world.dataset)


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("Test Topks:", world.topks)
print('===========end===================')


MODELS = {
    'EASE': model.EASE,
    'LAE_DAN': model.LAE_DAN,
    'EASE_DAN': model.EASE_DAN,
    'RLAE_DAN': model.RLAE_DAN
}