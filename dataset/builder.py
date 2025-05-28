from utils import Registry

DATASET = Registry("dataset")

def build_dataset(cfg):
    return DATASET.build(cfg)