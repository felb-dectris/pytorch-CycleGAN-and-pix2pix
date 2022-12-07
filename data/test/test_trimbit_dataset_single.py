import pytest as pytest

from data import create_dataset
from data.trimbit_dataset import TrimbitDataset
from data.trimbit_dataset_single import TrimbitDatasetSingle
from options.train_options import TrainOptions


@pytest.fixture
def opt():
    opt = TrainOptions()
    opt.dataset_mode = 'trimbit'
    opt.dataframe_name = 'df_1.3.5.pickle'
    opt.dataroot = '/Users/felix.bachmair/sdvlp/CAS_BDAI_Project/data'
    opt.phase = 'train'
    opt.max_dataset_size =  float("inf")
    opt.preprocess = 'none'
    opt.no_flip = True
    opt.batch_size = 1
    opt.serial_batches = True
    opt.num_threads=0
    opt.grayscale=True
    opt.num_workers=0
    opt.prediction_item=3
    return opt
    opt.prepare_parser()

    opt.gather_options()
    return opt

def test_create_dataset(opt):
    dataset = create_dataset(opt)
    #print(dataset.get(0))
    for d in dataset:
        print(d)

@pytest.fixture
def trimbit_dataset(opt):
    return TrimbitDatasetSingle(opt)

def test_init(trimbit_dataset):
    assert isinstance(trimbit_dataset, TrimbitDatasetSingle)

def test_get_item(trimbit_dataset):
    for i in range(4):
        data = trimbit_dataset[i]
        print(data['A'].shape)


def test_get_length(trimbit_dataset):
    assert len(trimbit_dataset) == 474 * 4
