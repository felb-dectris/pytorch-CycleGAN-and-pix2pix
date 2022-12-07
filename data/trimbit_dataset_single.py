import torch

from data.trimbit_dataset import TrimbitDataset
from data.trimbit_pandas_df_dataset import TrimbitPandasDfDataset


class TrimbitDatasetSingle(TrimbitPandasDfDataset):
    n_chips = 4

    def __getitem__(self, index):
        orig_index = index // self.n_chips
        chip_id = index % self.n_chips
        data = super(TrimbitDatasetSingle,self).__getitem__(orig_index)
        data['chip_id'] = chip_id
        data['orig_index'] = orig_index
        for k,v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v[:6,:256,chip_id*256:(chip_id+1)*256]
        return data

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)*self.n_chips
