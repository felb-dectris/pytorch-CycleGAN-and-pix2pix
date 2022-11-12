"""Dataset class for Trimbits

This module provides a implementation for a EIGER2 MCU trimbit  datasets.
You can specify '--dataset_mode trimbit' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.

You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import functools
import json
import os
import pickle

import numpy
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
from data.image_folder import make_dataset, is_image_file


class TrimbitPandasDfDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--prediction_item', type=int, default=3, help='new dataset option')
        parser.add_argument('--dataframe_name', type=str, default='trimbits_df.pickle', help='new dataset option')
        parser.set_defaults(
                            prediction_item=3,
                            dataroot='/Users/felix.bachmair/sdvlp/CAS_BDAI_Project/data')
        # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.data_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # get the image paths of your dataset;
        # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        try:
            self.keys = sorted(self.make_dataset(self.data_dir, max_dataset_size=opt.max_dataset_size,))
        except:
            self.data_dir = os.path.join(opt.dataroot)  # get the image directory
            self.keys = sorted(self.make_dataset(self.data_dir, max_dataset_size=opt.max_dataset_size, ))
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt, grayscale=True)
        self.prediction_item = opt.prediction_item
        self.opt = opt


    def make_dataset(self,dir, pickle_filename=None,max_dataset_size=None, energies=None):
            print(energies)
            if pickle_filename is None:
                pickle_filename = self.opt.dataframe_name
            if energies is None:
                energies=[8000]
            self.energies=energies
            if max_dataset_size is None:
                max_dataset_size = float("inf")
            pickle_fpath = os.path.join(dir, pickle_filename)
            self.pickle_path = pickle_fpath
            df = self.get_pickle(pickle_fpath)
            images = df

            return images.iloc[:min(max_dataset_size, len(images))]

    def get_pickle(self, pickle_fpath):
        try:
            with open(pickle_fpath, 'rb') as f:
                print(f'load pickle: {pickle_fpath}')
                df = pickle.load(f)
                self.df = df
            return df
        except AttributeError:
            import pickle5
            with open(pickle_fpath, 'rb') as f:
                print(f'load pickle: {pickle_fpath}')
                df = pickle5.load(f)
                self.df = df
            return df

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """


        series = self.df.iloc[index]
        images = {}
        for key in self.keys:
            keys = key.split('_')
            if len(keys) != 3:
                continue
            _, vrfp, i = keys
            i = int(i)
            vrfp = int(vrfp)
            if not vrfp in self.energies:
                continue
            img = Image.fromarray(series[key][2].reshape(256,1024))
            images[i] = img
        B_img = images.pop(self.prediction_item)

        B = self.transform(B_img)
        A = torch.cat([self.transform(x) for x in images.values()])
        return {'A': A, 'B': B, 'A_paths': self.pickle_path, 'B_paths': self.pickle_path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.df)

