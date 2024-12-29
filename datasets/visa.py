import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T3
# TODOs - 
# Dataset paths
# 




# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']


class ViSA(Dataset):
    # TODO: Check the dataset path
    def __init__(self, dataset_path='D:/dataset/VisA', class_name='cashew', is_train=True,
                 resize=256, cropsize=224):
        # assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        self.x, self.y, self.mask = self.load_dataset_folder()

        # set transforms
        self.transform_x = T3.Compose([T3.Resize(resize, Image.LANCZOS),
                                      T3.CenterCrop(cropsize),
                                      T3.ToTensor(),
                                      T3.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T3.Compose([T3.Resize(resize, Image.NEAREST),
                                         T3.CenterCrop(cropsize),
                                         T3.ToTensor()])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if y == 0:
            mask = torch.zeros([1, self.cropsize, self.cropsize])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, y, mask

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        # Dataset is not split with test/train
        x, y, mask = [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, 'Data', 'Images')
        gt_dir = os.path.join(img_dir, 'Normal')
        ab_dir =  os.path.join(img_dir, 'Anomaly')
        ms_dir =  os.path.join(self.dataset_path, self.class_name, 'Data', 'Masks')
        img_types = sorted(os.listdir(img_dir))

        img_good = sorted(os.listdir(gt_dir))
        img_ab = sorted(os.listdir(ab_dir))
        masks = sorted(os.listdir(ms_dir))

        y = [0] * len(img_good)
        y.extend([1] * len(img_ab))

        if not os.path.isdir(img_dir):
            return
        for type in ['Anomaly', 'Normal']:
            type_dir = os.path.join(img_dir, type)
            img_fpath_list = sorted([os.path.join(type_dir, f)
                                        for f in os.listdir(type_dir)
                                        if f.endswith('.JPG')])
            x.extend(img_fpath_list)


        assert len(x) == len(y), 'number of x and y should be same'

        return list(), list(y), list(masks)

#     def download(self):
#         """Download dataset if not exist"""

#         if not os.path.exists(self.mvtec_folder_path):
#             tar_file_path = self.mvtec_folder_path + '.tar.xz'
#             if not os.path.exists(tar_file_path):
#                 download_url(URL, tar_file_path)
#             print('unzip downloaded dataset: %s' % tar_file_path)
#             tar = tarfile.open(tar_file_path, 'r:xz')
#             tar.extractall(self.mvtec_folder_path)
#             tar.close()

#         return


# class DownloadProgressBar(tqdm):
#     def update_to(self, b=1, bsize=1, tsize=None):
#         if tsize is not None:
#             self.total = tsize
#         self.update(b * bsize - self.n)


# def download_url(url, output_path):
#     with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
#         urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
