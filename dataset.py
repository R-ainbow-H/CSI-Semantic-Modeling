import os
import numpy as np
import glob
import scipy.io as sio
import torch

from torch.utils.data import Dataset, DataLoader



# ============================
# UT_HAR Dataset
# ============================

def UT_HAR_dataset(root_dir):

    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')

    WiFi_data = {}

    for data_dir in data_list:

        data_name = os.path.basename(data_dir).split('.')[0]

        with open(data_dir, 'rb') as f:
            data = np.load(f)

            data = data.reshape(
                len(data),
                1,
                250,
                90
            )

            data_norm = (
                data - np.min(data)
            ) / (
                np.max(data)-np.min(data)
            )

        WiFi_data[data_name] = torch.Tensor(data_norm)



    for label_dir in label_list:

        label_name = os.path.basename(label_dir).split('.')[0]

        with open(label_dir, 'rb') as f:
            label = np.load(f)

        WiFi_data[label_name] = torch.Tensor(label)


    return WiFi_data




# ==========================================
# NTU-Fi HAR Dataset
# ==========================================


class CSI_Dataset(Dataset):

    """
    NTU-Fi HAR dataset

    Directory:

    train_amp
    |
    |-- box
    |     xxx.mat
    |
    |-- circle
    |     xxx.mat
    |
    ...
    """


    def __init__(
            self,
            root_dir,
            modal='CSIamp',
            transform=None,
            few_shot=False,
            k=5,
            single_trace=True):


        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform


        # all mat files

        self.data_list = glob.glob(
            root_dir + '/*/*.mat'
        )


        # class folders

        self.folder = glob.glob(
            root_dir + '/*/'
        )


        # ==========================
        # category mapping
        # ==========================

        self.category = {

            os.path.basename(
                os.path.normpath(folder)
            ): idx

            for idx, folder in enumerate(self.folder)

        }



        print("==============================")
        print("CSI Dataset Information")
        print("==============================")

        print("Root:", root_dir)

        print("Categories:")
        print(self.category)

        print("Total samples:",
              len(self.data_list))

        print("==============================")





    def __len__(self):

        return len(self.data_list)





    def __getitem__(self, idx):


        if torch.is_tensor(idx):

            idx = idx.tolist()



        sample_dir = self.data_list[idx]



        # ==========================
        # get label
        # ==========================

        class_name = os.path.basename(

            os.path.dirname(sample_dir)

        )


        y = self.category[class_name]



        # ==========================
        # load CSI
        # ==========================

        x = sio.loadmat(sample_dir)[self.modal]



        # normalize

        x = (
            x - 42.3199
        ) / 4.9802



        # sampling

        # 2000 -> 500

        x = x[:, ::4]



        # reshape

        x = x.reshape(
            3,
            114,
            500
        )



        if self.transform:

            x = self.transform(x)



        x = torch.FloatTensor(x)



        return x, y







# ==========================================
# Widar Dataset
# ==========================================


class Widar_Dataset(Dataset):


    def __init__(self, root_dir):


        self.root_dir = root_dir


        self.data_list = glob.glob(
            root_dir+'/*/*.csv'
        )


        self.folder = glob.glob(
            root_dir+'/*/'
        )



        self.category = {

            os.path.basename(
                os.path.normpath(folder)
            ):idx

            for idx, folder in enumerate(self.folder)

        }




    def __len__(self):

        return len(self.data_list)





    def __getitem__(self, idx):


        if torch.is_tensor(idx):

            idx = idx.tolist()



        sample_dir = self.data_list[idx]



        class_name = os.path.basename(

            os.path.dirname(sample_dir)

        )


        y = self.category[class_name]



        x = np.genfromtxt(
            sample_dir,
            delimiter=','
        )



        # normalize

        x = (
            x-0.0025
        )/0.0119



        # reshape

        x = x.reshape(
            22,
            20,
            20
        )



        x = torch.FloatTensor(x)



        return x, y