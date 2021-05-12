import os
from torchvision import transforms as tf
from PIL import Image
from  torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import math
class TrainSet(Dataset):
    def __init__(self,datadir=os.path.join('../usedata/', 'our480/')):
        super().__init__()
        self.datadir=datadir

        self.highdir=os.path.join(self.datadir,'high/')
        self.lowdir = os.path.join(self.datadir, 'low/')


        self.high_illumination_dir=os.path.join(self.highdir,'illumination/')
        self.high_orginal_dir = os.path.join(self.highdir, 'orginal/')
        self.high_reflectance_dir = os.path.join(self.highdir, 'reflectance/')

        self.low_illumination_dir=os.path.join(self.lowdir,'illumination/')
        self.low_orginal_dir = os.path.join(self.lowdir, 'orginal/')
        self.low_reflectance_dir = os.path.join(self.lowdir, 'reflectance/')


    def __len__(self):
        self.datanum=len(os.listdir(self.high_orginal_dir))
        return self.datanum

    def __getitem__(self, idx):
        self.res={}

        self.imghigh_orginal_path=self.high_orginal_dir+os.listdir(self.high_orginal_dir)[idx]
        self.imglow_orginal_path = self.low_orginal_dir + os.listdir(self.low_orginal_dir)[idx]
        self.imghigh_orginal=tf.ToTensor()(Image.open(self.imghigh_orginal_path))
        self.imglow_orginal=tf.ToTensor()(Image.open(self.imglow_orginal_path))


        self.imghigh_illumination_path=self.high_illumination_dir+os.listdir(self.high_illumination_dir)[idx]
        self.imglow_illumination_path = self.low_illumination_dir + os.listdir(self.low_illumination_dir)[idx]
        self.imghigh_illumination=tf.ToTensor()(Image.open(self.imghigh_illumination_path))
        self.imglow_illumination=tf.ToTensor()(Image.open(self.imglow_illumination_path))

        self.imghigh_reflectance_path=self.high_reflectance_dir+os.listdir(self.high_reflectance_dir)[idx]
        self.imglow_reflectance_path = self.low_reflectance_dir + os.listdir(self.low_reflectance_dir)[idx]
        self.imghigh_reflectance=tf.ToTensor()(Image.open(self.imghigh_reflectance_path))
        self.imglow_reflectance=tf.ToTensor()(Image.open(self.imglow_reflectance_path))

        self.res['illumination']=self.imglow_illumination,self.imghigh_illumination

        self.res['reflection']=self.imglow_reflectance,self.imghigh_reflectance

        self.res['orginal']=self.imglow_orginal,self.imghigh_orginal


        return self.res



class TestSet(Dataset):
    def __init__(self,datadir='../usedata/'+'eval15/'):
        super().__init__()
        self.datadir=datadir

        self.highdir=os.path.join(self.datadir,'high/')
        self.lowdir = os.path.join(self.datadir, 'low/')


        self.high_illumination_dir=os.path.join(self.highdir,'illumination/')
        self.high_orginal_dir = os.path.join(self.highdir, 'orginal/')
        self.high_reflectance_dir = os.path.join(self.highdir, 'reflectance/')

        self.low_illumination_dir=os.path.join(self.lowdir,'illumination/')
        self.low_orginal_dir = os.path.join(self.lowdir, 'orginal/')
        self.low_reflectance_dir = os.path.join(self.lowdir, 'reflectance/')


    def __len__(self):
        self.datanum=len(os.listdir(self.low_orginal_dir))
        return self.datanum

    def __getitem__(self, idx):
        self.res = {}

        self.imghigh_orginal_path = self.high_orginal_dir + os.listdir(self.high_orginal_dir)[idx]
        self.imglow_orginal_path = self.low_orginal_dir + os.listdir(self.low_orginal_dir)[idx]
        self.imghigh_orginal = tf.ToTensor()(Image.open(self.imghigh_orginal_path))
        self.imglow_orginal = tf.ToTensor()(Image.open(self.imglow_orginal_path))

        self.imghigh_illumination_path = self.high_illumination_dir + os.listdir(self.high_illumination_dir)[idx]
        self.imglow_illumination_path = self.low_illumination_dir + os.listdir(self.low_illumination_dir)[idx]
        self.imghigh_illumination = tf.ToTensor()(Image.open(self.imghigh_illumination_path))
        self.imglow_illumination = tf.ToTensor()(Image.open(self.imglow_illumination_path))

        self.imghigh_reflectance_path = self.high_reflectance_dir + os.listdir(self.high_reflectance_dir)[idx]
        self.imglow_reflectance_path = self.low_reflectance_dir + os.listdir(self.low_reflectance_dir)[idx]
        self.imghigh_reflectance = tf.ToTensor()(Image.open(self.imghigh_reflectance_path))
        self.imglow_reflectance = tf.ToTensor()(Image.open(self.imglow_reflectance_path))

        self.res['illumination'] = self.imglow_illumination, self.imghigh_illumination

        self.res['reflection'] = self.imglow_reflectance, self.imghigh_reflectance

        self.res['orginal'] = self.imglow_orginal, self.imghigh_orginal
        return self.res

class HatSet(Dataset):
    def __init__(self, datadir='../trainhat/'):
        super().__init__()
        self.datadir = datadir

        self.illdir = os.path.join(self.datadir, 'ill_hat/')
        self.recdir = os.path.join(self.datadir, 'rec_hat/')
        self.orginalhighdir=os.path.join(self.datadir,'orginal/')

    def __len__(self):
        self.datanum = len(os.listdir(self.illdir))
        return self.datanum

    def __getitem__(self, idx):
        self.res = {}

        self.imgill_path = self.illdir + os.listdir(self.illdir)[idx]
        self.imgrec_path = self.recdir + os.listdir(self.recdir)[idx]
        self.imgorg_path=self.orginalhighdir+os.listdir(self.orginalhighdir)[idx]
        self.imgill = tf.ToTensor()(Image.open(self.imgill_path))
        self.imgrec = tf.ToTensor()(Image.open(self.imgrec_path))
        self.org=tf.ToTensor()(Image.open(self.imgorg_path))



        self.res['rec']=self.imgrec
        self.res['ill']=self.imgill
        self.res['org']=self.org
        return self.res
