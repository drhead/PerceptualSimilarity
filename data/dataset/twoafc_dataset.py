import os.path
import torchvision.transforms as transforms
from data.dataset.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import time
from colour import XYZ_to_Oklab, sRGB_to_XYZ
from multiprocessing.shared_memory import SharedMemory
import signal
import atexit
import sys
# from IPython import embed

class TwoAFCDataset(BaseDataset):
    def initialize(self, dataroots, load_size=64, use_cache=False, colorspace='srgb'):
        if(not isinstance(dataroots,list)):
            dataroots = [dataroots,]
        self.roots = dataroots
        self.load_size = load_size
        self.use_cache = use_cache
        self.colorspace = colorspace

        # image directory
        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.ref_paths = make_dataset(self.dir_ref)
        self.ref_paths = sorted(self.ref_paths)

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)
        if self.use_cache:
            self.cache_shape = (len(self.p0_paths), 3, self.load_size, self.load_size)
            self.cache_nbytes = np.ndarray(self.cache_shape, np.float32).nbytes
            self.cache_shape_judge = (len(self.p0_paths), 1, 1, 1)
            self.cache_nbytes_judge = np.ndarray(self.cache_shape_judge, np.float32).nbytes

            try:
                print("Unlinking caches")
                SharedMemory("ref_cache").unlink()
                SharedMemory("p0_cache").unlink()
                SharedMemory("p1_cache").unlink()
                SharedMemory("judge_cache").unlink()
                SharedMemory("index_cache").unlink()
            except:
                pass

            # # Register the cleanup function to be called on termination signals
            # signal.signal(signal.SIGTERM, lambda signum, frame: self.cleanup_shared_memory())
            # signal.signal(signal.SIGINT, lambda signum, frame: self.cleanup_shared_memory())

            # # Ensure cleanup is called if the script exits normally
            # atexit.register(self.cleanup_shared_memory)

            self.shm_ref = SharedMemory("ref_cache", create=True, size=self.cache_nbytes)
            self.ref_cache = np.ndarray(self.cache_shape, np.float32, buffer=self.shm_ref.buf)
            self.ref_cache.fill(0)

            self.shm_p0 = SharedMemory("p0_cache", create=True, size=self.cache_nbytes)
            self.p0_cache = np.ndarray(self.cache_shape, np.float32, buffer=self.shm_p0.buf)
            self.p0_cache.fill(0)

            self.shm_p1 = SharedMemory("p1_cache", create=True, size=self.cache_nbytes)
            self.p1_cache = np.ndarray(self.cache_shape, np.float32, buffer=self.shm_p1.buf)
            self.p1_cache.fill(0)

            self.shm_judge = SharedMemory("judge_cache", create=True, size=self.cache_nbytes_judge)
            self.judge_cache = np.ndarray(self.cache_shape_judge, np.float32, buffer=self.shm_judge.buf)
            self.judge_cache.fill(0)

            self.shm_index = SharedMemory("index_cache", create=True, size=np.ndarray((len(self.p0_paths),), np.bool_).nbytes)
            self.index_cache = np.ndarray((len(self.p0_paths),), np.bool_, buffer=self.shm_index.buf)
            self.index_cache.fill(False)

        # judgement directory
        self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        self.judge_paths = make_dataset(self.dir_J,mode='np')
        self.judge_paths = sorted(self.judge_paths)

    def transform_totensor(self, array: np.ndarray):
        if self.colorspace == 'srgb':
            return (array.transpose((2,0,1)).astype(np.float32) / 127.5 - 1).clip(-1, 1)
        elif self.colorspace == 'oklab':
            oklab_array = XYZ_to_Oklab(sRGB_to_XYZ(array.astype(np.float32) / 255, chromatic_adaptation_transform='Bianco 2010'))
            oklab_array = oklab_array.transpose((2,0,1))
            # colour-science appears to already make a,b as [-1, 1]?
            oklab_array = oklab_array * 2 - 1
            # print(f"min/max: {oklab_array.min(1).min(1)}, {oklab_array.max(1).max(1)}")
            return oklab_array

    # def cleanup_shared_memory(self):
    #     try:
    #         SharedMemory("ref_cache").unlink()
    #         SharedMemory("p0_cache").unlink()
    #         SharedMemory("p1_cache").unlink()
    #         SharedMemory("judge_cache").unlink()
    #         SharedMemory("index_cache").unlink()
    #     except:
    #         pass
    #     time.sleep(3)
    #     sys.exit()

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p1_path = self.p1_paths[index]
        ref_path = self.ref_paths[index]
        judge_path = self.judge_paths[index]

        if not self.use_cache or not self.index_cache[index]:
            p0_img_ = np.array(Image.open(p0_path).convert('RGB').resize((self.load_size, self.load_size), Image.NEAREST))
            p1_img_ = np.array(Image.open(p1_path).convert('RGB').resize((self.load_size, self.load_size), Image.NEAREST))
            ref_img_ = np.array(Image.open(ref_path).convert('RGB').resize((self.load_size, self.load_size), Image.NEAREST))
            # judge_img = (np.load(judge_path)*2.-1.).reshape((1,1,1,)) # [-1,1]
            judge_img = np.load(judge_path).reshape((1,1,1,)) # [0,1]
            p0_img_ = self.transform_totensor(p0_img_)
            p1_img_ = self.transform_totensor(p1_img_)
            ref_img_ = self.transform_totensor(ref_img_)

        if self.use_cache:
            if not self.index_cache[index]:
                self.p0_cache[index] = p0_img_.astype(np.float32)
                self.p1_cache[index] = p1_img_.astype(np.float32)
                self.ref_cache[index] = ref_img_.astype(np.float32)
                self.judge_cache[index] = judge_img.astype(np.float32)
                self.index_cache[index] = True
            else:
                p0_img_ = self.p0_cache[index]
                p1_img_ = self.p1_cache[index]
                ref_img_ = self.ref_cache[index]
                judge_img = self.judge_cache[index]

        p0_img = p0_img_.astype(np.float32)
        p1_img = p1_img_.astype(np.float32)
        ref_img = ref_img_.astype(np.float32)
        judge_img = judge_img.astype(np.float32)

        return {'p0': p0_img, 'p1': p1_img, 'ref': ref_img, 'judge': judge_img,
            'p0_path': p0_path, 'p1_path': p1_path, 'ref_path': ref_path, 'judge_path': judge_path}

    def __len__(self):
        return len(self.p0_paths)
