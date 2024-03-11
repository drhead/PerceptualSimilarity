
from __future__ import absolute_import
from typing import Tuple

import numpy as np
import torch
from collections import OrderedDict
from scipy.ndimage import zoom
from tqdm import tqdm
import lpips
import os
from adan import Adan

class Trainer():
    def name(self):
        return self.model_name

    def initialize(self, model='lpips', net='alex', colorspace='Lab', pnet_rand=False, pnet_tune=False, model_path=None,
            device='cuda', spatial=False, 
            is_train=False, lr=.0001, beta1=0.5, version='0.1', T_max=23600, optimizer='adam', schedule='none', no_decay_bias=False):
        '''
        INPUTS
            model - ['lpips'] for linearly calibrated network
                    ['baseline'] for off-the-shelf network
                    ['L2'] for L2 distance in Lab colorspace
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original (with a bug)
            gpu_ids - int array - [0] by default, gpus to use
        '''

        self.device = device
        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.model_name = '%s [%s]'%(model,net)
        self.schedule = schedule

        if(self.model == 'lpips'): # pretrained net + linear layer
            self.net = lpips.LPIPS(pretrained=not is_train, net=net, version=version, lpips=True, spatial=spatial, 
                pnet_rand=pnet_rand, pnet_tune=pnet_tune, 
                use_dropout=True, model_path=model_path, eval_mode=False)
        elif(self.model=='baseline'): # pretrained network
            self.net = lpips.LPIPS(pnet_rand=pnet_rand, net=net, lpips=False)
        elif(self.model in ['L2','l2']):
            self.net = lpips.L2(colorspace=colorspace) # not really a network, only for testing
            self.model_name = 'L2'
        elif(self.model in ['DSSIM','dssim','SSIM','ssim']):
            self.net = lpips.DSSIM(colorspace=colorspace)
            self.model_name = 'SSIM'
        else:
            raise ValueError("Model [%s] not recognized." % self.model)

        self.parameters = list(self.net.parameters())

        if self.is_train: # training mode
            # extra network on top to go from distances (d0,d1) => predicted human judgment (h*)
            self.rankLoss = lpips.BCERankingLoss()
            self.parameters += list(self.rankLoss.net.parameters())
            self.lr = lr
            self.old_lr = lr
            self.net.train()

        else: # test mode
            self.net.eval()

        self.net = self.net.cuda()

        if self.is_train:
            self.rankLoss = self.rankLoss.cuda()
            if no_decay_bias:
                decay = []
                no_decay = []
                for name, param in self.net.named_parameters():
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)
                for name, param in self.rankLoss.net.named_parameters():
                    if not param.requires_grad:
                        continue

                    if param.ndim <= 1 or name.endswith(".bias"):
                        no_decay.append(param)
                    else:
                        decay.append(param)
                optim_params = [
                    {'params': no_decay, 'weight_decay': 0.},
                    {'params': decay, 'weight_decay': 0.02}]
            else: optim_params = self.parameters

            if optimizer == 'adam':
                self.optimizer_net = torch.optim.Adam(optim_params, lr=lr, betas=(beta1, 0.999))
            elif optimizer == 'adamw':
                self.optimizer_net = torch.optim.AdamW(optim_params, lr=torch.tensor(lr), betas=(beta1, 0.999))
            elif optimizer == 'adan':
                self.optimizer_net = Adan(optim_params, lr=torch.tensor(lr))

            if self.schedule == 'cosine':
                self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_net, T_max=T_max, eta_min=0)
            elif self.schedule == 'linear':
                self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer_net, start_factor=1.0, end_factor=0.0, total_iters=T_max)

    # ***** TRAINING FUNCTIONS *****
    @torch.compile(options={
            "triton.cudagraphs": True,
            "max_autotune": True,
            "max_autotune_pointwise": True,
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "ATEN,TRITON"
            }
        )
    def compiled_training(self, data: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            d0 = self.net.forward(data['ref'], data['p0'])
            d1 = self.net.forward(data['ref'], data['p1'])
            loss = self.rankLoss.forward(d0, d1, data['judge']*2.-1.)
            with torch.no_grad():
                acc_r = self.compute_accuracy(d0, d1, data['judge'])
        loss.backward()
        return loss.detach(), acc_r.mean()

    @torch.compile(fullgraph=False)
    def compiled_optimizer(self):
        self.optimizer_net.step()
        self.optimizer_net.zero_grad()
        self.clamp_weights()

    def optimize_parameters(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        loss, acc_r = self.compiled_training(data)
        self.compiled_optimizer()
        if self.schedule != 'none':
            self.lr_scheduler.step()

        return loss, acc_r

    def clamp_weights(self):
        for module in self.net.lins.modules():
            if(hasattr(module, 'weight') and hasattr(module, 'kernel_size') and module.kernel_size==(1,1)):
                module.weight.data.clamp_(min=0)

    def compute_accuracy(self, d0: torch.Tensor, d1: torch.Tensor, judge: torch.Tensor) -> torch.Tensor:
        ''' d0, d1 are Variables, judge is a Tensor '''
        d1_lt_d0 = (d1<d0).data.flatten()
        judge_per = judge.flatten()
        return d1_lt_d0*judge_per + (~d1_lt_d0)*(1-judge_per)

    def get_visuals(self, ref: torch.Tensor, p0: torch.Tensor, p1: torch.Tensor):
        import wandb
        zoom_factor = 256/ref.size()[2]

        ref_img = lpips.tensor2im(ref)
        p0_img = lpips.tensor2im(p0)
        p1_img = lpips.tensor2im(p1)

        ref_img_vis = zoom(ref_img,[zoom_factor, zoom_factor, 1],order=0)
        p0_img_vis = zoom(p0_img,[zoom_factor, zoom_factor, 1],order=0)
        p1_img_vis = zoom(p1_img,[zoom_factor, zoom_factor, 1],order=0)

        return OrderedDict([('img/ref', wandb.Image(ref_img_vis)),
                            ('img/p0', wandb.Image(p0_img_vis)),
                            ('img/p1', wandb.Image(p1_img_vis))])

    def save(self, path, label):
        self.save_network(self.net, path, '', label)
        self.save_network(self.rankLoss.net, path, 'rank', label)

    # helper saving function that can be used by subclasses
    def save_network(self, network, path, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(path, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print('Loading network from %s'%save_path)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self,nepoch_decay):
        lrd = self.lr / nepoch_decay
        lr = self.old_lr - lrd

        for param_group in self.optimizer_net.param_groups:
            param_group['lr'] = lr

        print('update lr [%s] decay: %f -> %f' % (type,self.old_lr, lr))
        self.old_lr = lr


    def get_image_paths(self):
        return self.image_paths

    def save_done(self, flag=False):
        np.save(os.path.join(self.save_dir, 'done_flag'),flag)
        np.savetxt(os.path.join(self.save_dir, 'done_flag'),[flag,],fmt='%i')


def score_2afc_dataset(data_loader, func, name=''):
    ''' Function computes Two Alternative Forced Choice (2AFC) score using
        distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a TwoAFCDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return numpy array of length N
    OUTPUTS
        [0] - 2AFC score in [0,1], fraction of time func agrees with human evaluators
        [1] - dictionary with following elements
            d0s,d1s - N arrays containing distances between reference patch to perturbed patches 
            gts - N array in [0,1], preferred patch selected by human evaluators
                (closer to "0" for left patch p0, "1" for right patch p1,
                "0.6" means 60pct people preferred right patch, 40pct preferred left)
            scores - N array in [0,1], corresponding to what percentage function agreed with humans
    CONSTS
        N - number of test triplets in data_loader
    '''

    d0s = []
    d1s = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name, leave=False):
        d0s+=func(data['ref'].cuda(),data['p0'].cuda()).data.cpu().numpy().flatten().tolist()
        d1s+=func(data['ref'].cuda(),data['p1'].cuda()).data.cpu().numpy().flatten().tolist()
        gts+=data['judge'].cpu().numpy().flatten().tolist()

    d0s = np.array(d0s)
    d1s = np.array(d1s)
    gts = np.array(gts)
    scores = (d0s<d1s)*(1.-gts) + (d1s<d0s)*gts + (d1s==d0s)*.5

    return(np.mean(scores), dict(d0s=d0s,d1s=d1s,gts=gts,scores=scores))

def score_jnd_dataset(data_loader, func, name=''):
    ''' Function computes JND score using distance function 'func' in dataset 'data_loader'
    INPUTS
        data_loader - CustomDatasetDataLoader object - contains a JNDDataset inside
        func - callable distance function - calling d=func(in0,in1) should take 2
            pytorch tensors with shape Nx3xXxY, and return pytorch array of length N
    OUTPUTS
        [0] - JND score in [0,1], mAP score (area under precision-recall curve)
        [1] - dictionary with following elements
            ds - N array containing distances between two patches shown to human evaluator
            sames - N array containing fraction of people who thought the two patches were identical
    CONSTS
        N - number of test triplets in data_loader
    '''

    ds = []
    gts = []

    for data in tqdm(data_loader.load_data(), desc=name):
        ds+=func(data['p0'].cuda(),data['p1'].cuda()).data.cpu().numpy().tolist()
        gts+=data['same'].cpu().numpy().flatten().tolist()

    sames = np.array(gts)
    ds = np.array(ds)

    sorted_inds = np.argsort(ds)
    ds_sorted = ds[sorted_inds]
    sames_sorted = sames[sorted_inds]

    TPs = np.cumsum(sames_sorted)
    FPs = np.cumsum(1-sames_sorted)
    FNs = np.sum(sames_sorted)-TPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    score = lpips.voc_ap(recs,precs)

    return(score, dict(ds=ds,sames=sames))
