import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn
import torch._dynamo.config
cudnn.benchmark=False
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision('high')
# torch._dynamo.config.verify_correctness = True
# torch._dynamo.config.repro_tolerance = 1e4
import lpips
from data import data_loader as dl
import argparse
from IPython import embed
from tqdm import tqdm
import wandb
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
use_wandb = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_mode', type=str, default='2afc', help='[2afc,jnd]')
parser.add_argument('--datasets', type=str, nargs='+', default=['train/traditional','train/cnn','train/mix'], help='datasets to train on: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
parser.add_argument('--batch_size', type=int, default=64, help='batch size to test image patches in')

parser.add_argument('--nThreads', type=int, default=12, help='number of threads to use in data loader')
parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
parser.add_argument('--save_latest_freq', type=int, default=25600, help='frequency (in instances) of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')
parser.add_argument('--name', type=str, default='tmp', help='directory name for training')

parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
parser.add_argument('--train_plot', action='store_true', help='plot saving')
parser.add_argument('--optimizer', type=str, default='adamw')
parser.add_argument('--lr_schedule', type=str, default='cosine')
parser.add_argument('--no_decay_bias', type=bool, default=True)
parser.add_argument('--color_space', type=str, default='srgb')

lr = 1e-4

opt = parser.parse_args()
opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
if(not os.path.exists(opt.save_dir)):
    os.mkdir(opt.save_dir)

if use_wandb:
    config = {
        'net': opt.net,
        'lr': lr,
        'batch_size': opt.batch_size,
        'epochs': opt.nepoch + opt.nepoch_decay,
        'optimizer': opt.optimizer,
        'no_decay_bias': opt.no_decay_bias,
        'lr_schedule': opt.lr_schedule,
        'color_space': opt.color_space
    }
    wandb.init(
        # set the wandb project where this run will be logged
        project="lpips",
        name=f"{opt.net}-{opt.optimizer}-srgb",
        config=config)

img_size = 56 if opt.net=='efficientnetv2' else 64

# load data from all training sets
data_loader = dl.CreateDataLoader(
    opt.datasets,
    dataset_mode='2afc',
    batch_size=opt.batch_size,
    load_size=img_size,
    serial_batches=False,
    nThreads=opt.nThreads,
    use_cache=True,
    colorspace=opt.color_space)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
D = len(dataset)
print('Loading %i instances from'%dataset_size,opt.datasets)

# initialize model
trainer = lpips.Trainer()
trainer.initialize(
    model=opt.model, 
    net=opt.net, 
    device='cuda', 
    is_train=True, 
    pnet_rand=opt.from_scratch, 
    pnet_tune=opt.train_trunk, 
    lr=lr, 
    T_max=dataset_size//opt.batch_size * (opt.nepoch + opt.nepoch_decay),
    schedule=opt.lr_schedule,
    optimizer=opt.optimizer,
    no_decay_bias=opt.no_decay_bias)

val_forward = torch.compile(
    trainer.forward,
    fullgraph=True,
    backend="inductor",
    options={
        "triton.cudagraphs": True,
        "max_autotune": True,
        "max_autotune_pointwise": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "ATEN,TRITON",
    })

total_steps = 0
fid = open(os.path.join(opt.checkpoints_dir,opt.name,'train_log.txt'),'w+')

ema_loss = None
ema_accuracy = None
alpha = 0.95

for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
    epoch_start_time = time.time()
    with tqdm(
            desc=f"Training (Epoch {epoch})...",
            total=dataset_size//opt.batch_size,
            dynamic_ncols=True,
            mininterval=1.0) as pbar:
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter = total_steps - dataset_size * (epoch - 1)

            ref, p0, p1, judge = trainer.set_input(data)
            loss, acc_r = trainer.optimize_parameters(ref, p0, p1, judge)

            loss = loss.detach().cpu().numpy()
            acc_r = np.mean(acc_r.detach().cpu().numpy())

            pbar.update(1)
            if ema_loss is None:
                ema_loss = loss
            else:
                ema_loss = alpha * ema_loss + (1 - alpha) * loss

            if ema_accuracy is None:
                ema_accuracy = acc_r
            else:
                ema_accuracy = alpha * ema_accuracy + (1 - alpha) * acc_r
            pbar.set_postfix(loss=f'{ema_loss:.4f}', accuracy=f'{ema_accuracy:.4f}', refresh=False)

            if use_wandb:
                wandb.log({
                    'loss_total': loss,
                    'acc_r': acc_r},
                    step=total_steps
                )
            # if total_steps % opt.display_freq == 0 and use_wandb:
            #     wandb.log(trainer.get_visuals(ref, p0, p1), commit=False)

            if total_steps % opt.save_latest_freq == 0:
                # print('saving the latest model (epoch %d, total_steps %d)' %
                #       (epoch, total_steps))
                trainer.save(opt.save_dir, 'latest')
            if use_wandb:
                wandb.log(data={}, step=total_steps)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        trainer.save(opt.save_dir, 'latest')
        trainer.save(opt.save_dir, epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

    if epoch > opt.nepoch and opt.lr_schedule == 'none':
        trainer.update_learning_rate(opt.nepoch_decay)
    # initialize data loader
    val_distortions = 0.0
    val_algorithms = 0.0
    for val_dataset in ['val/traditional','val/cnn','val/superres','val/deblur','val/color','val/frameinterp']:
        val_data_loader = dl.CreateDataLoader(val_dataset,dataset_mode=opt.dataset_mode, load_size=img_size, batch_size=50, nThreads=opt.nThreads, colorspace=opt.color_space)
        with torch.no_grad():
            # evaluate model on data
            if(opt.dataset_mode=='2afc'):
                (score, results_verbose) = lpips.score_2afc_dataset(val_data_loader, val_forward, name=val_dataset)
            elif(opt.dataset_mode=='jnd'):
                (score, results_verbose) = lpips.score_jnd_dataset(val_data_loader, val_forward, name=val_dataset)
        if val_dataset == 'val/traditional' or val_dataset == 'val/cnn':
            val_distortions += score / 2
        else:
            val_algorithms += score / 4
        if use_wandb:
            wandb.log({f"epoch-{val_dataset}": score}, step=total_steps)
            if epoch == opt.nepoch + opt.nepoch_decay:
                wandb.log({val_dataset: score}, step=total_steps)
        # print results
        print('  Dataset [%s]: %.2f'%(val_dataset,100.*score))
    if use_wandb:
        wandb.log({
            f"epoch-val/distortions": val_distortions,
            f"epoch-val/algorithms": val_algorithms}, step=total_steps)
        if epoch == opt.nepoch + opt.nepoch_decay:
            wandb.log({
                "val/algorithms": val_distortions,
                "val/distortions": val_algorithms}, step=total_steps)
    print('  Average [%s]: %.2f'%('distortions',100.*val_distortions))
    print('  Average [%s]: %.2f'%('algorithms',100.*val_algorithms))
# trainer.save_done(True)
fid.close()
data_loader.dataset.shm_index.unlink()
data_loader.dataset.shm_judge.unlink()
data_loader.dataset.shm_p0.unlink()
data_loader.dataset.shm_p1.unlink()
data_loader.dataset.shm_ref.unlink()