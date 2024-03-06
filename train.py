import torch.backends.cudnn as cudnn
cudnn.benchmark=False

import numpy as np
import time
import os
os.environ['ACCELERATE_DYNAMO_MODE'] = "max-autotune-no-cudagraphs"
# os.environ['ACCELERATE_DYNAMO_USE_FULLGRAPH'] = "True"
# os.environ["TORCHDYNAMO_REPRO_LEVEL"] = '4'
import torch

# torch._dynamo.config.verify_correctness = True
# torch._dynamo.config.repro_level = 4
# torch._dynamo.config.verbose = True
import lpips
from data import data_loader as dl
import argparse
from util.visualizer import Visualizer
from IPython import embed
from tqdm import tqdm
import wandb
# from accelerate import Accelerator
# from accelerate.utils import DynamoBackend
# accelerator = Accelerator(
#     mixed_precision='bf16',
#     dynamo_backend=DynamoBackend.INDUCTOR)
# device = accelerator.device
use_wandb = False
torch._inductor.list_options()
if use_wandb:
    wandb.init(
        # set the wandb project where this run will be logged
        project="lpips",
        name="vgg-adan-newbaseline")

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, nargs='+', default=['train/traditional','train/cnn','train/mix'], help='datasets to train on: [train/traditional],[train/cnn],[train/mix],[val/traditional],[val/cnn],[val/color],[val/deblur],[val/frameinterp],[val/superres]')
parser.add_argument('--model', type=str, default='lpips', help='distance model type [lpips] for linearly calibrated net, [baseline] for off-the-shelf network, [l2] for euclidean distance, [ssim] for Structured Similarity Image Metric')
parser.add_argument('--net', type=str, default='alex', help='[squeeze], [alex], or [vgg] for network architectures')
parser.add_argument('--batch_size', type=int, default=64, help='batch size to test image patches in')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='gpus to use')

parser.add_argument('--nThreads', type=int, default=16, help='number of threads to use in data loader')
parser.add_argument('--nepoch', type=int, default=5, help='# epochs at base learning rate')
parser.add_argument('--nepoch_decay', type=int, default=5, help='# additional epochs at linearly learning rate')
parser.add_argument('--display_freq', type=int, default=6400, help='frequency (in instances) of showing training results on screen')
parser.add_argument('--print_freq', type=int, default=6400, help='frequency (in instances) of showing training results on console')
parser.add_argument('--save_latest_freq', type=int, default=25600, help='frequency (in instances) of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--display_id', type=int, default=0, help='window id of the visdom display, [0] for no displaying')
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--display_port', type=int, default=8001,  help='visdom display port')
parser.add_argument('--use_html', action='store_true', help='save off html pages')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='checkpoints directory')
parser.add_argument('--name', type=str, default='tmp', help='directory name for training')

parser.add_argument('--from_scratch', action='store_true', help='model was initialized from scratch')
parser.add_argument('--train_trunk', action='store_true', help='model trunk was trained/tuned')
parser.add_argument('--train_plot', action='store_true', help='plot saving')

opt = parser.parse_args()
opt.save_dir = os.path.join(opt.checkpoints_dir,opt.name)
if(not os.path.exists(opt.save_dir)):
    os.mkdir(opt.save_dir)

# initialize model
trainer = lpips.Trainer()
trainer.initialize(model=opt.model, net=opt.net, device='cuda', is_train=True, 
    pnet_rand=opt.from_scratch, pnet_tune=opt.train_trunk)

img_size = 56 if opt.net=='efficientnetv2' else 64

# load data from all training sets
data_loader = dl.CreateDataLoader(opt.datasets,dataset_mode='2afc', batch_size=opt.batch_size, load_size=img_size, serial_batches=False, nThreads=opt.nThreads)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
D = len(dataset)
print('Loading %i instances from'%dataset_size,opt.datasets)
visualizer = Visualizer(opt)

total_steps = 0
fid = open(os.path.join(opt.checkpoints_dir,opt.name,'train_log.txt'),'w+')

ema_loss = None
ema_accuracy = None
alpha = 0.95

for epoch in range(1, opt.nepoch + opt.nepoch_decay + 1):
    epoch_start_time = time.time()
    with tqdm(
            desc=f"Training (Epoch {epoch})...",
            total=dataset_size//64,
            dynamic_ncols=True) as pbar:
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            total_steps += opt.batch_size
            epoch_iter = total_steps - dataset_size * (epoch - 1)

            ref, p0, p1, judge = trainer.set_input(data)
            errors = {}
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
            # if not use_wandb:
                # print(f"loss: {errors['loss_total'].detach().cpu().numpy()}, acc: {errors['acc_r'].detach().cpu().numpy().mean()}")
            # errors = trainer.get_current_errors()
            if use_wandb:
                wandb.log({
                    'loss_total': loss,
                    'acc_r': acc_r},
                    commit=False
                )
            if total_steps % opt.display_freq == 0 and use_wandb:
                wandb.log(trainer.get_visuals(), commit=False)

            # if total_steps % opt.print_freq == 0:
            #     # errors = trainer.get_current_errors()
            #     t = (time.time()-iter_start_time)/opt.batch_size
            #     t2o = (time.time()-epoch_start_time)/3600.
            #     t2 = t2o*D/(i+.0001)
            #     visualizer.print_current_errors(epoch, epoch_iter, errors, t, t2=t2, t2o=t2o, fid=fid)

            #     for key in errors.keys():
            #         visualizer.plot_current_errors_save(epoch, float(epoch_iter)/dataset_size, opt, errors, keys=[key,], name=key, to_plot=opt.train_plot)

            #     if opt.display_id > 0:
            #         visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                # print('saving the latest model (epoch %d, total_steps %d)' %
                #       (epoch, total_steps))
                trainer.save(opt.save_dir, 'latest')
            if use_wandb:
                wandb.log(commit=True)

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        trainer.save(opt.save_dir, 'latest')
        trainer.save(opt.save_dir, epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))

    # if epoch > opt.nepoch:
    #     trainer.update_learning_rate(opt.nepoch_decay)

# initialize data loader
for dataset in opt.datasets:
    data_loader = dl.CreateDataLoader(dataset,dataset_mode=opt.dataset_mode, load_size=img_size, batch_size=opt.batch_size, nThreads=opt.nThreads)

    # evaluate model on data
    if(opt.dataset_mode=='2afc'):
        (score, results_verbose) = lpips.score_2afc_dataset(data_loader, trainer.forward, name=dataset)
    elif(opt.dataset_mode=='jnd'):
        (score, results_verbose) = lpips.score_jnd_dataset(data_loader, trainer.forward, name=dataset)
    if use_wandb:
        wandb.log({dataset: score})
    # print results
    print('  Dataset [%s]: %.2f'%(dataset,100.*score))

# trainer.save_done(True)
fid.close()
