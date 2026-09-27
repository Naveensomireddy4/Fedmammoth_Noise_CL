import torch
from torch import nn
from torch.nn import functional as F
import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from _models import register_model
from typing import List
from _models._utils import BaseModel
from _networks.vit import VisionTransformer as Vit
from torch.func import functional_call
from copy import deepcopy
from utils.tools import str_to_bool, compute_fisher_expectation_fabric

from _models.lora import Lora, merge_AB, zero_pad
from _models.regmean import RegMean
from _models.lora import Lora
from tqdm import tqdm
from torchvision import transforms
from kornia import augmentation
import time, os, math
import torch.nn.init as init
from PIL import Image
import numpy as np
from torch.autograd import Variable
from abc import ABC
from _models.fedavg import FedAvg
import shutil

#dataset = "cifar100"
#
#if dataset =="cifar100":
synthesis_batch_size = 256
sample_batch_size = 256
g_steps = 10
is_maml = 1
kd_steps = 400
warmup = 2          # must be < syn_round
lr_g = 0.002
lr_z = 0.01
oh = 0.5
T = 20.0
act = 0.0
adv = 1.0
bn = 10.0
reset_l0 = 1
reset_bn = 0
bn_mmt = 0.9
syn_round = 2       # 2 × 256 = 512 ≈ 500 images per task
tau = 1
#data_normalize = dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
data_normalize = dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

train_transform = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    transforms.RandomResizedCrop(size=(224, 224), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(**dict(data_normalize)),
])
    
#else:
#    synthesis_batch_size = 16
#    sample_batch_size = 16
#    g_steps=50  
#    is_maml=0   
#    kd_steps=400     
#    warmup=20
#    lr_g=0.0002 
#    lr_z=0.01   
#    oh=0.1  
#    T=5     
#    act=0.0 
#    adv=1.0 
#    bn=0.1  
#    reset_l0=0 
#    reset_bn=0 
#    bn_mmt=0.9  
#    syn_round=200  
#    tau=1
#    data_normalize = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#    train_transform = transforms.Compose([
#        transforms.RandomCrop(64, padding=4),
#        transforms.RandomHorizontalFlip(),
#        transforms.ToTensor(),
#        transforms.Normalize(**dict(data_normalize)),
#    ])
#
    

def normalize(tensor, mean, std, reverse=False):
    if reverse:
        _mean = [ -m / s for m, s in zip(mean, std) ]
        _std = [ 1/s for s in std ]
    else:
        _mean = mean
        _std = std
    
    _mean = torch.as_tensor(_mean, dtype=tensor.dtype, device=tensor.device)
    _std = torch.as_tensor(_std, dtype=tensor.dtype, device=tensor.device)
    tensor = (tensor - _mean[None, :, None, None]) / (_std[None, :, None, None])
    return tensor


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x, reverse=False):
        return normalize(x, self.mean, self.std, reverse=reverse)

normalizer = Normalizer(**dict(data_normalize))


def _collect_all_images(nums, root, postfix=['png', 'jpg', 'jpeg', 'JPEG']):
    images = []
    if isinstance( postfix, str):
        postfix = [ postfix ]
    for dirpath, dirnames, files in os.walk(root):
        for pos in postfix:
            if nums != None:
                files.sort()
                # random.shuffle(files)
                files = files[:nums]
                # files = files[20*256:20*256+nums]       # discard the ealry-stage data
                # files = files[-nums:]  # 40*256 
            for f in files:
                if f.endswith(pos):
                    images.append( os.path.join( dirpath, f ) )
    return images


class DataIter(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._iter = iter(self.dataloader)
    
    def next(self):
        try:
            data = next( self._iter )
        except StopIteration:
            self._iter = iter(self.dataloader)
            data = next( self._iter )
        return data


class UnlabeledImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, nums=None):
        self.root = os.path.abspath(root)
        self.images = _collect_all_images(nums, self.root) #[ os.path.join(self.root, f) for f in os.listdir( root ) ]
        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open( self.images[idx] )
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return 'Unlabeled data:\n\troot: %s\n\tdata mount: %d\n\ttransforms: %s'%(self.root, len(self), self.transform)


def pack_images(images, col=None, channel_last=False, padding=1):
    # N, C, H, W
    if isinstance(images, (list, tuple) ):
        images = np.stack(images, 0)
    if channel_last:
        images = images.transpose(0,3,1,2) # make it channel first
    assert len(images.shape)==4
    assert isinstance(images, np.ndarray)

    N,C,H,W = images.shape
    if col is None:
        col = int(math.ceil(math.sqrt(N)))
    row = int(math.ceil(N / col))
    
    pack = np.zeros( (C, H*row+padding*(row-1), W*col+padding*(col-1)), dtype=images.dtype )
    for idx, img in enumerate(images):
        h = (idx // col) * (H+padding)
        w = (idx % col) * (W+padding)
        pack[:, h:h+H, w:w+W] = img
    return pack

def reptile_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(p.data - tar_p.data, alpha=67) # , alpha=40


def fomaml_grad(src, tar):
    for p, tar_p in zip(src.parameters(), tar.parameters()):
        if p.grad is None:
            p.grad = Variable(torch.zeros(p.size())).cuda()
        p.grad.data.add_(tar_p.grad.data)   #, alpha=0.67


def reset_l0_fun(model):
    for n,m in model.named_modules():
        if n == "l1.0" or n == "conv_blocks.0":
            nn.init.normal_(m.weight, 0.0, 0.02)
            nn.init.constant_(m.bias, 0)

def save_image_batch(imgs, output, col=None, size=None, pack=True,device="cuda"):
    if isinstance(imgs, torch.Tensor):
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
        # Then ALWAYS move to CPU for numpy conversion
        imgs = imgs.detach().clamp(0, 1).cpu().numpy()
        imgs = (imgs * 255).astype("uint8")
    base_dir = os.path.dirname(output)
    if base_dir!='':
        os.makedirs(base_dir, exist_ok=True)
    if pack:
        imgs = pack_images( imgs, col=col ).transpose( 1, 2, 0 ).squeeze()
        imgs = Image.fromarray( imgs )
        if size is not None:
            if isinstance(size, (list,tuple)):
                imgs = imgs.resize(size)
            else:
                w, h = imgs.size
                max_side = max( h, w )
                scale = float(size) / float(max_side)
                _w, _h = int(w*scale), int(h*scale)
                imgs = imgs.resize([_w, _h])
        imgs.save(output)
    else:
        output_filename = output.strip('.png')
        for idx, img in enumerate(imgs):
            img = Image.fromarray( img.transpose(1, 2, 0) )
            img.save(output_filename+'-%d.png'%(idx))

class DeepInversionHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module, mmt_rate):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = module
        self.mmt_rate = mmt_rate
        self.mmt = None
        self.tmp_val = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        if self.mmt is None:
            r_feature = torch.norm(module.running_var.data - var, 2) + \
                        torch.norm(module.running_mean.data - mean, 2)
        else:
            mean_mmt, var_mmt = self.mmt
            r_feature = torch.norm(module.running_var.data - (1 - self.mmt_rate) * var - self.mmt_rate * var_mmt, 2) + \
                        torch.norm(module.running_mean.data - (1 - self.mmt_rate) * mean - self.mmt_rate * mean_mmt, 2)

        self.r_feature = r_feature
        self.tmp_val = (mean, var)

    def update_mmt(self):
        mean, var = self.tmp_val
        if self.mmt is None:
            self.mmt = (mean.data, var.data)
        else:
            mean_mmt, var_mmt = self.mmt
            self.mmt = ( self.mmt_rate*mean_mmt+(1-self.mmt_rate)*mean.data,
                        self.mmt_rate*var_mmt+(1-self.mmt_rate)*var.data )

    def remove(self):
        self.hook.remove()


class ImagePool(object):
    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self._idx = 0

    def add(self, imgs,device="cuda", targets=None):
        save_image_batch(imgs, os.path.join( self.root, "%d.png"%(self._idx) ), pack=False,device=device)
        self._idx+=1

    def get_dataset(self, nums=None, transform=None, labeled=True):
        return UnlabeledImageDataset(self.root, transform=transform, nums=nums)


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()
        self.params = (nz, ngf, img_size, nc)
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

    # return a copy of its own
    def clone(self):
        clone = Generator(self.params[0], self.params[1], self.params[2], self.params[3])
        clone.load_state_dict(self.state_dict())
        return clone.cuda()


def kldiv( logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits/T, dim=1)
    p = F.softmax( targets/T, dim=1 )
    return F.kl_div( q, p, reduction=reduction ) * (T*T)

class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

class GlobalSynthesizer(ABC):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size,task_id,
                    init_dataset=None, iterations=100, lr_g=0.1,
                    synthesis_batch_size=128, sample_batch_size=128, 
                    adv=0.0, bn=1, oh=1,
                    save_dir='run/fast', transform=None, autocast=None, use_fp16=False,
                    normalizer=None, distributed=False, lr_z = 0.01,
                    warmup=10, reset_l0=0, reset_bn=0, bn_mmt=0,
                    is_maml=1, fabric = None,device="cuda"):#, args=None):
        super(GlobalSynthesizer, self).__init__()
        self.teacher = teacher
        self.task_id= task_id
        self.student = student
        self.save_dir = save_dir
        self.img_size = img_size 
        self.device = device
        self.iterations = iterations
        self.lr_g = lr_g
        self.lr_z = lr_z
        self.nz = nz
        self.adv = adv
        self.bn = bn
        self.oh = oh
        self.ismaml = is_maml
        #self.args = args

        self.num_classes = num_classes
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.normalizer = normalizer

        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.generator = generator.cuda().train()
        self.ep = 0
        self.ep_start = warmup
        self.reset_l0 = reset_l0
        self.reset_bn = reset_bn
        self.prev_z = None
        self.fabric = fabric

        if self.ismaml:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])
        else:
            self.meta_optimizer = torch.optim.Adam(self.generator.parameters(), self.lr_g*self.iterations, betas=[0.5, 0.999])


        self.aug = transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ])
        
        self.bn_mmt = bn_mmt
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m, self.bn_mmt) )



    def synthesize(self, targets=None):
        self.ep+=1
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6

        if (self.ep == 120+self.ep_start) and self.reset_l0:
            reset_l0_fun(self.generator)
        
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz)).cuda()
        z.requires_grad = True
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
        else:
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.cuda()

        fast_generator = self.generator.clone() 
        optimizer = torch.optim.Adam([
            {'params': fast_generator.parameters()},
            {'params': [z], 'lr': self.lr_z}
        ], lr=self.lr_g, betas=[0.5, 0.999])
        for it in range(self.iterations):
            inputs = fast_generator(z)
            inputs_aug = self.aug(inputs) # crop and normalize
            if it == 0:
                originalMeta = inputs
            t_out = self.teacher(inputs_aug,task_id=self.task_id)#["logits"]
            if targets is None:
                targets = torch.argmax(t_out, dim=-1)
                targets = targets.cuda()

            loss_bn = sum([h.r_feature for h in self.hooks])
            loss_oh = F.cross_entropy( t_out, targets )
            if self.adv>0 and (self.ep >= self.ep_start):
                s_out = self.student(inputs_aug,task_id=self.task_id)#["logits"]
                mask = (s_out.max(1)[1]==t_out.max(1)[1]).float()
                loss_adv = -(kldiv(s_out, t_out, reduction='none').sum(1) * mask).mean() # adversarial distillation
            else:
                loss_adv = loss_oh.new_zeros(1)
            loss = self.bn * loss_bn + self.oh * loss_oh + self.adv * loss_adv
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data.to(self.device) # mem, self.device
                    # save_data = best_inputs.clone()
                    # vutils.save_image(save_data[:200], 'real_{}.png'.format(dataset), normalize=True, scale_each=True, nrow=20)


            optimizer.zero_grad()
            loss.backward() if self.fabric is None else self.fabric.backward(loss)


            if self.ismaml:
                if it==0: self.meta_optimizer.zero_grad()
                fomaml_grad(self.generator, fast_generator)
                if it == (self.iterations-1): self.meta_optimizer.step()

            optimizer.step()

        if self.bn_mmt != 0:
            for h in self.hooks:
                h.update_mmt()

        # REPTILE meta gradient
        if not self.ismaml:
            self.meta_optimizer.zero_grad()
            reptile_grad(self.generator, fast_generator)
            self.meta_optimizer.step()

        self.student.train()
        self.prev_z = (z, targets)
        end = time.time()

        self.data_pool.add( best_inputs,self.device )       # add a batch of data


        


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def refine_as_not_true(logits, targets, num_classes):
    nt_positions = torch.arange(0, num_classes).cuda()
    nt_positions = nt_positions.repeat(logits.size(0), 1)
    nt_positions = nt_positions[nt_positions[:, :] != targets.view(-1, 1)]
    nt_positions = nt_positions.view(-1, num_classes - 1)

    logits = torch.gather(logits, 1, nt_positions)

    return logits



@register_model("target")
class Target(FedAvg):
    def __init__(
        self,
        fabric,
        network: Vit,
        device: str,
        optimizer: str = "AdamW",
        lr: float = 0.0003,
        wd_reg: float = 0.0,
        dataset: str = "seq-cifar100_224",
        num_clients: int = 10,
        tasks: int = 10,
        batch_size: int = 16,
        nums: int = 8000,
        kd_alpha: float = 25,
        save_dir: str = "synthetic_data",
    ) -> None:
        super().__init__(
            fabric,
            network, 
            device,
            optimizer, 
            lr, 
            wd_reg, 
            )
        self.num_clients = num_clients
        self.tasks = tasks
        self.batch_size = batch_size
        self.save_dir = save_dir
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        self.dataset = dataset
        if "cifar" in dataset:
            self.dataset_size = 50000
        elif "tiny" in dataset and "imagenet" in dataset:
            self.dataset_size = 100000
        elif "imagenet100" in dataset:
            self.dataset_size = 130000
        elif "imagenetr" in dataset:
            self.dataset_size = 30000
        elif "eurosat" in dataset:
            self.dataset_size = 27000
        else:
            self.dataset_size = 8000
        self.nums = nums
        self.total_classes = []
        self.syn_data_loader = None
        self.old_network = None
        self.kd_alpha = kd_alpha

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,task_id, update: bool = True) -> float:

        with self.fabric.autocast():
            inputs = self.augment(inputs)
            outputs = self.network(inputs,task_id)
            loss = self.loss(outputs, labels )
            if self.cur_task > 0 and hasattr(self, "syn_data_iters"):
                for old_task_id in range(self.cur_task):

                    try:
                        syn_inputs = next(self.syn_data_iters[old_task_id]).to(self.device)
                    except StopIteration:
                        # restart this task's loader when exhausted
                        self.syn_data_iters[old_task_id] = iter(self.syn_data_loaders[old_task_id])
                        syn_inputs = next(self.syn_data_iters[old_task_id]).to(self.device)

                    # Forward through matching task heads
                    syn_outputs = self.network(syn_inputs, task_id=old_task_id)

                    with torch.no_grad():
                        syn_old_outputs = self.old_network(syn_inputs, task_id=old_task_id)

                    # Knowledge distillation for that specific task
                    kd_loss = _KD_loss(syn_outputs, syn_old_outputs, T=2)
                    loss += self.kd_alpha * kd_loss



        if update:
            self.fabric.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad()


        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
        return loss.item(), preds

    def begin_task(self, task_id: int):
        super().begin_task(task_id)
        self.current_task = task_id
        self.cur_task = task_id
        #self.total_classes += [i for i in range(self.cur_task * self.cpt, (self.cur_task + 1) * self.cpt)]
        if self.cur_task > 0:
            self.old_network = deepcopy(self.network)
            for p in self.old_network.parameters():
                p.requires_grad = False
    
    def get_server_info(self):
        info = {
            "state_dict": self.network.state_dict()
        }
        # Include old_network if it exists
        if self.cur_task>0 and hasattr(self, "old_network") and self.old_network is not None:
            print(f"sending  old model wts from server task{self.cur_task} ")
            info["old_network"] = self.old_network.state_dict()
        return info

    
    def get_client_info(self, dataloader):
        return {
            "state_dict": self.network.state_dict(),
            "num_train_samples": len(dataloader.dataset),
        }

    def begin_round_client(self, dataloader: DataLoader, server_info: dict, task_id):
        super().begin_round_client(dataloader, server_info, task_id)
        
        if (self.cur_task > 0
            and hasattr(self, "old_network")
            and self.old_network is not None
            and "old_network" in server_info):
            print("loading old_network wts from server")
            self.old_network.load_state_dict(server_info["old_network"])
        
        self.network = self.network.to(self.device)
        params = [{"params": self.network.parameters()}]
        optimizer = self.optimizer_class(params, lr=self.lr, weight_decay=self.wd)
        self.optimizer = self.fabric.setup_optimizers(optimizer)
        
        if self.cur_task > 0:
            self.syn_data_loaders = self.get_syn_data_loader()

            for t in self.syn_data_loaders:
                self.syn_data_loaders[t] = self.fabric.setup_dataloaders(self.syn_data_loaders[t])

            self.syn_data_iters = {
                t: iter(loader) for t, loader in self.syn_data_loaders.items()
            }

            self.old_network = self.old_network.to(self.device)


    def end_round_client(self, dataloader: DataLoader, task):
        self.syn_data_loaders = {}
        self.syn_data_iters = {}
        self.syn_data_loader = None
        self.optimizer.zero_grad()
        self.optimizer = None
        if self.old_network is not None:
            self.old_network = self.old_network.to(self.device)
        self.network = self.network.to(self.device)
    
    def end_task_client(self, dataloader: DataLoader = None, server_info: dict = None,task_id=0):
        return 

    def kd_train(self, student, teacher, criterion, optimizer):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data() 
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            images = data_iter.next().cuda()  
            with torch.no_grad():
                t_out = teacher(images)#["logits"]
            s_out = student(images.detach())#["logits"]
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()

            self.fabric.backward(loss_s)
            self.fabric.clip_gradients(student, optimizer, max_norm=1.0, norm_type=2)
            optimizer.step()
        return loss_s.item()
    
    def kd_train_task(self, student, teacher, task_id, criterion, optimizer):
        student.train()
        teacher.eval()

        loader = self.get_all_syn_data(task_id)
        data_iter = DataIter(loader)

        for _ in range(kd_steps):
            images = data_iter.next().cuda()

            with torch.no_grad():
                t_out = teacher(images, task_id)

            s_out = student(images, task_id)

            loss_s = criterion(s_out, t_out.detach())

            optimizer.zero_grad()
            self.fabric.backward(loss_s)
            self.fabric.clip_gradients(student, optimizer, max_norm=1.0)
            optimizer.step()

        return loss_s.item()

        
    class TaskWrapper(nn.Module):
        def __init__(self, net, task_id):
            super().__init__()
            self.net = net
            self.register_buffer("task_id_tensor", torch.tensor(task_id))

        def forward(self, x):
            return self.net(x, int(self.task_id_tensor.item()))



    import shutil
    from copy import deepcopy
    from tqdm import tqdm
    import os

    def data_generation(self):
        nz = 256
        img_size = 224
        img_shape = (3, 224,224)

        # Fresh student model (shared across all old tasks)
        student = deepcopy(self.network).cuda()
        student.apply(weight_init)

        criterion = KLDiv(T=T)
        optimizer = self.fabric.setup_optimizers(
            torch.optim.SGD(student.parameters(), lr=0.2, weight_decay=1e-4, momentum=0.9)
        )

        # 🔁 LOOP THROUGH ALL PREVIOUS TASKS
        for task_id in range(self.cur_task +1):

            print(f"\n🔹 Generating synthetic data for OLD TASK {task_id}")

            # -------------------------
            # Ensure synthetic data folder exists (clear if exists)
            # -------------------------
            tmp_dir = os.path.join(self.save_dir, f"task_{task_id}")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

            # -------------------------
            # Frozen teacher snapshot (MULTI-HEAD)
            # -------------------------
            teacher = deepcopy(self.network).cuda().eval()
            for p in teacher.parameters():
                p.requires_grad = False

            # -------------------------
            # Fresh generator for this task
            # -------------------------
            generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3).cuda()
            num_classes = self.network.classifiers[task_id].out_features

            synthesizer = GlobalSynthesizer(
                teacher=teacher,
                student=student,
                generator=generator,
                nz=nz,
                num_classes=num_classes,
                img_size=img_shape,
                task_id=task_id,   # ✅ correct task ID
                init_dataset=None,
                save_dir=tmp_dir,
                transform=train_transform,
                normalizer=normalizer,
                synthesis_batch_size=synthesis_batch_size,
                sample_batch_size=sample_batch_size,
                iterations=g_steps,
                warmup=warmup,
                lr_g=lr_g,
                lr_z=lr_z,
                adv=adv,
                bn=bn,
                oh=oh,
                reset_l0=reset_l0,
                reset_bn=reset_bn,
                bn_mmt=bn_mmt,
                is_maml=is_maml,
                fabric=self.fabric,
                device=self.device
            )

            # -------------------------
            # Generate + Distill
            # -------------------------
            for it in tqdm(range(syn_round), desc=f"Synth Task {task_id}"):
                synthesizer.synthesize()

                if it >= warmup:
                    loss = self.kd_train_task(student, teacher, task_id, criterion, optimizer)
                    print(f"Task {task_id}, Epoch {it+1}/{syn_round}, KD Loss: {loss:.4f}")

            # -------------------------
            # Print number of samples generated and location
            # -------------------------
            generated_files = os.listdir(tmp_dir)
            print(f"Task {task_id} synthetic data saved at: {tmp_dir}")
            print(f"Number of synthetic samples generated: {len(generated_files)}")

        print("\n✅ Data generation and distillation complete for all previous tasks!")

            
    def get_syn_data_loader(self):
        """
        Returns a dict of DataLoaders for all previous tasks.
        Access by task_id: syn_loaders[task_id]
        """
        if self.cur_task == 0:
            return {}  # no previous tasks yet

        syn_loaders = {}
        dataset_size = self.dataset_size
        iters = math.ceil(dataset_size / (self.num_clients * self.tasks * self.batch_size))
        syn_bs = 16  # batch size for synthetic data

        for task_id in range(self.cur_task):
            data_dir = os.path.join(self.save_dir, f"task_{task_id}")
            print(f"Loading synthetic data for Task {task_id}: iters={iters}, syn_bs={syn_bs}, data_dir={data_dir}")

            syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
            syn_data_loader = torch.utils.data.DataLoader(
                syn_dataset, batch_size=syn_bs, shuffle=True, num_workers=0
            )
            syn_loaders[task_id] = syn_data_loader

        return syn_loaders


    def get_all_syn_data(self,task_id):
        data_dir = os.path.join(self.save_dir, "task_{}".format(task_id))
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True,
            num_workers=0, pin_memory=True, sampler=None)
        return loader

    def end_round_server(self, client_info,task):
        super().end_round_server(client_info)

    def end_task_server(self, client_info: List[dict] = None):
        super().end_task_server(client_info)
        self.data_generation()
