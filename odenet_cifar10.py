import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.vision import VisionDataset
import collections
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cifar10', metavar='N', help='dataset')
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--save', type=str, default='./experiment1')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument("--dir", type=str, default='cifar10_models/', required=False, help='model dir')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# download from https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
CIFAR10C_FOLDER = 'data/CIFAR-10-C/'

NOISE_TYPES = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    
    #"speckle_noise",
    
    "defocus_blur",
    "glass_blur",    
    "motion_blur",
    "zoom_blur",

    "snow",
    "frost",
    "fog",
    "brightness",
    
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    
    #"gaussian_blur",
    #"saturate",
    #"spatter",
]

SEVERITIES = [1, 2, 3, 4, 5]

class CIFARCorrupt(VisionDataset):

    def __init__(self,
                 root="data/CIFAR-10-C",
                 severity=[1, 2, 3, 4, 5],
                 noise=None,
                 transform=None,
                 target_transform=None):
        super(CIFARCorrupt, self).__init__(root, transform=transform, target_transform=target_transform)

        noise = NOISE_TYPES if noise is None else noise

        X = []
        for n in noise:
            D = np.load(os.path.join(root, f"{n}.npy"))
            D_s = np.split(D, 5, axis=0)
            for s in severity:
                X.append(D_s[s - 1])
        X = np.concatenate(X, axis=0)
        Y = np.load(os.path.join(root, "labels.npy"))
        Y_s = np.split(Y, 5, axis=0)
        Y = np.concatenate([Y_s[s - 1] for s in severity])
        Y = np.repeat(Y, len(noise))

        self.data = X
        self.targets = Y
        self.noise_to_nsamples = (noise, X.shape, Y.shape)
        print(f"Loaded {severity}-{noise}: X {X.shape} Y: {Y.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        #img = ToPILImage()(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

def get_calibration(data_loader, model, debug=False, n_bins=None):
    model.eval()
    mean_conf, mean_acc = [], []
    ece = []

    # New code to compute ECE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ------------------------------------------------------------------------
    logits_list = []
    labels_list = []
    #mean_conf, mean_acc = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)  # TODO: one hot or class number? We want class number
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(target)

    logits = torch.cat(logits_list).cuda()
    labels = torch.cat(labels_list).cuda()

    ece_criterion = _ECELoss(n_bins=15).cuda()
    ece = ece_criterion(logits, labels).item()
    #print('ECE (new implementation):', ece * 100)

    #calibration_dict = _get_calibration(labels, torch.nn.functional.softmax(logits, dim=1), debug=False, num_bins=n_bins)
    #mean_conf = calibration_dict['reliability_diag'][0]
    #mean_acc = calibration_dict['reliability_diag'][1]
    #print(mean_conf.shape)
    
    #calibration_results = {'reliability_diag': (torch.vstack(mean_conf), torch.vstack(mean_acc)), 'ece': ece}
    #calibration_results = {'reliability_diag': ((mean_conf), (mean_acc)), 'ece': ece}

    return ece

def cls_validate(val_loader, model, time_begin=None):
    model.eval()
    acc1_val = 0
    n = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = model(images)
            model_logits = output[0] if (type(output) is tuple) else output
            pred = model_logits.data.max(1, keepdim=True)[1]
            acc1_val += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            n += len(images)
    avg_acc1 = (acc1_val / n)
    return avg_acc1


def evaluate(folder, dataset, save_dir):

    datasetc = dataset + str("c")
    os.makedirs(os.path.join(save_dir, datasetc), exist_ok=True)

    models = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    models = sorted(models)
    #print(models)

    results = collections.defaultdict(dict)
    _, test_loader_clean = get_cifar10C_loaders(test_bs=1024)

    for index, m in enumerate(models):
        model = torch.load(folder + m)

        #model.load_state_dict(torch.load(folder + m))
        model.eval()
        
        # Clean Accuracy
        clean_test_acc = cls_validate(test_loader_clean, model, time_begin=None)
        print('Test Accuracy: ', clean_test_acc)
        ece = get_calibration(test_loader_clean, model, debug=False)
        print('Calibration: ', ece* 100)
        
        # Robust Accuracy
        rob_test_acc = []
        ece = []
        for noise in NOISE_TYPES:
            results[m][noise] = collections.defaultdict(dict)
            
            temp_results = []
            for severity in SEVERITIES:
                _, test_loader = get_cifar10C_loaders(test_bs=1024, severity=severity, noise=noise)
                result_m = cls_validate(test_loader, model)
                results[m][noise][severity] = result_m
                rob_test_acc.append(result_m)
                ece.append( get_calibration(test_loader, model, debug=False))
                temp_results.append(result_m)
            
            print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(noise, 100 * np.mean(temp_results)))

            print(temp_results)
            
            
        with open(f"{save_dir}/{datasetc}/robust_{m}.pickle", "wb") as f:
            np.save(f, result_m)

        print('***')
        print('Average Robust Accuracy: ', np.mean(rob_test_acc))
        print('ECE (%): {:.2f}'.format(np.mean(ece)* 100))
        print('***')
        
    return results


#########################################################

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixerBlock(nn.Module):
    def __init__(self, dim=64, kernel_size=8):
        super(ConvMixerBlock, self).__init__()
        self.mixer = nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) 

    def forward(self, x):
        return self.mixer(x)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d 
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.gelu = nn.GELU()
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1) 
        self.norm2 = nn.BatchNorm2d(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1) 
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(t, x)
        out = self.gelu(out)
        out = self.norm1(out)
        out = self.conv2(t, out)
        out = self.gelu(out)
        out = self.norm2(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol, method='dopri5')
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def get_cifar10_loaders(batch_size=128, test_batch_size=1000, perc=1.0):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True
    )

    train_eval_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, train_eval_loader

def get_cifar10C_loaders(test_bs=512, severity=0, noise='fog'):

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_loader = None

    testset = CIFARCorrupt(root=CIFAR10C_FOLDER,
                                        severity=[severity],
                                        noise=[noise],
                                        transform=transform_test)
    
    test_loader = torch.utils.data.DataLoader(testset,
                                                batch_size=test_bs,
                                                shuffle=False,
                                                num_workers=4,
                                                pin_memory=True)

    train_loader = None

    return train_loader, test_loader 


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    makedirs(args.save)
    logger = get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    dim = args.dim 

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(3, dim, kernel_size=8, stride=2),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        ]
    elif args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, dim, 3, 1),
            ResBlock(dim, dim, stride=2, downsample=conv1x1(dim, dim, 2)),
            ResBlock(dim, dim, stride=2, downsample=conv1x1(dim, dim, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(dim))] if is_odenet else [ConvMixerBlock(dim, 8) for _ in range(6)]
    fc_layers = [nn.Conv2d(dim, dim, kernel_size=1), nn.GELU(), nn.BatchNorm2d(dim), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim, 10)] if is_odenet else [nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(dim, 10)] 

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)
    model = nn.DataParallel(model).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_cifar10_loaders(
        args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch, boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    end = time.time()

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        if is_odenet:
            nfe_forward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        if itr % batches_per_epoch == 0:
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save({'state_dict': model.state_dict(), 'args': args}, os.path.join(args.save, 'model.pth'))
                    best_acc = val_acc
                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f}".format(
                        itr // batches_per_epoch, batch_time_meter.val, batch_time_meter.avg, f_nfe_meter.avg,
                        b_nfe_meter.avg, train_acc, val_acc
                    )
                )

    DESTINATION_PATH = args.data + '_models/'
    OUT_DIR = os.path.join(DESTINATION_PATH, f'final_arch_{args.network}_seed_{args.seed}')
    if not os.path.isdir(DESTINATION_PATH):
        os.mkdir(DESTINATION_PATH)
    torch.save(model, OUT_DIR+'.pt') 

    test_batch_size = args.batch_size
    os.makedirs('eval_results', exist_ok=True)
    evaluate(args.dir, args.data, 'eval_results')