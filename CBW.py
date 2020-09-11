# Torch
" 这个模型是用于测试 "
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# utils
import math
import os
import datetime
import numpy as np
from sklearn.externals import joblib
import sklearn
from tqdm import tqdm
from utils import grouper, sliding_window, count_sliding_window,\
                  camel_to_snake

def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        models: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault('device', torch.device('cuda'))
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']

    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'CBW':
        patch_size = kwargs.setdefault('patch_size', 5)
        center_pixel = True
        model = MODEL(n_bands, n_classes, patch_size=patch_size)
        lr = kwargs.setdefault('lr', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)
        kwargs.setdefault('batch_size', 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])   # loss function

    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    # model = model.cuda()
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch//4,
                                                                        verbose=True))
    #kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs

class CBW_BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1, padding=1, downsample=None, patch_size=11, count=1):
        super(CBW_BasicBlock, self).__init__()
        self.stride = stride
        self.count = count
        self.padding = padding
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.globalAvgPool = nn.AvgPool2d(patch_size, stride=1, padding=0)
        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(1, 1, 7, 1, 3, groups=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        self.conv1d_12 = nn.Sequential(
            nn.Conv1d(1, 1, 7, 1, 3, groups=1, bias=False),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            )
        self.sigmoid = nn.Sigmoid()
        self.relu_ = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        original_out = x
        out = self.globalAvgPool(x)
        out = out.squeeze(3)
        out0 = out.transpose(2, 1)
        out1 = self.conv1d_1(out0)
        out2 = self.conv1d_12(out1)
        out = out1 + out2 + out0
        out = self.bn(out)
        out = self.sigmoid(out)
        out = out.unsqueeze(3).transpose(2, 1)
        out = out * original_out
        return out

class MODEL(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=21):
        super(MODEL, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        kernel_size = 3
        nb_filter = 16

        self.cbw = nn.Sequential(
            CBW_BasicBlock(self.input_channels, self.input_channels, stride=1, padding=1, patch_size=patch_size, count=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, nb_filter*4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(nb_filter*4),
            nn.LeakyReLU(),
        )
        self.maxpool = nn.MaxPool2d((2, 2), 2, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(nb_filter*4, nb_filter*8, kernel_size, padding=1),
            nn.BatchNorm2d(nb_filter*8),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(nb_filter * 8, nb_filter * 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(nb_filter*16),
            nn.LeakyReLU()
        )
        self.flattened_size = self.flattened()
        self.fc_1 = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.BatchNorm1d(1024),
        )
        self.fc_2 = nn.Sequential(
            nn.Linear(1024, n_classes),
            nn.BatchNorm1d(n_classes)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def flattened(self):
        with torch.no_grad():
            x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size,))
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.conv2(x)
            x = self.maxpool(x)
            x = self.conv3(x)
            x = self.maxpool(x)
            t, w, l, b = x.size()
            return t*w*l*b

    def forward(self, x):
        x = self.cbw(x)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.reshape(-1, self.flattened_size)
        x = self.fc_1(x)
        out = self.fc_2(x)
        return out

def train(net, optimizer, criterion, data_loader, epoch, scheduler=None,
          display_iter=100, device=torch.device('cuda'), display=None,
          val_loader=None, supervision='full'):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """

    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)
    # net = nn.DataParallel(net)
    save_epoch = epoch
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []
    for e in tqdm(range(1, epoch + 1), desc="Training the network"):
        # Set the network to training mode
        net.train()
        avg_loss = 0.
        if e==30:
            for p in optimizer.param_groups:
                p['lr'] *= 0.5
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == 'full':
                output = net(data)
                loss = criterion(output, target)

            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[1](rec, data)
            else:
                raise ValueError("supervision mode \"{}\" is unknown.".format(supervision))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if display_iter and iter_ % display_iter == 0:
                string = 'Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
                string = string.format(
                    e, epoch, batch_idx *
                    len(data), len(data) * len(data_loader),
                    100. * batch_idx / len(data_loader), mean_losses[iter_])
                update = None if loss_win is None else 'append'
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter:iter_],
                    win=loss_win,
                    update=update,
                    opts={'title': "Training loss",
                          'xlabel': "Iterations",
                          'ylabel': "Loss"
                         }
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(Y=np.array(val_accuracies),
                                           X=np.arange(len(val_accuracies)),
                                           win=val_win,
                                           opts={'title': "Validation accuracy",
                                                 'xlabel': "Epochs",
                                                 'ylabel': "Accuracy"
                                                 })
            iter_ += 1
            del(data, target, loss, output)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        if e == 100:
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e, metric=abs(metric))


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str('wk') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        # torch.save(model.state_dict(), model_dir + filename + '.pth')
        torch.save(model, model_dir + filename + '.pth')
    else:
        filename = str('wk')
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + '.pkl')


def test(net, img, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams['patch_size']
    center_pixel = hyperparams['center_pixel']
    batch_size, device = hyperparams['batch_size'], hyperparams['device']
    n_classes = hyperparams['n_classes']

    kwargs = {'step': hyperparams['test_stride'], 'window_size': (patch_size, patch_size)}
    probs = np.zeros(img.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img, **kwargs) // batch_size
    for batch in tqdm(grouper(batch_size, sliding_window(img, **kwargs)),
                      total=(iterations),
                      desc="Inference on the image"
                      ):
        with torch.no_grad():
            if patch_size == 1:
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)

            indices = [b[1:] for b in batch]
            data = data.to(device)
            output = net(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.to('cpu')  # 将cpu 改为 cuda

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x:x + w, y:y + h] += out
    return probs


def val(net, data_loader, device='cpu', supervision='full'):
    # TODO : fix me using metrics()
    accuracy, total = 0., 0.
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, target = data.to(device), target.to(device)
            if supervision == 'full':
                output = net(data)
            elif supervision == 'semi':
                outs = net(data)
                output, rec = outs
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total