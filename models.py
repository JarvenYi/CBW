# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
from torch.autograd import Variable
# utils
import math
import os
import datetime
import numpy as np
from sklearn.externals import joblib
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
    device = kwargs.setdefault('device', torch.device('cuda'))  # 给字典添加键值
    n_classes = kwargs['n_classes']
    n_bands = kwargs['n_bands']

    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs['ignored_labels'])] = 0.
    weights = weights.to(device)
    weights = kwargs.setdefault('weights', weights)

    if name == 'hamida':
        patch_size = kwargs.setdefault('patch_size', 5)     # 给传入的字典写入一个'patch_size'键，键值为5
        center_pixel = True
        model = HamidaEtAl(n_bands, n_classes, patch_size=patch_size)   # 模型
        lr = kwargs.setdefault('learning_rate', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0005)   # 优化器
        kwargs.setdefault('batch_size', 100)
        criterion = nn.CrossEntropyLoss(weight=kwargs['weights'])   # loss function     # weight参数分别代表n类的权重

    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    epoch = kwargs.setdefault('epoch', 100)
    kwargs.setdefault('scheduler', optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=epoch//4,
                                                                        verbose=True))    # 学习率调整，该方法提供了一些基于训练过程中的某些测量值对学习率进行动态的下降
    #kwargs.setdefault('scheduler', None)
    kwargs.setdefault('batch_size', 100)
    kwargs.setdefault('supervision', 'full')
    kwargs.setdefault('flip_augmentation', False)
    kwargs.setdefault('radiation_augmentation', False)
    kwargs.setdefault('mixture_augmentation', False)
    kwargs['center_pixel'] = center_pixel
    return model, optimizer, criterion, kwargs


class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)      # weight kaiming函数初始化
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        # The first layer is a (3,3,3) kernel sized Conv characterized
        # by a stride equal to 1 and number of neurons equal to 20
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = 1 #(dilation, 1, 1)     # 卷积核膨胀系数(空洞卷积)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1)
        # else:
        #     self.conv1 = nn.Sequential(
        #         nn.Conv3d(1, 16, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1, 1, 1)),            # 0 这里的输入channels为什么是1
        #         nn.BatchNorm3d(16),
        #         nn.LeakyReLU(),
        #         nn.Conv3d(16, 32, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1, 1, 1)),   # (1, 0, 0)
        #         nn.BatchNorm3d(32)
        #         )
        # # Next pooling is applied using a layer identical to the previous one
        # # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # # equal to 2 in order to reduce the spectral dimension
        # self.pool1 = nn.Sequential(
        #     nn.Conv3d(32, 32, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 1, 1)),       # stride=(2,1,1)(1, 0, 0)光谱数据维度pool
        #     nn.BatchNorm3d(32))
        # # Then, a duplicate of the first and second layers is created with
        # # 35 hidden neurons per layer.
        # self.conv2 = nn.Sequential(
        #     nn.Conv3d(32, 64, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 1, 1)),   # (1, 0, 0)
        #     nn.BatchNorm3d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1, 1, 1)),      # (1, 0, 0)
        #     nn.BatchNorm3d(128)
        # )
        # self.pool2 = nn.Sequential(
        #     nn.Conv3d(128, 128, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 1, 1)),     # stride=(2, 1, 1)  (1, 0, 0)
        #     nn.BatchNorm3d(128))
        # # Finally, the 1D spatial dimension is progressively reduced
        # # thanks to the use of two Conv layers, 35 neurons each,
        # # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # # respectively equal to (1,1,1) and (1,1,2)
        # self.conv3 = nn.Sequential(nn.Conv3d(
        #     128, 64, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 1, 1)),        # (1, 0, 0)
        #     nn.BatchNorm3d(64),
        #     nn.LeakyReLU(),
        #     nn.Conv3d(64, 32, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=(1, 0, 0)),   # (1, 0, 0)
        #     nn.BatchNorm3d(32)
        # )
        # self.pool3 = nn.Sequential(nn.Conv3d(
        #     32, 32, (2, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 1, 1)),     # stride=(2, 1, 1), padding=(1, 0, 0)
        #     nn.BatchNorm3d(32))

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(self.input_channels, 128, 3, stride=1, dilation=dilation, padding=1),            # 0 这里的输入channels为什么是1
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, self.input_channels, 3, stride=1, dilation=dilation, padding=1),   # (1, 0, 0)
                nn.BatchNorm2d(self.input_channels)
                )
        # Next pooling is applied using a layer identical to the previous one
        # with the difference of a 1D kernel size (1,1,3) and a larger stride
        # equal to 2 in order to reduce the spectral dimension
        self.pool1 = nn.Sequential(
            # nn.Conv2d(self.input_channels, 128, 1, dilation=dilation, stride=1, padding=1),       # stride=(2,1,1)(1, 0, 0)光谱数据维度pool
            nn.MaxPool2d(2, 1, padding=1),
            nn.Conv2d(self.input_channels, 128, 1, stride=1),
            nn.BatchNorm2d(128)
        )
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, dilation=dilation, stride=1, padding=1),   # (1, 0, 0)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=1, dilation=dilation, padding=1),      # (1, 0, 0)
            nn.BatchNorm2d(128)
        )
        self.pool2 = nn.Sequential(
            # nn.Conv2d(128, 64, 1, dilation=dilation, stride=1, padding=1),     # stride=(2, 1, 1)  (1, 0, 0)
            nn.MaxPool2d(2, 1, 1),
            nn.Conv2d(128, 64, 1, stride=1),
            nn.BatchNorm2d(64)
        )
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Sequential(nn.Conv2d(
            64, 32, 1, dilation=dilation, stride=1, padding=1),        # (1, 0, 0)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, 3, stride=1, dilation=dilation, padding=0),   # (1, 0, 0)
            nn.BatchNorm2d(64)
        )
        self.pool3 = nn.Sequential(
            # nn.Conv2d(64, 32, 1, dilation=dilation, stride=1, padding=1),     # stride=(2, 1, 1), padding=(1, 0, 0)
            nn.MaxPool2d(2, 1, 1),
            nn.Conv2d(64, 32, 1, stride=1),
            nn.BatchNorm2d(32)
        )

        #self.dropout = nn.Dropout(p=0.5)

        self.features_size = self._get_final_flattened_size()
        # The architecture ends with a fully connected layer where the number
        # of neurons is equal to the number of input classes.
        self.fc = nn.Linear(self.features_size, n_classes)
        self.apply(self.weight_init)

        def cascade_block(input_channels, nb_filter, kernels=3):
            conv1_1 = nn.Sequential(
                nn.Conv2d(input_channels, nb_filter * 2, (kernels, kernels), padding=1),
                nn.BatchNorm2d(nb_filter * 2),
                nn.LeakyReLU()
            )
            conv1_2 = nn.Sequential(
                nn.Conv2d(nb_filter*2, (1, 1), )
            )
    def _get_final_flattened_size(self):
        with torch.no_grad():       # 作用是在上下文环境中切断梯度计算，在此模式下，每一步的计算结果中requires_grad都是False
            res = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
            x = self.conv1(res)
            x = res+x

            res = self.pool1(x)
            x = self.conv2(res)
            x = res + x

            res = self.pool2(x)
            x = self.conv3(res)
            x = res+x

            x = self.pool3(x)

            # x = self.pool1(self.conv1(x))
            # x = self.pool2(self.conv2(x))
            # x = self.conv3(x)
            # x = self.pool3(x)
            # _, t, c, w, h = x.size()
            t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        res = self.conv1(x)
        x = res+x
        x = F.leaky_relu(x)
        x = self.pool1(x)
        res = self.conv2(x)
        x = res+x
        x = F.leaky_relu(x)
        x = self.pool2(x)
        res = self.conv3(x)
        x = res+x
        x = F.leaky_relu(x)
        x = self.pool3(x)
        x = x.view(-1, self.features_size)      # flatten
        #x = self.dropout(x)
        x = self.fc(x)
        return x

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

    save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=4e-08)  #

    for e in tqdm(range(1, epoch + 1), desc="Training the network"):       # tqdm(list)方法可以传入任意一种list   tqdm进展显示
        # Set the network to training mode
        lr_scheduler.step()

        net.train()
        avg_loss = 0.
        # if e//50 == 0 and e != 0:      # lr 衰减
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.1

        # Run the training loop for one epoch
        for batch_idx, (data, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # Load the data into the GPU if required
            data, target = Variable(data), Variable(target)     #自己加的Variable()   （好像加没加没看出有什么影响）
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

            if display_iter and iter_ % display_iter == 0:      # 每迭代100次更新一下 绘图
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
        if e % save_epoch == 0:
            save_model(net, camel_to_snake(str(net.__class__.__name__)), data_loader.dataset.name, epoch=e, metric=abs(metric))


def save_model(model, model_name, dataset_name, **kwargs):
    model_dir = './checkpoints/' + model_name + "/" + dataset_name + "/"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = str('wk') + "_epoch{epoch}_{metric:.2f}".format(**kwargs)
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + '.pth')
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
                # data = data.unsqueeze(1)      # Conv3D的维度要求

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


def val(net, data_loader, device='cuda', supervision='full'):   # 将cpu 改为 cuda
    # TODO : fix me using metrics()
    accuracy, total, acount= 0., 0., 0.
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
                    acount += 1
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total

