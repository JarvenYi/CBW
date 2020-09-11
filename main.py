# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import torch
import torch.utils.data as data
from torchsummary import summary

# Numpy, scipy, scikit-image, spectral
import numpy as np
import sklearn.svm
import sklearn.model_selection
from skimage import io
# VisualizationSE-
import seaborn as sns
import visdom       # 可视化工具
import torchvision
import scipy.io as scio
import os
from utils import metrics, convert_to_color_, convert_from_color_,\
    display_dataset, display_predictions, explore_spectrums, plot_spectrums, plot_spectrums_, \
    sample_gt, build_dataset, show_results, compute_imf_weights, get_device
from datasets import get_dataset, HyperX, open_file, DATASETS_CONFIG
from CBW import get_model, train, test, save_model
import argparse

dataset_names = [v['name'] if 'name' in v.keys() else k for k, v in DATASETS_CONFIG.items()]    # 提取dataset的名称

# 利用argparse设置参数
# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default='Salinas', choices=dataset_names,     # 数据集！！！
                    help="Dataset to use. IndianPines; PaviaU; Salinas")
parser.add_argument('--model', type=str, default="CBW",
                    help="Model to train. Available:\n"
                    "CBW (CBW + 2D CNN classifier), ")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, #default="./checkpoints/model/IndianPines/IndianPines_CBW.pth",
                    help="Weights to use for initialization, e.g. a checkpoint"
                         "./checkpoints/model/IndianPines/IndianPines_CBW.pth"
                         "./checkpoints/model/PaviaU/PaviaU_CBW.pth"
                         "./checkpoints/model/Salinas/Salinas_CBW.pth")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=0.01,           # 训练样本比率！！！
                    help="percentage of samples to use for training (0.-1.)(default: 0.5)")
group_dataset.add_argument('--sampling_mode', type=str, help="Sampling mode"
                    " (random sampling or disjoint, default: random)",
                    default='random')
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, default=100, help="Training epochs optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int, default=11,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")     # 这里对patch_size的设置并没有读取
group_train.add_argument('--lr', type=float, default=0.001,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true', default=True,
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int, default=100,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,     # 翻转
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true', default=False,    # 加噪声
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true', default=False,
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true', default=False,
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, nargs='+',
                    choices=dataset_names,
                    help="Download the specified datasets and quits.")
parser.add_argument('--train_sample_extend', type=bool, default=False,
                    help="train sample extended by flip.")

args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride
# Training sample extended by flip
TRAIN_SAMPLE_EXTEND = args.train_sample_extend

if args.download is not None and len(args.download) > 0:    # 下载数据集
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER, patch_size=PATCH_SIZE)
    quit()

viz = visdom.Visdom(env=DATASET + ' ' + MODEL)  # visdom可视化窗口
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")


hyperparams = vars(args)
# Load the dataset
img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER, patch_size=PATCH_SIZE)    # IGNORED_LABELS就是 undefined类别

# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = img.shape[-1]

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype='uint8'))
invert_palette = {v: k for k, v in palette.items()}
def convert_to_color(x):
    return convert_to_color_(x, palette=palette)
def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)

# Instantiate the experiment based on predefined networks
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

# Show the image and the ground truth
display_dataset(img, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
# display_dataset(ori_img, ori_gt, RGB_BANDS, LABEL_VALUES, palette, viz)
color_gt = convert_to_color(gt)

if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums, std_spectrums = explore_spectrums(img, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS)
    with open("mean_spectrum_Salinas.txt", 'w') as f:
        for ln, lv in mean_spectrums.items():
            f.write(str(lv))
            f.write('\n')
    plot_spectrums(mean_spectrums, viz, title='Mean spectrum/class')
    plot_spectrums_(std_spectrums, viz, title='Std spectrum/class')

results = []
# run the experiment several times
for run in range(N_RUNS):
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = open_file(TEST_GT)
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
    # Sample random training spectral
        gt_ = gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2),
              (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
        train_gt, test_gt = sample_gt(gt_, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE)
    # ----------------------------------------------------------------------------------
        mask = np.zeros_like(gt)
        for l in set(hyperparams['ignored_labels']):
            mask[gt == l] = 0
        x_pos, y_pos = np.nonzero(train_gt)
        indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        for x, y in indices:
            if mask[x+PATCH_SIZE//2, y+PATCH_SIZE//2] is not 0:
                mask[x+PATCH_SIZE//2, y+PATCH_SIZE//2] = gt[x+PATCH_SIZE//2, y+PATCH_SIZE//2]
        train_gt = mask
    # ----------------------------------------------------------------------------------
        test_gt = gt  # all of sample to be test sample
    # -----------------------------------------------------------------------------------------------------
    print("{} samples selected (over {})".format(np.count_nonzero(train_gt), np.count_nonzero(gt)))
    print("Running an experiment with the {} model".format(MODEL),
          "run {}/{}".format(run + 1, N_RUNS))

    display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
    display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")

    if MODEL == 'SGD':
        X_train, y_train = build_dataset(img, train_gt,
                                         ignored_labels=IGNORED_LABELS)
        X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        class_weight = 'balanced' if CLASS_BALANCING else None
        clf = sklearn.linear_model.SGDClassifier(class_weight=class_weight, learning_rate='optimal', tol=1e-3, average=10)
        clf.fit(X_train, y_train)
        save_model(clf, MODEL, DATASET)
        prediction = clf.predict(scaler.transform(img.reshape(-1, N_BANDS)))
        prediction = prediction.reshape(img.shape[:2])
    else:   # 自定义算法
        # Neural network
        model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)
        if CLASS_BALANCING: #
            weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
            hyperparams['weights'] = torch.from_numpy(weights)
        # Split train set in train/val
        # train_gt, val_gt = sample_gt(train_gt, 0.9, mode='random')
        # Generate the dataset
        train_dataset = HyperX(img, train_gt, **hyperparams)
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       pin_memory=hyperparams['device'],
                                       shuffle=True)
        # val_dataset = HyperX(img, val_gt, **hyperparams)
        # val_loader = data.DataLoader(val_dataset,
        #                              pin_memory=hyperparams['device'],
        #                              batch_size=hyperparams['batch_size'])

        print(hyperparams)
        print("Network :")
        with torch.no_grad():
            for input, _ in train_loader:
                break
            # summary(model.to(hyperparams['device']), input.size()[1:], device=hyperparams['device'])
            summary(model.to(hyperparams['device']), input.size()[1:])

        if CHECKPOINT is not None:
            """ load model """
            # model.load_state_dict(torch.load(CHECKPOINT))
            model = torch.load(CHECKPOINT)

        try:
            train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                  scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                  supervision=hyperparams['supervision'],
                  display=viz)
        except KeyboardInterrupt:
            # Allow the user to stop the training
            pass

        probabilities = test(model, img, hyperparams)
        prediction = np.argmax(probabilities, axis=-1)

    prediction = prediction[(PATCH_SIZE // 2):-(PATCH_SIZE // 2),
                 (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
    test_gt = test_gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2),
              (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]
    gt = gt[(PATCH_SIZE // 2):-(PATCH_SIZE // 2),
         (PATCH_SIZE // 2):-(PATCH_SIZE // 2)]

    run_results = metrics(prediction, test_gt, ignored_labels=hyperparams['ignored_labels'], n_classes=N_CLASSES)

    mask = np.zeros(gt.shape, dtype='bool')
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0
    color_prediction = convert_to_color(prediction)
    display_predictions(color_prediction, viz, gt=convert_to_color(gt), caption="Prediction vs. ground truth")

    results.append(run_results)
    show_results(run_results, viz, label_values=LABEL_VALUES)

if N_RUNS > 1:
    show_results(results, viz, label_values=LABEL_VALUES, agregated=True)
