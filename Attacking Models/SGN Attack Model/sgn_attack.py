import argparse
import time
import shutil
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import os.path as osp
import csv
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm

np.random.seed(1337)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from model import SGN
from data import NTUDataLoaders, AverageMeter
from util import make_dir, get_num_classes
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

parser = argparse.ArgumentParser(
    description='SGN Attack Model for Skeleton Anonymization')
# Data should be a dictionary where the key is the name of the actor and the value is a list of skeletons
parser.add_argument('--data', type=str, default='data/X.pkl', help='dataset name')
parser.add_argument('--action_data', type=str, default=None, help='dataset name')

parser.add_argument('--model', type=str,
                    default='results/NTU/SGN/0_best.pth', help='model name')

parser.add_argument('--action_model', type=str, default='results/NTU/SGN/1_best.pth', help='action model name')

parser.add_argument('--labels', type=str, default='data/Genders.csv',
                    help='Actor,Label CSV file')

parser.add_argument('--source', type=str,
                    default='motion_privacy', help='skele_anon,motion_privacy: Used to control how data is parsed')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')

parser.add_argument('--num_workers', type=int, default=16, help='number of workers')

args = parser.parse_args()

# Below is directly from SGN.ipynb, adapted for testing only
network='SGN'
dataset='NTU'
start_epoch=0
batch_size=args.batch_size
max_epochs=120
monitor='val_acc'
lr=args.lr
weight_decay=0.0001
lr_factor=0.1
workers=args.num_workers
print_freq = 20
seg=20


def test(test_loader, model, checkpoint, lable_path, pred_path):
    acces = AverageMeter()
    # load learnt model that obtained best performance on validation set
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    model.eval()

    label_output = list()
    pred_output = list()

    t_start = time.time()
    for i, t in enumerate(test_loader):
        inputs = t[0]
        target = t[1]
        with torch.no_grad():
            output = model(inputs.cuda())
            output = output.view(
                (-1, inputs.size(0)//target.size(0), output.size(1)))
            output = output.mean(1)

        label_output.append(target.cpu().numpy())
        pred_output.append(output.cpu().numpy())

        acc = accuracy(output.data, target.cuda())
        acces.update(acc[0], inputs.size(0))

    label_output = np.concatenate(label_output, axis=0)
    np.savetxt(lable_path, label_output, fmt='%d')
    pred_output = np.concatenate(pred_output, axis=0)
    np.savetxt(pred_path, pred_output, fmt='%f')

    label_indices = np.argmax(label_output, axis=1)
    pred_indices = np.argmax(pred_output, axis=1)

    print(confusion_matrix(label_indices, pred_indices))
    print('F1: ' + str(f1_score(label_indices, pred_indices, average='macro')))
    print('Precision: ' + str(precision_score(label_indices, pred_indices, average='macro')))
    print('Recall: ' + str(recall_score(label_indices, pred_indices, average='macro')))
    print('Accuracy: ' + str(accuracy_score(label_indices, pred_indices)))

    print('Test: accuracy {:.3f}, time: {:.2f}s'
          .format(acces.avg, time.time() - t_start))


def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    target = torch.argmax(target, dim=1)  # Add this line to convert one-hot targets to class indices
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct = correct.view(-1).float().sum(0, keepdim=True)
    return correct.mul_(100.0 / batch_size)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def load_data(case):
    if case == 0:
        print(f"Loading data from {args.data}")
        with open(args.data, 'rb') as f:
            Actors = pickle.load(f)
        print("Data loaded")
        return Actors
    elif case == 1:
        print(f"Loading action data from {args.action_data}")
        with open(args.action_data, 'rb') as f:
            Actors = pickle.load(f)
        print("Data loaded")
        return Actors
    else:
        raise Exception("Invalid case, this error should be impossible")

def anonymizer_to_sgd(t, max_frames=300):
    # (300, 150)
    # [x:0,y:1,z:2][300][joints][actors]
    xyz,frames,joints,actors = t.shape
    # transpose to make loop simple
    # [frame][actors][joints][xyz]
    t = t.transpose(1,3,2,0)
    # make empty array
    frames = []
    
    joints_per_frame = xyz*joints*actors
    
    # crazy loop
    for frame in t:
        f = []
        for actor in frame:
            for joint in actor:
                for xyz in joint:
                    f.append(xyz)
        
        # Pad 0's to 150 joints (2 actors)
        if len(f) < joints_per_frame:
            f = np.pad(f, (0, joints_per_frame-len(f)), 'constant')
            
        frames.append(f)
        
    # to numpy array
    X = np.array(frames, dtype=np.float32)
    
    if X.shape[0] < max_frames:
        X = np.pad(X, ((0, max_frames-X.shape[0]), (0, 0)), 'constant')
        
    return X

def gen_labels(Actors, case):
    if case == 0:
        # Load the Genders
        Genders = pd.read_csv(args.labels)

        # Convert M to 1 and F to 0
        Genders = Genders.replace('M', 1).replace('F', 0)

        # Convert dataframe to oject where P is the key, and Gender is the value
        Genders = Genders.set_index('P').T.to_dict('list')

        # Convert to X and Y
        print("Generating labels")
        X = []
        Y = []
        for actor in tqdm(Actors, desc="Actor"):
            # Account for skeleton anonymization source
            if args.source == 'skele_anon':
                for skeleton in Actors[actor]:
                    s = anonymizer_to_sgd(skeleton)
                    if s is not None:
                        X.append(s)
                        Y.extend([Genders[actor]])
            else:
                X.extend(Actors[actor])
                Y.extend([Genders[actor]]*len(Actors[actor]))

        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int8)
        print("Labels generated")
        return X, Y
    elif case == 1:
        X = []
        Y = []
        for action in tqdm(Actors, desc="Action"):
            if args.source == 'skele_anon':
                for skeleton in Actors[action]:
                    s = anonymizer_to_sgd(skeleton)
                    if s is not None:
                        X.append(s)
                        # Get the action index
                        action_index = int(action[1:4])-1
                        # Convert to onehot encoding
                        y_ = np.zeros(120)
                        y_[action_index] = 1
                        # Append to X and Y
                        Y.extend([y_])
            else:
                # Get the action index
                action_index = int(action[1:4])-1
                # Convert to onehot encoding
                y_ = np.zeros(120)
                y_[action_index] = 1
                # Append to X and Y
                Y.extend([y_]*len(Actors[action]))
                X.extend(Actors[action])
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.int8)
        return X, Y
    else:
        raise ValueError("Invalid case, must be 0 or 1. This error shouldnt be hit :)")

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            target = torch.argmax(target, dim=1)  # Add this line to convert one-hot targets to class indices
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def to_categorical(y):
    return np.array([np.array([1, 0]) if i == 0 else np.array([0, 1]) for i in y])

def main(train_x, train_y, test_x, test_y, val_x, val_y, case):
    num_classes = get_num_classes(dataset, case)
    model = SGN(num_classes, dataset, seg, batch_size, 0)

    total = get_n_params(model)
    # print(model)
    print('The number of parameters: ', total)
    print('The modes is:', network)

    if torch.cuda.is_available():
        print('It is using GPU!')
        model = model.cuda()

    # Data loading
    ntu_loaders = NTUDataLoaders(dataset, case, seg=seg, train_X=train_x, train_Y=train_y, test_X=test_x, test_Y=test_y, val_X=val_x, val_Y=val_y, aug=0)

    test_loader = ntu_loaders.get_test_loader(32, workers)

    print('Testing with on %d samples' %
          (ntu_loaders.get_test_size()))

    output_dir = make_dir(dataset)

    save_path = os.path.join(output_dir, network)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint = osp.join(save_path, '%s_best.pth' % case)

    lable_path = osp.join(save_path, '%s_lable.txt' % case)
    pred_path = osp.join(save_path, '%s_pred.txt' % case)

    # Test
    model = SGN(num_classes, dataset, seg, batch_size, 0)
    model = model.cuda()
    test(test_loader, model, checkpoint, lable_path, pred_path)

if __name__ == '__main__':
    Actors = load_data(0)
    X, Y = gen_labels(Actors, 0)
    Y = to_categorical(Y)
    print(X.shape, Y.shape)
    try:
        assert len(X) == len(Y)
    except AssertionError:
        print("X and Y are not the same length")
    print("Evaluating model")

    # Create empty train/val sets
    train_x = np.zeros((batch_size, 300, 150))
    train_y = np.zeros((batch_size, 1))
    val_x = np.zeros((batch_size, 300, 150))
    val_y = np.zeros((batch_size, 1))

    # Gender Classification Attack
    print('\n\nPerforming Gender Classification Attack')
    main(train_x, train_y, X, Y, val_x, val_y, 0)


    # Action Classification Attack
    Actors = load_data(1)
    X, Y = gen_labels(Actors, 1)
    print(X.shape, Y.shape)
    try:
        assert len(X) == len(Y)
    except AssertionError:
        print("X and Y are not the same length")
    print("Evaluating model")
    print('\n\nPerforming Action Classification Attack')
    main(train_x, train_y, X, Y, val_x, val_y, 1)

