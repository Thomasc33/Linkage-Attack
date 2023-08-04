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

def load_data(source):
    print(f"Loading action data from {source}")
    with open(source, 'rb') as f:
        d = pickle.load(f)
    print("Data loaded")
    return d

def anonymizer_to_sgd(t, max_frames=300):
    # (300, 150)
    # [x:0,y:1,z:2][300][joints][actors]
    xyz, frames, joints, actors = t.shape
    # transpose to make loop simple
    # [frame][actors][joints][xyz]
    t = t.transpose(1, 3, 2, 0)
    
    # Reshape t into the desired 2D array
    t_reshaped = t.reshape(t.shape[0], -1)
    
    # Pad 0's to 150 joints (2 actors)
    if t_reshaped.shape[1] < xyz * joints * actors:
        t_reshaped = np.pad(t_reshaped, ((0, 0), (0, xyz * joints * actors - t_reshaped.shape[1])), 'constant')
    
    # Pad 0's to max_frames
    if t_reshaped.shape[0] < max_frames:
        t_reshaped = np.pad(t_reshaped, ((0, max_frames - t_reshaped.shape[0]), (0, 0)), 'constant')
        
    return t_reshaped.astype(np.float32)

def gen_labels(Actors, case, key_is_file=False, source=args.source):
    if key_is_file:
        X = {}
        if case == 0:
            for file in Actors:
                if file[8:12] not in X:
                    X[file[8:12]] = []
                X[file[8:12]].append(Actors[file])
        elif case == 1:
            for file in Actors:
                if file[16:20] not in X:
                    X[file[16:20]] = []
                X[file[16:20]].append(Actors[file])
        for key in X:
            X[key] = np.array(X[key])
            if source == 'skele_anon':
                X[key] = np.squeeze(X[key], axis=1)
        Actors = X

        if source == 'raw':
            # pad data to (None, 300, 150)
            for key in Actors:
                t=[]
                for ex in Actors[key]:
                    pad_width = ((0, 300 - ex.shape[0]), (0, 150 - ex.shape[1]))
                    ex_padded = np.pad(ex, pad_width=pad_width, mode='constant', constant_values=0)
                    t.append(ex_padded)
                Actors[key] = np.array(t)
                
    if case == 0:
        # Load the Genders
        Genders = pd.read_csv(args.labels)

        # Convert M to 1 and F to 0
        Genders = Genders.replace('M', 1).replace('F', 0)

        # Convert dataframe to oject where P is the key, and Gender is the value
        Genders = Genders.set_index('P').T.to_dict('list')

        # Convert to X and Y
        X = []
        Y = []
        for actor in tqdm(Actors, desc="Actor"):
            # Account for skeleton anonymization source
            if source == 'skele_anon':
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
        return X, Y
    elif case == 1:
        X = []
        Y = []
        for action in tqdm(Actors, desc="Action"):
            if source == 'skele_anon':
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

def main(train_x, train_y, test_x, test_y, val_x, val_y, case, num_classes = None):
    if num_classes is None:
        num_classes = get_num_classes(dataset, case)
    model = SGN(num_classes, dataset, seg, batch_size, 0)

    if torch.cuda.is_available():
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

# Temp for evaluation
data = [
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\Linkage Attack\\RF Based Linkage Attack\\X.pkl', # Raw Data
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\External Repositories\\Skeleton-anonymization\\X_resnet_file.pkl', # Moon ResNet
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\External Repositories\\Skeleton-anonymization\\X_unet_file.pkl', # Moon UNet
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\Defense Models\\Mean Skeleton\\X_FileNameKey_SingleActor_filtered.pkl' # Classical MR
]

file_names = [ # Arrays of file names to use for each data source
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\Skeleton Info\\File Sorting\\ntu60.pkl',
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\Linkage Attack\\SGN Based Linkage Attack\\data\\ntu120.pkl',
    'C:\\Users\\Carrt\\OneDrive\\Code\\Motion Privacy\\Linkage Attack\\SGN Based Linkage Attack\\data\\ntu120_no_dupe_actors.pkl',
    None # Use all data (ntu60+120)
]
# Delete this when done with evaluations

if __name__ == '__main__':
    # Temp for evaluation
    for source in data:
        for i in range(len(file_names)):
            print('\n\n\n-------------------------------------------------------------------------------\n')

            Actors = load_data(source)

            if file_names[i] is not None:
                with open(file_names[i], 'rb') as f:
                    files_to_use = pickle.load(f)

                to_del = []
                for file in Actors:
                    if file not in files_to_use:
                        to_del.append(file)

                for to_delete in to_del:
                    del Actors[to_delete]

            source_ = 'rf'
            if 'Skeleton-anonymization' in source: source_ = 'skele_anon'
            elif 'X.pkl' in source: source_ = 'raw'

            X, Y = gen_labels(Actors, 0, key_is_file=True, source=source_)

            with open('x_dump.pkl', 'wb') as f:
                pickle.dump(X, f)
            Y = to_categorical(Y)
            try:
                assert len(X) == len(Y)
            except AssertionError:
                print("X and Y are not the same length")

            # Create empty train/val sets
            train_x = np.zeros((batch_size, 300, 150))
            train_y = np.zeros((batch_size, 1))
            val_x = np.zeros((batch_size, 300, 150))
            val_y = np.zeros((batch_size, 1))

            # Gender Classification Attack
            print('\n\nPerforming Gender Classification Attack\n\n')
            main(train_x, train_y, X, Y, val_x, val_y, 0)

            # Action Classification Attack
            X, Y = gen_labels(Actors, 1, key_is_file=True, source=source_)
            print(X.shape, Y.shape)
            try:
                assert len(X) == len(Y)
            except AssertionError:
                print("X and Y are not the same length")
            print("Evaluating model")
            print('\n\nPerforming Action Classification Attack\n\n')
            main(train_x, train_y, X, Y, val_x, val_y, 1)


    # Remove above when done with evaluation



    # Actors = load_data(args.data)
    # X, Y = gen_labels(Actors, 0)
    # Y = to_categorical(Y)
    # print(X.shape, Y.shape)
    # try:
    #     assert len(X) == len(Y)
    # except AssertionError:
    #     print("X and Y are not the same length")
    # print("Evaluating model")

    # # Create empty train/val sets
    # train_x = np.zeros((batch_size, 300, 150))
    # train_y = np.zeros((batch_size, 1))
    # val_x = np.zeros((batch_size, 300, 150))
    # val_y = np.zeros((batch_size, 1))

    # # Gender Classification Attack
    # print('\n\nPerforming Gender Classification Attack')
    # main(train_x, train_y, X, Y, val_x, val_y, 0)


    # # Action Classification Attack
    # Actors = load_data(args.action_data)
    # X, Y = gen_labels(Actors, 1)
    # print(X.shape, Y.shape)
    # try:
    #     assert len(X) == len(Y)
    # except AssertionError:
    #     print("X and Y are not the same length")
    # print("Evaluating model")
    # print('\n\nPerforming Action Classification Attack')
    # main(train_x, train_y, X, Y, val_x, val_y, 1)

