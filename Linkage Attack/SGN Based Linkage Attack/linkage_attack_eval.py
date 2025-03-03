#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standalone script for evaluating a Linkage Attack model on skeleton data,
with:
  1) Optimized data sampling for same/diff pairs,
  2) Padding from (T,75)->(T,150) if only one actor is present,
  3) NO transposition to (150, T). We keep (T,150) => [frames, features],
  4) Debug prints to confirm shape,
  5) SGN initialization exactly as requested,
  6) Default device='cuda'.

Usage Example:
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "C:\\Users\\Carrt\\OneDrive\\Code\\Motion Retargeting\\NTU\\SGN\\X_full.pkl" --max_frames 75 --compute_auc
  
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "C:\\Users\\Carrt\\OneDrive\\Code\\Motion Retargeting\\NTU\\SGN\\X_full.pkl" --max_frames 75
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "E:/LocalCode/Linkage Attack/External Repositories/Skeleton-anonymization/X_resnet_file.pkl" --max_frames 75
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "E:/LocalCode/Linkage Attack/External Repositories/Skeleton-anonymization/X_unet_file.pkl" --max_frames 75
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "C:/Users/Carrt/OneDrive/Code/Motion Retargeting/results/DMR_X_hat_constant_CA.pkl" --max_frames 75
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "C:/Users/Carrt/OneDrive/Code/Motion Retargeting/results/X_hat_constant_CA.pkl" --max_frames 75
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "C:/Users/Carrt/OneDrive/Code/Motion Retargeting/results/NTU_120_DMR_X_hat_constant_CA.pkl" --max_frames 75
  python linkage_attack_eval.py --model models/2unfrozen_67acc.pt --data "C:/Users/Carrt/OneDrive/Code/Motion Retargeting/results/DMR_X_hat_constant_CA.pkl" --max_frames 75
"""

import argparse
import pickle
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    roc_auc_score
)
from tqdm import tqdm

# If your "SGN" implementation is in a separate file "model.py", make sure
# it's in the same folder or on your PYTHONPATH.
from model import SGN

##############################################################################
# 1. Utility Function: Parse the File Name
##############################################################################

def parse_file_name(file_name):
    """
    Extracts numeric tags from an NTU-style skeleton file name.
    Returns a dict with S, C, P, R, A keys.

    Examples:
      - S003C001P004R002A008 => { 'S':3, 'C':1, 'P':4, 'R':2, 'A':8 }
      - b-S003C001P004R002A008 => same values
    """
    file_name = str(file_name)
    if file_name[0] == 'b':  # SGN preprocessing style
        S = int(file_name[3:6])
        C = int(file_name[7:10])
        P = int(file_name[11:14])
        R = int(file_name[15:18])
        A = int(file_name[19:22])
    else:
        S = int(file_name[1:4])
        C = int(file_name[5:8])
        P = int(file_name[9:12])
        R = int(file_name[13:16])
        A = int(file_name[17:20])
    return {'S': S, 'C': C, 'P': P, 'R': R, 'A': A}


##############################################################################
# 2. Optimized Data Generation (Pairing Skeletons)
##############################################################################

def data_generator_per_actor(
    data_dict,
    same_samples_per_actor=100,
    diff_samples_per_actor=100,
    seed=42
):
    """
    Produces pairs of skeleton sequences for a linkage task:
      - label=1 if from the same actor,
      - label=0 if from different actors.

    Args:
      data_dict: { filename : np.array([...]) }
      same_samples_per_actor: # of same-actor pairs per actor
      diff_samples_per_actor: # of diff-actor pairs per actor
      seed: reproducibility seed

    Returns:
      list of (pair_array, label), each pair_array has shape [2, frames, features].
    """
    rng = np.random.default_rng(seed)

    # 1) Build a global list of (actor_id, skeleton_array).
    all_seqs = []
    actor_indices = {}  # actor_id -> global indices in all_seqs

    print("Building global list of skeleton sequences...")
    idx_counter = 0
    for filename, skeleton in data_dict.items():
        actor_id = parse_file_name(filename)['P']
        all_seqs.append((actor_id, skeleton))
        actor_indices.setdefault(actor_id, []).append(idx_counter)
        idx_counter += 1

    all_seqs = np.array(all_seqs, dtype=object)
    all_indices = np.arange(len(all_seqs))

    # 2) For each actor, sample same/diff pairs
    pairs = []
    actor_ids = sorted(actor_indices.keys())

    print("Generating same/diff pairs for each actor (fast method)...")
    for actor_id in tqdm(actor_ids, desc="Actors", leave=True):
        idxs_actor = actor_indices[actor_id]
        M = len(idxs_actor)
        if M == 0:
            continue

        # ----- Same-actor pairs -----
        if same_samples_per_actor > 0 and M > 0:
            same_draws = rng.integers(0, M, size=2 * same_samples_per_actor)
            first_half = same_draws[:same_samples_per_actor]
            second_half = same_draws[same_samples_per_actor:]
            for i in range(same_samples_per_actor):
                idx1 = idxs_actor[first_half[i]]
                idx2 = idxs_actor[second_half[i]]
                x1 = all_seqs[idx1][1]
                x2 = all_seqs[idx2][1]
                pair_array = np.array([x1, x2], dtype=np.float32)
                pairs.append((pair_array, 1))

        # ----- Different-actor pairs -----
        if diff_samples_per_actor > 0:
            mask = np.ones(len(all_seqs), dtype=bool)
            mask[idxs_actor] = False
            valid_diff = all_indices[mask]
            if len(valid_diff) == 0:
                continue

            diff_draws_x1 = rng.integers(0, M, size=diff_samples_per_actor)
            diff_draws_x2 = rng.integers(0, len(valid_diff), size=diff_samples_per_actor)
            for i in range(diff_samples_per_actor):
                idx1 = idxs_actor[diff_draws_x1[i]]
                idx2 = valid_diff[diff_draws_x2[i]]
                x1 = all_seqs[idx1][1]
                x2 = all_seqs[idx2][1]
                pair_array = np.array([x1, x2], dtype=np.float32)
                pairs.append((pair_array, 0))

    rng.shuffle(pairs)
    print(f"Total pairs generated: {len(pairs)}")
    return pairs


##############################################################################
# 3. Dataset: SGN_Linkage_Dataset
##############################################################################

class SGN_Linkage_Dataset(Dataset):
    """
    For each (x_a, x_b, label):
      1) If shape is (T,75), pad to (T,150).
      2) Trim or extend to 'max_frames' along T dimension.
      3) Remove zero frames.
      4) Segment-based random sampling => shape (seg, 150).
      5) Return shape (seg, 150) as a torch.Tensor => [frames=seg, features=150].

    Stacking yields [batch_size, seg, 150], as SGN typically expects [B, T, 150].
    """

    def __init__(self, pairs, seg=20, device='cuda', max_frames=300):
        """
        Args:
            pairs: list of (np.array([2, frames, features]), label)
            seg: # of segments for the SGN extraction
            device: CPU or CUDA device string (default 'cuda')
            max_frames: T to fix each sequence
        """
        self.pairs = pairs
        self.seg = seg
        self.device = device
        self.max_frames = max_frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair_array, label = self.pairs[idx]
        x_a, x_b = pair_array[0], pair_array[1]

        # Possibly shape is (T,75). If so, pad => (T,150).
        x_a_fixed = self._fix_and_pad_actors(x_a)
        x_b_fixed = self._fix_and_pad_actors(x_b)

        # Preprocess for SGN
        x_a_tensor = self._preprocess_single(x_a_fixed)
        x_b_tensor = self._preprocess_single(x_b_fixed)

        # Return on the same device
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return (x_a_tensor.to(self.device),
                x_b_tensor.to(self.device),
                label_tensor.to(self.device))

    def _fix_and_pad_actors(self, arr):
        """
        1) If arr.shape[1] == 75 => pad to (T,150) with zero columns.
        2) Then fix #frames => (max_frames, 150).
        """
        # if arr.shape[1] == 75:
        #     zeros_2nd = np.zeros((arr.shape[0], 75), dtype=arr.dtype)
        #     arr = np.concatenate([arr, zeros_2nd], axis=1)  # => (T,150)

        if arr.shape[1] == 150:
            arr = arr[:, :75]

        # If we see (T,150) we keep it. Next we trim or pad frames:
        frames = arr.shape[0]
        if frames > self.max_frames:
            arr = arr[:self.max_frames]
        elif frames < self.max_frames:
            needed = self.max_frames - frames
            last_frame = arr[-1:].copy()
            arr = np.concatenate((arr, np.repeat(last_frame, needed, axis=0)), axis=0)

        return arr

    def _preprocess_single(self, arr):
        """
        1) Remove zero frames if present,
        2) If < seg => repeat last frame,
        3) Segment-based random sampling => shape (seg, 150),
        4) Return torch.Tensor => (seg,150).
        """
        mask = ~(np.all(arr == 0, axis=1))
        arr = arr[mask]
        if arr.shape[0] == 0:
            arr = np.zeros((1, arr.shape[1]), dtype=np.float32)

        if arr.shape[0] < self.seg:
            needed = self.seg - arr.shape[0]
            last_frame = arr[-1:].copy()
            arr = np.concatenate((arr, np.repeat(last_frame, needed, axis=0)), axis=0)

        # Random segment-based sampling
        frames = arr.shape[0]
        avg_duration = frames // self.seg
        if avg_duration == 0:
            indices = np.arange(min(frames, self.seg))
        else:
            rng = np.random.default_rng()
            offsets = rng.integers(0, avg_duration, size=self.seg)
            base_offsets = np.arange(self.seg) * avg_duration
            indices = base_offsets + offsets
            indices = np.clip(indices, 0, frames - 1)

        sub_seq = arr[indices]  # shape (seg, 150)

        return torch.tensor(sub_seq, dtype=torch.float32)


##############################################################################
# 4. Linkage Attack Model
##############################################################################

class SGN_Linkage_Attack(nn.Module):
    """
    Combines two SGN encoders (model_a, model_b) for binary classification:
    same actor or not.
    """

    def __init__(self, model_a, model_b, output_size=1):
        super(SGN_Linkage_Attack, self).__init__()
        self.model_a = model_a
        self.model_b = model_b

        self.conv1 = nn.Conv1d(1024, 512, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x_a, x_b):
        """
        x_a, x_b => shape: [batch_size, seg, 150]
        The SGN model typically does .view(bs, seg, 50, 3) inside model_a & model_b.
        """
        feat_a = self.model_a(x_a)  # (B, 512)
        feat_b = self.model_b(x_b)  # (B, 512)

        concat = torch.cat((feat_a, feat_b), dim=1)  # (B, 1024)
        concat = concat.unsqueeze(-1)                # (B, 1024, 1)

        out = F.relu(self.bn1(self.conv1(concat)))   # -> (B, 512, 1)
        out = self.dropout1(out)
        out = out.view(out.size(0), -1)              # -> (B, 512)
        out = F.relu(self.bn2(self.fc1(out)))        # -> (B, 256)
        out = self.dropout2(out)
        out = self.fc2(out)                          # -> (B, 128)
        out = self.dropout3(out)
        out = self.fc3(out)                          # -> (B, 1)
        out = torch.sigmoid(out)
        return out


##############################################################################
# 5. Evaluation Routine
##############################################################################

def evaluate_model(model, data_loader, device='cuda', compute_auc=False):
    """
    Iterates over data_loader, collects predictions, computes:
      - BCELoss
      - Accuracy, Precision, Recall, F1
      - Confusion Matrix
      - (optional) ROC-AUC
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    criterion = nn.BCELoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_a, x_b, labels in tqdm(data_loader, desc="Evaluating", leave=True):
            # We assume x_a,x_b,labels are already on device (cuda)
            outputs = model(x_a, x_b).squeeze(dim=1)  # shape (batch_size,)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_samples += labels.size(0)

            predicted = (outputs >= 0.5).long()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_probs.extend(outputs.cpu().numpy().tolist())

    avg_loss = total_loss / (total_samples + 1e-8)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    auc_score = None
    if compute_auc and len(set(all_labels)) == 2:
        auc_score = roc_auc_score(all_labels, all_probs)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'auc': auc_score,
    }


##############################################################################
# 6. Main: Argument Parsing & Execution
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Evaluate a Linkage Attack model on skeleton data (no transpose).")
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the saved Linkage Attack model (.pt).')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to the pickled dictionary of skeleton data.')
    # Default device is now 'cuda' (instead of 'cpu').
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu).')
    parser.add_argument('--samples_same', type=int, default=200,
                        help='Number of same-actor pairs per actor.')
    parser.add_argument('--samples_diff', type=int, default=200,
                        help='Number of diff-actor pairs per actor.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation.')
    parser.add_argument('--segments', type=int, default=20,
                        help='Number of segments for the SGN input (T).')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Max frames T for each sample (trim or extend).')
    parser.add_argument('--compute_auc', action='store_true',
                        help='Compute and display ROC-AUC as well.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of evaluation runs to perform for statistics.')
    args = parser.parse_args()

    # If the user explicitly did --device cpu but there's a GPU, that's fine.
    # If they do not have a GPU but default is 'cuda', it will fail unless 
    # they override with --device cpu.
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu'
                          else 'cpu')

    print("==========================================")
    print(" Linkage Attack Evaluation (No Transpose) ")
    print("==========================================")
    print(f"Model file:         {args.model}")
    print(f"Data file:          {args.data}")
    print(f"Device:             {device}")
    print(f"Samples (same/diff) {args.samples_same}/{args.samples_diff}")
    print(f"Batch size:         {args.batch_size}")
    print(f"Segments (SGN):     {args.segments}")
    print(f"Max Frames (T):     {args.max_frames}")
    print(f"Compute AUC:        {args.compute_auc}")
    print(f"Base Seed:          {args.seed}")
    print(f"Number of runs:     {args.runs}")
    print("------------------------------------------")

    # 1) Load skeleton data
    print("Loading skeleton data from pickle...")
    with open(args.data, 'rb') as f:
        data_dict = pickle.load(f)
    print(f"Loaded {len(data_dict)} skeleton sequences.")

    # Initialize model once
    num_classes = 1
    SGN_Encoder1 = SGN(
        num_classes=num_classes,
        dataset='NTU',
        seg=args.segments,
        batch_size=args.batch_size
    ).to(device)

    SGN_Encoder2 = SGN(
        num_classes=num_classes,
        dataset='NTU',
        seg=args.segments,
        batch_size=args.batch_size
    ).to(device)

    model = SGN_Linkage_Attack(SGN_Encoder1, SGN_Encoder2, output_size=1).to(device)

    # Load saved model
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Could not find model file: {args.model}")
    print(f"Loading model from {args.model} ...")
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    
    # Store results across multiple runs
    all_results = []
    
    # Run experiment multiple times
    for run in range(args.runs):
        current_seed = args.seed + run
        print(f"\n--- Run {run+1}/{args.runs} (seed: {current_seed}) ---")
        
        # Generate pairs with a different seed each run
        print("Generating pairs (same/diff)...")
        pairs = data_generator_per_actor(
            data_dict=data_dict,
            same_samples_per_actor=args.samples_same,
            diff_samples_per_actor=args.samples_diff,
            seed=current_seed
        )

        # Create dataset & loader
        dataset_eval = SGN_Linkage_Dataset(
            pairs,
            seg=args.segments,
            device=device,
            max_frames=args.max_frames
        )
        loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size,
                                 shuffle=False, drop_last=False)

        # Evaluate
        print("Starting evaluation loop...")
        results = evaluate_model(model, loader_eval, device=device, compute_auc=args.compute_auc)
        all_results.append(results)
        
        # Print individual run results
        print(f"\n=== Run {run+1} Results ===")
        print(f"Loss:      {results['loss']:.4f}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"F1 Score:  {results['f1']:.4f}")
    
    # Calculate statistics
    accuracy_values = [r['accuracy'] for r in all_results]
    f1_values = [r['f1'] for r in all_results]
    loss_values = [r['loss'] for r in all_results]
    
    mean_accuracy = np.mean(accuracy_values)
    std_accuracy = np.std(accuracy_values)
    mean_f1 = np.mean(f1_values)
    std_f1 = np.std(f1_values)
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    
    # Print summary statistics
    print("\n========================================")
    print("          EXPERIMENT SUMMARY            ")
    print("========================================")
    print(f"Data file: {args.data}")
    print(f"Model file: {args.model}")
    print(f"Number of runs: {args.runs}")
    print(f"Samples per actor: {args.samples_same} same, {args.samples_diff} diff")
    print("\n--- Performance Metrics ---")
    print(f"Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
