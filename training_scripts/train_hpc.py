# =============================================================================
# train_hpc.py  —  Illustrative training script (run on HPC with GPU)
# =============================================================================

import argparse, json, math, os, random, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

# ── Model imports (from models/ directory) ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from cnn_large_script import CNN_large
from cnn_small_script import CNN_small
from vit_large_script import ViT_large
from vit_small_script import ViT_small
from cct_large_script import CCT_large
from cct_small_script import CCT_small

# ── Model registry ────────────────────────────────────────────────────────────
def _vit_large(num_classes=10):
    return ViT_large(emb_dim=320, n_layers=6, heads=4, mlp_dim=640,
                     dropout=0.1, out_dim=num_classes)

MODEL_REGISTRY = {
    'cnn_large': (CNN_large,  'CNN', '5M'),
    'cnn_small': (CNN_small,  'CNN', '0.75M'),
    'vit_large': (_vit_large, 'ViT', '5M'),
    'vit_small': (ViT_small,  'ViT', '0.75M'),
    'cct_large': (CCT_large,  'CCT', '5M'),
    'cct_small': (CCT_small,  'CCT', '0.75M'),
}

FRACTIONS  = [0.1, 0.25, 0.5, 0.75, 1.0]
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_datasets(augment: bool):
    """Returns train/test datasets with or without augmentation."""
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        # No augmentation — only normalise (ablation condition)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True,  download=True, transform=transform_train)
    test_dataset  = torchvision.datasets.CIFAR10(
        root='./dataset', train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset


def get_nested_subset(dataset, seed: int, fraction: float):
    """
    Nested subset sampling: smaller fractions are strict subsets of larger ones.
    Guarantees that accuracy differences reflect data quantity, not identity.
    """
    rng     = np.random.RandomState(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    n = int(len(dataset) * fraction)
    return Subset(dataset, indices[:n])


# =============================================================================
# Train / Eval
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


# =============================================================================
# Train one fraction
# =============================================================================

def train_fraction(model_name, fraction, args, train_dataset, test_dataset, device):
    ModelClass, arch, regime = MODEL_REGISTRY[model_name]
    aug_tag = 'aug' if args.augment else 'noaug'

    print(f"\n  [{model_name}] frac={fraction} | {aug_tag} | "
          f"{int(len(train_dataset) * fraction):,} samples")

    set_seed(args.seed)

    train_loader = DataLoader(
        get_nested_subset(train_dataset, args.seed, fraction),
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4,
    )

    model     = ModelClass(num_classes=10).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=5e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Linear warm-up -> cosine annealing
    warmup = max(1, args.epochs // 20)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda ep: (
        (ep + 1) / warmup if ep < warmup
        else 0.5 * (1 + math.cos(math.pi * (ep - warmup) / (args.epochs - warmup)))
    ))

    history = []
    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        acc  = evaluate(model, test_loader, device)
        scheduler.step()
        history.append({'epoch': epoch, 'loss': loss, 'acc': acc})
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"    ep {epoch:3d} | loss {loss:.4f} | acc {acc:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    # Naming convention mirrors parameters_aug/ and parameters_noaug/
    base = os.path.join(os.path.dirname(__file__), '..')
    if args.augment:
        pt_path  = os.path.join(base, 'parameters_aug',
                                f'{model_name}_seed{args.seed}_frac{fraction}.pt')
        json_dir = os.path.join(base, f'resultsxepochs_aug/{arch}({regime})')
        json_path = os.path.join(json_dir,
                                 f'{model_name}_seed{args.seed}_frac{fraction}.json')
    else:
        pt_path  = os.path.join(base, 'parameters_noaug',
                                f'{model_name}_noaug_seed{args.seed}_frac{fraction}.pt')
        json_dir = os.path.join(base, f'resultsxepochs_noaug/{arch}({regime})')
        json_path = os.path.join(json_dir,
                                 f'{model_name}_noaug_seed{args.seed}_frac{fraction}.json')

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    torch.save(model.state_dict(), pt_path)
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"    Saved: {pt_path}")
    return history[-1]['acc']


# =============================================================================
# Main
# =============================================================================

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device   : {device}")
    print(f"Model    : {args.model}")
    print(f"Augment  : {args.augment}")
    print(f"Seed     : {args.seed}")
    print(f"Epochs   : {args.epochs}")
    print(f"Fractions: {FRACTIONS}")

    train_dataset, test_dataset = get_datasets(args.augment)

    for fraction in FRACTIONS:
        train_fraction(args.model, fraction, args,
                       train_dataset, test_dataset, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train one architecture on CIFAR-10 across all data fractions.')
    parser.add_argument('--model',      required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Architecture to train')
    parser.add_argument('--augment',    action='store_true',  default=True)
    parser.add_argument('--no-augment', action='store_false', dest='augment')
    parser.add_argument('--seed',       type=int,   default=0)
    parser.add_argument('--epochs',     type=int,   default=150)
    parser.add_argument('--batch_size', type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
