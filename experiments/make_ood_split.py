"""Create OOD train/test splits for VGGSound by holding out N classes.

Usage: python experiments/make_ood_split.py --train_tsv sets/vgg3-train.tsv \
           --test_tsv sets/vgg3-test.tsv --n_holdout 30 --seed 0 \
           --out_dir sets/

Writes:
  <out_dir>/vgg3-train-ood.tsv  (training tsv with heldout classes removed)
  <out_dir>/vgg3-test-ood.tsv   (test tsv with ONLY heldout classes)
  <out_dir>/vgg3-ood-classes.txt (list of held-out class names)

The same logic is applied to the extracted-latent tsvs if --memmap_dir is given.
"""
import argparse
import random
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_tsv', required=True)
    p.add_argument('--test_tsv', required=True)
    p.add_argument('--n_holdout', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--label_col', default='label')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_tsv, sep='\t')
    test = pd.read_csv(args.test_tsv, sep='\t')

    classes = sorted(train[args.label_col].unique().tolist())
    rng = random.Random(args.seed)
    rng.shuffle(classes)
    held = set(classes[:args.n_holdout])

    (out / 'vgg3-ood-classes.txt').write_text('\n'.join(sorted(held)) + '\n')

    train_ood = train[~train[args.label_col].isin(held)]
    test_ood = test[test[args.label_col].isin(held)]

    train_ood.to_csv(out / 'vgg3-train-ood.tsv', sep='\t', index=False)
    test_ood.to_csv(out / 'vgg3-test-ood.tsv', sep='\t', index=False)

    print(f'Held out {len(held)} classes.')
    print(f'Train OOD rows: {len(train_ood)} / {len(train)}')
    print(f'Test OOD rows: {len(test_ood)} / {len(test)}')


if __name__ == '__main__':
    main()
