"""Aggregate test-output_metrics.json across all output/<exp_id>/ runs.

Walks output/, reads each test-output_metrics.json, prints a markdown table
sorted by FAD-PaSST (lower = better). Also writes the table to
output/_analysis/results.md and a CSV to output/_analysis/results.csv.

Usage:
    python experiments/collect_metrics.py
    python experiments/collect_metrics.py --pattern 'output/gw_var_*'
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('output'),
                   help='Where the per-run dirs live (default: output)')
    p.add_argument('--pattern', default='*',
                   help="glob pattern under --root (default: '*')")
    p.add_argument('--out_dir', type=Path, default=Path('output/_analysis'))
    p.add_argument('--sort_by', default='FD_PaSST',
                   help='Column to sort ascending by (default: FD_PaSST)')
    args = p.parse_args()

    rows: list[dict] = []
    for run_dir in sorted(args.root.glob(args.pattern)):
        if not run_dir.is_dir():
            continue
        metrics_file = run_dir / 'test-output_metrics.json'
        if not metrics_file.is_file():
            continue
        with open(metrics_file) as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                print(f'  skipping {metrics_file}: malformed JSON', file=sys.stderr)
                continue
        rows.append({'exp_id': run_dir.name, **metrics})

    if not rows:
        print(f'No test-output_metrics.json under {args.root}/{args.pattern}',
              file=sys.stderr)
        return 1

    # Union of metric columns across runs
    metric_keys = sorted({k for r in rows for k in r if k != 'exp_id'})
    sort_key = args.sort_by if args.sort_by in metric_keys else metric_keys[0]
    rows.sort(key=lambda r: r.get(sort_key, float('inf')))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = args.out_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['exp_id'] + metric_keys)
        writer.writeheader()
        writer.writerows(rows)

    # Markdown
    md_path = args.out_dir / 'results.md'
    headers = ['exp_id'] + metric_keys
    md_lines = ['| ' + ' | '.join(headers) + ' |',
                '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        cells = [r['exp_id']]
        for k in metric_keys:
            v = r.get(k)
            cells.append(f'{v:.4f}' if isinstance(v, (int, float)) else '—')
        md_lines.append('| ' + ' | '.join(cells) + ' |')
    md_text = '\n'.join(md_lines)
    md_path.write_text(md_text + '\n')

    print(md_text)
    print(f'\nSaved to {csv_path} and {md_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
