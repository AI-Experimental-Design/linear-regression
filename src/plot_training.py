#!/usr/bin/env python3
"""Plot columns (loss, w, b, acc, ...) over epochs from a params.tsv file.

Adapted from AI-Experimental-Design/neural-networks plot_interval_training.py.
Reads the params.tsv that train_line.py or train_logistic.py writes.

Usage:
    python plot_training.py -i out/line.params.tsv -o out/line_training.png
    python plot_training.py -i out/line.params.tsv -o out/line_wb.png --columns w,b
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import regression as reg


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--params_file', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('--columns', default='loss',
                   help='comma-separated column names to plot')
    p.add_argument('--colors', default='blue,green,red,magenta,orange,purple')
    p.add_argument('--ylog', action='store_true')
    p.add_argument('--plot_width', type=float, default=4)
    p.add_argument('--plot_height', type=float, default=3)
    p.add_argument('--title')
    return p.parse_args()


def main():
    args = get_args()
    rows = reg.read_params(args.params_file)
    epochs = [r['epoch'] for r in rows]
    columns = args.columns.split(',')
    colors = args.colors.split(',')

    fig, ax = plt.subplots(figsize=(args.plot_width, args.plot_height), dpi=300)
    ax.tick_params(axis='both', which='major', labelsize=8, width=0.5, length=2)
    for i, col in enumerate(columns):
        if col not in rows[0]:
            raise SystemExit(f'column {col!r} not in {args.params_file}')
        ax.plot(epochs, [r[col] for r in rows], '-', color=colors[i % len(colors)],
                lw=1, label=col)
    if args.ylog:
        ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=8)
    if args.title:
        ax.set_title(args.title, fontsize=8)
    ax.set_xlabel('epoch', fontsize=8)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_linewidth(0.5)
    plt.savefig(args.output, bbox_inches='tight')


if __name__ == '__main__':
    main()
