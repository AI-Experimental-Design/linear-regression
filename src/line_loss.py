#!/usr/bin/env python3
"""Score one line y = w*x + b against a dataset.

Prints the loss. With --nudge, also bumps w and b up and down by a small
amount and prints the loss at each, which shows which direction to move
each parameter to lower the loss. That direction is what the gradient
gives you without the guessing, and --nudge prints the gradient too so
you can compare.

Usage:
    python line_loss.py --data out/line.data.tsv --w 0.5 --b 8
    python line_loss.py --data out/line.data.tsv --w 0.5 --b 8 --nudge 0.1
    python line_loss.py --data out/line.data.tsv --w 0.5 --b 8 --loss mae
"""
import argparse

import regression as reg


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--x_col',
                   help='x column name (default: first column)')
    p.add_argument('--y_col',
                   help='y column name (default: second column)')
    p.add_argument('--w',
                   type=float,
                   required=True)
    p.add_argument('--b',
                   type=float,
                   required=True)
    p.add_argument('--loss',
                   choices=['mse', 'mae'],
                   default='mse')
    return p.parse_args()


def main():
    args = get_args()

    x, y = reg.read_xy(args.data,
                       args.x_col,
                       args.y_col)

    L = reg.loss(x,
                 y,
                 args.w,
                 args.b,
                 args.loss)

    print(f'{args.loss}={L:.4f}')

if __name__ == '__main__':
    main()
