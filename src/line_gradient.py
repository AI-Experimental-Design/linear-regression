#!/usr/bin/env python3
"""Compute the gradient of the loss for one line y = w*x + b.

Prints dL/dw and dL/db, how much the loss changes for a tiny increase in
each parameter. A negative value means increasing that parameter lowers
the loss. With --points N, also prints each point's contribution for the
first N points so you can see where the averages come from.

Usage:
    python line_gradient.py --data out/line_2x_1_1.5_noise.data.tsv -w 0.5 -b 8
    python line_gradient.py --data out/line_2x_1_1.5_noise.data.tsv -w 0.5 -b 8 --points 5
"""
import argparse

import numpy as np

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
    p.add_argument('--points',
                   type=int,
                   default=0,
                   help='print the per-point terms for the first N points')
    return p.parse_args()


def main():
    args = get_args()

    x, y = reg.read_xy(args.data,
                       args.x_col,
                       args.y_col)

    if args.points:
        y_hat = reg.predict(x, args.w, args.b)
        r = y_hat - y
        g = 2 * r if args.loss == 'mse' else np.sign(r)
        print(f'{"i":>3} {"x":>7} {"y":>7} {"y_hat":>7} {"resid":>7} '
              f'{"w term":>8} {"b term":>8}')
        for i in range(min(args.points, len(x))):
            print(f'{i + 1:>3} {x[i]:7.2f} {y[i]:7.2f} {y_hat[i]:7.2f} '
                  f'{r[i]:7.2f} {g[i] * x[i]:8.2f} {g[i]:8.2f}')
        print(f'{"sum":>3} {"":>7} {"":>7} {"":>7} {"":>7} '
              f'{np.sum(g * x):8.2f} {np.sum(g):8.2f}')
        print(f'n={len(x)}')

    dw, db = reg.gradient(x,
                          y,
                          args.w,
                          args.b,
                          args.loss)

    print(f'dL/dw={dw:.4f} dL/db={db:.4f}')


if __name__ == '__main__':
    main()
