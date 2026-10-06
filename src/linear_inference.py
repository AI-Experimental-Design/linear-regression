#!/usr/bin/env python3
"""Run a trained linear regression model on new x values.

For each x, predicts y using y_hat = wx + b.

Usage:
    python src/line_inference.py --w 2.068 --b 0.710 --x 0 2 4 6 8 10
"""
import argparse


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--w', type=float, required=True)
    p.add_argument('--b', type=float, required=True)
    p.add_argument('--x', type=float, nargs='+', required=True)
    return p.parse_args()


def main():
    args = get_args()

    print(f'{"x":>8} {"y_hat":>10}')

    for x in args.x:
        y_hat = args.w * x + args.b
        print(f'{x:8.2f} {y_hat:10.3f}')


if __name__ == '__main__':
    main()

