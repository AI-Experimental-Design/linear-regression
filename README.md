# Linear Regression

- State the basic idea of learning a function from data to help use predict values we dont have data for
- Give a few examples
- Linear regression is the most basic learning task
- Fitting a straight line ($y = wx + b$, we are intetionally using $w$ ineaste of the more traditional $m$ to better align with future lesons) to data
- learning two parmaters, the slope $w$  and $y$-intercets $b$
- The process is start with a random asignment of values to the parameters
measure how well that line and those paramters fit the data then update the
paramters to improve the fit.

Data

will start by fitting lines to data where we know what $w$ and $b$ should
be 
data generated from knows values of $w$ the line $y=2x+1$ with noise


| $y=0.5x+3$ no noise | $y=2x+1$ no noise | $y=2x+1$ some noise  |$y=2x+1$ more noise |
| - | - | - | - |
| <img src="line_0.5x_3_no_noise.png"> | <img src="line_2x_1_no_noise.png"> | <img src="line_2x_1_1.5_noise.png"> | <img src="line_2x_1_5_noise.png"> |


<details>

```
python src/make_line_dataset.py \
    -w 0.5 \
    -b 3 \
    --noise 0 \
    --out out/line_0.5x_3_no_noise.tsv

tail -n +2 out/line_0.5x_3_no_noise.tsv \
| python src/plot_line.py \
    -o img/line_0.5x_3_no_noise.png \
    -x x \
    -y y \
    --x_min 0 --x_max 10 \
    --y_min 0 --y_max 10 \
    --point_size 3 \
    --markeredgecolor tab:blue \
    --markerfacecolor tab:blue \
    --width 3 \
    --height 2.5 \
    --line_style o

python src/make_line_dataset.py \
    -w 2 \
    -b 1 \
    --noise 0 \
    --out out/line_2x_1_no_noise.tsv

tail -n +2 out/line_2x_1_no_noise.tsv \
| python src/plot_line.py \
    -o img/line_2x_1_no_noise.png \
    -x x \
    -y y \
    --point_size 3 \
    --markeredgecolor tab:blue \
    --markerfacecolor tab:blue \
    --width 3 \
    --height 2.5 \
    --line_style o

python src/make_line_dataset.py \
    -w 2 \
    -b 1 \
    --noise 1.5 \
    --out out/line_2x_1_1.5_noise.tsv

tail -n +2 out/line_2x_1_1.5_noise.tsv \
| python src/plot_line.py \
    -o img/line_2x_1_1.5_noise.png \
    -x x \
    -y y \
    --point_size 3 \
    --markeredgecolor tab:blue \
    --markerfacecolor tab:blue \
    --width 3 \
    --height 2.5 \
    --line_style o

python src/make_line_dataset.py \
    -w 2 \
    -b 1 \
    --noise 5 \
    --out out/line_2x_1_5_noise.tsv

tail -n +2 out/line_2x_1_5_noise.tsv \
| python src/plot_line.py \
    -o img/line_2x_1_5_noise.png \
    -x x \
    -y y \
    --point_size 3 \
    --markeredgecolor tab:blue \
    --markerfacecolor tab:blue \
    --width 3 \
    --height 2.5 \
    --line_style o

```

</details>







k


