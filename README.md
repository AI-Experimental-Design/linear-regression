# Linear Regression

- State the basic idea of learning a function from data to help use predict values we dont have data for
- Give a few examples
- Linear regression is the most basic learning task
- Fitting a straight line ($y = wx + b$, we are intetionally using $w$ ineaste of the more traditional $m$ to better align with future lesons) to data
- learning two parmaters, the slope $w$  and $y$-intercets $b$
- The process is start with a random asignment of values to the parameters
measure how well that line and those paramters fit the data then update the
paramters to improve the fit.

## Data

We start with synthetic data, where we choose the line ahead of time.  Because
we know what $w$ and $b$ should be, we can check whether training finds them.
Each dataset has 50 points where noise is random scatter added to each $y$
value. With no noise, every point falls exactly on the line. With
more noise, the line gets harder to see.

| $y=0.5x+3$, no noise | $y=2x+1$, no noise | $y=2x+1$, some noise  |$y=2x+1$, more noise |
| - | - | - | - |
| <img src="img/line_0.5x_3_no_noise.png"> | <img src="img/line_2x_1_no_noise.png"> | <img src="img/line_2x_1_1.5_noise.png"> | <img src="img/line_2x_1_5_noise.png"> |


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

## Loss

To judge how well a line fits the data, we measure the distance from each point
to the line and combine those distances into one number. The distance we use is
vertical, the gap between the observed $y$ and the line's prediction $\hat{y} =
wx + b$. That gap is called the residual. Squaring each residual and taking the
mean gives the mean squared error (MSE).

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2$$

A function that scores a model's predictions like this is called a loss
function. A lower loss means a better fit, and a perfect line has a loss of 0.

Suppose we pick the line $y=05.x+8$, then the loss for the different data sets would be

| 25.0 | 19.9273 | 24.3077 | 52.3390 |
| - | - | - | - |
| <img src="img/line_0.5x_3_no_noise.residuals.png"> | <img src="img/line_2x_1_no_noise.residuals.png"> | <img src="img/line_2x_1_1.5_noise.residuals.png"> | <img src="img/line_2x_1_5_noise.residuals.png"> |

<details>

```
for data in out/line_0.5x_3_no_noise.tsv out/line_2x_1_1.5_noise.tsv out/line_2x_1_5_noise.tsv out/line_2x_1_no_noise.tsv; do
    echo $data

    python src/line_loss.py \
        --data $data \
        -w 0.5 \
        -b 8

    base=$(basename $data .tsv)

    python src/plot_fit.py \
        --data $data \
        --w 0.5 \
        --b 8 \
        --residuals \
        -o img/${base}.residuals.png \
        -x x \
        -y y 
done

out/line_0.5x_3_no_noise.tsv
mse=25.0000
out/line_2x_1_1.5_noise.tsv
mse=24.3077
out/line_2x_1_5_noise.tsv
mse=52.3390
out/line_2x_1_no_noise.tsv
mse=19.9273
```

</details>


