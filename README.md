# Linear Regression

- State the basic idea of learning a function from data to help use predict values we dont have data for
- Give a few examples
- Linear regression is the most basic learning task
- Fitting a straight line ($y = wx + b$, we are intetionally using $w$ ineaste of the more traditional $m$ to better align with future lesons) to data
- learning two parmaters, the slope $w$  and $y$-intercets $b$
- The process is start with a random asignment of values to the parameters
measure how well that line and those paramters fit the data then update the
paramters to improve the fit.

## Training

Training starts with a guess for the model parameters. We measure how well the
guess fits the data, adjust the parameters to improve it, and repeat. We stop
when the improvements level out and each new round barely changes the loss.
Ideally that is also the point where the model fits well. It doesn't have to
be. A model can stop improving because it has found the best answer it can
reach, even if that answer is still poor.

### Data

We start with synthetic data, where we choose the line ahead of time.  Because
we know what $w$ and $b$ should be, we can check whether training finds them.
Each dataset has 50 points where noise is random scatter added to each $y$
value. With no noise, every point falls exactly on the line. With
more noise, the line gets harder to see.

| Generating fucntion | noise | Plot|
|-|-|-|
| $y=0.5x+3$ | 0   | <img src="img/line_0.5x_3_no_noise.png" height="2in"> |
| $y=2x+1$   | 0   | <img src="img/line_2x_1_no_noise.png" height="2in">   |
| $y=2x+1$   | 1.5 | <img src="img/line_2x_1_1.5_noise.png" height="2in">  | 
| $y=2x+1$   | 5   | <img src="img/line_2x_1_5_noise.png" height="2in">   |

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

| MSE | Plot|
|-|-|
| 25.0    | <img src="img/line_0.5x_3_no_noise.residuals.png"> |
| 19.9273 | <img src="img/line_2x_1_no_noise.residuals.png"> | 
| 24.3077 | <img src="img/line_2x_1_1.5_noise.residuals.png"> |
| 52.3390 | <img src="img/line_2x_1_5_noise.residuals.png"> |

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


## Updating paramters


To improve the line we need to know which direction to move $w$ and $b$. We
could do this by nudging each parameter up and down by a small amount and see
what happens to the loss.

| change | w | b | MSE | vs. start |
|-|-|-|-|-|
| none    | 0.500 | 8.000 | 24.3077 | 0 |
| w + 0.1 | 0.600 | 8.000 | 20.9850 | -3.3227 |
| w - 0.1 | 0.400 | 8.000 | 28.3553 | +4.0476 |
| b + 0.1 | 0.500 | 8.100 | 24.1272 | -0.1805 |
| b - 0.1 | 0.500 | 7.900 | 24.5081 | +0.2004 |

Increasing $w$ lowers the loss the most, and increasing $b$ lowers less.  In
the next step we should increase both should go up, ith $w$ mattering the most.

<details>

```
data=out/line_2x_1_1.5_noise.tsv
python src/line_loss.py \
    --data $data \
    -w 0.5 \
    -b 8
mse=24.3077

python src/line_loss.py \
    --data $data \
    -w 0.6 \
    -b 8
mse=20.9850

echo "20.9850-24.3077" | bc
-3.3227

python src/line_loss.py \
    --data $data \
    -w 0.4 \
    -b 8
mse=28.3553

echo "28.3553-24.3077" | bc
4.0476

python src/line_loss.py \
    --data $data \
    -w 0.5 \
    -b 8.1
mse=24.1272

echo "24.1272-24.3077" | bc
-.1805

python src/line_loss.py \
    --data $data \
    -w 0.5 \
    -b 7.9 
mse=24.5081

echo "24.5081-24.3077" | bc
.2004

```

</details>


