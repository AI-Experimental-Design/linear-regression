# Classification

## Learning goals

After this module you should be able to

1. Explain the difference between regression and classification.
2. Explain why a linear model with mean squared error is not a good classifier.
3. Describe how logistic regression uses a sigmoid to predict probabilities.
4. Explain binary cross-entropy and why it is useful for classification.
5. Describe how gradient descent trains a classification model.

## Introduction


Machine learning uses examples to make predictions about things we have not
observed. In linear regression, we predict a number, like how square footage
predicts price. In classification, we want to predict which category an example
belongs to, like whether an email is spam or whether a patient will develop a
disease.  

But we already know how to fit a line to data. Can we use the same approach to
classify things?

## Data

We start with a small synthetic dataset containing 60 observations and one
feature, $x$.  Observations with smaller values of $x$ are mostly class 0.
Observations with larger values are mostly class 1. There is some overlap
between the classes.

<img src="img/class.data.png" style="height: 2in;">

<details>

```
python src/make_class_dataset.py \
    --out out/class.data.tsv

python src/plot_fit.py \
    --data out/class.data.tsv \
    --w 0 \
    --b -100 \
    --color_by_label \
    -o img/class.data.png \
    -x x \
    -y label \
    --width 3 \
    --height 2.2 \
    --y_min -0.15 \
    --y_max 1.15 \
    --title "Two classes;10;center"
```

</details>


## Linear regression as a classifier

We can fit a line to these data just like we did for linear regression. The
difference is that now the observed values are either 0 or 1.

To turn the fitted line into a classifier, we need a rule that maps its output
to a class. Here we use 0.5 as the decision boundary: if $\hat{y} < 0.5$, we
predict class 0; otherwise, we predict class 1.

The plots below show the loss decreasing during training, the values of $w$ and
$b$ changing, and the final fitted line. At the end of training, the model has
learned $w=0.135$ and $b=-0.087$.

|Loss | $w$, $b$ | Fit |
|-|-|-l|
| <img src="img/class.data.params.training.png"> | <img src="img/class.data.params.png"> | <img src="img/class.data.tsv.residuals.png"> |

<details>

```
python src/train_line.py \
    --data out/class.data.tsv \
    --w0 0.5 \
    --b0 8 \
    --lr 0.02 \
    --epochs 500 \
    --out_prefix out/class.data
epoch 0000 mse=100.0553 w=+0.500 b=+8.000 dL/dw=+106.779 dL/db=+19.887
epoch 0001 mse=29.4520 w=-1.636 b=+7.602 dL/dw=-43.897 dL/db=-2.492
epoch 0002 mse=17.6992 w=-0.758 b=+7.652 dL/dw=+16.898 dL/db=+6.481
epoch 0005 mse=14.5802 w=-0.989 b=+7.382 dL/dw=-1.707 dL/db=+3.604
epoch 0010 mse=13.1789 w=-0.917 b=+7.014 dL/dw=-0.520 dL/db=+3.590
epoch 0020 mse=10.7832 w=-0.816 b=+6.328 dL/dw=-0.481 dL/db=+3.244
epoch 0050 mse=5.9165 w=-0.566 b=+4.640 dL/dw=-0.355 dL/db=+2.397
epoch 0100 mse=2.1985 w=-0.285 b=+2.748 dL/dw=-0.215 dL/db=+1.448
epoch 0200 mse=0.3477 w=-0.013 b=+0.914 dL/dw=-0.078 dL/db=+0.528
epoch 0300 mse=0.1015 w=+0.086 b=+0.246 dL/dw=-0.029 dL/db=+0.193
epoch 0500 mse=0.0644 w=+0.135 b=-0.087 dL/dw=-0.004 dL/db=+0.026
wrote out/class.data.params.tsv

python src/plot_training.py \
    -i out/class.data.params.tsv \
    -o img/class.data.params.training.png \
    --columns loss \
    --ylog \
    --title "MSE over training"

python src/plot_training.py \
    -i out/class.data.params.tsv \
    -o img/class.data.params.png \
    --columns w,b \
    --title "w and b over training"
    
python src/plot_fit.py \
    --data  out/class.data.tsv \
    --w 0.135 \
    --b -0.087 \
    --residuals \
    -o img/class.data.tsv.residuals.png \
    -x x \
    -y y


```

</details>


To turn the fitted line into a classifier, we need a rule that maps its output
to a class. Here we use 0.5 as the decision boundary: if $\hat{y} < 0.5$, we
predict class 0; otherwise, we predict class 1.

We can evaluate the classifier using accuracy, the fraction of observations
assigned to the correct class.

$$
\text{accuracy} =
\frac{\text{number correctly classified}}
{\text{total number of observations}}
$$

After training, we can use the fitted line to classify every observation and
calculate its accuracy. This tells us how well the line works as a classifier,
rather than just how closely it fits the 0 and 1 values.

