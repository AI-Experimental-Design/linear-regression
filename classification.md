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
feature, $x$.  Observations with smaller values of $x$ are mostly class 0,
while observations with larger values are mostly class 1. There is some overlap
near the boundary.

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


