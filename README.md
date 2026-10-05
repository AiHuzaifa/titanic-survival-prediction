# Logistic Regression From Scratch — Mathematical Notes

> Complete mathematical notes for understanding and implementing Binary Logistic Regression from scratch using NumPy.
>
> The goal is to understand the mathematics behind Logistic Regression before implementing it manually.

---

# 1. What is Logistic Regression?

Logistic Regression is a supervised machine learning algorithm primarily used for **binary classification**.

Binary classification means that the target has two possible classes:

$$
y \in \{0,1\}
$$

Examples:

- Spam / Not Spam
- Disease / No Disease
- Pass / Fail
- Survived / Did Not Survive
- Fraud / Not Fraud

The main goal is to estimate:

$$
P(y=1\mid x)
$$

In simple words:

> Given the input features $x$, what is the probability that the observation belongs to class 1?

---

# 2. Why Not Use Linear Regression?

Linear Regression produces:

$$
\hat{y} = wx+b
$$

The problem is that its output can be any real number:

$$
-\infty < \hat{y} < +\infty
$$

But a probability must satisfy:

$$
0 \leq P \leq 1
$$

For example, Linear Regression could produce:

$$
\hat{y}=1.7
$$

or:

$$
\hat{y}=-0.4
$$

These cannot represent probabilities.

Therefore, Logistic Regression first calculates a linear score and then passes that score through the **Sigmoid function**.

---

# 3. Overall Logistic Regression Pipeline

The basic process is:

$$
x
\rightarrow
z
\rightarrow
\sigma(z)
\rightarrow
P(y=1\mid x)
\rightarrow
\text{Class}
$$

For multiple features:

$$
X
\rightarrow
Xw+b
\rightarrow
\sigma(Xw+b)
\rightarrow
\hat{y}
$$

During training:

$$
\text{Prediction}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Parameter Update}
\rightarrow
\text{Repeat}
$$

---

# 4. Binary Classification Setup

We have:

- $m$ = number of training examples
- $n$ = number of features

The target is binary:

$$
y_i \in \{0,1\}
$$

For example:

$$
y=
\begin{bmatrix}
1\\
0\\
1\\
1\\
0
\end{bmatrix}
$$

Each value represents the actual class of one observation.

---

# 5. Simple Logistic Regression — One Feature

First consider a dataset with only one feature.

For example:

$$
x = \text{Age}
$$

We have two parameters:

$$
w
$$

and:

$$
b
$$

where:

- $w$ = weight / coefficient
- $b$ = bias / intercept

---

## 5.1 Linear Score

The model first calculates a linear score:

$$
\boxed{z=wx+b}
$$

For example:

$$
w=0.5
$$

$$
b=-10
$$

and:

$$
x=25
$$

Then:

$$
z=(0.5)(25)-10
$$

$$
z=2.5
$$

But $z=2.5$ is not a probability.

We need to convert it into a value between 0 and 1.

---

# 6. Sigmoid Function

The **Sigmoid function**, also called the **Logistic function**, converts any real number into a value between 0 and 1.

$$
\boxed{
\sigma(z)=\frac{1}{1+e^{-z}}
}
$$

Therefore:

$$
\boxed{
P(y=1\mid x)=\sigma(wx+b)
}
$$

or:

$$
\boxed{
P(y=1\mid x)
=
\frac{1}{1+e^{-(wx+b)}}
}
$$

The model's predicted probability is usually written as:

$$
\boxed{\hat{y}=\sigma(z)}
$$

---

# 7. Properties of the Sigmoid Function

The Sigmoid function behaves as follows:

$$
z\rightarrow+\infty
\Rightarrow
\sigma(z)\rightarrow1
$$

$$
z\rightarrow-\infty
\Rightarrow
\sigma(z)\rightarrow0
$$

And:

$$
\sigma(0)=0.5
$$

Some examples:

| $z$ | $\sigma(z)$ |
|---:|---:|
| -5 | 0.0067 |
| -2 | 0.1192 |
| -1 | 0.2689 |
| 0 | 0.5000 |
| 1 | 0.7311 |
| 2 | 0.8808 |
| 5 | 0.9933 |

Therefore, Sigmoid transforms:

$$
(-\infty,+\infty)
$$

into:

$$
(0,1)
$$

---

# 8. Probability Interpretation

Suppose:

$$
\hat{y}=0.82
$$

Then:

$$
P(y=1\mid x)=0.82
$$

This means:

> The model estimates an 82% probability that the observation belongs to class 1.

The output is a probability, not yet necessarily the final class label.

---

# 9. Converting Probability Into a Class

For binary classification, we commonly use a threshold of:

$$
0.5
$$

Therefore:

$$
\boxed{
\hat{c}=
\begin{cases}
1 & \text{if } \hat{y}\geq0.5\\
0 & \text{if } \hat{y}<0.5
\end{cases}
}
$$

For example:

$$
\hat{y}=0.83
\Rightarrow
\text{Class}=1
$$

$$
\hat{y}=0.21
\Rightarrow
\text{Class}=0
$$

### Important

The threshold is used for **classification**, not for calculating the training gradient.

During training, we keep the continuous probability:

$$
\hat{y}
$$

---

# 10. Decision Boundary

The classification boundary occurs at:

$$
\hat{y}=0.5
$$

Since:

$$
\sigma(0)=0.5
$$

the boundary occurs when:

$$
z=0
$$

For one feature:

$$
wx+b=0
$$

Solving for $x$:

$$
\boxed{
x=-\frac{b}{w}
}
$$

This is the decision boundary for a one-feature Logistic Regression model.

---

# 11. Multi-Variable Logistic Regression

Now suppose we have multiple features.

For example:

$$
x_1=\text{Age}
$$

$$
x_2=\text{Fare}
$$

$$
x_3=\text{Pclass}
$$

The linear equation becomes:

$$
\boxed{
z=w_1x_1+w_2x_2+w_3x_3+b
}
$$

For $n$ features:

$$
\boxed{
z=
w_1x_1+w_2x_2+\cdots+w_nx_n+b
}
$$

---

# 12. Vector Form

Instead of writing every feature separately, we use vectors.

Feature vector:

$$
\boxed{
x=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
}
$$

Weight vector:

$$
\boxed{
w=
\begin{bmatrix}
w_1\\
w_2\\
\vdots\\
w_n
\end{bmatrix}
}
$$

Bias:

$$
\boxed{
b\in\mathbb{R}
}
$$

### Important

The weight vector and bias are separate:

$$
w=\text{vector}
$$

$$
b=\text{scalar}
$$

We do not combine them into one parameter vector.

---

# 13. Dot Product Form

The linear score can be written as:

$$
\boxed{
z=w^Tx+b
}
$$

Because:

$$
w^Tx
=
w_1x_1+w_2x_2+\cdots+w_nx_n
$$

we get:

$$
\boxed{
z=w^Tx+b
}
$$

---

# 14. Probability in Vector Form

Apply the Sigmoid function:

$$
\boxed{
\hat{y}=\sigma(w^Tx+b)
}
$$

Therefore:

$$
\boxed{
\hat{y}
=
\frac{1}{1+e^{-(w^Tx+b)}}
}
$$

This gives the predicted probability for one observation.

---

# 15. Complete Dataset Representation

For $m$ observations and $n$ features:

$$
\boxed{
X\in\mathbb{R}^{m\times n}
}
$$

For example, if we have:

- 712 training examples
- 3 features

then:

$$
X\in\mathbb{R}^{712\times3}
$$

A typical feature matrix looks like:

$$
X=
\begin{bmatrix}
x_{11}&x_{12}&x_{13}\\
x_{21}&x_{22}&x_{23}\\
\vdots&\vdots&\vdots\\
x_{m1}&x_{m2}&x_{m3}
\end{bmatrix}
$$

Each row represents one observation.

Each column represents one feature.

---

# 16. Weight Vector Dimensions

For $n$ features:

$$
\boxed{
w\in\mathbb{R}^{n\times1}
}
$$

For three features:

$$
w=
\begin{bmatrix}
w_1\\
w_2\\
w_3
\end{bmatrix}
$$

Therefore:

$$
w\in\mathbb{R}^{3\times1}
$$

The bias remains a separate scalar:

$$
\boxed{
b\in\mathbb{R}
}
$$

---

# 17. Vectorized Linear Equation

For the entire dataset:

$$
\boxed{
z=Xw+b
}
$$

Dimensions:

$$
X:(m,n)
$$

$$
w:(n,1)
$$

Therefore:

$$
Xw:(m,1)
$$

Then the scalar bias $b$ is added to every element:

$$
z:(m,1)
$$

Therefore:

$$
\boxed{
z\in\mathbb{R}^{m\times1}
}
$$

---

# 18. Vectorized Prediction

Apply Sigmoid element-wise:

$$
\boxed{
\hat{y}=\sigma(z)
}
$$

Therefore:

$$
\boxed{
\hat{y}=\sigma(Xw+b)
}
$$

where:

$$
\hat{y}\in\mathbb{R}^{m\times1}
$$

Each element of $\hat{y}$ is the predicted probability for one training example.

---

# 19. What Is the Model Trying to Learn?

The model needs to learn two things:

$$
w
$$

and:

$$
b
$$

The objective is to find values of $w$ and $b$ that produce probabilities close to the actual labels.

We need a mathematical way to measure how wrong our predictions are.

That is the purpose of the **Loss Function**.

---

# 20. Loss Function — Binary Cross-Entropy

The standard loss function used for Binary Logistic Regression is:

> **Binary Cross-Entropy Loss**

It is also commonly called:

> **Log Loss**

For one observation:

$$
\boxed{
L_i=
-\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
}
$$

---

# 21. Understanding Binary Cross-Entropy

There are two possible cases.

## Case 1: Actual Class Is 1

If:

$$
y=1
$$

then:

$$
L=-\log(\hat{y})
$$

If the model predicts:

$$
\hat{y}=0.9
$$

then:

$$
L=-\log(0.9)
$$

which is small.

But if:

$$
\hat{y}=0.01
$$

then:

$$
L=-\log(0.01)
$$

which is very large.

Therefore:

> Confidently wrong predictions receive a large penalty.

---

## Case 2: Actual Class Is 0

If:

$$
y=0
$$

then:

$$
L=-\log(1-\hat{y})
$$

If:

$$
\hat{y}=0.1
$$

the loss is small.

If:

$$
\hat{y}=0.99
$$

the loss is very large.

Again:

> The model is heavily punished for being confidently wrong.

---

# 22. Cost Function for the Entire Dataset

For $m$ observations, we calculate the average loss:

$$
\boxed{
J(w,b)
=
-\frac{1}{m}
\sum_{i=1}^{m}
\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
}
$$

This is the:

> **Binary Cross-Entropy Cost Function / Log-Loss Cost Function**

Our objective is:

$$
\boxed{
\min_{w,b}J(w,b)
}
$$

In simple words:

> Find $w$ and $b$ that minimize the average prediction error.

---

# 23. Why Not Use Accuracy as the Loss?

Accuracy is useful for evaluating a model, but it is not suitable as the optimization objective.

For example:

$$
0.51
$$

and:

$$
0.99
$$

are both classified as class 1 using a threshold of 0.5.

Accuracy treats them the same.

But Log Loss understands that:

$$
0.99
$$

is much more confident than:

$$
0.51
$$

Therefore, Logistic Regression trains using:

$$
\boxed{\text{Binary Cross-Entropy / Log Loss}}
$$

rather than accuracy.

---

# 24. Maximum Likelihood Interpretation

Logistic Regression can also be derived using:

> **Maximum Likelihood Estimation (MLE)**

For each observation:

$$
P(y_i\mid x_i)
=
\hat{y}_i^{y_i}
(1-\hat{y}_i)^{1-y_i}
$$

This works because:

### If $y_i=1$

$$
P(y_i\mid x_i)=\hat{y}_i
$$

### If $y_i=0$

$$
P(y_i\mid x_i)=1-\hat{y}_i
$$

---

# 25. Likelihood

Assuming the observations are independent:

$$
\boxed{
L(w,b)
=
\prod_{i=1}^{m}
\hat{y}_i^{y_i}
(1-\hat{y}_i)^{1-y_i}
}
$$

We want to maximize:

$$
\boxed{
\max_{w,b}L(w,b)
}
$$

In other words:

> Find parameters that make the observed training data as probable as possible.

---

# 26. Log-Likelihood

Products are difficult to work with, so we take the logarithm.

$$
\log L(w,b)
=
\sum_{i=1}^{m}
\left[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
\right]
$$

We want to maximize:

$$
\log L
$$

But machine learning optimization is often expressed as minimization.

Therefore, we take the negative:

$$
-\log L
$$

and average it:

$$
\boxed{
J(w,b)
=
-\frac{1}{m}\log L(w,b)
}
$$

This gives Binary Cross-Entropy.

Therefore:

$$
\boxed{
\text{Maximum Likelihood}
\Longleftrightarrow
\text{Minimizing Log Loss}
}
$$

---

# 27. Gradient Descent

We now have our cost function:

$$
J(w,b)
$$

But we need to find the values of $w$ and $b$ that minimize it.

We use:

> **Gradient Descent**

The gradient tells us the direction in which the loss increases most rapidly.

Therefore, to decrease the loss, we move in the opposite direction.

---

# 28. Learning Rate

The learning rate is represented by:

$$
\boxed{\alpha}
$$

It controls how large each parameter update is.

For a parameter $\theta$:

$$
\boxed{
\theta\leftarrow
\theta-\alpha\nabla_{\theta}J
}
$$

If $\alpha$ is too small:

> Training can be very slow.

If $\alpha$ is too large:

> Training can overshoot the minimum or become unstable.

A common starting point can be:

$$
\alpha=0.01
$$

but the appropriate value depends on the dataset and feature scaling.

---

# 29. Gradient Descent for Weights

For the weight vector:

$$
\boxed{
w\leftarrow
w-\alpha\nabla_wJ
}
$$

where:

$$
\nabla_wJ
=
\frac{\partial J}{\partial w}
$$

---

# 30. Gradient Descent for Bias

The bias is separate:

$$
\boxed{
b\leftarrow
b-\alpha\nabla_bJ
}
$$

where:

$$
\nabla_bJ
=
\frac{\partial J}{\partial b}
$$

Remember:

$$
w=\text{vector}
$$

while:

$$
b=\text{scalar}
$$

---

# 31. Derivative of the Sigmoid Function

The Sigmoid function is:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Its derivative has a very useful form:

$$
\boxed{
\sigma'(z)
=
\sigma(z)(1-\sigma(z))
}
$$

Since:

$$
\hat{y}=\sigma(z)
$$

we can also write:

$$
\boxed{
\sigma'(z)
=
\hat{y}(1-\hat{y})
}
$$

This derivative is important when deriving the Logistic Regression gradient.

---

# 32. Deriving the Gradient

We have:

$$
z=w^Tx+b
$$

and:

$$
\hat{y}=\sigma(z)
$$

and:

$$
L=
-\left[
y\log(\hat{y})
+
(1-y)\log(1-\hat{y})
\right]
$$

We want:

$$
\frac{\partial L}{\partial w}
$$

Using the Chain Rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial\hat{y}}
\frac{\partial\hat{y}}{\partial z}
\frac{\partial z}{\partial w}
$$

---

# 33. First Derivative of the Loss

The derivative of the loss with respect to $\hat{y}$ is:

$$
\boxed{
\frac{\partial L}{\partial\hat{y}}
=
-\frac{y}{\hat{y}}
+
\frac{1-y}{1-\hat{y}}
}
$$

The Sigmoid derivative is:

$$
\boxed{
\frac{\partial\hat{y}}{\partial z}
=
\hat{y}(1-\hat{y})
}
$$

And:

$$
z=w^Tx+b
$$

so:

$$
\boxed{
\frac{\partial z}{\partial w}=x
}
$$

Combining these terms produces a very important simplification:

$$
\boxed{
\frac{\partial L}{\partial w}
=
(\hat{y}-y)x
}
$$

---

# 34. Gradient for the Entire Dataset

For $m$ examples:

$$
\boxed{
\frac{\partial J}{\partial w}
=
\frac{1}{m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)x_i
}
$$

This is the gradient for the weight vector.

---

# 35. Bias Gradient

Since:

$$
z=w^Tx+b
$$

we have:

$$
\frac{\partial z}{\partial b}=1
$$

Therefore:

$$
\boxed{
\frac{\partial L}{\partial b}
=
\hat{y}-y
}
$$

For the entire dataset:

$$
\boxed{
\frac{\partial J}{\partial b}
=
\frac{1}{m}
\sum_{i=1}^{m}
(\hat{y}_i-y_i)
}
$$

---

# 36. Vectorized Gradient

Define the prediction error:

$$
\boxed{
e=\hat{y}-y
}
$$

Then:

$$
e\in\mathbb{R}^{m\times1}
$$

The weight gradient is:

$$
\boxed{
dw=
\frac{1}{m}X^Te
}
$$

or equivalently:

$$
\boxed{
\nabla_wJ
=
\frac{1}{m}
X^T(\hat{y}-y)
}
$$

The bias gradient is:

$$
\boxed{
db=
\frac{1}{m}
\sum(\hat{y}-y)
}
$$

---

# 37. Why Does $X^T$ Appear?

Suppose:

$$
X\in\mathbb{R}^{m\times n}
$$

Then:

$$
X^T\in\mathbb{R}^{n\times m}
$$

and:

$$
e\in\mathbb{R}^{m\times1}
$$

Therefore:

$$
X^Te
$$

has dimensions:

$$
(n,m)(m,1)
$$

which gives:

$$
\boxed{
(n,1)
}
$$

This exactly matches the shape of the weight vector:

$$
w\in\mathbb{R}^{n\times1}
$$

This allows all weight gradients to be calculated simultaneously.

---

# 38. Gradient Descent Updates

Weight update:

$$
\boxed{
w\leftarrow w-\alpha dw
}
$$

Substituting the gradient:

$$
\boxed{
w
\leftarrow
w-
\frac{\alpha}{m}
X^T(\hat{y}-y)
}
$$

Bias update:

$$
\boxed{
b\leftarrow b-\alpha db
}
$$

or:

$$
\boxed{
b
\leftarrow
b-
\frac{\alpha}{m}
\sum(\hat{y}-y)
}
$$

---

# 39. Complete Vectorized Training Algorithm

The complete Binary Logistic Regression training process is:

## Step 1 — Initialize Parameters

Initialize:

$$
w=0
$$

and:

$$
b=0
$$

---

## Step 2 — Calculate Linear Score

$$
\boxed{
z=Xw+b
}
$$

---

## Step 3 — Calculate Predictions

Apply Sigmoid:

$$
\boxed{
\hat{y}
=
\frac{1}{1+e^{-z}}
}
$$

---

## Step 4 — Calculate Loss

Binary Cross-Entropy / Log Loss:

$$
\boxed{
J=
-\frac1m
\sum
\left[
y\log(\hat{y})
+
(1-y)\log(1-\hat{y})
\right]
}
$$

---

## Step 5 — Calculate Error

$$
\boxed{
e=\hat{y}-y
}
$$

---

## Step 6 — Calculate Weight Gradient

$$
\boxed{
dw=
\frac1mX^Te
}
$$

---

## Step 7 — Calculate Bias Gradient

$$
\boxed{
db=
\frac1m\sum e
}
$$

---

## Step 8 — Update Weights

$$
\boxed{
w\leftarrow w-\alpha dw
}
$$

---

## Step 9 — Update Bias

$$
\boxed{
b\leftarrow b-\alpha db
}
$$

---

## Step 10 — Repeat

Repeat the forward pass, loss calculation, gradient calculation, and parameter updates until the model converges or reaches the desired number of iterations.

---

# 40. Dimensions — Extremely Important

Let:

- $m$ = number of examples
- $n$ = number of features

Then:

$$
\boxed{
X:(m,n)
}
$$

$$
\boxed{
w:(n,1)
}
$$

$$
\boxed{
b:\text{scalar}
}
$$

$$
\boxed{
z:(m,1)
}
$$

$$
\boxed{
\hat{y}:(m,1)
}
$$

$$
\boxed{
y:(m,1)
}
$$

$$
\boxed{
e:(m,1)
}
$$

$$
\boxed{
dw:(n,1)
}
$$

$$
\boxed{
db:\text{scalar}
}
$$

---

# 41. Matrix Dimension Flow

A useful way to remember the complete vectorized calculation is:

$$
\boxed{
(m,n)(n,1)=(m,1)
}
$$

Therefore:

$$
Xw=z
$$

Then:

$$
z\rightarrow\sigma(z)\rightarrow\hat{y}
$$

Then:

$$
\hat{y}-y=e
$$

Then:

$$
\boxed{
(n,m)(m,1)=(n,1)
}
$$

Therefore:

$$
X^Te=dw
$$

This is the mathematical reason vectorization works.

---

# 42. Example With Titanic Dataset

Suppose we use:

$$
X=[Age,Fare,Pclass]
$$

Therefore:

$$
n=3
$$

If there are 712 training examples:

$$
X\in\mathbb{R}^{712\times3}
$$

The weight vector is:

$$
w=
\begin{bmatrix}
w_{Age}\\
w_{Fare}\\
w_{Pclass}
\end{bmatrix}
$$

Therefore:

$$
w\in\mathbb{R}^{3\times1}
$$

The bias is still separate:

$$
b\in\mathbb{R}
$$

The linear score becomes:

$$
z=Xw+b
$$

which produces:

$$
z\in\mathbb{R}^{712\times1}
$$

After Sigmoid:

$$
\hat{y}=\sigma(z)
$$

and:

$$
\hat{y}\in\mathbb{R}^{712\times1}
$$

---

# 43. Interpretation of Weights

The weights determine how the features influence the linear score:

$$
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
$$

For example, if:

$$
w_1>0
$$

then increasing $x_1$ increases $z$, assuming the other features remain fixed.

Since Sigmoid is increasing, increasing $z$ also increases:

$$
P(y=1\mid x)
$$

If:

$$
w_1<0
$$

then increasing $x_1$ decreases the probability of class 1, all else equal.

---

# 44. Log-Odds Interpretation

One of the most important mathematical interpretations of Logistic Regression is through **log-odds**.

The odds of class 1 are:

$$
\boxed{
\text{Odds}
=
\frac{P(y=1\mid x)}
{1-P(y=1\mid x)}
}
$$

Log-odds are:

$$
\boxed{
\log
\left(
\frac{P(y=1\mid x)}
{1-P(y=1\mid x)}
\right)
}
$$

Logistic Regression models these log-odds as a linear function:

$$
\boxed{
\log
\left(
\frac{P(y=1\mid x)}
{1-P(y=1\mid x)}
\right)
=
w^Tx+b
}
$$

This is called the **logit**.

Therefore:

$$
\boxed{
\text{Logit}(P)=w^Tx+b
}
$$

This is the mathematical reason Logistic Regression is called a regression model even though it is commonly used for classification.

---

# 45. Odds Interpretation of a Weight

For a one-unit increase in feature $x_j$, the log-odds change by:

$$
w_j
$$

The odds themselves are multiplied by:

$$
\boxed{
e^{w_j}
}
$$

Therefore:

$$
\boxed{
e^{w_j}
}
$$

is called the **odds ratio** associated with a one-unit increase in feature $x_j$, holding other features constant.

For example, if:

$$
w_j=0.693
$$

then:

$$
e^{0.693}\approx2
$$

So a one-unit increase in that feature approximately doubles the odds of class 1, assuming the other features stay constant.

---

# 46. Decision Boundary for Multiple Features

The decision boundary occurs when:

$$
P(y=1\mid x)=0.5
$$

Because:

$$
\sigma(0)=0.5
$$

the boundary is:

$$
w^Tx+b=0
$$

For two features:

$$
w_1x_1+w_2x_2+b=0
$$

Solving for $x_2$:

$$
\boxed{
x_2=
-\frac{w_1x_1+b}{w_2}
}
$$

This produces a straight line.

For more features, the decision boundary becomes a hyperplane.

---

# 47. Why Feature Scaling Helps

Suppose one feature has values:

$$
0\text{ to }1
$$

while another has values:

$$
0\text{ to }100000
$$

Gradient Descent can have difficulty because the parameters can operate on very different scales.

Standardization transforms a feature approximately as:

$$
\boxed{
x_{scaled}
=
\frac{x-\mu}{\sigma}
}
$$

where:

- $\mu$ = mean of the feature
- $\sigma$ = standard deviation

This often makes Gradient Descent more stable and efficient.

---

# 48. Important Data Preprocessing Rule

When using train/test data:

1. Split the dataset first.
2. Calculate scaling parameters using the training data only.
3. Transform the training data.
4. Transform the test data using the same training parameters.

Conceptually:

$$
X_{train}
\rightarrow
\text{fit scaler}
\rightarrow
X_{train,scaled}
$$

Then:

$$
X_{test}
\rightarrow
\text{same scaler}
\rightarrow
X_{test,scaled}
$$

Do not calculate scaling statistics from the entire dataset before splitting because that can introduce **data leakage**.

---

# 49. Numerical Stability of the Sigmoid

For very large positive or negative values of $z$, directly calculating:

$$
e^{-z}
$$

can cause numerical issues.

For example, extremely large values can cause overflow or underflow.

In practical implementations, numerical stability should be considered when implementing the Sigmoid function.

Also, when calculating Binary Cross-Entropy, probabilities extremely close to 0 or 1 can cause:

$$
\log(0)
$$

which is undefined.

Therefore, practical implementations often clip probabilities to a very small range away from exactly 0 and 1.

---

# 50. Training vs Prediction

It is important to distinguish between **training** and **prediction**.

## During Training

We calculate:

$$
z=Xw+b
$$

$$
\hat{y}=\sigma(z)
$$

$$
J=\text{Binary Cross-Entropy}
$$

Then:

$$
dw
$$

and:

$$
db
$$

Then update:

$$
w
$$

and:

$$
b
$$

---

## During Prediction

We only need:

$$
z=Xw+b
$$

then:

$$
\hat{y}=\sigma(z)
$$

and finally apply a threshold:

$$
\hat{c}=
\begin{cases}
1 & \hat{y}\geq0.5\\
0 & \hat{y}<0.5
\end{cases}
$$

We do not calculate gradients during prediction.

---

# 51. Core Mathematical Equations

These are the equations that are most important to remember.

## Linear Score

$$
\boxed{
z=Xw+b
}
$$

## Sigmoid

$$
\boxed{
\hat{y}=\frac{1}{1+e^{-z}}
}
$$

## Binary Cross-Entropy / Log Loss

$$
\boxed{
J=
-\frac1m
\sum_{i=1}^{m}
[
y_i\log(\hat{y}_i)
+
(1-y_i)\log(1-\hat{y}_i)
]
}
$$

## Prediction Error

$$
\boxed{
e=\hat{y}-y
}
$$

## Weight Gradient

$$
\boxed{
dw=
\frac1mX^T(\hat{y}-y)
}
$$

## Bias Gradient

$$
\boxed{
db=
\frac1m
\sum(\hat{y}-y)
}
$$

## Weight Update

$$
\boxed{
w\leftarrow w-\alpha dw
}
$$

## Bias Update

$$
\boxed{
b\leftarrow b-\alpha db
}
$$

---

# 52. The Complete Mathematical Picture

The entire model can be summarized as:

$$
\boxed{
z=Xw+b
}
$$

then:

$$
\boxed{
\hat{y}=\sigma(z)
}
$$

then:

$$
\boxed{
J=
-\frac1m
\sum
[
y\log(\hat{y})
+
(1-y)\log(1-\hat{y})
]
}
$$

then:

$$
\boxed{
e=\hat{y}-y
}
$$

then:

$$
\boxed{
dw=\frac1mX^Te
}
$$

and:

$$
\boxed{
db=\frac1m\sum e
}
$$

then:

$$
\boxed{
w\leftarrow w-\alpha dw
}
$$

and:

$$
\boxed{
b\leftarrow b-\alpha db
}
$$

Then repeat.

---

# 53. The Most Important Conceptual Flow

Remember this sequence:

$$
\boxed{
X
\rightarrow
Xw+b
\rightarrow
\sigma(z)
\rightarrow
\hat{y}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Update}
}
$$

More explicitly:

$$
\boxed{
X
\rightarrow
z=Xw+b
\rightarrow
\hat{y}=\sigma(z)
\rightarrow
J=\text{Binary Cross-Entropy}
}
$$

Then:

$$
\boxed{
e=\hat{y}-y
}
$$

Then:

$$
\boxed{
dw=\frac1mX^Te
}
$$

$$
\boxed{
db=\frac1m\sum e
}
$$

Then:

$$
\boxed{
w\leftarrow w-\alpha dw
}
$$

$$
\boxed{
b\leftarrow b-\alpha db
}
$$

Repeat until the loss converges.

---

# 54. Binary Logistic Regression Cheat Sheet

| Concept | Formula |
|---|---|
| Linear score | $z=Xw+b$ |
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ |
| Prediction | $\hat{y}=\sigma(Xw+b)$ |
| Loss | Binary Cross-Entropy |
| Log Loss | Same as Binary Cross-Entropy |
| Error | $e=\hat{y}-y$ |
| Weight gradient | $dw=\frac1mX^Te$ |
| Bias gradient | $db=\frac1m\sum e$ |
| Weight update | $w\leftarrow w-\alpha dw$ |
| Bias update | $b\leftarrow b-\alpha db$ |
| Classification | Threshold, commonly $0.5$ |
| Decision boundary | $w^Tx+b=0$ |
| Logit | $\log\frac{P}{1-P}=w^Tx+b$ |
| Optimization | Gradient Descent |
| Statistical interpretation | Maximum Likelihood Estimation |

---

# 55. Dimensions Cheat Sheet

For:

- $m$ = number of examples
- $n$ = number of features

we have:

$$
\boxed{
X:(m,n)
}
$$

$$
\boxed{
w:(n,1)
}
$$

$$
\boxed{
b:\text{scalar}
}
$$

$$
\boxed{
z:(m,1)
}
$$

$$
\boxed{
\hat{y}:(m,1)
}
$$

$$
\boxed{
y:(m,1)
}
$$

$$
\boxed{
e:(m,1)
}
$$

$$
\boxed{
dw:(n,1)
}
$$

$$
\boxed{
db:\text{scalar}
}
$$

The two most important matrix multiplications are:

$$
\boxed{
(m,n)(n,1)=(m,1)
}
$$

and:

$$
\boxed{
(n,m)(m,1)=(n,1)
}
$$

---

# 56. Important Points to Remember Before Coding

### 1. $w$ and $b$ are separate

$$
w=\text{weight vector}
$$

$$
b=\text{scalar bias}
$$

Do not think of them as the same parameter.

---

### 2. Sigmoid gives probability

$$
\hat{y}=\sigma(Xw+b)
$$

The output is between 0 and 1.

---

### 3. Threshold is only for classification

Training uses:

$$
\hat{y}
$$

not the thresholded 0/1 prediction.

---

### 4. Training uses Binary Cross-Entropy

The model minimizes:

$$
J(w,b)
$$

not accuracy.

---

### 5. Gradient tells us how to change parameters

$$
dw
$$

tells us how to change:

$$
w
$$

while:

$$
db
$$

tells us how to change:

$$
b
$$

---

### 6. Gradient Descent moves opposite the gradient

$$
\boxed{
\theta\leftarrow
\theta-\alpha\nabla J
}
$$

The negative sign is important.

---

### 7. Vectorization removes unnecessary loops

Instead of calculating each example separately:

$$
z_i=w^Tx_i+b
$$

we calculate all examples simultaneously:

$$
\boxed{
z=Xw+b
}
$$

Similarly:

$$
\boxed{
dw=\frac1mX^T(\hat{y}-y)
}
$$

---

### 8. Always check dimensions

If the matrix dimensions do not make sense, the mathematical implementation is wrong.

---

### 9. Scaling can help Gradient Descent

Standardized features generally make optimization easier:

$$
x_{scaled}
=
\frac{x-\mu}{\sigma}
$$

---

### 10. Avoid data leakage

Split the dataset first.

Then fit preprocessing transformations using the training set only.

---

# 57. Final Mental Model

If you remember only one thing, remember this:

### Forward Pass

$$
\boxed{
z=Xw+b
}
$$

$$
\boxed{
\hat{y}=\sigma(z)
}
$$

### Measure Error

$$
\boxed{
J=\text{Binary Cross-Entropy}(\hat{y},y)
}
$$

### Calculate Error

$$
\boxed{
e=\hat{y}-y
}
$$

### Calculate Gradients

$$
\boxed{
dw=\frac1mX^Te
}
$$

$$
\boxed{
db=\frac1m\sum e
}
$$

### Update Parameters

$$
\boxed{
w\leftarrow w-\alpha dw
}
$$

$$
\boxed{
b\leftarrow b-\alpha db
}
$$

### Repeat

$$
\boxed{
\text{Forward}
\rightarrow
\text{Loss}
\rightarrow
\text{Gradient}
\rightarrow
\text{Update}
\rightarrow
\text{Repeat}
}
$$

That is the mathematical foundation of **Binary Logistic Regression from scratch**.
