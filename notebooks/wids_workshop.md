---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: retmar-ym1P-xdI-py3.10
    language: python
    name: python3
---

<a target="_blank" href="https://colab.research.google.com/github/YuXia-SR/numpyro-example/blob/main/notebooks/snack_learn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```python
!pip install numpyro
!pip install arviz
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
!pip install statsmodels
```

```python
import sys
sys.path.append('../')
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import Predictive
from numpyro.distributions.transforms import SigmoidTransform
from jax import random
from numpyro.infer import MCMC, NUTS
import arviz as az

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error

from utils.numypro_example_utils import (
    compute_purchase_probability,
    convert_purchase_prob_to_df,
    generate_features_and_target,
    generate_purchase_observation,
    convert_features_and_target_to_df
)
```

# Overview

Probabilistic programming is a powerful tool for data scientists and statisticians that allows them to build complex models to make predictions based on uncertain or incomplete data. In this tutorial notebook, we will use the probabilistic programming library, Numpyro, to build a probabilistic model for predicting customer purchase probability in a retail marketing setup. The model will be based on historical data and will take into account several simple factors such as customer gender, product categories, and loyalty score in one category. Through this session, we will work to implement the model, analyze the results, and discuss potential improvements. By the end of the notebook, you will have an understanding of probabilistic programming and the ability to use Numpyro to build probabilistic models for your own predictive analysis.

## Review the model
Following the same definition from the reference [1]
1. $C_u$ means consumer $u$ selects the category $c$
2. $B_{i, u}$ means consumer $u$ purchases product $i$
3. $Q_{i, u}=q$ means consumer $u$’s purchase quantity of $i$ is q

The joint probability of product demand is captured by a product of three conditional probabilities which represent the preferences in previous purchase stages.
$$
P(Q_{i, u}=q, B_{i, u}, C_u) = P(C_u) P(B_{i, u}| C_u) P(Q_{i, u}=q | B_{i, u}, C_u)
$$

Each purchase stage follows a similar modeling algorithm, thus we will focus on the category choice as an example to illustrate the method in this session.


# Prepare the synthetic dataset

```python
# We're going to define our own dataset
# number of categories
n_category = 2
# number of customers
n_customer = 100
n_customer_test = 100
# number of observations in each combination of (customer, category)
# basically have an observation for each customer per month over 3 years
n_samples = 40
# set random seed
np.random.seed(0)
```

## Generate features and observations


We're going to construct a synthetic on 100 customers' purchase decision of 2 categories. While we hold a naive assumptions of purchase probability based on the observations from daily life. These following numbers could not represent the actual purchase probability from person to person, but only serves to illustrate the model we're going to implement shortly.

```python
normal_dist = [(1.5, 0.1),(0.2, 0.1), (0.2, 0.1), (2.4, 0.1)]
# we set the values of alpha and beta as following
alpha = jnp.array([-0.8, 0.5])
beta = jnp.array([jnp.array([np.random.normal(-0.8, 0.1)]) if i % 2 == 0 else jnp.array([np.random.normal(-2, 0.1)]) for i in range(n_customer)])
```

```python
# create training data
global_feature, category_feature, purchase_prob = generate_features_and_target(
    n_category, n_customer, normal_dist, alpha, beta
)
sampling_y = generate_purchase_observation(purchase_prob, n_samples)
# create testing data
global_feature_test, category_feature_test, purchase_prob_test = generate_features_and_target(
    n_category, n_customer_test, normal_dist, alpha, beta
)
sampling_y_test = generate_purchase_observation(purchase_prob_test, n_samples)

```

Generated data are in array format, but hard to inspect them. Let's use a helper method to convert them to a pandas dataframe.

```python
training_data_with_purchase_prob = convert_features_and_target_to_df(
    global_feature, purchase_prob
)
training_data_with_purchase_obs = convert_features_and_target_to_df(
    global_feature, sampling_y[0], target_column_name='purchase_decision'
)
testing_data_with_purchase_prob = convert_features_and_target_to_df(
    global_feature_test, purchase_prob_test
)
```

## Inspect the feature table

We could define modeling goal to be learning the purchase behavior of female and male customers on two categories: razor and makeup foundations. The gender column has been contrast encoded with 1 representing male and -1 representing female. The column of loyalty score is continuous number column.

```python
training_data_with_purchase_prob.head()
```

```python
def plot_distribution_from_df(data, target_column, ax):
    sns.violinplot(data, x='gender', y=target_column, hue='category', ax=ax)
    target_column = 'Loyalty score' if target_column == 'loyalty_score' else 'Purchase probability'
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Female', 'Male'])
    ax.set_xlabel('Gender')
    ax.set_ylabel(target_column)
    ax.set_title(f'{target_column} distribution across category')
```

In the synthetic dataset, we observe difference purchase probabilities on each category across customers. Female customers usually have a larger probability to purchase foundation products than razor products, while male customers will be more likely to purchase razor products.

```python
f, ax = plt.subplots(1, 1, figsize=(6, 5))
plot_distribution_from_df(training_data_with_purchase_prob, 'purchase_probability', ax)
```

We could directly pass the purchase probability matrix to bernouli distribution and sample the category choice for each customer.

```python
training_data_with_purchase_obs.head()
```

# Model structure: Bayesian Hierarchical Linear Regression

For the category decision, we use a hierarchical linear model to approximate the purchase probability, and aim to estimate the coefficient $\alpha_1, \alpha_2, \beta_u$ defined below. For each customer $u$ and category $i$,

$$
\begin{align}
s_{ui} &= \alpha_1*Gender_{ui} + \alpha_2*LoyaltyScore_{ui} + \beta_u  \\
p_{ui} &= sigmoid(s_{ui}), \\
y_{ui} &\sim Bernouli(p_{ui})
\end{align}
$$

where we use the following notations
1. Gender is a global effect, and we'd like to see if customers with different genders have different purchase probability on categories. The corresponding coefficient $\alpha_1$ is shared across all customers and all categories.
2. Loyalty score is also a global effect, and the corresponding coefficient $\alpha_2$ remains the same as well across all customers and all categories.
3. The interception term is with subscription of customer $u$, aiming to capture category effects caused by different customers. Thus, $\beta$ will be in shape of (n_customer, )

<div style="text-align:center" heighr="400">

![causalDAG](../images/DAG.png)

</div>

We will model the first two terms together as the global effect and last term as the category effect. The preference score $s_{ui}$ will be the summation of these two effects. We assume all coefficient priors follow a normal distribution with different mean and standard deviation.
1. $\alpha_{1} \sim Normal(\mu_{\alpha_1}, \sigma^2_{\alpha_1})$, $\alpha_{2} \sim Normal(\mu_{\alpha_2}, \sigma^2_{\alpha_2})$
2. We need to initialize a coefficient matrix for the term of category effect, with shape of (n_customer, ). For each $\beta_u$, $\beta_u \sim Normal(\mu_{\beta_u}, \sigma^2_{\beta_u})$


<div style="text-align:center">

![modelDAG](../images/modelDAG.png)

</div>


## Implement the category choice model

```python
def category_choice(
    X_global_effect_feature: jnp.DeviceArray,
    X_category_effect_feature: jnp.DeviceArray,
    y: jnp.DeviceArray=None
)-> jnp.DeviceArray:
    # read the shape of the input
    n_sample, n_customer, n_category, n_global_feature = X_global_effect_feature.shape
    n_category_feature = X_category_effect_feature.shape[-1]
    """
    Sample the alphas and compute global effect

    """
    # define the transformed distribution for alpha
    alpha_mu = numpyro.sample(
        "alpha_mu", 
        dist.Normal(loc=jnp.zeros(n_global_feature), 
        scale=jnp.ones(n_global_feature)))
    alpha_sigma = numpyro.sample(
        "alpha_sigma", 
        dist.HalfNormal(scale=jnp.ones(n_global_feature)))
    # generate coefficients for global effect
    alpha = numpyro.sample(
        'alpha',
        dist.Normal(alpha_mu, alpha_sigma),
    )
    # since we have two features for the global effects, alpha is in shape (2,)
    # compute effect components separately
    global_effect = jnp.matmul(X_global_effect_feature, alpha)

    """
    Sample the betas and compute category effect

    """
    # this plate restricts the row dimension of the customer coefficient matrix
    plate_customer = numpyro.plate("customer", n_customer, dim=-2)
    # this plate restricts the column dimension of the customer coefficient matrix
    plate_category_feature = numpyro.plate("category_feature", n_category_feature, dim=-1)
    with plate_customer, plate_category_feature:
        beta_mu = numpyro.sample(
            "beta_mu", 
            dist.Normal(loc=jnp.zeros((n_customer, n_category_feature)), 
            scale=jnp.ones((n_customer, n_category_feature))))
        beta_sigma = numpyro.sample(
            "beta_sigma", 
            dist.HalfNormal(scale=jnp.ones((n_customer, n_category_feature))))
        beta = numpyro.sample(
            'beta',
            dist.Normal(beta_mu, beta_sigma),
        )
        # adjust the shape of beta array
        beta = jnp.expand_dims(beta, axis=-1)
        # since we have only one feature for the category effects, beta is in shape (100, 1)
        # we extend a third dimension of beta to (100, 1, 1) such that it is compatible with the category feature
    category_effect = jnp.matmul(X_category_effect_feature, beta).squeeze(axis=-1)

    """
    Add effects term together and calculate the purchase probability

    """
    # this plate defines the first dimension of the output observation
    plate_sample_size = numpyro.plate("sample_size", n_sample, dim=-3)
    # this plate restricts the last dimension of the output observation
    plate_category = numpyro.plate("category", n_category, dim=-1)
    # generate the probability of purchase
    with plate_sample_size, plate_customer, plate_category:
        # compute the probability of purchase
        sigmoid = SigmoidTransform()
        purchase_prob = sigmoid(global_effect + category_effect)
        return numpyro.sample("category_choice", dist.Bernoulli(probs=purchase_prob), obs=y)
```

## Train the model using MCMC simulation

```python
X_global_feature = jnp.expand_dims(global_feature, axis=0).repeat(n_samples, axis=0)
X_category_feature = jnp.expand_dims(category_feature, axis=0).repeat(n_samples, axis=0)
nuts_kernel = NUTS(category_choice)
mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=1000)
rng_key = random.PRNGKey(0)
mcmc.run(rng_key, X_global_feature, X_category_feature, sampling_y)
```

```python
mcmc.print_summary()
```

## Check the model

### Inspecting the learned parameters
We will check the states recorded in the MCMC object to see if all coefficients are converged. We will also use the Arviz package to visualize the trace.

```python
# check if MCMC is converging
assert ~ mcmc._states['diverging'].all()
```

```python
data = az.from_numpyro(mcmc)
az.plot_trace(data, var_names=['alpha', 'beta'])
```

Looks like the model is able to learn alpha and beta for each customer.


## Use the model for inference


## Extract learnt coefficient
If the model coefficients all converge, we could extract the learnt value from the object that recorded the trace of MCMC, and compute the mean of each coefficient as the learnt value. In this way, we obtain a full control of the linear model and are able to compute the purchase probability ourselves.

```python
# extract the learnt parameters
learnt_alpha = mcmc._states['z']['alpha'].mean(axis=1).reshape(2, )
learnt_beta = mcmc._states['z']['beta'].mean(axis=1).reshape(n_customer, 1)
```

```python
# Use the learnt parameter to compute the purchase probability
learnt_prob = compute_purchase_probability(global_feature_test, category_feature_test, learnt_alpha, learnt_beta)
learnt_prob_df = convert_purchase_prob_to_df(learnt_prob, column_name='purchase_probability')
learnt_prob_df.head()
```

We put the dataframe of true purchase probability side-by-side with the dataframe of predicted purchase probability

```python
prob_comparison_df = pd.concat([testing_data_with_purchase_prob, learnt_prob_df])
prob_comparison_df.loc[:, 'label'] = np.repeat(['true', 'pred'], n_customer * 2)

def plot_prob_comparison(df: pd.DataFrame, group_column:str, filter: tuple, ax):
    data = df[df[filter[0]]==filter[1]]
    mape = data.groupby(group_column).apply(lambda x: mean_absolute_percentage_error(x[x.label == 'true'].purchase_probability, x[x.label == 'pred'].purchase_probability))
    sns.violinplot(x=group_column, y='purchase_probability', hue='label', data=data, split=True, ax=ax)
    ax.text(0.1, 0.1, s='mape: {:.2f}'.format(mape.iloc[0]), fontsize=10)
    ax.text(1.1, 0.1, s='mape: {:.2f}'.format(mape.iloc[1]), fontsize=10)
    ax.set_title('purchase probability for {} {}'.format(*filter))
    ax.set_ylim(0.05, 0.75)

f, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_prob_comparison(prob_comparison_df, 'category', ('gender', 1), ax[0])
plot_prob_comparison(prob_comparison_df, 'category', ('gender', -1), ax[1])
```

## Use posterior samples

We could also leverage the samples from MCMC simulation for prediction or even creating large amount of synthetic data.

```python
# get posterior samples
X_test_global_feature = jnp.expand_dims(global_feature_test, axis=0).repeat(n_samples, axis=0)
X_test_category_feature = jnp.expand_dims(category_feature_test, axis=0).repeat(n_samples, axis=0)

posterior_samples = mcmc.get_samples()
predictive = Predictive(category_choice, posterior_samples)
prediction = predictive(rng_key, X_test_global_feature, X_test_category_feature)["category_choice"]
# compute the mean of purchase choice for each customer to approximate the purchase probability
prob_mean = prediction.mean(axis=[0,1])
prob_mean_df = convert_purchase_prob_to_df(prob_mean, column_name='purchase_probability')
```

```python
def draw_qq_plot(true, pred, axis_range=(-0.1, 1.1)):

    # Compute the quantiles for x and y
    x_quantiles = sm.distributions.ECDF(true)(true)
    y_quantiles = sm.distributions.ECDF(pred)(pred)

    # Create a QQ plot
    _, ax = plt.subplots(figsize=(6,6))
    ax.scatter(x_quantiles, y_quantiles)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='--', c='k')
    plt.xlim(*axis_range)
    plt.ylim(*axis_range)
    # Set plot title and labels
    plt.title('QQ Plot')
    plt.xlabel('Quantiles of true purchase probability')
    plt.ylabel('Quantiles of predicted purchase probability')
draw_qq_plot(training_data_with_purchase_prob.purchase_probability, prob_mean_df.purchase_probability)
```

# Wrap-up

Hope this notebook provides a starting point for you to explore Numpyro and probabilistic programming. We find the following pages useful to quickly get a taste of Numpyro:

1. "Numpyro distributions". [webpage](https://num.pyro.ai/en/stable/distributions.html)
2. "Bayesian Regression Using Numpyro". [notebook](https://num.pyro.ai/en/stable/tutorials/bayesian_regression.html)
3. Charlos et.al. "Bayesian Hierarchical Linear Regression". [notebook](https://num.pyro.ai/en/stable/tutorials/bayesian_hierarchical_linear_regression.html)

For the research model we will test with retail dataset and the application in the retail environment simulation, please stay tuned with the update in the [Retmar](https://github.com/Bain/retmar) repository



# References
1. Mengting et. al. 2017. “Modeling Consumer Preferences and Price Sensitivities from Large-Scale Grocery Shopping Transaction Logs”. Proceedings of the 26th International Conference on World Wide Web​
2. "Numpyro documentation". Uber Technologies, Inc. https://num.pyro.ai/en/stable/index.html



