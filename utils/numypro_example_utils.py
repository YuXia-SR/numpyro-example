import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
import numpyro.distributions as dist
from numpyro.distributions.transforms import SigmoidTransform



def generate_global_features(
        n_category: int, 
        n_customer: int, 
        normal_dist: list = [(1.5, 0.1),(0.2, 0.1), (0.2, 0.1), (2.4, 0.1)]
    ) -> jnp.array:
    """ create global features for corresponding customers and categories

    Parameters:
        n_category (int): number of categories
        n_customer (int): number of customers
        normal_dist (list): list of tuples, each tuple is a pair of mean and std for normal distribution to create loyalty score
    
    Returns:
        jnp.array: global feature
    """
    # On the gender info, we shall use a contrast encoding vector to represent. [1] means male, [-1] means female.
    # odd index -> male, even index -> female
    gender_array = jnp.array([jnp.array([1]) if i % 2 == 0 else jnp.array([-1]) for i in range(n_customer)])
    # if the customer is a male
    # 1. his loyalty score on foundations is sampling from a normal distribution with mean 0.2 and std 0.1
    # 2. his loyalty score on razor is sampling from a normal distribution with mean 1.5 and std 0.1
    # if the customer is a female
    # 1. her loyalty score on foundations is sampling from a normal distribution with mean 2.4 and std 0.1
    # 2. her loyalty score on razor is sampling from a normal distribution with mean 0.2 and std 0.1
    money_spent_last_year_array = jnp.array([jnp.array([[np.random.normal(*normal_dist[0])], [np.random.normal(*normal_dist[1])]]) if i % 2 == 0 \
                                     else jnp.array([[np.random.normal(*normal_dist[2])], [np.random.normal(*normal_dist[3])]]) for i in range(n_customer)])

    global_feature = jnp.concatenate([gender_array.repeat(2, axis=0).reshape(n_customer, n_category, 1), money_spent_last_year_array], axis=-1)
    return global_feature

def generate_category_feature(n_category, n_customer):
    """ create category features for corresponding customers and categories

    Parameters:
        n_category (int): number of categories
        n_customer (int): number of customers
    
    Returns:
        jnp.array: category feature
    """
    # we assume that the category feature is a constant vector
    return jnp.ones((n_customer, n_category, 1))

def compute_purchase_probability(
        global_feature: jnp.array, 
        category_feature: jnp.array, 
        global_coef: np.array, 
        category_effect_coef: np.array
    ) -> jnp.array:
    """ Method to compute purchase probability according to the formula

    Parameter:
        global_feature (jnp.array)
        category_feature (jnp.array)
        global_coef (np.array)
        category_effect_coef (np.array)

    Returns:
        jnp.array: purchase probability for each customer and category
    """
    # global_feature: (n_customer, n_category, n_feature)
    # category_feature: (n_customer, n_category, n_feature)
    # global_coef: (n_feature)
    # category_effect_coef: (n_customer, n_feature)``
    # return: (n_customer, n_category)
    preference_score = jnp.matmul(global_feature, global_coef) + jnp.matmul(category_feature, jnp.expand_dims(category_effect_coef, axis=-1)).squeeze(axis=-1)
    purchase_prob = SigmoidTransform()(preference_score)
    return purchase_prob

def generate_features_and_target(
        n_category: int, 
        n_customer: int, 
        normal_dist: list, 
        global_coef: np.array, 
        category_coef: np.array
    ) -> tuple:
    """ helper method to create dataset with one step

    Parameters:
        n_category (int): number of categories
        n_customer (int): number of customers
        normal_dist (list): list of tuples, each tuple is a pair of mean and std for normal distribution to create loyalty score
        global_coef (np.array): global coefficient
        category_coef (np.array): category coefficient

    Returns:
        tuple(jnp.array, jnp.array, jnp.array): global feature, category feature, purchase probability
    """
    global_feature = generate_global_features(n_category, n_customer, normal_dist)
    category_feature = generate_category_feature(n_category, n_customer)
    purchase_prob = compute_purchase_probability(global_feature, category_feature, global_coef, category_coef)
    return global_feature, category_feature, purchase_prob

def generate_purchase_observation(
        purchase_prob: jnp.array, 
        n_samples: int=40
    ):
    """ generate purchase observation according to the purchase probability

    Parameters:
        purchase_prob (jnp.array): purchase probability for each customer and category
        n_samples (int): number of samples to generate per customer and category
    """
    # purchase_prob: (n_customer, n_category)
    # assume that the purchase follows a bernoulli distribution
    sampling_y = dist.Bernoulli(probs=purchase_prob).sample(random.PRNGKey(0), sample_shape=(n_samples, ))
    return sampling_y

def convert_global_feature_to_df(
        global_feature: jnp.array, 
        categories: list) -> pd.DataFrame:
    """ helper method to convert global feature to dataframe for easy inspection

    Parameters:
        global_feature (jnp.array)
        categories (list): names of categories

    Returns:
        pd.DataFrame: global feature dataframe
    """
    n_customer, n_category, _ = global_feature.shape
    examine_data = pd.DataFrame(jnp.concatenate([global_feature[:, i, :] for i in range(len(categories))]), columns=['gender', 'loyalty_score'])
    examine_data.loc[:, 'customer_id'] = np.tile(np.arange(n_customer), n_category)
    examine_data.loc[:, 'category'] = np.repeat(categories, n_customer)
    examine_data = examine_data.loc[:, ['customer_id', 'category', 'gender', 'loyalty_score']].sort_values(by='customer_id', ignore_index=True)
    return examine_data

def convert_purchase_prob_to_df(
        purchase_prob: jnp.array, 
        categories: list=['razor', 'foundation'], 
        column_name: str='purchase_probability'
    ):
    """ helper method to convert observation feature to dataframe for easy inspection

    Parameters:
        purchase_prob (jnp.array)
        categories (list): names of categories
        column_name (str): name of the column to have in the output df

    Returns:
        pd.DataFrame: purchase observation dataframe
    """
    n_customer, n_category = purchase_prob.shape
    examine_data = pd.DataFrame(jnp.concatenate([purchase_prob[:, i] for i in range(len(categories))]), columns=[column_name])
    examine_data.loc[:, 'customer_id'] = np.tile(np.arange(n_customer), n_category)
    examine_data.loc[:, 'gender'] = np.tile([1, -1], n_customer)
    examine_data.loc[:, 'category'] = np.repeat(categories, n_customer)
    examine_data = examine_data.loc[:, ['customer_id', 'category', 'gender', column_name]].sort_values(by='customer_id', ignore_index=True)
    return examine_data

def convert_features_and_target_to_df(
        global_feature: jnp.array, 
        sampling_y: jnp.array, 
        category_names: list=['razor', 'foundation'], 
        target_column_name: str='purchase_probability'
    ) -> pd.DataFrame:
    """ helper method to convert features and target to dataframe for easy inspection at one step

    Parameters:
        global_feature (jnp.array)
        sampling_y (jnp.array)
        category_names (list, optional): Defaults to ['razor', 'foundation'].
        target_column_name (str, optional): Defaults to 'purchase_probability'.

    Returns:
        pd.DataFrame: features and target dataframe merged together
    """
    feature_table = convert_global_feature_to_df(global_feature, category_names)
    target_table = convert_purchase_prob_to_df(sampling_y, category_names, target_column_name)
    return feature_table.merge(target_table)