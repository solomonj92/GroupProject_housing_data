#!/usr/bin/env python
# coding: utf-8

"""
Created on Sun Sep 28 12:07:44 2021

@author:
Himani Gadve
Jacob Greenbaum
Mike Arbuzov
"""

# In[ ]:
import itertools

import gower
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from prince import MCA
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

# In[ ]:
"""
PROJECT PART 1.
1. Prepare data & split into test and train.
2. Train regression (train, val tune) on data with PCA & MCA transforemed categorical variables
3. Train Ridge (5CV tune) on data with hot encoded categorical variables
"""
# In[ ]:
"""
PROJECT PART 1.1. Prepare data & split into test and train.
"""


def d_type_split_num_str(df):
    """Takes in a single dataframe and outputs two dataframes.
    One with all of the numeric features and one
    with all of the non-numeric features

    Args:
        df (pd.DataFrame): dataframe to split

    Returns:
        pd.DataFrame: Scaled numerical features
        pd.DataFrame: Non numeric features
    """

    num = [is_numeric_dtype(df[x]) for x in df.columns]
    lst = zip(num, df.columns)
    cols = list(lst)
    is_num = []
    is_str = []

    for x in cols:
        if x[0] is True:
            is_num.append(x[1])
        else:
            is_str.append(x[1])

    return StandardScaler().fit_transform(df[is_num]), df[is_str]


# In[ ]:
# Load data set
data = pd.read_csv(r"project1_train.csv")
data = data.drop("Id", axis=1)

# Remove columns that have too many missing values
data = data.drop(data.columns[data.isnull().sum() > 30], axis=1)

# Remove missing values
data.dropna(inplace=True)

# Define response variable and features
y = data.SalePrice
del data["SalePrice"]

# In[ ]:
# Converting the following features to strings
# as they are best suited as categorical variables:
data["MoSold"] = data["MoSold"].astype(str)
data["OverallQual"] = data["OverallQual"].astype(str)
data["OverallCond"] = data["OverallCond"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(data, y, random_state=42)

# In[ ]:
# Create 6 datasets, one training set, one validation set, one
# testing set for the categorical features and the numeric features
# Using the function d_type_split_num_str to create
# 2 datasets from one, one categorical one numeric.

X_train_num, X_train_cat = d_type_split_num_str(X_train)
X_test_num, X_test_cat = d_type_split_num_str(X_test)

# In[ ]:
# Create test, validation sand training sets
# for both categorical and numeric features.

# Note that the random state is the same for both to ensure
# the same indecies are split for each set on both training and testing

# Making sure that the same categoircal features are in both
# the training and testing AND validation sets.
keep = X_train_cat.nunique() == X_test_cat.nunique()
X_train_cat = X_train_cat[X_train_cat.columns[keep]]
X_test_cat = X_test_cat[X_test_cat.columns[keep]]

# In[ ]:
X_val_train_num, X_val_test_num, y_val_train, y_val_test = train_test_split(
    X_train_num, y_train, random_state=42, test_size=0.5, train_size=0.5
)

X_val_train_cat, X_val_test_cat, y_val_train, y_val_test = train_test_split(
    X_train_cat, y_train, random_state=42, test_size=0.5, train_size=0.5
)

# For categorical features that have same levels,
# make sure the classes are the same.
keep = []
for clmn in X_train_cat.columns:

    train_cats = set(X_val_train_cat[clmn].unique())
    val_cats = set(X_val_test_cat[clmn].unique())
    test_cats = set(X_test_cat[clmn].unique())

    keep.append(train_cats == val_cats == test_cats)

keep_columns = X_train_cat.columns[keep]

X_train_cat = X_train_cat[keep_columns]
X_test_cat = X_test_cat[keep_columns]
X_val_train_cat = X_val_train_cat[keep_columns]
X_val_test_cat = X_val_test_cat[keep_columns]

print(X_val_train_cat.shape, X_val_test_cat.shape, X_test_cat.shape)
print(X_val_train_num.shape, X_val_test_num.shape, X_test_num.shape)

# In[ ]:
""""
PROJECT PART 1.2. Train regression (train, val tune)
on data with PCA & MCA transforemed categorical variables
"""

# In[ ]:
# Create an array of hyperparameters for both PCA and
# MCA given the quantity of original features in each
n_comps_pca = np.geomspace(6, 30, 10).astype(int)
n_comps_mca = np.geomspace(6, 26, 10).astype(int)

# Create a list of all possible combinations of the two hyperparameters,
# will be used to find best hyperparameters
combos = list(itertools.product(n_comps_pca, n_comps_mca))

# Make an empty array whos size is that of the length
# of combos to store validation scores
params = np.zeros(shape=len(combos))

# In[ ]:
# Initiate variable to iterate over params varriable above
p = 0

# Nested for loop to train a regression model over
# all possible hyperparameters for MCA and PCA
for n, m in combos:
    pca = PCA(n_components=n)  # Initiate PCA for training hyperparameters
    pca.fit(X_val_train_num)
    X_val_train_num_transformed = pca.transform(X_val_train_num)
    X_val_test_num_transformed = pca.transform(X_val_test_num)

    mca = MCA(n_components=m)  # Initiate MCA for training
    mca.fit(X_val_train_cat)
    if X_val_train_cat.shape[1] == X_val_test_cat.shape[1]:
        X_val_train_cat_transformed = mca.transform(X_val_train_cat)
        X_val_test_cat_transformed = mca.transform(X_val_test_cat)

    # Concatenate the categorical and numeric datasets
    # to train model and test on validation set
    transformed_df_train = np.concatenate(
        (X_val_train_num_transformed, X_val_train_cat_transformed), axis=1
    )
    transformed_df_test = np.concatenate(
        (X_val_test_num_transformed, X_val_test_cat_transformed), axis=1
    )
    reg = LinearRegression().fit(transformed_df_train, y_val_train)
    params[p] = reg.score(transformed_df_test, y_val_test)
    p += 1


# In[ ]:
# Find the best set of n_components within the nested for loop of MCA and PCA
best_score_idx = np.argmax(params)
best_pca_components = combos[best_score_idx][0]
best_mca_components = combos[best_score_idx][1]

# Retrain using full training data set
# First step is to transform using best PCA parameters
pca = PCA(n_components=best_pca_components)
pca.fit(X_train_num)
X_train_num_transformed = pca.transform(X_train_num)
X_test_num_transformed = pca.transform(X_test_num)

# Fit the training data using best MCA parameters
mca = MCA(n_components=best_mca_components)
mca.fit(X_train_cat)
X_train_cat_transformed = mca.transform(X_train_cat)
X_test_cat_transformed = mca.transform(X_test_cat)

# Combine the categorical and numeric features into one dataframe
# for each training and testing set.
transformed_df_train = np.concatenate((X_train_num_transformed, X_train_cat_transformed), axis=1)
transformed_df_test = np.concatenate((X_test_num_transformed, X_test_cat_transformed), axis=1)

# Retrain the model using fully transformed
# training data with tuned hyperparameters
reg = LinearRegression().fit(transformed_df_train, y_train)
test_score = reg.score(transformed_df_test, y_test)

# In[]
""""
PROJECT PART 1.3. Train Ridge (5CV tune)
on data with hot encoded categorical variables
"""
# In[]
X_train_cat_dummies = pd.get_dummies(X_train_cat)
X_test_cat_dummies = pd.get_dummies(X_test_cat)

X_train = np.concatenate((X_train_num, X_train_cat_dummies), axis=1)
X_test = np.concatenate((X_test_num, X_test_cat_dummies), axis=1)

# In[]
# Set hyperparameters to tune
a_vals = np.geomspace(0.00001, 10000, 50)
tol_vals = np.geomspace(0.00001, 1, 8)
max_iters = np.geomspace(10000, 20000, 4)

# Create a list of all possible combinations of the two hyperparameters,
# will be used to find best hyperparameters
combos = list(itertools.product(a_vals, tol_vals, max_iters))

# Make an empty array whos size is that of
# the length of combos to store validation scores
params = []
k = 5

for a, t, m in combos:
    clf = Ridge(alpha=a, tol=t, max_iter=m)
    cv_results = cross_validate(clf, X_train, y_train, cv=k)
    params += [np.mean(cv_results["test_score"])]

# In[]
best_score_idx = np.argmax(params)
best_alpha, best_tol_val, best_max_iter = combos[best_score_idx]

clf = Ridge(alpha=best_alpha, tol=best_tol_val, max_iter=best_max_iter)
clf.fit(X_train, y_train)

reg_l2_testing_score = clf.score(X_test, y_test)
# In[]
print(f"Regression with PCA & MCA score: {test_score}")
print(f"Regression with dummies score: {reg_l2_testing_score}")
# %%


# In[]
"""
Discussion of part 1 outcomes:
Ridge with dummies outperformed regression with PCA and MCA.
Ridge was also less time consuming as it did not have to do
additional computations except one hot encoding.
"""

# In[]
"""
PART 2.
While we perform dimension reduction separately for numerical and categorical
data, there are methods that can perform clustering analysis with numerical
and categorical data combined. As usual, the most important aspect is the
distance metric to use. For mixed data types, researchers have proposed to
use the Gower distance. The Gower distance is essentially a special distance
metric that measures numerical data and categorical data separately,
then combine them to form a distance calculation.

PART 2.1. Load data set
"""
print()
print("PROJECT 1. PART 2.")

data_path = "project1_train.csv"
data = pd.read_csv(data_path)
data = data.drop("Id", axis=1)

# Remove columns that have too many missing values
data = data.drop(data.columns[data.isnull().sum() > 30], axis=1)

# Remove rows with missing values
data.dropna(inplace=True)
print(data.head())

# In[3]:
X = data.copy()
del X["SalePrice"]
y = data["SalePrice"]
X.shape  # (1451, 63)
y.shape  # (1451,)

# In[3]:
"""
PART 2.2. Compute the Gower distance of the full predictors set,
i.e. no train/test split.
"""
gd_matrix = gower.gower_matrix(X)
gd_matrix

# In[3]:
"""
PART 2.3. Apply K-medoids using the gower distance matrix as input.
"""
# To use the K-medoid function, provide some initial centers. Let's take k =5.
# Randomly sample 5 observations as centers.
np.random.seed(50)
k = 5
center_index = np.random.randint(0, len(y), k)
print("centers: ", center_index)

kmedoids_gd = kmedoids(gd_matrix, center_index, data_type="distance_matrix")
kmedoids_gd.process()
medoids = kmedoids_gd.get_medoids()
clusters = kmedoids_gd.get_clusters()
print("Medoids:", medoids)

# Cluster output for each cluster of size 5
for i in range(k):
    print(f"cluster in {i} is {clusters[i]}")
    print(clusters[i])

# result:
# clustering result tells us which observations belong to cluster k

# In[3]:
"""
PART 2.4.a first create an array that records the cluster membership of each
observation. Assign labels to Clusters"""
labels = np.zeros([len(X)], dtype=int)
for i in range(k):
    labels[clusters[i]] = i

"""
PART 2.4.b Bin the response variable (of the original data set)
# into the number of categories you used for k-medoids"""
bins = pd.qcut(y, k, range(k))
print("response variable binned into 5 categories :", bins)

# In[]:
"""
PART 2.4.optional observation:
As kmedoids return unordered labels and qcut returns ordered labels,
we need to make sure that NMI does not depend on lables permutations.

Observation:
means for each cluster using kmedoids is non sequential whereas
mean of each cluster using qcut is sequential. However,
that does not affect NMI score calculations!
"""
for i in range(5):
    print(y[labels == i].mean())

# labels
# Categories (5, int64): [0 < 1 < 2 < 3 < 4]
# 146266.26962457338
# 130985.00852272728
# 238547.92651757188
# 236582.3855799373
# 132112.3563218391

for i in range(5):
    print(y[bins == i].mean())

# bins
# Categories (5, int64): [0 < 1 < 2 < 3 < 4]
# 100760.70169491526
# 135884.31632653062
# 163317.5809859155
# 201066.3676975945
# 304943.1010452962

# In[3]:
"""
PART 2.4.c Compute the normalized mutual information (NMI)
between your clustering results and the binned categories.
"""
NMI_5 = normalized_mutual_info_score(bins, labels)
print("NMI score for K = 5 is", NMI_5)
# NMI score for K = 5 is 0.21221029395430538

# In[3]:
"""
PART 2. Additional analysis:
As results for 5 clusters are not very good, we will try
to find better number of clusters:
 -repeat the clustering for k in [4, 3 ,2]
"""
for k in [4, 3, 2]:
    # Let's compute NMI for several K values to compare the results
    # Randomly sample 4 observations as initial centers.
    np.random.seed(50)
    center_index = np.random.randint(0, len(y), k)
    print("centers: ", center_index)

    # 3. Apply K-medoids using the gower distance matrix as input.
    kmedoids_gd = kmedoids(gd_matrix, center_index, data_type="distance_matrix")

    # Run cluster analysis and obtain results
    kmedoids_gd.process()
    # finding new medoids
    medoids = kmedoids_gd.get_medoids()
    # finding new clusters
    clusters = kmedoids_gd.get_clusters()

    print("Medoids:", medoids)

    # Cluster output for each cluster of size 5
    for i in range(k):
        print(f"cluster in {i} is {clusters[i]}")

    # 4.a first create an array that records
    # the cluster membership of each observation
    # Assign labels to Clusters
    labels = np.zeros([len(X)], dtype=int)
    for i in range(k):
        labels[clusters[i]] = i
        print(labels)

    # 4.b Bin the response variable (of the original data set) into the number
    # of categories you used for k-medoids
    bins = pd.qcut(y, k, range(k))
    print("response variable binned into 4 categories :", bins)

    # 4. c Compute the normalized mutual information (NMI) between your
    # clustering results and the binned categories.
    NMI_4 = normalized_mutual_info_score(bins, labels)
    print(f"NMI score for K = {k} is", NMI_4)

# NMI score for K = 4 is 0.24005278162951577
# NMI score for K = 3 is 0.28863032518077514
# NMI score for K = 2 is 0.39546405815174285
# In[]

"""
Discussion of part 2 outcomes::
Using NMI score we are determining the score of clustering
NMI score increases as the number of cluster decreases.
NMI score is highest for clustter size of 2
and NMI score is lowest for cluster size as 5
"""
