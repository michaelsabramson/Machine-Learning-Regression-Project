#!/usr/bin/env python
# coding: utf-8

# In[62]:


# timeit

# Student Name : Michael Abramson
# Cohort       : 3

# Note: You are only allowed to submit ONE final model for this assignment.


################################################################################
# Import Packages
################################################################################

#must install catboost with pip
#pip install catboost --no-cache-dir
import pandas as pd # data science essentials
from sklearn.model_selection import train_test_split # train/test split
import numpy as np #tools for log transformation
import sklearn.linear_model #linear modeling package
from catboost import CatBoostRegressor #best gradient regression package
from contextlib import contextmanager #first package for output suppression
import sys, os #second package for output suppression

################################################################################
# Load Data
################################################################################

file = 'Apprentice_Chef_Dataset.xlsx' #locating dataset

original_df = pd.read_excel(file) #reading dataset into a dataframe

################################################################################
# Feature Engineering and (optional) Dataset Standardization
################################################################################

#setting outliers
original_df = original_df.loc[original_df['AVG_PREP_VID_TIME']<700]
original_df = original_df.loc[original_df['UNIQUE_MEALS_PURCH']<9.5]
original_df = original_df.loc[original_df['TOTAL_MEALS_ORDERED']<450]
original_df = original_df.loc[original_df['AVG_TIME_PER_SITE_VISIT']<180]
original_df = original_df.loc[original_df['EARLY_DELIVERIES']<6]
original_df = original_df.loc[original_df['AVG_PREP_VID_TIME']<=295] 
original_df = original_df.loc[original_df['AVG_CLICKS_PER_VISIT']>8]
original_df = original_df.loc[original_df['LARGEST_ORDER_SIZE']<10] 

# preparing response variable data
chef_target = original_df.loc[:, 'REVENUE']
chef_target = np.log(chef_target) #log transformation

#preparing explanatory variable data
chef_data   = original_df.drop(['REVENUE',
                            'NAME',
                           'EMAIL',
                           'FIRST_NAME',
                           'FAMILY_NAME',
                           'MOBILE_NUMBER'],
                           axis=1)
chef_data['AVG_TIME_PER_SITE_VISIT'] = np.log(chef_data['AVG_TIME_PER_SITE_VISIT']) #log transformation
chef_data['TOTAL_PHOTOS_VIEWED'] = chef_data['TOTAL_PHOTOS_VIEWED']+1 #preparation for log transformation
chef_data['TOTAL_PHOTOS_VIEWED'] = np.log(chef_data['TOTAL_PHOTOS_VIEWED']) #log transformation

################################################################################
# Train/Test Split
################################################################################

# preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            chef_data,
            chef_target,
            test_size = 0.25,
            random_state = 222)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

#function for suppressing unnecessary iteration output
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# INSTANTIATING a model object with chosen hyperparamaters
cat_model = CatBoostRegressor(learning_rate=.007,iterations=4000,depth =5,  cat_features = [0,8,14,15], l2_leaf_reg=6,thread_count=4,
                             border_count=50, random_strength=.9, grow_policy='Depthwise', min_data_in_leaf=1)

with suppress_stdout():
    # FITTING the training data
    cat_fit = cat_model.fit(X_train, y_train,
                           use_best_model=True,
                          eval_set= (X_test, y_test))


    # PREDICTING on new data
    cat_pred = cat_model.predict(X_test)

# saving scoring data for future use
cat_train_score = cat_model.score(X_train, y_train).round(4)
cat_test_score  = cat_model.score(X_test, y_test).round(4)

################################################################################
# Final Model Score (score)
################################################################################

#printing train and test score
print('Training Score:', cat_model.score(X_train, y_train).round(4))
print('Testing Score:',  cat_model.score(X_test, y_test).round(4))

# saving scoring data for future use
cat_train_score = cat_model.score(X_train, y_train).round(4)
cat_test_score  = cat_model.score(X_test, y_test).round(4)
test_score = cat_test_score



# In[ ]:




