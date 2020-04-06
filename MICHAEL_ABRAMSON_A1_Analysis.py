#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries for data analysis including dataframe, graphing, and modeling tools.

# In[1]:


# importing libraries
#must install catboost with pip
#pip install catboost --no-cache-dir
import pandas as pd # data science essentials
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # data visualization
import statsmodels.formula.api as smf # linear regression (statsmodels)
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression # linear regression (scikit-learn)
import numpy as np #log and math tools
from sklearn.linear_model import Ridge #ridge regression
from sklearn.linear_model import Lasso #lasso regression
import sklearn.linear_model #contains ARD regression
from catboost import CatBoostRegressor #best gradient regression package
from contextlib import contextmanager #first package for output suppression
import sys, os #second package for output suppression


# Loading the data set for analysis.

# In[2]:


#locating dataset
file = 'Apprentice_Chef_Dataset.xlsx'

#reading dataset into a dataframe
dataset = pd.read_excel(file)


# Histograms for outlier analysis. Histograms reveal information about normalcy of variables as well as distribution of categorical variables.

# In[3]:


#prepares dataset for histogram creation, removes non-numerical or non-continuous variables
histset = dataset.drop(['REVENUE',
                           'NAME',
                           'EMAIL',
                           'FIRST_NAME',
                           'FAMILY_NAME',
                           'MOBILE_NUMBER'],
                           axis = 1)

#histset.hist( figsize = (10,100), bins=100, grid=False, layout=(23,1))


# Regression plots to examine explanatory variables' relationships with the response. This reveals possible candidates for log transformations and possible variables that should be removed from the regression. Possible outlier thresholds are also revealed in these plots.

# In[4]:


#preparing data for regplots
regplots_df = dataset.drop(['NAME',
                           'EMAIL',
                           'FIRST_NAME',
                           'FAMILY_NAME',
                           'MOBILE_NUMBER'],
                        axis = 1)

#regplots for each variable vs revenue, must run one at a time and comment all others out
#sns.regplot(x=regplots_df["WEEKLY_PLAN"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["CROSS_SELL_SUCCESS"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["TOTAL_PHOTOS_VIEWED"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["AVG_CLICKS_PER_VISIT"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["MEDIAN_MEAL_RATING"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["MASTER_CLASSES_ATTENDED"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["LARGEST_ORDER_SIZE"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["AVG_PREP_VID_TIME"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["FOLLOWED_RECOMMENDATIONS_PCT"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["REFRIGERATED_LOCKER"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["PACKAGE_LOCKER"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["LATE_DELIVERIES"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["EARLY_DELIVERIES"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["MOBILE_LOGINS"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["PC_LOGINS"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["TASTES_AND_PREFERENCES"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["CANCELLATIONS_AFTER_NOON"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["CANCELLATIONS_BEFORE_NOON"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["AVG_TIME_PER_SITE_VISIT"], y=regplots_df["REVENUE"]) 
#sns.regplot(x=regplots_df["PRODUCT_CATEGORIES_VIEWED"], y=regplots_df["REVENUE"])     
#sns.regplot(x=regplots_df["CONTACTS_W_CUSTOMER_SERVICE"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["UNIQUE_MEALS_PURCH"], y=regplots_df["REVENUE"])
#sns.regplot(x=regplots_df["TOTAL_MEALS_ORDERED"], y=regplots_df["REVENUE"])


# Analysis of the histograms and regplots revealed outlier thresholds for 'AVG_PREP_VID_TIME', 'UNIQUE_MEALS_PURCH', 'TOTAL_MEALS_ORDERED', 'AVG_TIME_PER_SITE_VISIT', 'EARLY_DELIVERIES', 'AVG_PREP_VID_TIME', 'AVG_CLICKS_PER_VISIT', and 'LARGEST_ORDER_SIZE'. Analysis revealed log transformation in 'AVG_TIME_PER_SITE_VISIT' and 'TOTAL_PHOTOS_VIEWED'.

# Next I examined a heatmap to look for high correlations between explanatory variables. Sometimes highly correlated explanatory variables either need to be removed or need to be given an interaction term such as variable1*variable2.

# In[5]:


#set size of figure
#plt.figure(figsize=(10,10))

#heatmap
#sns.heatmap(histset.corr())


# Next I moved onto testing various regression models to see which had the highest baseline performance with this dataset. In order to begin this phase in my analysis, I identified by my x and y variable data and did a train_test_split to create training and testing data.

# In[6]:


# preparing explanatory variable data
chef_data   = dataset.drop(['REVENUE',
                           'NAME',
                           'EMAIL',
                           'FIRST_NAME',
                           'FAMILY_NAME',
                           'MOBILE_NUMBER'],
                           axis = 1)

# preparing response variable data
chef_target = dataset.loc[:, 'REVENUE']

# preparing training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
            chef_data,
            chef_target,
            test_size = 0.25,
            random_state = 222)


# Next I moved on to preparing and executing my linear regression. First I used statsmodels to get an R^2 and then I use sklearn to fit and predict a model.

# In[7]:


# declaring set of x-variables
x_variables = chef_data.columns

# looping to make x-variables suitable for statsmodels
#for val in x_variables:
    #print(f"chef_train['{val}'] +")


# In[8]:


# merging X_train and y_train so that they can be used in statsmodels
chef_train = pd.concat([X_train, y_train], axis = 1)


# Step 1: build a model
lm_best = smf.ols(formula =  """REVENUE ~chef_train['CROSS_SELL_SUCCESS'] +
                                            chef_train['TOTAL_MEALS_ORDERED'] +
                                            chef_train['UNIQUE_MEALS_PURCH'] +
                                            chef_train['CONTACTS_W_CUSTOMER_SERVICE'] +
                                            chef_train['PRODUCT_CATEGORIES_VIEWED'] +
                                            chef_train['AVG_TIME_PER_SITE_VISIT'] +
                                            chef_train['CANCELLATIONS_BEFORE_NOON'] +
                                            chef_train['CANCELLATIONS_AFTER_NOON'] +
                                            chef_train['TASTES_AND_PREFERENCES'] +
                                            chef_train['PC_LOGINS'] +
                                            chef_train['MOBILE_LOGINS'] +
                                            chef_train['WEEKLY_PLAN'] +
                                            chef_train['EARLY_DELIVERIES'] +
                                            chef_train['LATE_DELIVERIES'] +
                                            chef_train['PACKAGE_LOCKER'] +
                                            chef_train['REFRIGERATED_LOCKER'] +
                                            chef_train['FOLLOWED_RECOMMENDATIONS_PCT'] +
                                            chef_train['AVG_PREP_VID_TIME'] +
                                            chef_train['LARGEST_ORDER_SIZE'] +
                                            chef_train['MASTER_CLASSES_ATTENDED'] +
                                            chef_train['MEDIAN_MEAL_RATING'] +
                                            chef_train['AVG_CLICKS_PER_VISIT'] +
                                            chef_train['TOTAL_PHOTOS_VIEWED']""",
                                            data = chef_train)


# Step 2: fit the model based on the data
results = lm_best.fit()

# Step 3: analyze the summary output. uncomment out the print statement below for a summary with R^2 for the linear regression
#print(results.summary())


# Next I instantiated, fit, and predicted my standard linear regression model. I saved and printed the test and train scores for this model.

# In[9]:


# INSTANTIATING a model object
lr = LinearRegression()


# FITTING to the training data
lr_fit = lr.fit(X_train, y_train)


# PREDICTING on new data
lr_pred = lr_fit.predict(X_test)


# SCORING the results
#print('Training Score:', lr.score(X_train, y_train).round(4))
#print('Testing Score:',  lr.score(X_test, y_test).round(4))

# saving scoring data for future use
lr_train_score = lr.score(X_train, y_train).round(4)
lr_test_score  = lr.score(X_test, y_test).round(4)


# Next I instantiated, fit, and predicted my ridge regression model. I also saved and printed these results.

# In[10]:


# INSTANTIATING a model object
ridge_model = Ridge()

# FITTING the training data
ridge_fit = ridge_model.fit(X_train, y_train)


# PREDICTING on new data
ridge_pred = ridge_model.predict(X_test)

#print('Training Score:', ridge_model.score(X_train, y_train).round(4))
#print('Testing Score:',  ridge_model.score(X_test, y_test).round(4))


# saving scoring data for future use
ridge_train_score = ridge_model.score(X_train, y_train).round(4)
ridge_test_score  = ridge_model.score(X_test, y_test).round(4)


# Next I instantiated, fit, and predicted my lasso regression model. I also saved and printed these results.

# In[11]:


# INSTANTIATING a model object
lasso_model = Lasso()

# FITTING the training data
lasso_fit = lasso_model.fit(X_train, y_train)


# PREDICTING on new data
lasso_pred = lasso_model.predict(X_test)

#print('Training Score:', lasso_model.score(X_train, y_train).round(4))
#print('Testing Score:',  lasso_model.score(X_test, y_test).round(4))


# saving scoring data for future use
lasso_train_score = lasso_model.score(X_train, y_train).round(4)
lasso_test_score  = lasso_model.score(X_test, y_test).round(4)


# Next I instantiated, fit, and predicted my ARD regression model. Once again I saved and printed these results.

# In[12]:


# INSTANTIATING a model object
ard_model = sklearn.linear_model.ARDRegression()


# FITTING the training data
ard_fit = ard_model.fit(X_train, y_train)


# PREDICTING on new data
ard_pred = ard_model.predict(X_test)


#print('Training Score:', ard_model.score(X_train, y_train).round(4))
#print('Testing Score:',  ard_model.score(X_test, y_test).round(4))


# saving scoring data for future use
ard_train_score = ard_model.score(X_train, y_train).round(4)
ard_test_score  = ard_model.score(X_test, y_test).round(4)


# Now I tried my CatBoostRegression. As always, I print and save my results.

# In[13]:


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
cat_model = CatBoostRegressor(learning_rate=.01,iterations=4000,depth = 6,  cat_features = [0,8,14,15], l2_leaf_reg=6,thread_count=4,
                             border_count=50)

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

#print('Training Score:', cat_model.score(X_train, y_train).round(4))
#print('Testing Score:',  cat_model.score(X_test, y_test).round(4))


# Next I compared the results of these models and selected my final model type which I would tweak based on the findings in this analysis.

# In[14]:


# comparing results

print(f"""
Model      Train Score      Test Score
-----      -----------      ----------
OLS        {lr_train_score}           {lr_test_score}
Ridge      {ridge_train_score}           {ridge_test_score}
Lasso      {lasso_train_score}           {lasso_test_score}
ARD        {ard_train_score}           {ard_test_score}
Cat        {cat_train_score}           {cat_test_score}
""")


# creating a dictionary for model results
model_performance = {'Model'    : ['OLS', 'Ridge', 'Lasso', 'ARD', 'Cat'],
           
                     'Training' : [lr_train_score, ridge_train_score,
                                   lasso_train_score, ard_train_score, cat_train_score],
           
                     'Testing'  : [lr_test_score, ridge_test_score,
                                   lasso_test_score, ard_test_score, cat_test_score]}


# converting model_performance into a DataFrame
model_performance = pd.DataFrame(model_performance)


# sending model results to Excel
model_performance.to_excel('regression_model_performance.xlsx',
                           index = False)


# I chose to use CatBoostRegression which is a form of gradient boost regression.                     

# Below I will show a number of variables that I created and tested. For the most part, these variables increase the test scores of linear, ridge, lasso, and ARD regressions but do not increase the score of my Cat regression. Additionally, the Cat regression test score remains higher despite the increases in the test scores of the other regressions. Therefore, I stuck with Cat for my final model.

# In[ ]:


#I tried creating a series of categorical variables to mark turning points in revenue trends for the following variables
#contacts with customer service > 10 marks a turning point in trend
#dataset['CWCSover10'] = [1 if x >10 else 0 for x in dataset['CONTACTS_W_CUSTOMER_SERVICE']]

#clicksover15 marks a turning point in trend
#dataset['Clicksover15'] = [1 if x >15 else 0 for x in dataset['AVG_CLICKS_PER_VISIT']]

#clicksover15 marks a turning point in trend
#dataset['latedeliveriesover13'] = [1 if x >13 else 0 for x in dataset['LATE_DELIVERIES']]

#dummy variable for mobile logins trend threshold
#dataset['mobilecat'] = [1 if x == 1 or x==2 else 0 for x in dataset['MOBILE_LOGINS']]

#dummyvariable for unique meals purchased trend threshold
#dataset['uniquecat'] = [1 if x>9.5 else 0 for x in dataset['UNIQUE_MEALS_PURCH']]

#I tried one hot encoding median meal rating
#dataset = pd.concat([dataset, pd.get_dummies(dataset['MEDIAN_MEAL_RATING'])],axis=1)
#dataset.columns = [                     'REVENUE',           'CROSS_SELL_SUCCESS',
#                               'NAME',                        'EMAIL',
#                         'FIRST_NAME',                  'FAMILY_NAME',
#                'TOTAL_MEALS_ORDERED',           'UNIQUE_MEALS_PURCH',
#        'CONTACTS_W_CUSTOMER_SERVICE',    'PRODUCT_CATEGORIES_VIEWED',
#            'AVG_TIME_PER_SITE_VISIT',                'MOBILE_NUMBER',
#          'CANCELLATIONS_BEFORE_NOON',     'CANCELLATIONS_AFTER_NOON',
#            'TASTES_AND_PREFERENCES',                    'PC_LOGINS',
#                      'MOBILE_LOGINS',                  'WEEKLY_PLAN',
#                   'EARLY_DELIVERIES',              'LATE_DELIVERIES',
#                     'PACKAGE_LOCKER',          'REFRIGERATED_LOCKER',
#       'FOLLOWED_RECOMMENDATIONS_PCT',            'AVG_PREP_VID_TIME',
#                 'LARGEST_ORDER_SIZE',      'MASTER_CLASSES_ATTENDED',
#                 'MEDIAN_MEAL_RATING',         'AVG_CLICKS_PER_VISIT',
#                'TOTAL_PHOTOS_VIEWED',                   'CWCSover10',
#                       'Clicksover15',         'latedeliveriesover13',
#                          'mobilecat',                    'uniquecat',
#                           'rating_1',                     'rating_2',
#                           'rating_3',                     'rating_4',
#                           'rating_5']

#I tried one hot encoding masters classes attended
#dataset = pd.concat([dataset, pd.get_dummies(dataset['MASTER_CLASSES_ATTENDED'])],axis=1)
#dataset.columns = [                     'REVENUE',           'CROSS_SELL_SUCCESS',
#                               'NAME',                        'EMAIL',
#                         'FIRST_NAME',                  'FAMILY_NAME',
#                'TOTAL_MEALS_ORDERED',           'UNIQUE_MEALS_PURCH',
#        'CONTACTS_W_CUSTOMER_SERVICE',    'PRODUCT_CATEGORIES_VIEWED',
#            'AVG_TIME_PER_SITE_VISIT',                'MOBILE_NUMBER',
#          'CANCELLATIONS_BEFORE_NOON',     'CANCELLATIONS_AFTER_NOON',
#             'TASTES_AND_PREFERENCES',                    'PC_LOGINS',
#                      'MOBILE_LOGINS',                  'WEEKLY_PLAN',
#                   'EARLY_DELIVERIES',              'LATE_DELIVERIES',
#                     'PACKAGE_LOCKER',          'REFRIGERATED_LOCKER',
#       'FOLLOWED_RECOMMENDATIONS_PCT',            'AVG_PREP_VID_TIME',
#                 'LARGEST_ORDER_SIZE',      'MASTER_CLASSES_ATTENDED',
#                 'MEDIAN_MEAL_RATING',         'AVG_CLICKS_PER_VISIT',
#                'TOTAL_PHOTOS_VIEWED',                   'CWCSover10',
#                       'Clicksover15',         'latedeliveriesover13',
#                          'mobilecat',                    'uniquecat',
#                           'rating_1',                     'rating_2',
#                           'rating_3',                     'rating_4',
#                           'rating_5',                      'class_0',
#                            'class_1',                      'class_2',
#                            'class_3']

#I tried splitting email into domain addresses and one hot encoding this variable into cateogricals for each domain
#dataset['domain'] = dataset['EMAIL'].str.split('@').str[1]
#dataset = pd.concat([dataset, pd.get_dummies(dataset['domain'])],axis=1)

#this loop creates interaction terms between every explanatory variable
#this helped for linear, lasso, ridge, and ARD but did not help for Cat
#L=[(x, y) for x, y in itertools.product(chef_data.columns,chef_data.columns) if x != y]
#interaction_data = pd.concat([pd.DataFrame({''.join(i):chef_data.loc[:,i].prod(axis=1)}) for i in L],axis=1)
#chef_data = pd.concat([chef_data, interaction_data], axis=1)


#                             .O°o. .o°O________________________________O°o. .o°O.
#                             .°o.O.o° ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯.°o.O.o°
#                             ░░░░░╔══╦╗░░░░╔╗░░░░░░╔╗╔╗░░░░░░░░
#                             ░░░░░╚╗╔╣╚╦═╦═╣║╔╗░░░░║║║╠═╦╦╗░░░░
#                             ░░░░░░║║║║╠╝║║║╠╝║░░░░║╚╝║║║║║░░░░
#                             ░░░░░░║║║║║║║║║╔╗╣░░░░╚╗╔╣║║║║░░░░
#                             ░░░░░░╚╝╚╩╩═╩╩╩╝╚╝░░░░░╚╝╚═╩═╝░░░░
#                             .O°o. .o°O________________________________O°o. .o°O.
#                             .°o.O.o° ¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯.°o.O.o°..

# In[ ]:




