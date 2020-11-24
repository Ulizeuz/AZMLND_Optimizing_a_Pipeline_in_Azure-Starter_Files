# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about phone calls from a bank marketing campaing with 21 features like age, job, marital status, etc. We create and optimize ML Pipeline in two different ways: first, we use hyperdrive to get the best run and secondly we use AutoML to find the best model

## Pipeline architecture Hyperdrive

### Train script

The script train.py help us to preprocess the data:

- Load and storage the data into Tabular Dataset 
- The function clean_data helps us to clean the data (for example: handle missing values) and load the data into Pandas Dataframe 
- Additionally, split data into train and test datasets. We use 20% - 80% ratio for train and test respectively  
- Finally, apply Scikit-Learn model to fit the training data and compute the accuracy. 

### Hyperdrive

This tool helps us to find the best hyperparameters for the logistic regression model. We use the next configuration:

- Sampling. We use RandomParameterSampling in spite of grid sweep is exhaustive but consumes more time, whereas random sweep can get a good results without taking as much time, reducing computation cost. The parameters values are chosen with the function choice and we select values around the default values for C (Inverse of regularization strength) and max_iter (Maximum number of iterations to converge)

- Early stopping policy. We define with Bandit policy based on slack factor and evaluation interval. We use this policy to terminate runs if the primary metric is non within the slack factor contrast with best run.We choose a 0.1 slack factor, but if we want to have more aggresive savings it could be smaller. In the same way, we could opt to small evaluation_interval, but we decide to choose the value 2 to evaluate every 2 runs.  

- Estimator. SKLearn creates an estimator for training in Scikit-learn experiments. The parameter used are C (Inverse of regularization strength and max_iter (Maximum number of iterations to converge)

- Primary Metric Name and Primary Metric Goal. With this parameters we define what HyperDrive tries to maximize, in this case was the ACCURACY

At the end we save the hyper parameters optimized by Hyperdrive: ['--C', '1', '--max_iter', '100'] with 0.9109 of Accuracy


## AutoML
The AutoML pipeline is the following:

- Tabular Dataset. We create this dataset with the bank data. 
- Data preparation. In tis step we uses OneHotEncoder for preprocessing to encode categorical features as a one-hot numerical array.
- AutoML. We can evaluate different models in this step
- AutoML Model. Using AutoML we can find another optimized model 

AutoML generated 27 iterations, the best model with 0.9184 of accuracy was VotingEnsemble with the following classifiers:

-  min_samples_leaf=0.01
-  min_samples_split=0.01
-  min_weight_fraction_leaf=0.0
-  n_estimators=25
-  n_jobs=1
-  oob_score=False
-  random_state=None
-  verbose=0
-  warm_start=False

The second best model was MaxAbsScaler LightGBM with 0.9159 accuracy.

## Pipeline comparison

Comparing the two models, AutoML has better performance in accuracy (0.9184 vs 0.9109) and in architecture (seamless pipeline, less steps). About the runnings AutoML run for 27 iteration yet the hyperdrive run for 100.

## Future work
I'd like to clean up the project (remove innecesary comments, include outputs after each steps) and find another options for hyperparameters to improve SciKit-learn pipeline. Because the data has class inbalance problem, other area to work is dealing with it.  

Finally, for future work we can use different values for C parameters and observe the performance
