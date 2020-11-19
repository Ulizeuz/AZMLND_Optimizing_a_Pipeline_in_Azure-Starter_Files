# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains data about phone calls from a bank marketing campaing with 21 features like age, job, marital status, etc. We create and optimize ML Pipeline in two different ways: first, we use hyperdrive to get the best run and secondly we use AutoML to find the best model

## Pipeline architecture Hyperdrive
The Hyperdrive pipeline is the following:

- Train script. It's has many functions, for example clean_data to use the information. 
- Tabular Dataset. In the train script the data is storaged in a Tabular Dataset.
- Scikit Learn Logistic Regression. The Tabular Dataset is evaluated to train this model.
- Hyperdrive. This tool helps us to find the best hyperparameters for the logistic regression model
- Optimized model. We save the hyper parameters optimized by Hyperdrive  

The following are the parameters:

--C: Inverse of regularization strength
--max_iter: Maximum number of iterations to converge

We choose the RandomParameterSampling because we can choose the choices:

- For the first parameter we use the default value (1.0) and two more (0.5 and 1.5)

- For the second we use also the default value (100) and (50 and 150)

And we use the folliwing parameter for the early stopping policy:

- evaluation_interval = 2. With this value every 2 excercises apply the policy
- slack_factor = 0.1. To calculate the allowed distance from the best performing experiment run

## AutoML
The AutoML pipeline is the following:

- Tabular Dataset. We create this dataset with the bank data
- AutoML. We can evaluate different models in this step
- AutoML Model. Using AutoML we can find another optimized model 

AutoML generated 27 iterations, the best model with 0.9184 of accuracy was VotingEnsemble 

## Pipeline comparison

Comparing the two models, AutoML has better performnace in accuracy (0.9184) and in architecture (seamless pipeline, less steps)

## Future work
I'd like to clean up the project (remove innecesary comments, include outputs after each steps, delete the compute cluster) and find another options for hyperparameters to improve SciKit-learn pipeline
