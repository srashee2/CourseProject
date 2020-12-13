# Iterative Topic Modeling with Time Series Feedback

## Team Starks
Saad Rasheed - srashee2
Javier Huamani - huamani2 
Sai Allu - allu2

## Purpose
Team starks set out to recreate "Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback"

## Introduction 
In our final project we create a text mining application to find causal topics from corpus data. Our application takes a probabilistic topic model and using time-series data we explore topics that are causally correlated with said time-series data. We improve on the topics at each iteration by using prior distributions. To determine causality we are using Granger causality tests which is a popular testing mechanism used with lead and lag relationships across time series.

## Libraries
pandas
sklearn
nltk
pyLDAvis
statsmodels

## Files
**Corpus Data**
NYT2000_1.csv
NYT2000_1.csv

**IEM Stock Data**
IEM2000.xlsx

**Main Application**
SKLearn Test.py

## Code Walkthrough
We begin by reading in the corpus data that we segmented into two files to be able to store alongside the code. We then clean the data by removing NaN values and other filtering. We do more filtering and remove unneccesary characters and then lemmatize the data. 

We then read in the IEM data and normalize it.

We then take the corpus data and create a document term matrix



## How to run
The easiest way to run our code is to download Anaconda (https://www.anaconda.com/products/individual) and run it through jupyter notebook

clone this repository
$ cd CourseProject
$ jupyter notebook

You can now click on SKLearn Test.py and click run all cells. Give it about 10 minutes to let all the code run.


