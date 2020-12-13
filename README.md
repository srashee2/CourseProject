# CS410 Final Project: Iterative Topic Modeling with Time Series Feedback

## Team Starks<br>
Saad Rasheed - srashee2<br>
Javier Huamani - huamani2 <br>
Sai Allu - allu2<br>

## Demonstration
(Put Video Link here)

## Purpose
Team starks set out to recreate **Mining Causal Topics in Text Data**

**Title**<br>
Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback

**Authors**<br>
Hyun Duk Kim, Malu Castellanos, Meichun Hsu, ChengXiang Zhai, Thomas Rietz, and Daniel Diermeier

**Citation**<br>
Mining causal topics in text data: Iterative topic modeling with time series feedback. In Proceedings of the 22nd ACM international conference on information & knowledge management (CIKM 2013). ACM, New York, NY, USA, 885-890. DOI=10.1145/2505515.2505612

## Introduction 
In our final project we create a text mining application to find causal topics from corpus data. Our application takes a probabilistic topic model and using time-series data we explore topics that are causally correlated with said time-series data. We improve on the topics at each iteration by using prior distributions. To determine causality we are using Granger causality tests which is a popular testing mechanism used with lead and lag relationships across time series.

## Libraries
* `pandas` - Used for data manipulation and analysis
* `scikit-learn` - Used for classification, regression and clustering algorithms
* `nltk` - Used for symbolic and statistical natural language processesing
* `pyLDAvis` - Used to help interptret topics from a LDA topic model
* `statsmodels` - Used for statistical computations

## Files
**Corpus Data**
* `NYT2000_1.csv`
* `NYT2000_2.csv`

**IEM Stock Data**
* `IEM2000.xlsx`

**Main Application**
* `LDA.py`

## Code Walkthrough
>We begin by reading in the corpus data that we segmented into two files to be able to store alongside the code. We then clean the data by removing NaN values and other filtering. We do more filtering and remove unneccesary characters and then lemmatize the data. 

>We then read in the IEM data and normalize it.

>We then take the corpus data and generate the counts and vocabulary to create a document term matrix

>Now we're able to fit the LDA model, we use 15 topics and have found that number to be optimal. For each date we create a topic stream and aggregate topic coverages and plot them.

>To evaluate causality we then run Granger tests against each topic and output the p values for the f Tests against each lag. To determine the optimal lag value we aggregate p values. We sort the p values in ascending order.

>Of the top 25 words for each topic we run granger causality tests and pearson coefficient tests. We only continue if we get a p value of less than .05. To actually create the priors we evaluate a topic based on its negative or positive bias. If a topic has a dominated negative or positive bias we create a prior for each word and assign it to a single topic. Conversely, if there is no negative or positive bias we split the word into two topics and assign it to a single topic.

>Our code then iterates using the generated prior (On the first iteration the priors are empty) and fits the LDA model again according to the max iteration.



## How to run
The easiest way to run our code is to download [Anaconda](https://www.anaconda.com/products/individual) and run it through jupyter notebook


```$ git clone https://github.com/srashee2/CourseProject.git```<br>
```$ cd CourseProject```<br>
```$ jupyter notebook```<br>

You can now click on LDA.ipynb and click run all cells. It will take some time to run through the code, approximately 1 hour.

## Contributions
Team Starks came together over the course of a few months with weekly meetings to understand, learn and recreate Iterative Topic Modeling with Time Series Feedback. More specific contributions for the team members can be found below.<br>

All team members did the following: library research, paper breakdown and documentation. <br>
<br>
>Saad Rasheed - Logistical work, Corpus Text Extraction, Presentation, and LDA modeling iteration<br>
>Javier Huamani - Text Filtering and Manipulation, LDA Modeling, Granger Causality, and Pearson Coefficient Tests <br>
>Sai Allu - Text Filtering and Manipulation, LDA Modeling, Granger Causality, and Presentation <br>
