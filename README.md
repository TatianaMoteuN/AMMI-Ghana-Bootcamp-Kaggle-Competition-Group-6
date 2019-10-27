# Wine price predicting kaggle competition 
This project was made for the (AMMI Ghana Bootcamp Kaggle competition)

## Problem statement
Given the following features:
1. country (String) The country that the wine is from
2. province (String) The province or state that the wine is from
3. region_1 (String) The wine growing area in a province or state (ie Napa)
4. region_2 (String) Sometimes there are more specific regions within the wine growing area (ie Rutherford inside the Napa Valley), but this value can sometimes be blank
5. winery (String) The winery that made the wine
6. variety (String) The type of grapes used to make the wine (ie Pinot Noir)
7. designation (String) The vineyard within the winery where the grapes that made the wine are from
8. taster_name (String) taster name
9. taster_twitter_handle (String) taster twitter account name
10. description (String) A few sentences from a sommelier describing the wine's taste, smell, look, feel, etc.
11. points (Numeric) Number of points WineEnthusiast rated the wine on a scale of 1-100

We need to predict the price (Numeric) The cost for a bottle of wine.

## dependencies
```
pip3 install -r requirements.txt
```

## feature engnieering
for the models to be able to deal with the categorical features some preprocessing was made.
* country, region_2, province, taster_name and variety were encoded as one hot vectors
* title, region_1 and designation were vectorized using CountVectorizer from ```sklearn```
* taster_twitter_handle was ignored due to it's redundant contribution to the data (see visualisation.ipynb)
* And finally the description feature was encoded using Word2Vec (by summing the vectors representing all of a training example description)

## Models
1. Linear regression
2. Dicision Trees
3. Random Forests
4. Neural networks

## Techniques used
* K-Fold cross validation
* Word Embeddings
* GridSearch hyper-parameters optimization
* One Hot Enconding
* CountVectorizer
* PCA
