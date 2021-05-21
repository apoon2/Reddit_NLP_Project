# Reddit Web API and NLP Project
by Ashley Poon

![alt text](stocks-wsb.png?raw=true)

## Problem Statement

In late January 2021, Gamestop's stock prices rose dramatically despite the company's shares being shorted at high rates. The forces behind the movement were a handful of r/WallStreetBets users who convinced Redditors en masse to buy up a bunch of GameStop stock, therefore increasing its value. As with any day-trade, this resulted in large profits for some and large losses for others, and the risks were especially magnified when the stock value was driven by the whims of a small group of Redditors rather than market inputs.

This project seeks to create a model that will help every day investors find less risky, long-term investments backed by financial and business results. In other words, it helps investors distinguish investment advice from r/Stocks vs r/WallStreetBets. The goal is to minimize instances of investors being told to buy a regular stock when it is actually a "meme stock" (minimize false negatives and optimize for recall/sensitivity).

[source](https://time.com/5933242/gamestop-stock-gme/)

## Data Collection via API

[r/stocks](https://www.reddit.com/r/stocks/)
[r/wallstreetbets](https://www.reddit.com/r/wallstreetbets/)

1. Leveraged Pushshift API to collect comments from r/stocks and r/wallstreetbets.

2. Created a function that iterates the collection of 100 comments at a time per subreddit, using a sleep timer of 5 seconds. With each iteration, we find the minimum epoch time and start there for the next iteration, and so on until the desired number of comments were collected.

3. Combined all pulls into a single csv file to use for cleaning and modeling.

## Cleaning & EDA 

1. Dropped all rows where comments were removed.

2. Checked the balance of classes / baseline accuracy (51% stocks, 49% wsb)

3. Binarized subreddits (0 for stocks, 1 for wsb)

4. Using CountVectorizer to transform comments into bag-of-words, we observed the top most frequent words, excluding stop-words, were 'just', 'like', 'stocks', 'stock', 'don' (likely don't), 'buy', 'https', 'com', and 'money'.

## The Modeling Process

This project used NLP to train various classification models to find the best accuracy score that also minimizes false negatives.

### 1) Baseline Models with Default Parameters

Fitting several classification models with default parameters to identify ones to move forward in hyperparameter tuning.

|Model                          |Train Score|Test Score|Continue?                             |
|-------------------------------|-----------|----------|--------------------------------------|
|CountVectorizer + BernoulliNB  |73%        |71%       |Yes - similar scores                  |
|CountVectorizer + MultinomialNB|84%        |76%       |Yes - scores higher than Bernoulli    |                        
|TfidfVectorizer + MultinomialNB|84%        |74%       |No - test score lower than cvec       |
|CountVectorizer + KNN          |73%        |61%       |No - low test score                   |
|CountVectorizer + LogReg       |91%        |77%       |Yes - high test score but needs tuning|
|CountVectorizer + SVC          |84%        |76%       |Yes - high test score but needs tuning|
|CountVectorizer + Decision Tree|99%        |67%       |Yes - test performance with tuning    |

Of the models above, did not move foward with TfidfVectorizer for NLP as that performed slightly worse than CountVectorizer. Also did not move forward with KNN as that performed the worst out of all classifiers. As for Decision Trees, although that performed poorly relative to other models, it is worth testing with hyperparameter tuning since it is prone to overfitting.

### 2) Naive Bayes

Overall, Naive Bayes models performed relatively well. The Bernoulli model produced the same train and test score of 73% which means no over or under-fitting. The Multinomial model produced higher scores than Bernoulli but the train score is 82% vs 77% for test, showing signs of slight overfitting.
    
|Model                          |Train Score|Test Score|Best Params                           |
|-------------------------------|-----------|----------|--------------------------------------|
|CountVectorizer + BernoulliNB  |73%        |73%       |{'cvec__max_features': 5000, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': 'english'}|
|CountVectorizer + MultinomialNB|82%        |77%       |{'cvec__max_features': 10000, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': ['just', 'like', 'stocks', 'stock']}|                        

The Bernoulli model minimizes false negatives and optimizes for recall/sensitivity (what we want) - predicted 203 false negatives versus 1240 false postives.

The Multinomial model has fairly close number of false negatives and false positives - predicted 660 false negatives versus 574 false positives.

#### Lemmatizing
Lemmatizer on Bernoulli model resulted in minimal changes to scores: 74% train, 73% test

#### Stemming
Stemming on Bernoulli model also resulted in minimal changes to scores: 74% train, 73% test

### 3) Logistic Regression & Support Vector Machines

Overall, the LogReg and SVC models performed similarly to each other and to the Multinomial NB model. These models have about 4-5 percentage points of overfitting.
    
|Model                          |Train Score|Test Score|Best Params                           |
|-------------------------------|-----------|----------|--------------------------------------|
|CountVectorizer + LogReg       |80%        |76%       |{'cvec__max_df': 0.8, 'cvec__min_df': 20, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': ['just', 'like', 'stocks', 'stock'], 'logreg__C': 0.3, 'logreg__penalty': 'l2', 'logreg__solver': 'liblinear'}|
|CountVectorizer + SVC          |80%        |75%       |{'cvec__max_features': 10000, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': 'english', 'svc__C': 0.4, 'svc__gamma': 'scale', 'svc__kernel': 'rbf'}|                        

The LogReg model minimizes false negatives and optimizes for recall/sensitivity (what we want) - predicted 475 false negatives versus 826 false postives.

The SVC model also minimizes false negatives and optimizes for recall/sensitivity (what we want) - predicted 396 false negatives versus 972 false postives.

### 3) Decision Trees & Ensemble Models

Overall, decision trees were not as strong as Multinomial, LogReg, or SVC. The Boosting model was the strongest out of the ensemble models, with the highest score and no significant overfitting. 

|Model                          |Train Score|Test Score|Best Params                           |
|-------------------------------|-----------|----------|--------------------------------------|
|CountVectorizer + Single Tree  |68%        |66%       |{'cvec__max_features': 8000, 'cvec__ngram_range': (1, 1), 'cvec__stop_words': ['just', 'like', 'stocks', 'stock'], 'dt__ccp_alpha': 0.001, 'dt__max_depth': 15, 'dt__min_samples_leaf': 2, 'dt__min_samples_split': 3}|
|CountVectorizer + Bagging      |96%        |70%       |{'bag__n_estimators': 20, 'cvec__max_df': 0.85, 'cvec__max_features': 10000, 'cvec__min_df': 100, 'cvec__ngram_range': (1, 2), 'cvec__stop_words': ['just', 'like', 'stocks', 'stock']}
|CountVectorizer + Boosting     |77%        |74%       |{'abc__base_estimator': LogisticRegression(), 'abc__n_estimators': 150, 'cvec__max_features': 15000, 'cvec__ngram_range': (1, 2), 'cvec__stop_words': ['just', 'like', 'stocks', 'stock']}|
|CountVectorizer + Random Forest|67%        |66%       |{'cvec__max_features': 15000, 'cvec__ngram_range': (1, 2), 'cvec__stop_words': ['just', 'like', 'stocks', 'stock'], 'rf__max_depth': 3, 'rf__max_features': 0.3, 'rf__min_samples_leaf': 3, 'rf__n_estimators': 150}|  

The Single Decision Tree model minimizes false positives (not what we want) - predicted 1031 false negatives versus 819 false postives.

The Bagging model minimizes false positives (not what we want) - predicted 963 false negatives versus 662 false postives.

The Boosting model minimizes false negatives and optimizes for recall/sensitivity (what we want) - predicted 429 false negatives versus 976 false postives.

The Random Forest model also minimizes false negatives and optimizes for recall/sensitivity (what we want) - predicted 612 false negatives versus 1211 false postives.

## Model Evaluation

The Logistic Regression model was the best performing in terms of having the highest test score, minimal overfitting, while optimizing for recall/sensitivity. The test score was 76% compared to 80% for train, showing slight overfitting. The accuracy score for the model is 77%, f1-score is 78%, and the recall score is 82%. By contrast, the specificity score is 72% and our goal of optimizing towards recall is met.

The runner up models were the Support Vector Machine and Boosting models. 
- SVM had a test score of 75% and a train score of 80%, which is both slightly lower and more overfitting than LogReg. The accuracy score for the model is 74%, f1-score is 77%, and the recall score is 85% compared to 64% for specificity.
- Boosting had a test score of 74% and a train score of 77% which are lower but less overfit than LogReg. The accuracy score for the model is 74%, f1-score is 76%, and the recall score is 84% compared to 65% for specificity.

## Conclusion & Recommendations

After generating the predictions from the LogReg model and comparing it to the actual subreddits, there is greater confidence that this model is performing near the upper limit given the similar content of both forums and the bag-of-words modeling. Many of the misclassified comments are more general or vague and do not necessarily lean towards one subreddit or the other. 

For the false negatives (predicting it was stocks when it was actually wsb), there are comments that could have been accurately classified if the model was able to pick up on sarcasm, which is a common tone in wsb. Additionally, there are a few wsb terminology that the model missed as unique to wsb (ie. GUH, diamond hands).

For the false positives (predicting it was wsb when it was actually stocks), the model misclassified several comments with strong or sarcastic language as wsb which is what a human might have guessed as well. Additionally, some comments using common wsb terms (ie. GME, apes) were also misclassified as wsb.

To improve the model, we can use neural networks and sequence modeling the NLP to help pick up on nuances and sarcasm in the comments. Also, this may help improve classification of posts that use wsb's lingo.
