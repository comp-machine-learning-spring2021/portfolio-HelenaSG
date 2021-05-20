# Music Genre Classification
![studio-vibe](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Music-Genre-Classification/theme.png)
_Image via [Official Account of "Twenty Twenty"](https://m.weibo.cn/5123810627/4550137152279582)_

## :musical_note: Introduction

As a dancer, music is an essential component of my life. In the 21st century, music streaming services have been the most popular means of entertainment for people to listen to their favorite music. Over the past few years, more advanced features have emerged. Among these features, making personalized recommendations is one of the most magical ones to me. How does it know my tastes so well? Why are the recommendations so on point?

To unfold this mystery, I selected the data provided by a research group from The Echo Nest, which is available on [Kaggle](https://www.kaggle.com/rashikrahmanpritom/177k-english-song-data-from-20082017). One dataset contains metadata about tracks, and the second one includes the track metrics such as danceability and acoustics on a scale from -1 to 1. The former is in CSV format, and the latter is in JSON format. I merged them using the ids of tracks as the primary key. The idea is that each song is essentially a deliberately designed audio. With musical features derived from this raw audio information, we can score the song on various musical features and use it for further analysis. Our goal is to build a model to classify songs as either 'Hip-Hop' or 'Rock' based on just the audio features.

:small_blue_diamond: **A complete walk-through on the machine learning process:** 

  * Data Preprocessing
    * Merging tables
    * Handling missing/categorical data
    * Bringing features onto the same scale
  * Selecting meaningful features
  * Partitioning a dataset into separate training and test datasets
  * Using k-fold cross-validation to assess model performance
  * Fine-tuning machine learning models via grid search
  * Performance evaluation

## :musical_note: Data Preprocessing

The real-world data is messy, with missing values and undesired formats, hence it is necessary to preprocess the data before fitting the model. In order to perform classification with the models, I converted the information of the genre feature so that "Rock" becomes 0 and "Hip-hop" becomes 1. I also scaled the features, given the fact that large ranges could make an influence on the performance of kNN and SVM classifiers. There are no missing values in this dataset so we do not need to worry about that. 

## :musical_note: Feature Selection

To improve interpretability and reduce the computation cost of the model, we typically want to remove irrelevant variables and avoid using variables that are correlated with each other. By creating a correlation matrix to see the pairwise correlation of columns, I selected three features that have the strongest relationship with the target variable: `speechiness`, `danceability`, and `instrumentalness`. 

## :musical_note: Modeling

Due to the variations in data, a model's performance can vary wildly depending on what data is in the training set and what data is in the testing set. If the wrong training happens to be a bad one, the model is at risk of overfitting. Therefore, I used a technique called k-fold cross-validation that calculates the metrics based on different combinations of training and testing data. In this way, the resulting model is robust to variations in the data. Specifically, for each classifier, I conducted 6-fold cross-validation on 90% of the data, setting aside 10% of the data as the test set that would be used in the end to evaluate the final model's performance.

In the 6-fold cross-validation, the data was divided into 6 folds. During the first iteration, the first fold served as the test data, and the model was trained on the combined set of the remaining five folds. In the second iteration, the second fold served as the test data, and so on. In the end, I took the average CV-scores of the 6 folds and used this information to see which one gives the best accuracy for the classification task. The CV-scores are the percentage of correctly labeled data.

Among the four models, It appeared that the ensemble method, Random Forest, worked the best with the data. I conducted a grid search to get the optimized hyperparameters for the model and tuned it.
## :musical_note: Results 

  The final model has an accuracy of about 91.06% on the test set. That's to say, on new data, we can expect this model to tell whether the song is a hip-hop song or a rock song with about 91.06% of the answers being correct. 
  
## :musical_note: Conclusion

In this project, I built a machine learning model using sklearn implementations to label the genres of each song. I performed cross-validation to choose between k-NN, SVM, and Random Forests. Based on the results, I selected the Random Forests. After fine-tuning the hyperparameters, I retrained the model and tested the general accuracy of the final model on the test set. The model achieved 91.06%  accuracy. 

