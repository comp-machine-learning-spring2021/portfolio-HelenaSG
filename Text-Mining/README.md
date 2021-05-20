# Text Mining
![detective](https://i.guim.co.uk/img/media/15ec744af8cdebb8e64e66948948a980b7dee359/0_37_2148_1289/master/2148.jpg?width=1020&quality=45&auto=format&fit=max&dpr=2&s=cf6dc98d30f5317c7c76f0f73735ea96)

> “My name is Sherlock Holmes. It is my business to know what other people do not know.” — Arthur Conan Doyle, _The Adventure of the Blue Carbuncle_
     
> “It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts.” — Arthur Conan Doyle, _A Scandal in Bohemia_

## :pencil: Introduction
When I think of a detective, the image that shows up in my mind is a man with a magnifying glass sitting in front of a table piled high with news articles. As I learn more about machine learning and data analysis, I feel fascinated by the idea of discovering patterns from facts and using this information to guide decisions. This makes me feel like a detective: there are always some gems hidden in the data. However, I found text data to be more complicated. Unlike typical data frames that I used to work with plotting, text data can be written resources, chat room conversations, social media posts, etc. They are often unstructured and do not fit in a spreadsheet with rows and columns. Also, it would be impossible for me to read all relevant articles given a certain amount of time. Out of curiosity, I started to look into the world of text mining. 

:small_blue_diamond: **An exploration on transfer learning-based sentiment analysis:** 

  * Understanding the background and framing the question
  * Conducting sentiment analysis at document-level 
  * Classifying texts in a time-saving way: transfer learning
  * Populating a data frame with the information extracted
  * Creating data visualization to find insights in the results 

## :pencil: Setting the stage 

This project is motivated by [VAST Challenge 2021](https://vast-challenge.github.io/2021/MC1.html). The data consist of current and historical news reports from multiple domestic and translated foreign sources in text file format. Specifically, these files are dealing with the kidnapping of the GASTech employees by members of the social movement group POK. One of the tasks is to characterize some biases in these news reports, concerning their representation of specific people, places, and events. As contextual mining of texts, sentiment analysis can identify and extract subjective information in source materials, which I think could be helpful. 

## :pencil: Sentiment analysis and transfer learning

Our goal is to have the news report data classified as "positive" or "negative" without having to read through them ourselves. Since there is not enough labeled data to train a reliable model for the task, the traditional supervised learning paradigm would not apply in this scenario. What we could do instead is that we could adopt a "semi-supervised" method to leverage a model that has been trained on a similar domain. This process would fall under what is called transfer learning. 

Since the news data are in text file format, a document-level sentiment analysis would be appropriate. Unfortunately, all labeled news data I found is either at sentence-level or using a different set of labels like "true" and "fake." I was able to find the [polarity dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) that looked the most suitable for the task we cared about. This dataset contains 1000 positive and 1000 negative processed reviews on movies. Movie reviews and news reports are similar because both are a mixture of quotes, descriptions, and opinions.

I split the labeled reviews data into 90% train set and 10% test set, trained three classifiers on the train set, and evaluate the results in terms of time and accuracy. The linear SVM model turned out to be the best. Then, I used all of the data for training to get the final model. The next step was to read the texts from all the news report files for classifications. After I formatted the data, I obtained the sentiment labels by applying the model. With the results, I populated a new data frame with each news report as the observation, and the columns to be the file names, polarity labels, contents, and the sources of each report. The code for this section can be found [here](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Text-Mining/Sentiment-analysis.ipynb).

## :pencil: Data visualization

In practice, when we use transfer learning to make the model generalize to a new domain, we risk deterioration or collapse in the performance. To further investigate the data and verify the results, I made a word cloud for the collection of reports labeled as `positive`, and a separate cloud for the `negative` class. It is particularly reassuring to see the giant "January Update" and "employee" in the negative cloud. Because given as ground truth, the key incident is the disappearance of several employees of GAStech, which happened in January 2014. Therefore, the sentiment classification seems to be reasonably successful. For phrases mentioned more frequently than others in the negative cloud, it would be helpful to go back to the articles for clues as to what is bad about them. And I believe it would also be fruitful to look into the names that appear in the positive cloud. The code and visualizations for this section can be found [here](https://github.com/comp-machine-learning-spring2021/portfolio-HelenaSG/blob/main/Text-Mining/Wordcloud-visualization.ipynb).


