# Text Mining
![detective](https://i.guim.co.uk/img/media/15ec744af8cdebb8e64e66948948a980b7dee359/0_37_2148_1289/master/2148.jpg?width=1020&quality=45&auto=format&fit=max&dpr=2&s=cf6dc98d30f5317c7c76f0f73735ea96)

> “My name is Sherlock Holmes. It is my business to know what other people do not know.” — Arthur Conan Doyle, _The Adventure of the Blue Carbuncle_
     
> “It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts.” — Arthur Conan Doyle, _A Scandal in Bohemia_

## :pencil: Introduction
When I think of a detective, the image that shows up in my mind is a man with a magnifying glass sitting in front of a table piled high with news articles. As I learn more about machine learning and data analysis, I feel fascinated by the idea of discovering patterns from facts and using this information to guide my decisions. This makes me feel like a detective: there is always something worth knowing about hidden behind the data, and the gems I end up discovering will never let me down. 

As technology advances, people's focus gradually shifts from how they collect data to how to process that data efficiently in real-time. With this in mind, I surf through vast amounts of data as fast as possible to bring insights to the stakeholders in time. However, I found text data to be more complicated. Unlike typical data frames that I used to work with plotting, text data can be written resources, chat room conversations, social media posts, etc. They are often unstructured and do not fit in a spreadsheet with rows and columns. Also, it would be impossible for me to read all relevant articles. Therefore, I started to look into the world of text mining. 

:small_blue_diamond: **An exploration on transfer learning-based sentiment analysis:** 

  * Understanding the background and framing the question
  * Conducting sentiment analysis at document level 
  * Classifying texts in a time-saving way: transfer learning
  * Populating a data frame with the information extracted
  * Creating data visualization to find insights in the results

## :pencil: Setting the stage

This project is motivated by [VAST Challenge 2021](https://vast-challenge.github.io/2021/MC1.html). The data provided consisted of a collection of current and historical news reports from multiple domestic and translated foreign sources, in text file format. Specifically, these files are dealing with the kidnapping of the GASTech employees by members of the social movement group POK. I wanted to characterize any biases in these news reports, concerning their representation of specific people, places, and events. As contextual mining of texts, sentiment analysis can identify and extract subjective information in source materials, which I think could be helpful. Therefore, I decided to conduct sentiment analysis on the news reports.

## :pencil: Sentiment analysis and transfer learning

Our goal is to have the news report data classified as “positive” or “negative” without having to read through them ourselves. Since there is not enough labeled data to train a reliable model for the task, the traditional supervised learning paradigm would not apply in this scenario. What we could do instead is that we could adopt a “semi-supervised” method to leverage a model that has been trained on a similar domain. This process would fall under what is called transfer learning. 


## :pencil: Data visualization
