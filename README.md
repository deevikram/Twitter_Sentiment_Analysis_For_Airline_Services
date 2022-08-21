# US Airline Sentiment: analysis and classification for social media strategies

#### Skills used in this project: Natural Language Processing (NLP) model, Vectorization (Count vectorizer and tf-idf vectorizer), Sentiment Analysis, Parameter tuning, Confusion matrix-based model evaluation

### Introduction
A key strategy to understand customers and discover new trends is to understand customer sentiment and opinions of the brand as well as the competitors. This can also help in ascertaining issues with the airline products and services and help improve customer service and expand customer base.

Background: Twitter is a widely used social media platform. It can and is also used by customers of airline companies to voice their experience while flying or using other airline company products. However, this information is buried under other information as the platform involves users interacting and posting information on a wide variety of topics. Thus sentiment analysis and classification of positive/negative tweets will be helpful in understanding customer trends.

According to the Twitter help center, a tweet is any message that is posted to the platform Twitter. It may contain photos, videos, links and text.

### Objective
- Build a natural language processing (NLP) model
- Sentiment analysis and classification of tweets for US airlines

### Data Information/Variables
- tweet_id - unique ID assigned to each tweet (composed of timestamp,sequence number, and worker number)
- airline_sentiment - categorical description of customer's opinion of the airline(positive, negative, neutral)
- airline_sentiment_confidence - score indicating confidence level in customer's sentiment. Values are between 0 and 1.
- negativereason - negative reasons given by the customer regarding the airline
- negativereason_confidence - score indicating confidence level in the negative reason. Values between 0 and 1.
- airline - name/brand of the airline
- airline_sentiment_gold - has 4 unique values (negative,positive,neutral and NaN)
- name - customer handle who tweeted about the airline
- negativereason_gold - contains negative reasons given by customers
- retweet_count - count of how many time the original tweet was reposted on Twitter
- text - description/text of the review of the airlines
- tweet_coord - latitude/longtitude coordinates of the location of the tweet
- tweet_created - the time that the tweet was created/posted
- tweet_location- city location data of the tweets
- user_timezone- timezone information (EST/PST etc)

### The analysis below has the following sections:
- Loading and importing packages
- Removing warnings from python notebooks
- Loading the dataset
- Preview of the dataset
- Descriptive statistics for the dataset
- Exploratory Data Analysis - distribution of tweets among airlines, distribution of sentiment across tweets, plots of negative reasons, word clouds for positive and negative sentiments
- Data preprocessing/model preparation - html tag removal, removal of numbers, removal of special characters, and punctuations, removal of stop words, conversion to lowercase, lemmatization/stemming, tokenization, joining the words to convert back to text, vectorization with countVectorizer and TfidfVectorizer
- NLP Model - building the model (random forest)
- NNLP model performance improvement - analysis of model performance, improvement of the model
- Summary and key takeaways - final conclusions and summary of the analysis

### EDA excerpts
![image](https://user-images.githubusercontent.com/83994337/164948996-8a9fb40a-d16c-41dd-b6bb-79f5bdda4e7a.png)
![image](https://user-images.githubusercontent.com/83994337/164949045-32421d78-6a7a-49ce-9cbe-025d89c4aac6.png)


### Insights and summary
- From the 14,640 tweet data, the majority of the posts had a negative sentiment in it (62.69% of the posts). This indicates a high customer dissatisfaction with the airline products and services.
- United airlines and US airways lead the most tweets that are having a negative sentiment attached to it. In general, negative sentiment reviews outnumber positive and neutral reviews by a large margin. This is not true just for airline industry but also for other industries. One underlying hypothesis could be that people take the effort to write write reviews if they have had a bad experience with the product or service, rather than taking the effort to write a neutral review or a positive review. When the service is exceptional, people might take the effort again to write a review.
- Most of the negative sentiment surrounding the use of airline services and products deal with customer service, delays, lost baggage etc (as seen in the negative sentiment wordcloud in the EDA section). While delays due to weather and other external events are unavoidable, the airlines can develop strategies to make delays or dealing with customer service an easier operation - for example, being proactive in reaching out to customers about issues.
- Overall, the preprocessing and vectorization helped to break down text data (14,640 tweets) into smaller components that enabled an NLP model to classify text. The tfidf model performed slightly better than the countVectorizer for NLP models. This can be attributed to the process where tfidfVectorizer puts emphasis on the more important words, as compared to countVectorizer where the basis of vectorization is based on the frequency of the word.
- There are a couple of strategies can be utilized to improve the model performance - we used a custom stopwords list for preprocessing, however the list can be improved. Techniques such as gridsearch, while being computationally expensive, can yield improvement in performance.



