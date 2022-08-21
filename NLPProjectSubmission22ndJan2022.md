# US Airline Sentiment: analysis and classification for social media strategies

## Introduction
<p> A key strategy to understand customers and discover new trends is to understand customer sentiment and opinions of the brand as well as the competitors. This can also help in ascertaining issues with the airline products and services and help improve customer service and expand customer base. </p>
<p> Background: Twitter is a widely used social media platform. It can and is also used by customers of airline companies to voice their experience while flying or using other airline company products. However, this information is buried under other information as the platform involves users interacting and posting information on a wide variety of topics. Thus sentiment analysis and classification of positive/negative tweets will be helpful in understanding customer trends. </p>
<p> According to the Twitter help center, a tweet is any message that is posted to the platform Twitter. It may contain photos, videos, links and text.</p>

### Objective
<ol>
    <li>Build a natural language processing (NLP) model</li>
    <li>Sentiment analysis and classification of tweets for US airlines</li>

### Data Information/Variables
<p> <strong>tweet_id</strong> - unique ID assigned to each tweet (composed of timestamp,sequence number, and worker number)<br>
    <strong>airline_sentiment</strong> - categorical description of customer's opinion of the airline(positive, negative, neutral)<br>
    <strong>airline_sentiment_confidence</strong> - score indicating confidence level in customer's sentiment. Values are between 0 and 1.<br>
    <strong>negativereason </strong> - negative reasons given by the customer regarding the airline<br>
    <strong>negativereason_confidence </strong> - score indicating confidence level in the negative reason. Values between 0 and 1.<br>
    <strong>airline</strong> - name/brand of the airline<br>
    <strong>airline_sentiment_gold</strong> - has 4 unique values (negative,positive,neutral and NaN) <br>
    <strong>name </strong> - customer handle who tweeted about the airline<br>
    <strong>negativereason_gold</strong> - contains negative reasons given by customers<br>
    <strong>retweet_count</strong> - count of how many time the original tweet was reposted on Twitter<br>
    <strong>text</strong> - description/text of the review of the airlines<br>
    <strong>tweet_coord</strong> - latitude/longtitude coordinates of the location of the tweet<br>
    <strong>tweet_created</strong> - the time that the tweet was created/posted<br>
    <strong>tweet_location</strong>- city location data of the tweets<br>
    <strong>user_timezone</strong>- timezone information (EST/PST etc)<br>
    </p>

### The analysis below has the following sections:
<ol>
    <li> Loading and importing packages </li>
    <li> Removing warnings from python notebooks </li>
    <li> Loading the dataset </li>
    <li> Preview of the dataset </li>
    <li> Descriptive statistics for the dataset </li>
    <li> Exploratory Data Analysis - distribution of tweets among airlines, distribution of sentiment across tweets, plots of negative reasons, word clouds for positive and negative sentiments </li>
    <li> Data preprocessing/model preparation - html tag removal, removal of numbers, removal of special characters, and punctuations, removal of stop words, conversion to lowercase, lemmatization/stemming, tokenization, joining the words to convert back to text, vectorization with countVectorizer and TfidfVectorizer</li>
    <li> <strong>NLP Model</strong> - building the model (random forest)</li>
    <li> <strong>NNLP model performance improvement </strong> - analysis of model performance, improvement of the model</li>
    <li> Summary and key takeaways - final conclusions and summary of the analysis </li>

## 1. Loading and importing packages


```python
# Import the necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean

sns.set(color_codes=True)  # For background of the graphs
%matplotlib inline

# For previewing the data, the columns can be set to limit of None and 100 for the rows.
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# For Regex, string and unicodedata preprocessing
import re, string, unicodedata 
# For preprocessing of contractions in text
import contractions   
# For text cleaning with in-built parsers
from bs4 import BeautifulSoup                          
# For analysis and processing of text data
import nltk                                             
# To remove stopwords
nltk.download('stopwords')
# Tokenizer
nltk.download('punkt')
# lexical database of words 
nltk.download('wordnet')

from nltk.corpus import stopwords                       # Import stopwords.
from nltk.tokenize import word_tokenize, sent_tokenize  # Import Tokenizer.
from nltk.stem.wordnet import WordNetLemmatizer         # Import Lemmatizer.
from wordcloud import WordCloud,STOPWORDS

#This is to reduce the variation in the results everytime the notebook is run
import random
random.seed(1)
np.random.seed(1)

# For preparing data for the NLP model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# For model building
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/deeptivikram/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to
    [nltk_data]     /Users/deeptivikram/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/deeptivikram/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!


### 2. Removing warnings from python notebook


```python
# Removing warnings from the notebook
import warnings

warnings.filterwarnings("ignore")
```

### 3. Loading the dataset


```python
# The dataset is stored in a CSV file and we want to read it into a pandas dataframe
tweetData = pd.read_csv("Tweets.csv",header=0)
```


```python
print(f"The dataframe has {tweetData.shape[0]} rows and {tweetData.shape[1]} columns.")
```

    The dataframe has 14640 rows and 15 columns.


<p> The dataset has 14,640 rows, so there are 14640 tweets about airline companies in the dataset. </p>

### 4. Previewing the dataset


```python
# Preview of 10 random rows of the dataset
# To see random 10 rows, numpy's random seed was used.
# Putting random.seed to be 1 (in section 1) will return the same random 10 rows everytime we execute the code.
pd.set_option('display.max_colwidth', None) #To facilitate printing of the full text column (no truncation)
tweetData.sample(n=10)
# The Reason we want to return random rows is so that we can see the typical values from a random sample.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>airline_sentiment</th>
      <th>airline_sentiment_confidence</th>
      <th>negativereason</th>
      <th>negativereason_confidence</th>
      <th>airline</th>
      <th>airline_sentiment_gold</th>
      <th>name</th>
      <th>negativereason_gold</th>
      <th>retweet_count</th>
      <th>text</th>
      <th>tweet_coord</th>
      <th>tweet_created</th>
      <th>tweet_location</th>
      <th>user_timezone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8515</th>
      <td>568198336651649027</td>
      <td>positive</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Delta</td>
      <td>NaN</td>
      <td>GenuineJack</td>
      <td>NaN</td>
      <td>0</td>
      <td>@JetBlue I'll pass along the advice. You guys rock!!</td>
      <td>NaN</td>
      <td>2015-02-18 16:00:14 -0800</td>
      <td>Massachusetts</td>
      <td>Central Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>3439</th>
      <td>568438094652956673</td>
      <td>negative</td>
      <td>0.7036</td>
      <td>Lost Luggage</td>
      <td>0.7036</td>
      <td>United</td>
      <td>NaN</td>
      <td>vina_love</td>
      <td>NaN</td>
      <td>0</td>
      <td>@united I sent you a dm with my file reference number.. I just want to know if someone has located my bag even if it's not here yet.</td>
      <td>NaN</td>
      <td>2015-02-19 07:52:57 -0800</td>
      <td>ny</td>
      <td>Quito</td>
    </tr>
    <tr>
      <th>6439</th>
      <td>567858373527470080</td>
      <td>positive</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Southwest</td>
      <td>NaN</td>
      <td>Capt_Smirk</td>
      <td>NaN</td>
      <td>0</td>
      <td>@SouthwestAir Black History Commercial is really sweet. Well done.</td>
      <td>NaN</td>
      <td>2015-02-17 17:29:21 -0800</td>
      <td>La Florida</td>
      <td>Eastern Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>5112</th>
      <td>569336871853170688</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Late Flight</td>
      <td>1.0000</td>
      <td>Southwest</td>
      <td>NaN</td>
      <td>scoobydoo9749</td>
      <td>NaN</td>
      <td>0</td>
      <td>@SouthwestAir why am I still in Baltimore?! @delta is doing laps around us and laughing about it. # ridiculous</td>
      <td>[39.1848041, -76.6787131]</td>
      <td>2015-02-21 19:24:22 -0800</td>
      <td>Tallahassee, FL</td>
      <td>America/Chicago</td>
    </tr>
    <tr>
      <th>5645</th>
      <td>568839199773732864</td>
      <td>positive</td>
      <td>0.6832</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Southwest</td>
      <td>NaN</td>
      <td>laurafall</td>
      <td>NaN</td>
      <td>0</td>
      <td>@SouthwestAir SEA to DEN. South Sound Volleyball team on its way! http://t.co/tN5cXCld6M</td>
      <td>NaN</td>
      <td>2015-02-20 10:26:48 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>1380</th>
      <td>569748884164988929</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Flight Attendant Complaints</td>
      <td>0.6818</td>
      <td>United</td>
      <td>NaN</td>
      <td>JacquieMae08</td>
      <td>NaN</td>
      <td>1</td>
      <td>@united One of your workers refused to give me her name as a reference for my notes. Her tone &amp;amp; language was very unprofessional.</td>
      <td>NaN</td>
      <td>2015-02-22 22:41:34 -0800</td>
      <td>SD || CA</td>
      <td>Arizona</td>
    </tr>
    <tr>
      <th>12674</th>
      <td>570066226233417728</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Bad Flight</td>
      <td>0.6966</td>
      <td>American</td>
      <td>NaN</td>
      <td>lczand</td>
      <td>NaN</td>
      <td>0</td>
      <td>@AmericanAir seats that were assigned are inappropriate for child this age. AA knew age of child.</td>
      <td>NaN</td>
      <td>2015-02-23 19:42:34 -0800</td>
      <td>SE USA</td>
      <td>Central Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>13475</th>
      <td>569853969469734912</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Customer Service Issue</td>
      <td>0.3642</td>
      <td>American</td>
      <td>NaN</td>
      <td>Davitossss</td>
      <td>NaN</td>
      <td>0</td>
      <td>@AmericanAir now you change my gate and don't tell me? What the fuck is wrong with you people. Learn to do your fucken job</td>
      <td>NaN</td>
      <td>2015-02-23 05:39:08 -0800</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9630</th>
      <td>569771270839013376</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Late Flight</td>
      <td>1.0000</td>
      <td>US Airways</td>
      <td>NaN</td>
      <td>jakenemmasmom</td>
      <td>NaN</td>
      <td>0</td>
      <td>@USAirways What a mess caused by the computer systems. Flight 719 in 3 hours Late Flight and now no gate for us. Est 26 min wait.</td>
      <td>NaN</td>
      <td>2015-02-23 00:10:31 -0800</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3745</th>
      <td>568157451729526784</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Customer Service Issue</td>
      <td>0.6803</td>
      <td>United</td>
      <td>NaN</td>
      <td>HeHaithMe</td>
      <td>NaN</td>
      <td>1</td>
      <td>@united How come you are the ONLY airline, of 90+ flights last year that makes me check my carry-on. Not even gate check...baggage claim?!?</td>
      <td>NaN</td>
      <td>2015-02-18 13:17:47 -0800</td>
      <td>Born/Raised in 314/Home is 317</td>
      <td>Eastern Time (US &amp; Canada)</td>
    </tr>
  </tbody>
</table>
</div>



#### Observation
<p> The random rows show that several columns have NaNs. The airline column shows the name of the airline companies. the airline_sentiment column has categorical values and will be the target variable for this analysis. the text column which provides the description of the tweet posted by the customer contains punctuation marks such as commas, periods. There are also contractions present (example - 'I'll') as well as special characters such as '@'. As evident from the preview of the data, significant amount of preprocessing will be needed before the data can be used for the NLP model. </p>


```python
tweetData.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 14640 entries, 0 to 14639
    Data columns (total 15 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   tweet_id                      14640 non-null  int64  
     1   airline_sentiment             14640 non-null  object 
     2   airline_sentiment_confidence  14640 non-null  float64
     3   negativereason                9178 non-null   object 
     4   negativereason_confidence     10522 non-null  float64
     5   airline                       14640 non-null  object 
     6   airline_sentiment_gold        40 non-null     object 
     7   name                          14640 non-null  object 
     8   negativereason_gold           32 non-null     object 
     9   retweet_count                 14640 non-null  int64  
     10  text                          14640 non-null  object 
     11  tweet_coord                   1019 non-null   object 
     12  tweet_created                 14640 non-null  object 
     13  tweet_location                9907 non-null   object 
     14  user_timezone                 9820 non-null   object 
    dtypes: float64(2), int64(2), object(11)
    memory usage: 1.7+ MB


#### Observation
<p>All columns in the data set are either int64, float64 or objects. There are total 14,640 rows in the data set so text and airline_sentiment columns have no missing values. </p>


```python
# Finding the sum total of missing values in each of the columns
tweetData.isnull().sum().sort_values(ascending=False)
```




    negativereason_gold             14608
    airline_sentiment_gold          14600
    tweet_coord                     13621
    negativereason                   5462
    user_timezone                    4820
    tweet_location                   4733
    negativereason_confidence        4118
    tweet_created                       0
    text                                0
    retweet_count                       0
    name                                0
    airline                             0
    airline_sentiment_confidence        0
    airline_sentiment                   0
    tweet_id                            0
    dtype: int64



#### Observation
<p>We can see that negativereason_gold column has mostly missing values (14608/14640 i.e. >99% of the data is missing). This is followed by airline_sentiment_gold and tweet_coord having the next highest number of missing values (>90%). negativereason, user_timezone,tweet_location,negativereason_confidence have less than 30% values missing. Our target variable airline_sentiment and the text column which will be used in the model have no missing data. </p>

### 5. Descriptive Summary Statistics for the dataset


```python
# Before starting preprocessing, a look at summary statistics
tweetData.describe().T  # taking transpose since it is easier to view
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tweet_id</th>
      <td>14640.0</td>
      <td>5.692184e+17</td>
      <td>7.791112e+14</td>
      <td>5.675883e+17</td>
      <td>5.685592e+17</td>
      <td>5.694779e+17</td>
      <td>5.698905e+17</td>
      <td>5.703106e+17</td>
    </tr>
    <tr>
      <th>airline_sentiment_confidence</th>
      <td>14640.0</td>
      <td>9.001689e-01</td>
      <td>1.628300e-01</td>
      <td>3.350000e-01</td>
      <td>6.923000e-01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>negativereason_confidence</th>
      <td>10522.0</td>
      <td>6.382983e-01</td>
      <td>3.304398e-01</td>
      <td>0.000000e+00</td>
      <td>3.606000e-01</td>
      <td>6.706000e-01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
    </tr>
    <tr>
      <th>retweet_count</th>
      <td>14640.0</td>
      <td>8.265027e-02</td>
      <td>7.457782e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.400000e+01</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations 
<ol>
    <li>The tweet_id is a unique identifier and will be dropped in the preprocessing section. </li>
    <li>The airline_sentiment_confidence variable is a continous variable that has values between 0 and 1. The mean was 0.9 (highly confident about the sentiment) while the median was 1 indicating an almost symmetric distribution.</li>
    <li>The negativereason_confidence is also a continous variable with values between 0 and 1. The mean was approximately 0.63 while median was 0.67, also indicating a symmetric distribution.</li>
    <li> The retweet count is a continous variable, and the mean was 0.82 times, versus a median of 0. The maximum was 44 retweets. </li> 
    </ol>


```python
#Summary statistics for the non-numerical columns
tweetData.describe(include=["object"]).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>airline_sentiment</th>
      <td>14640</td>
      <td>3</td>
      <td>negative</td>
      <td>9178</td>
    </tr>
    <tr>
      <th>negativereason</th>
      <td>9178</td>
      <td>10</td>
      <td>Customer Service Issue</td>
      <td>2910</td>
    </tr>
    <tr>
      <th>airline</th>
      <td>14640</td>
      <td>6</td>
      <td>United</td>
      <td>3822</td>
    </tr>
    <tr>
      <th>airline_sentiment_gold</th>
      <td>40</td>
      <td>3</td>
      <td>negative</td>
      <td>32</td>
    </tr>
    <tr>
      <th>name</th>
      <td>14640</td>
      <td>7701</td>
      <td>JetBlueNews</td>
      <td>63</td>
    </tr>
    <tr>
      <th>negativereason_gold</th>
      <td>32</td>
      <td>13</td>
      <td>Customer Service Issue</td>
      <td>12</td>
    </tr>
    <tr>
      <th>text</th>
      <td>14640</td>
      <td>14427</td>
      <td>@united thanks</td>
      <td>6</td>
    </tr>
    <tr>
      <th>tweet_coord</th>
      <td>1019</td>
      <td>832</td>
      <td>[0.0, 0.0]</td>
      <td>164</td>
    </tr>
    <tr>
      <th>tweet_created</th>
      <td>14640</td>
      <td>14247</td>
      <td>2015-02-24 09:54:34 -0800</td>
      <td>5</td>
    </tr>
    <tr>
      <th>tweet_location</th>
      <td>9907</td>
      <td>3081</td>
      <td>Boston, MA</td>
      <td>157</td>
    </tr>
    <tr>
      <th>user_timezone</th>
      <td>9820</td>
      <td>85</td>
      <td>Eastern Time (US &amp; Canada)</td>
      <td>3744</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations 
<ol>
    <li> The dataset contains large number of columns that have categorical data.</li>
    <li> The column airline_sentiment has no missing values, 3 unique values and negative was the most frequent word in the dataset.</li>
    <li> The column "negativereason" had 10 unique values with "Customer Service Issue" occuring most frequently.</li>
    <li> The column airline which contains names of airline companies had 6 unique values, so 6 different airline companies were mentioned by customers in their tweets. Out of this 'United' occured 3822 times. </li>
    <li> The column airline_sentiment_gold only had a count of 40 (out of 14,640 rows) so this column contains very little information. The 40 counts contained the word "negative" which occured 32 times. It is not known how the data for this column was collected, and it is possible that the data was duplicated with the negativereason column or with the airline_sentiment column. </li>
    <li> The column name is expected to have multiple unique values, as the data would be twitter handlers or names of the people posting on the social media platform. The unique count being only 7701 out of 14640 rows indicates that the same poster/person has tweeted multiple times on the topic of airlines.</li>
    <li> The column text has 14,427 unique values. This column contains the information of the text that makes up the tweet. The top frequency was "@united thanks" which occured 6 times. </li>
    <li> The tweet_coord is represented here as a categorical column but represents location co-ordinates. There are only 832 unique values, which indicates that multiple people tweet from the similar co-ords.</li>
    <li> The tweet_created column has a large number of unique values (14,247).</li>
    <li> The tweet_location column has 3081 unique cities out of which Boston, MA occured 157 times. </li>
    <li> The user_timezone is a column that has the timezone information and we can see that there are 85 unique timezones represented in the dataset, out of which Eastern Time (US and Canada) occured 3744 times. So a large number of tweeters/posters live in a geographic area where the timezone is EST. </li>


```python
# Descriptive statistics on target categorical variables
# Let us take a look at how many of the current customers had positive, negative or neutral reviews
sentimentCount = pd.crosstab(index=tweetData["airline_sentiment"], columns="count")
sentimentCount
negativesentimentCount = (sentimentCount.iloc[0] / (tweetData.shape[0])) * 100
negativesentimentCount
print(f"The percentage of customers who had a negative sentiment was {float(negativesentimentCount)}.")
```

    The percentage of customers who had a negative sentiment was 62.69125683060109.


#### Observations
<p> We can see from the data that the majority of the airline sentiment is negative where 62.69% of the data deals with a negative sentiment.</p>


```python
neutralsentimentCount = (sentimentCount.iloc[1] / (tweetData.shape[0])) * 100
neutralsentimentCount
print(f"The percentage of customers who had a neutral sentiment was {float(neutralsentimentCount)}.")
```

    The percentage of customers who had a neutral sentiment was 21.168032786885245.


#### Observations
<p> 21.16% of the posts had a neutral sentiment according to the data in the airline_sentiment column.</p>


```python
positivesentimentCount = (sentimentCount.iloc[2] / (tweetData.shape[0])) * 100
positivesentimentCount
print(f"The percentage of customers who had a positive sentiment was {float(positivesentimentCount)}.")
```

    The percentage of customers who had a positive sentiment was 16.140710382513664.


#### Observations
<p> 16.14% of the posts had a positive sentiment according to the data in the airline_sentiment column.</p>


```python
# To look at the number of unique values in each column:
tweetData.nunique()
```




    tweet_id                        14485
    airline_sentiment                   3
    airline_sentiment_confidence     1023
    negativereason                     10
    negativereason_confidence        1410
    airline                             6
    airline_sentiment_gold              3
    name                             7701
    negativereason_gold                13
    retweet_count                      18
    text                            14427
    tweet_coord                       832
    tweet_created                   14247
    tweet_location                   3081
    user_timezone                      85
    dtype: int64



### 6. Exploratory Data Analysis - univariate, bivariate and multivariate analysis

<p> The section will have the following subsections: 
    <li> distribution of tweets among airlines </li>
    <li> distribution of sentiment across tweets </li>
    <li> plots of negative reasons </li>
    <li> word clouds for positive and negative sentiments </li>


```python
# Copying the dataframe again before we use it for EDA
tweetData1 = tweetData.copy()
```


```python
# Visualizing missing values using heatmaps
plt.figure(figsize=(12,5))
sns.heatmap(tweetData.isnull(), cmap = "Reds")                       
plt.title("Missing values?", fontsize = 10)
plt.show()
```


    
![png](output_34_0.png)
    


#### Observations
<p> We are utilizing a heatmap to visualize the missing values in the dataset. The darker the red color and close to each other, the more missing values those columns have. As we can see above, the airline_sentiment_gold and negativereason_gold are almost full of missing values. Thus, these two columns will be dropped for preprocessing. The tweet_coord column also has a lot of missing values, so that will also be dropped for preprocessing.</p>


```python
# Pie plot of distribution of positive, negative and neutral tweets

colors = ['#ff8863', '#ffcc99', '#99ff99']

sns.set(rc={'figure.figsize':(11.7,8.27)})
plot = plt.pie(tweetData1['airline_sentiment'].value_counts(), labels=tweetData1['airline_sentiment'].value_counts().index, colors=colors, startangle=90,  autopct='%.2f')
fig = plt.gcf()
plt.title('Pie plot for Twitter sentiment labels')
plt.axis('equal')
plt.tight_layout()
plt.show()
```


    
![png](output_36_0.png)
    


#### Observation
<p> As we can see above, the overall sentiment is tending towards being negative for the customer experience for airline companies products and services. </p>


```python
# Univariate and bivariate analysis
# We can start with analysis of 'airline_sentiment_confidence' - histogram 
# And box plots of 'airline_sentiment' versus 'retweet_count' and 'airline_sentiment_confidence'
# And also the box plot of 'airline_sentiment_confidence' versus the brand airline

# Display histogram and boxplots
# Make a grid of 2 rows and 2 columns
fig, axs = plt.subplots( nrows = 2, ncols = 2,figsize=(15, 8))
# Put a figure title
fig.suptitle("airline_sentiment_confidence and airline_sentiment analysis", fontsize=30)
# Histogram of airline_sentiment_confidence variable
sns.histplot(ax=axs[0, 0], data=tweetData1, x="airline_sentiment_confidence", kde=True)
# Box plot for airline_sentiment_confidence and airline_sentiment
sns.boxplot(
    ax=axs[0, 1],
    data=tweetData1,
    x=tweetData1["airline_sentiment"],
    y=tweetData1["airline_sentiment_confidence"],
)


sns.boxplot(
    ax=axs[1, 0],
    data=tweetData1,
    x=tweetData1["airline"],
    y=tweetData1["airline_sentiment_confidence"],
)

sns.boxplot(
    ax=axs[1, 1],
    data=tweetData1,
    x=tweetData1["airline_sentiment"],
    y=tweetData1["retweet_count"],
)

# We put semicolons to supress the axis output
```




    <AxesSubplot:xlabel='airline_sentiment', ylabel='retweet_count'>




    
![png](output_38_1.png)
    


#### Observations
<p> In the histogram of the airline_sentiment_confidence, we can see that the count is higher for airline_sentiment_confidence scores of 1. So overall, majority of posters are confident about their evaluation of opinion of the airline and its products/services. For the box plot of airline_sentiment_confidence versus airline_sentiment, we can see that there are whiskers only for one part of the box plots. This indicates that the data is skewed (having a long tail). There are also outliers present for the negative sentiments. For the box plot of airline company versus the airline_sentiment_confidence, we can see that the box plots have only one whisker (again indicating a skewed distribution). There are also outliers present for US Airways and American. </p>


```python
# Analysis between category type variables
sns.countplot(x="airline_sentiment", hue="airline", data=tweetData1)
```




    <AxesSubplot:xlabel='airline_sentiment', ylabel='count'>




    
![png](output_40_1.png)
    


#### Observations
<p> From the above graph, we can see that United airlines has the highest negative sentiments in customer reviews and it is followed by US Airways and American. When it comes to positive sentiments, Southwest, Delta and United were highest and close to each other in number of positive reviews. For Virgin America, the distribution of tweets for neutral, positive and negative was roughly the same. </p>


```python
# Distribution of tweets across airline companies
sns.countplot(x="airline",data=tweetData1)
```




    <AxesSubplot:xlabel='airline', ylabel='count'>




    
![png](output_42_1.png)
    


#### Observations
<p> The distribution of tweets across airlines is shown above. As we can see,people tweeted most about United airlines (generating more than 3500 tweets in the time period of the dataset). This was followed by US airways, and then American airlines. The number of tweets were lowest for Virgin America. </p>


```python
# Distribution of tweets across sentiment
sns.countplot(x="airline_sentiment",data=tweetData1)
```




    <AxesSubplot:xlabel='airline_sentiment', ylabel='count'>




    
![png](output_44_1.png)
    


#### Observations
<p> The maximum number of tweets were negative with respect to airline sentiment (> 8000). The least amount of tweets were for positive reviews. This indicates that people tend to tweet when they have a negative experience of the airline service or product and less likely to tweet if they have had a positive experience. Some of this can be analyzed on the basis that the tweets may be directed to the customer service departments of those airlines, and people tend to write to customer service department when something is amiss or incorrect. These tweets will also get classified as negative, because customer service departments would deal with lost baggage, delays, or other events where the airline did not meet the customer's expectation. </p>


```python
# Distribution of tweets across negative reasons
plt.figure(figsize=(20,10))
sns.countplot(x="negativereason",data=tweetData1)
plt.show()
#sns.catplot(x="negativereason", kind="count", palette="ch:.25", data=tweetData1)
```


    
![png](output_46_0.png)
    


#### Observations
<p> When we plot the negative reasons, we can see that the highest count was for customer service issues (>2500). The next highest count was for a late flight. The third category is "Can't Tell" which is ambiguous but it is possible that the data collection/interpretation process put every tweet that didn't match a defined column such as "Booking problems", "Lost Luggage" into a category called "Can't Tell". Overall, we can see that customers tweeted about customer service issues, delays, booking problems, lost luggaes, canceled flights, longlines, flight attendent complaints as well as the experience of a bad flight. </p>


```python
#Word cloud for negative sentiments
negative_tweets=tweetData1[tweetData1['airline_sentiment']=='negative']
words = ' '.join(negative_tweets['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
```


```python
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='red',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)
```


```python
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
```


    
![png](output_50_0.png)
    


#### Observations
<p> In the word cloud for negative sentiment, we can see the most frequent words that occur in the tweets. "flight" occurs most frequently (since the tweets are about airlines, the word "flight" would be occuring frequently). The next most frequent words are "help", "plane", "hour", "time", "now", "hold", "bag", "customer service". As we saw in the EDA, most of the negative reasons had customer service and delays as the most frequently occuring issues. We can also see that some of the issues like "reservation", "trying", "need", "stuck", luggage", "lost" and "delayed" occur in the word cloud. The word cloud gives a quick visual representation of the issues faced by customers when using the airlines and their services and products.</p>
<p> The airline names were removed from the tweets to form the word cloud to get a good gauge of the issues.</p>


```python
#Word cloud for positive sentiments
positive_tweets=tweetData1[tweetData1['airline_sentiment']=='positive']
words1 = ' '.join(negative_tweets['text'])
cleaned_word1 = " ".join([word for word in words1.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
```


```python
wordcloud1 = WordCloud(stopwords=STOPWORDS,
                      background_color='blue',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word1)
```


```python
plt.figure(1,figsize=(12, 12))
plt.imshow(wordcloud1)
plt.axis('off')
plt.show()
```


    
![png](output_54_0.png)
    


#### Observation
<p> For positive sentiments, the words "flight" occur most frequently. The next set of most frequently occuring words are "hour", "hold", "now", "bag", "help", "plane" and "customer service". So there is some overlap between the most frequently occuring words between the tweets that are mapped to negative sentiment and tweets that are mapped to positive sentiment. </p>

### 7. Data pre-processing section

<p> In this section, the following will be tackled:
    <li> html tag removal </li>
    <li> removal of numbers </li>
    <li> removal of special characters and punctuations </li>
    <li> removal of stop words </li>
    <li> conversion to lowercase </li>
    <li> lemmatization/stemming </li>
    <li> tokenization </li>
    <li> joining the words to convert back to text </li>
    <li> vectorization with countVectorizer and TfidfVectorizer </li>


```python
# Copying the dataframe again before we use it for preprocessing
tweetData2 = tweetData1.copy()
```


```python
#Print one sample of text before preprocessing
print (tweetData2["text"][10154])
```

    @USAirways you keep suggesting people call 800-428-4322 but we just end up on hold for hours. You seriously don't have a better method?!


#### Observations
<p> We can see that the tweet text has numbers, special characters such as "@", periods,contractions, and other punctuation marks. </p>


```python
#This section deals with the following:
#removal of html tags, removing contractions (such as "I'm"), removing numbers as they will not provide 
# meaningful information for the model, removing URLs, and removing "@"

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")                    
    return soup.get_text()

#expand the contractions
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

#remove the numericals present in the text
def remove_numbers(text):
  text = re.sub(r'\d+', '', text)
  return text

# remove the url's present in the text
def remove_url(text): 
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)
    return text

# remove the mentions in the tweets
def remove_mention(text):
    text = re.sub(r'@\w+','',text)
    return text

def clean_text(text):
    text = strip_html(text)
    text = replace_contractions(text)
    text = remove_numbers(text)
    text = remove_url(text)
    text = remove_mention(text)
    return text

tweetData2['text'] = tweetData2['text'].apply(lambda x: clean_text(x))
tweetData2.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>airline_sentiment</th>
      <th>airline_sentiment_confidence</th>
      <th>negativereason</th>
      <th>negativereason_confidence</th>
      <th>airline</th>
      <th>airline_sentiment_gold</th>
      <th>name</th>
      <th>negativereason_gold</th>
      <th>retweet_count</th>
      <th>text</th>
      <th>tweet_coord</th>
      <th>tweet_created</th>
      <th>tweet_location</th>
      <th>user_timezone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>570306133677760513</td>
      <td>neutral</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>cairdin</td>
      <td>NaN</td>
      <td>0</td>
      <td>What  said.</td>
      <td>NaN</td>
      <td>2015-02-24 11:35:52 -0800</td>
      <td>NaN</td>
      <td>Eastern Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>570301130888122368</td>
      <td>positive</td>
      <td>0.3486</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>plus you have added commercials to the experience... tacky.</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:59 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>570301083672813571</td>
      <td>neutral</td>
      <td>0.6837</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>yvonnalynn</td>
      <td>NaN</td>
      <td>0</td>
      <td>I did not today... Must mean I need to take another trip!</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:48 -0800</td>
      <td>Lets Play</td>
      <td>Central Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>570301031407624196</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Bad Flight</td>
      <td>0.7033</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>it is really aggressive to blast obnoxious "entertainment" in your guests' faces &amp; they have little recourse</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:36 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570300817074462722</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Can't Tell</td>
      <td>1.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>and it is a really big bad thing about it</td>
      <td>NaN</td>
      <td>2015-02-24 11:14:45 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Print one sample of text after cleaning up the text
print (tweetData2["text"][10154])
# Before preprocessing, the output was:
#@USAirways you keep suggesting people call 800-428-4322 but we just end up on hold for hours. You seriously don't have a better method?!
```

     you keep suggesting people call -- but we just end up on hold for hours. You seriously do not have a better method?!


#### Observations
<p> As we can see, we have successfully removed html tags, contractions, numbers and tweet mentions. </p>


```python
#We will tokenize the words in the text column
tweetData2['text'] = tweetData2.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
tweetData2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>airline_sentiment</th>
      <th>airline_sentiment_confidence</th>
      <th>negativereason</th>
      <th>negativereason_confidence</th>
      <th>airline</th>
      <th>airline_sentiment_gold</th>
      <th>name</th>
      <th>negativereason_gold</th>
      <th>retweet_count</th>
      <th>text</th>
      <th>tweet_coord</th>
      <th>tweet_created</th>
      <th>tweet_location</th>
      <th>user_timezone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>570306133677760513</td>
      <td>neutral</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>cairdin</td>
      <td>NaN</td>
      <td>0</td>
      <td>[What, said, .]</td>
      <td>NaN</td>
      <td>2015-02-24 11:35:52 -0800</td>
      <td>NaN</td>
      <td>Eastern Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>570301130888122368</td>
      <td>positive</td>
      <td>0.3486</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>[plus, you, have, added, commercials, to, the, experience, ..., tacky, .]</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:59 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>570301083672813571</td>
      <td>neutral</td>
      <td>0.6837</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>yvonnalynn</td>
      <td>NaN</td>
      <td>0</td>
      <td>[I, did, not, today, ..., Must, mean, I, need, to, take, another, trip, !]</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:48 -0800</td>
      <td>Lets Play</td>
      <td>Central Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>570301031407624196</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Bad Flight</td>
      <td>0.7033</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>[it, is, really, aggressive, to, blast, obnoxious, ``, entertainment, '', in, your, guests, ', faces, &amp;, they, have, little, recourse]</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:36 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570300817074462722</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Can't Tell</td>
      <td>1.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>[and, it, is, a, really, big, bad, thing, about, it]</td>
      <td>NaN</td>
      <td>2015-02-24 11:14:45 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations
<p> Tokenization splits the text in the "text" column of the dataset into tokens (smaller units). We have used the NLTK library here. This will help further preprocessing of the data. </p>


```python
#Print one sample of text after cleaning up the text
print (tweetData2["text"][10154])
# Before preprocessing, the output was:
#@USAirways you keep suggesting people call 800-428-4322 but we just end up on hold for hours. You seriously don't have a better method?!
```

    ['you', 'keep', 'suggesting', 'people', 'call', '--', 'but', 'we', 'just', 'end', 'up', 'on', 'hold', 'for', 'hours', '.', 'You', 'seriously', 'do', 'not', 'have', 'a', 'better', 'method', '?', '!']


#### Observations
<p> We can see the text in above example being tokenized and that can be used for further preprocessing such as removal
of special characters and removal of stop words. </p>


```python
# We will customize the stopwords for removal because some words like "not" matter in determining sentiment
stopwords = stopwords.words('english')

customlist = ['not', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
        "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

stopwords = list(set(stopwords) - set(customlist)) 
```


```python
#In this section, we will tackle removal of special characters,
#removal of stop words,conversion to lowercase, lemmatization and stemming
#and removal of non-ascii characters as well as removal of punctuation
lemmatizer = WordNetLemmatizer()

#remove the non-ASCII characters
def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

# convert all characters to lowercase
def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words


# Remove the punctuations
def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

# Remove the stop words
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    return new_words

# lemmatize the words
def lemmatize_list(words):
    new_words = []
    for word in words:
        new_words.append(lemmatizer.lemmatize(word, pos='v'))
    return new_words

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    words = lemmatize_list(words)
    return ' '.join(words)

tweetData2['text'] = tweetData2.apply(lambda row: normalize(row['text']), axis=1)
```


```python
tweetData2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>airline_sentiment</th>
      <th>airline_sentiment_confidence</th>
      <th>negativereason</th>
      <th>negativereason_confidence</th>
      <th>airline</th>
      <th>airline_sentiment_gold</th>
      <th>name</th>
      <th>negativereason_gold</th>
      <th>retweet_count</th>
      <th>text</th>
      <th>tweet_coord</th>
      <th>tweet_created</th>
      <th>tweet_location</th>
      <th>user_timezone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>570306133677760513</td>
      <td>neutral</td>
      <td>1.0000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>cairdin</td>
      <td>NaN</td>
      <td>0</td>
      <td>say</td>
      <td>NaN</td>
      <td>2015-02-24 11:35:52 -0800</td>
      <td>NaN</td>
      <td>Eastern Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>570301130888122368</td>
      <td>positive</td>
      <td>0.3486</td>
      <td>NaN</td>
      <td>0.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>plus add commercials experience tacky</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:59 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>570301083672813571</td>
      <td>neutral</td>
      <td>0.6837</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>yvonnalynn</td>
      <td>NaN</td>
      <td>0</td>
      <td>not today must mean need take another trip</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:48 -0800</td>
      <td>Lets Play</td>
      <td>Central Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>570301031407624196</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Bad Flight</td>
      <td>0.7033</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>really aggressive blast obnoxious entertainment guests face little recourse</td>
      <td>NaN</td>
      <td>2015-02-24 11:15:36 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>570300817074462722</td>
      <td>negative</td>
      <td>1.0000</td>
      <td>Can't Tell</td>
      <td>1.0000</td>
      <td>Virgin America</td>
      <td>NaN</td>
      <td>jnardino</td>
      <td>NaN</td>
      <td>0</td>
      <td>really big bad thing</td>
      <td>NaN</td>
      <td>2015-02-24 11:14:45 -0800</td>
      <td>NaN</td>
      <td>Pacific Time (US &amp; Canada)</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Print one sample of text after cleaning up the text
print (tweetData2["text"][10154])
# Before preprocessing, the output was:
#@USAirways you keep suggesting people call 800-428-4322 but we just end up on hold for hours. You seriously don't have a better method?!
```

    keep suggest people call end hold hours seriously not better method


#### Observations
<p> As we can see, after further preprocessing we have removed stop words, punctuation marks, kept words in its dictionary or canonical form (lemmatization), and also converted all text to lowercase. </p>


```python
#Dropping columns
tweetData2.drop("tweet_id", axis=1, inplace=True)
tweetData2.drop("airline_sentiment_confidence", axis=1, inplace=True)
tweetData2.drop("negativereason", axis=1, inplace=True)
tweetData2.drop("negativereason_confidence", axis=1, inplace=True)
tweetData2.drop("airline", axis=1, inplace=True)
tweetData2.drop("airline_sentiment_gold", axis=1, inplace=True)
tweetData2.drop("name", axis=1, inplace=True)
tweetData2.drop("negativereason_gold", axis=1, inplace=True)
tweetData2.drop("retweet_count", axis=1, inplace=True)
tweetData2.drop("tweet_coord", axis=1, inplace=True)
tweetData2.drop("tweet_created", axis=1, inplace=True)
tweetData2.drop("tweet_location", axis=1, inplace=True)
tweetData2.drop("user_timezone", axis=1, inplace=True)
```

#### Observations
<p> We will be dropping the following columns: 
    <li> tweet_id: this column provides no meaningful data for a model which has to learn to classify positive versus negative sentiment </li>
    <li> airline_sentiment_confidence and negativereason_confidence: these columns provide scores about the confidence level of the airline_sentiment data and negativereason data. However, since we are not sure of the data collection aspect, or how these confidence levels were derived, we can drop these columns for the model. </li>
    <li> negativereason - this information is already covered in the text of the tweet, hence we will drop this column </li>
    <li> retweet_count - this has no meaningful contribution in trying to understand whether a certain text is positive or negative, hence this column will be dropped </li>
    <li> name, airline - the name of the airline company is not meaningful information to determine if a customer's tweet is negative or positive, hence this column will be dropped too. Neither is the poster's handle or name provide meaningful information.</li>
    <li> tweet_coord, tweet_location, user_timezone, tweet_created: these columns are unlikely to provide useful information about predicting whether a tweet will be positive or negative, hence will be dropped as well. </li>
    


```python
# print 5 rows of data
tweetData2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline_sentiment</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>neutral</td>
      <td>say</td>
    </tr>
    <tr>
      <th>1</th>
      <td>positive</td>
      <td>plus add commercials experience tacky</td>
    </tr>
    <tr>
      <th>2</th>
      <td>neutral</td>
      <td>not today must mean need take another trip</td>
    </tr>
    <tr>
      <th>3</th>
      <td>negative</td>
      <td>really aggressive blast obnoxious entertainment guests face little recourse</td>
    </tr>
    <tr>
      <th>4</th>
      <td>negative</td>
      <td>really big bad thing</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Preparing data for the model using countVectorizer
# Vectorization (Convert text data to numbers). The countVectorizer does not treat 
# some words as more important, but transforms text to the vector format based on frequency
# max_features: this will build a vocabulary that only considers the top max_features ordered by 
# term frequency across the corpus.

# increasing the max_features will increase the processing time
# For this iteration, we will choose max_features as 30,000

bow_vec = CountVectorizer(max_features=30000)               
data_features = bow_vec.fit_transform(tweetData2['text'])

data_features = data_features.toarray()
```


```python
data_features.shape
```




    (14640, 10013)



#### Observations
<p> Using the count vectorizer, which is based on the frequency of the words that occur in the text, we arrive at the data_features which has 14640 rows and 10,013 columns. </p>


```python
# Using TfidfVectorizer to convert text data to numbers.
# The TfidfVectorizer is based on term frequency times inverse document frequency
# This tries to capture how important a word is and how many documents have that word

vectorizer = TfidfVectorizer(max_features=30000)
data_featurestf = vectorizer.fit_transform(tweetData2['text'])

data_featurestf = data_featurestf.toarray()

data_featurestf.shape
```




    (14640, 10013)




```python
# checking the countVectorizer unique values
np.unique(data_features,return_counts=True)
```




    (array([0, 1, 2, 3, 4, 5, 6]),
     array([146466737,    119419,      3909,       233,        20,         1,
                    1]))




```python
# checking the tfidfVectorizer unique values
np.unique(data_featurestf,return_counts=True)
```




    (array([0.        , 0.06020025, 0.0649467 , ..., 0.96395479, 0.97368132,
            1.        ]),
     array([146466737,         1,         1, ...,         1,         1,
                  432]))



#### Observations
<p> We can see the countVectorizer and tfidfVectorizer yield different formats (int versus float). The tfidfVectorizer will capture how important a word is, compared to countVectorizer which will capture how many times a word will occur among the documents.We will use both the tfidfVectorizer and the countVectorizer for the NLP and compare their performances. </p>

<p> This concludes the preprocessing of the dataset. </p>

### 8. Building  the NLP model

<p> There are a couple of options for building the natural language processing model. The model will have to use a multi-classifer (since we have to predict whether the airline sentiment is positive, negative or neutral so 3 classes) for a categorical prediction problem. We will use a random forest with cross validation as the NLP model here.</p> 


```python
# Copying the dataframe again before we build the NLP model
tweetData3 = tweetData2.copy()
```


```python
tweetData3.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>airline_sentiment</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>neutral</td>
      <td>say</td>
    </tr>
    <tr>
      <th>1</th>
      <td>positive</td>
      <td>plus add commercials experience tacky</td>
    </tr>
    <tr>
      <th>2</th>
      <td>neutral</td>
      <td>not today must mean need take another trip</td>
    </tr>
    <tr>
      <th>3</th>
      <td>negative</td>
      <td>really aggressive blast obnoxious entertainment guests face little recourse</td>
    </tr>
    <tr>
      <th>4</th>
      <td>negative</td>
      <td>really big bad thing</td>
    </tr>
    <tr>
      <th>5</th>
      <td>negative</td>
      <td>seriously would pay flight seat not play really bad thing fly va</td>
    </tr>
    <tr>
      <th>6</th>
      <td>positive</td>
      <td>yes nearly every time fly vx ear worm not go away</td>
    </tr>
    <tr>
      <th>7</th>
      <td>neutral</td>
      <td>really miss prime opportunity men without hat parody</td>
    </tr>
    <tr>
      <th>8</th>
      <td>positive</td>
      <td>well notbut</td>
    </tr>
    <tr>
      <th>9</th>
      <td>positive</td>
      <td>amaze arrive hour early good</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating the y column for the model 
labels = tweetData3['airline_sentiment']
# We will use data_featurestf for X_train and X_test
labels = labels.replace({"positive":1,"neutral":2,"negative":3})
```

<p> We will build the model with tfidf feature set first </p>


```python
labels.value_counts()
```




    3    9178
    2    3099
    1    2363
    Name: airline_sentiment, dtype: int64




```python
# Split data into training and testing set (30% will be used for testing)
X_train, X_test, y_train, y_test = train_test_split(data_featurestf, labels, test_size=0.3, random_state=42)
```


```python
print("Shape of Training set : ", X_train.shape)
print("Shape of Testing set : ", X_test.shape)
print("Percentage of classes in training set:")
print(y_train.value_counts(normalize=True))
print("Percentage of classes in test set:")
print(y_test.value_counts(normalize=True))
```

    Shape of Training set :  (10248, 10013)
    Shape of Testing set :  (4392, 10013)
    Percentage of classes in training set:
    3    0.620999
    2    0.216140
    1    0.162861
    Name: airline_sentiment, dtype: float64
    Percentage of classes in test set:
    3    0.640710
    2    0.201275
    1    0.158015
    Name: airline_sentiment, dtype: float64


#### Observations
<p> 30% of the dataset is used for testing. The distribution of classes between training and test set are similar. </p>


```python
# Using Random Forest to build model for the classification of sentiment tweets. 
# Also calculating the cross validation score.
# n_jobs is number of jobs to run in parallel
# n_estimators are the number of trees in the forest
forest = RandomForestClassifier(n_estimators=20, n_jobs=10)
forest = forest.fit(X_train, y_train)
print(forest)
print(mean(cross_val_score(forest, data_featurestf, labels, cv=10)))
```

    RandomForestClassifier(n_estimators=20, n_jobs=10)
    0.753483606557377


#### Observation 
<p> The score using cross-validation here is 75.35% where the number of folds in stratified kfold was 10. </p>


```python
# Finding optimal number of base learners using k-fold CV. Here, we will set up base learners to be 25 and will
# evaluate the graph to determine what is the best number for base learners. 
base_ln = [x for x in range(1, 25)]
base_ln
```




    [1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24]




```python
# K-Fold Cross - validation 
cv_scores = []
for b in base_ln:
    clf = RandomForestClassifier(n_estimators = b,n_jobs=10,warm_start=True)
    scores = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
```


```python
# plotting the error as k increases
error = [1 - x for x in cv_scores]                                 #error corresponds to each nu of estimator
optimal_learners = base_ln[error.index(min(error))]                #Selection of optimal nu of n_estimator corresponds to minimum error.
plt.plot(base_ln, error)                                           #Plot between each nu of estimator and misclassification error
xy = (optimal_learners, min(error))
plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
plt.xlabel("Number of base learners")
plt.ylabel("Misclassification Error")
plt.show()
```


    
![png](output_98_0.png)
    


#### Observations
<p> As we can see from the graph above, as the number of base learners increase, the misclassification error goes down and reaches its lowest point at 19 base learners (i.e. 19 trees). Thus, optimal learners will take the value 19 and we will calculate the score on the test set. </p>



```python
# Training the best model and calculating accuracy on test data .
clf = RandomForestClassifier(n_estimators = optimal_learners,warm_start=True)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
tf_idf_predicted = clf.predict(X_test)
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
print(classification_report(y_test , tf_idf_predicted , target_names = ['1' , '2','3']))
print("Accuracy of the model is : ",accuracy_score(y_test,tf_idf_predicted))
```

                  precision    recall  f1-score   support
    
               1       0.78      0.57      0.66       694
               2       0.63      0.48      0.54       884
               3       0.81      0.92      0.86      2814
    
        accuracy                           0.78      4392
       macro avg       0.74      0.66      0.69      4392
    weighted avg       0.77      0.78      0.77      4392
    
    Accuracy of the model is :  0.7789162112932605


#### Observations
<p> The accuracy on the test set is 77.89% i.e. 77.89% of the texts were classified with the correct label. </p>


```python
result =  clf.predict(X_test) 
```


```python
# Print and plot Confusion matirx to get an idea of how the distribution of the prediction is, among all the classes.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, result)

print(conf_mat)

print(metrics.f1_score(y_test, result,average='micro'))

df_cm = pd.DataFrame(conf_mat, index = [i for i in "123"],
                  columns = [i for i in "123"])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
```

    [[ 399   86  209]
     [  59  423  402]
     [  52  163 2599]]
    0.7789162112932606



    
![png](output_103_1.png)
    


#### Observations
<p> The accuracy metric here can be interpreted to be: the count of texts that were correctly identified as positive, negative or neutral. </p>
<p> Here, in the confusion matrix, for the first label "1" (which would correspond to "positive"), there were a total of 694 items (399+86+209).So approximately 57% of items belonging to class 1 were identified correctly. The model that used tfidfVectorizer marked 399 of those correctly as label "1", but put 86 of those in label "2" and 209 of those in label "3". Similarly, for label "2", the model identified 423 of those items correctly, but misclassified 59 as label "1" and 402 as label "2". For the third class which is the "negative" sentiment, 2599 items were correctly labeled as "3" out of 2814 items (92.3%). </p>


```python
all_features = vectorizer.get_feature_names()              #Instantiate the feature from the vectorizer (tfidf)
top_features=''                                            # Addition of top 40 feature into top_feature after training the model
feat=clf.feature_importances_
features=np.argsort(feat)[::-1]
for i in features[0:40]:
    top_features+=all_features[i]
    top_features+=' '
    
    

from wordcloud import WordCloud
wordcloud = WordCloud(background_color="white",colormap='viridis',width=2000, 
                          height=1000).generate(top_features)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.figure(1, figsize=(14, 11), frameon='equal')
plt.title('Top 40 features WordCloud', fontsize=20)
plt.axis("off")
plt.show()
```


    
![png](output_105_0.png)
    


#### Observations
<p> We can see that the words "hour", "thank", "service", "delay", "flight", "love" stand out in the word cloud. </p>

<p> We will now build the model with the feature set from countVectorizer </p>


```python
# Split data into training and testing set.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.3, random_state=42)
```


```python
# Using Random Forest to build model for the classification of reviews.
# Also calculating the cross validation score.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

forest = RandomForestClassifier(n_estimators=20, n_jobs=10)

forest = forest.fit(X_train, y_train)

print(forest)

print(np.mean(cross_val_score(forest, data_features, labels, cv=10)))
```

    RandomForestClassifier(n_estimators=20, n_jobs=10)
    0.7523907103825136


#### Observation 
<p> The score using cross-validation here is 75.23% where the number of folds in stratified kfold was 10. </p>


```python
# Finding optimal number of base learners using k-fold CV. Here, we will set up base learners to be 25 and will
# evaluate the graph to determine what is the best number for base learners. 
base_ln = [x for x in range(1, 25)]
base_ln
```




    [1,
     2,
     3,
     4,
     5,
     6,
     7,
     8,
     9,
     10,
     11,
     12,
     13,
     14,
     15,
     16,
     17,
     18,
     19,
     20,
     21,
     22,
     23,
     24]




```python
# K-Fold Cross - validation .
cv_scores = []
for b in base_ln:
    clf1 = RandomForestClassifier(n_estimators = b,n_jobs=10)
    scores = cross_val_score(clf1, X_train, y_train, cv = 10, scoring = 'accuracy')
    cv_scores.append(scores.mean())
```


```python
# plotting the error as k increases
error = [1 - x for x in cv_scores]                                 #error corresponds to each nu of estimator
optimal_learners = base_ln[error.index(min(error))]                #Selection of optimal nu of n_estimator corresponds to minimum error.
plt.plot(base_ln, error)                                           #Plot between each nu of estimator and misclassification error
xy = (optimal_learners, min(error))
plt.annotate('(%s, %s)' % xy, xy = xy, textcoords='data')
plt.xlabel("Number of base learners")
plt.ylabel("Misclassification Error")
plt.show()
```


    
![png](output_113_0.png)
    


#### Observations
<p> As we can see from the graph above, as the number of base learners increase, the misclassification error goes down and reaches its lowest point at 21 base learners (i.e. 21 trees). Thus, optimal learners will take the value 21 and we will calculate the score on the test set. </p>


```python
# Training the best model and calculating accuracy on test data .
clf1 = RandomForestClassifier(n_estimators = optimal_learners)
clf1.fit(X_train, y_train)
clf1.score(X_test, y_test)
count_vectorizer_predicted = clf1.predict(X_test)
print(classification_report(y_test , count_vectorizer_predicted , target_names = ['1' , '2','3']))
print("Accuracy of the model is : ",accuracy_score(y_test,count_vectorizer_predicted))
```

                  precision    recall  f1-score   support
    
               1       0.74      0.62      0.67       694
               2       0.60      0.53      0.56       884
               3       0.83      0.89      0.86      2814
    
        accuracy                           0.78      4392
       macro avg       0.72      0.68      0.70      4392
    weighted avg       0.77      0.78      0.77      4392
    
    Accuracy of the model is :  0.7750455373406193


#### Observations
<p> The accuracy on the test set is 77.50% i.e. 77.50% of the texts were classified with the correct label. </p>


```python
result1 =  clf1.predict(X_test) 
```


```python
# Print and plot Confusion matirx to get an idea of how the distribution of the prediction is, among all the classes.

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, result1)

print(conf_mat)

print(metrics.f1_score(y_test, result1,average='micro'))

df_cm = pd.DataFrame(conf_mat, index = [i for i in "123"],
                  columns = [i for i in "123"])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, fmt='g')
```

    [[ 429   88  177]
     [  72  472  340]
     [  78  233 2503]]
    0.7750455373406193





    <AxesSubplot:>




    
![png](output_118_2.png)
    


#### Observations
<p> The accuracy metric here can be interpreted to be: the count of texts that were correctly identified as positive, negative or neutral. </p>
<p> Here, in the confusion matrix, for the first label "1" (which would correspond to "positive"), there were a total of 694 items (429+88+177).So approximately 61.8% of tweets belonging to class 1 were identified correctly. The model that used countVectorizer marked 429of those correctly as label "1", but put 88 of those in label "2" and 177 of those in label "3". Similarly, for label "2", the model identified 472 of those items correctly, but misclassified 72 as label "1" and 340 as label "3". For the third class which is the "negative" sentiment, 2503 items were correctly labeled as "3" out of 2814 items (88.9%). </p>


```python
all_features = bow_vec.get_feature_names()              #Instantiate the feature from the vectorizer
top_features=''                                            # Addition of top 40 feature into top_feature after training the model
feat=clf1.feature_importances_
features=np.argsort(feat)[::-1]
for i in features[0:40]:
    top_features+=all_features[i]
    top_features+=' '
    
    

from wordcloud import WordCloud
wordcloud = WordCloud(background_color="white",colormap='viridis',width=2000, 
                          height=1000).generate(top_features)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.figure(1, figsize=(14, 11), frameon='equal')
plt.title('Top 40 features WordCloud', fontsize=20)
plt.axis("off")
plt.show()
```


    
![png](output_120_0.png)
    


#### Observations
<p> We can see that the words "hour", "thank","great", "service", "delay", "flight", "love" stand out in the word cloud. </p>


```python
#convert the test samples into a dataframe where the columns are
#the y_test(ground truth labels),tf-idf model predicted labels(tf_idf_predicted),Count Vectorizer model predicted labels(count_vectorizer_predicted)
df = pd.DataFrame(y_test.tolist(),columns =['y_test'])
df['count_vectorizer_predicted'] = count_vectorizer_predicted
df['tf_idf_predicted'] = tf_idf_predicted
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>y_test</th>
      <th>count_vectorizer_predicted</th>
      <th>tf_idf_predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### Observations
<p> Comparing the two vectorizers (count versus tfidf) we see above in the sample of 10 predicted values, that both vectorizers are pretty close to each other in terms of performance. In the above table, if the actual value was "3" which corresponds to "negative", both models predicted it as "negative" (row number 2). However, for row number 4, the actual value is "3" which is "negative" sentiment, however both models predicted it as a "2" which corresponds to the "neutral" label. Thus, the models performance, while being approximately 77%, still needs further strategies to improve the performance. </p>


```python
#create bar plot to compare the accuaracies of Count Vectorizer and TF-IDF
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(7,5))
ax = fig.add_axes([0,0,1,1])
subjects = ['Count_Vectorizer', 'TF-IDF']

# calculation accuracies of Count Vectorizer and TF-IDF using accuracy_score metrics
scores = [accuracy_score(y_test,count_vectorizer_predicted),accuracy_score(y_test,tf_idf_predicted)]
ax.bar(subjects,scores)
ax.set_ylabel('scores',fontsize= 12)    # y axis label
ax.set_xlabel('models',fontsize= 12)    # x axis label
ax.set_title('Accuaracies of Supervised Learning Methods')  # tittle
for i, v in enumerate(scores):
    ax.text( i ,v+0.01, '{:.2f}%'.format(100*v), color='black', fontweight='bold')     
    plt.savefig('barplot_1.png',dpi=100, format='png', bbox_inches='tight')
plt.show()
```


    
![png](output_124_0.png)
    


#### Observations
<p> Both vectorizers performed similarly on the dataset. The TfidfVectorizer had a slightly better performance (77.89%) over the countVectorizer when the random forest parameters were kept the same. Overall, the model performance can be improved by using the TfidfVectorizer but with a thorough search of the gridspace using gridsearch teachniques. </p>

### Insights and summary
<ol>
    <li> From the 14,640 tweet data, the majority of the posts had a negative sentiment in it (62.69% of the posts). This indicates a high customer dissatisfaction with the airline products and services. </li>
    <li> United airlines and US airways lead the most tweets that are having a negative sentiment attached to it. In general, negative sentiment reviews outnumber positive and neutral reviews by a large margin. This is not true just for airline industry but also for other industries. One underlying hypothesis could be that people take the effort to write write reviews if they have had a bad experience with the product or service, rather than taking the effort to write a neutral review or a positive review. When the service is exceptional, people might take the effort again to write a review. </li>
    <li> Most of the negative sentiment surrounding the use of airline services and products deal with customer service, delays, lost baggage etc (as seen in the negative sentiment wordcloud in the EDA section). While delays due to weather and other external events are unavoidable, the airlines can develop strategies to make delays or dealing with customer service an easier operation - for example, being proactive in reaching out to customers about issues.
</li> 
    <li>Overall, the preprocessing and vectorization helped to break down text data (14,640 tweets) into smaller components that enabled an NLP model to classify text. The tfidf model performed slightly better than the countVectorizer for NLP models. This can be attributed to the process where tfidfVectorizer puts emphasis on the more important words, as compared to countVectorizer where the basis of vectorization is based on the frequency of the word.</li>
    <li> There are a couple of strategies can be utilized to improve the model performance - we used a custom stopwords list for preprocessing, however the list can be improved. Techniques such as gridsearch, while being computationally expensive, can yield improvement in performance. </li>


```python

```
