'''
    CAPSTONE PROJECT

    In this task, you will develop a Python program that performs sentiment analysis
    on a dataset of product reviews.

    Follow these steps:

        -Download a dataset of product reviews https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
         You can save it as a CSV file, naming it: amazon_product_reviews.csv.

        -Create a Python script, naming it: sentiment_analysis.py. Develop a Python script for sentiment analysis. Within
         the sript, you will perform the following tasks using the spaCy library:

         1. Implement a sentiment analysis model using spaCy: 
                Load the 'en_core_web_sm' spaCy model to enable natural language processing tasks.
                This model will help you analyse and classify the sentiment of the product reviews.

         2. Preprocess the text data:
                Remove stopwords, and perform any
                necessarty text cleaning to prepare the reviews for analysis.

                    2.1 To select the 'review.text' column form the dataset and retrieve its data, you can simply use
                        the square brackets notation. Here is the basic syntax:

                        reviews_data = dataframe['review.text']

                        This column, 'review.text', represents the feature variable containing the product reviews we will use
                        for sentiment analysis.
                    2.2
                        To remove all missing values from this column, you can simply use the dropna() function from Pandas using
                        the following code:

                        clean_data = dataframe.dropna(subset=['reviews.text'])       

         3. Create a function for sentiment analysis: 
                Define a function that takes a product review as input and predicts its sentiment.

         4. Test your model on sample product reviews:
                Test the sentiment analysis function on a few sample product reviews to verify its accuracy
                in predicting sentiment.
         
         5. Write a brief report or summary in a PDF file:
                sentiment_analysis_report.pdf that must include:
                    5.1 A desription of the dataset used.
                    5.2 Details of the preprocessing steps
                    5.3 Evaluation of results
                    5.4 Insights into the model's strngths and limitations
    
    Additional Instructions:

        -Some helpful guidelines on cleaning text:
            
            -To remove stopwords, you can utilise the .is_stop attribute in spaCy.
             This attribute helps identify whether a word in a text qualifies as a
             stop word or not. Stopwords are common words that do not add
             much meaning to a sentence, such as 'the', 'is', and 'of'.
             Subsequently, you can then employ the filterd list of tokens or
             words(words with no stop words) for conducting sentiment analysis.

            -You can also make use of the 'lower()', 'strip()' and 'str()' methods to
             perform some basic text cleaning.

        -You can use the spaCy model and the '.sentiment' attribute to analyse the 
         review and determine whether it expresses a positive, negative, or neutral
         sentiment. To use the '.polarity' attribute, you will need to install the
         
         'TextBlob library' You can do this with the following commands:
                - #Install spacytextblob
                - pip install spacytextblob
            
            -Textblob requires additional data vefore getting started, download the data
             using the following code:
                - python -m textblob.download_corpora
            
            -Once you have installed TextBlob, you can use the .setntiment and .polarity attribute
             to analyse the review and determine whether it expresses a positive, negative, or neutral
             sentiment. You can also incorporate this code to get yourself started:
                - # Using the polarity attribute
                - polartity = doc._.blob.polarity
                - # Using the sentiment attribute
                - sentiment = doc._.blob.sentiment

    FYI: The underscore in the code just above is a 'Python convetnion' for naming private attributes. 
         Private attributes are not meant to be accessed directly by the user, but can be accessed through public
         methods.

         - You can use the '.polarity' attribute to measure the strngth of the sentiment in a product review. A polarity
           score of 1 indicates a very positive sentiment, while a polarity score of -1 indicates a very negative sentiment.
           A polarity score of 0 indicates a neutral sentiment.

         - You can also use the 'similarity' function to compare the similarity of two product reviews. A similarity score
           compare their similarity. To select a specific review from this column, simply use indexing, as shown 
           in the code below:

                my_review_of_choice = data['reviews.text'][0]

        - The above code retrieves a review from the 'review.text' column at index 0. You can select two reviews of your choice 
          using indexing.
          However, please be cautious not to use an index that is out of bounds, meaning it exceedes the number of data points
          or rows in our dataset.

        -Include informative comments that clarify the rationale behind each line of code  
'''
#Import pandas, spacy, textblob and load 'en_core_web_sm' spaCy model
import pandas as pd
import spacy
from textblob import TextBlob

                                            
nlp = spacy.load('en_core_web_sm')  #Create nlp object with 'en_core_web_sm' statistical model


amazon_df = pd.read_csv('amazon_product_reviews.csv')   #Create 'amazon_df" data frame with 'amazon_product_review.csv' data
                                                        #Enter Your own path to 'amazon_product_reviews.csv' 



#Block with defintions of main 'sentiment_analysis' function which takes data frame as an argument
#and two subfnuctions; 'preprocess' and 'analyze_polarity'.
#Both funtctions take preprocessed text data 'reviews.text' from 'amazon_df' data frame
def sentiment_analysis(df_input):
    
    def preprocess(text):           #Defintion of 'preprocess' function. 
                                    #Function takes string from every row of 'reviews.text' as an argument
        
        
        doc = nlp(text)             #Using nlp with "en_core_web_sm" model creates 'doc' object with tokenized
                                    #string from every row of 'reviews.text'
        
        no_stop_words = [token.lemma_.lower() for token in doc             #Create 'no_stop_words' list with tokens 
                        if not token.is_stop and not token.is_punct]       #from every row of 'reveiws.text' wich aren't 'stop words'
                                                                           #or 'puncation marks' using 'list comprehension' method. 
 
        return " ".join(no_stop_words)              #Join all 'non_stop_words' items and return

    #Definition of 'analyze_polarity' function
    #Function takes every string from every row of 'processed.text' as an argument
    def analyze_polarity(text):
      
        polarity = TextBlob(text).sentiment.polarity    #Using 'TextBlob' get the polarity from 'text' and write in 'polarity' object
    
        return polarity     #Return received polarity


    #[1]
    #Preprocessing of 'df_input'.
    #First, create object 'amazon_reviews_df' with  'reviews.text' column content
    #Second, create 'amazon_reviews_cleaned_df' object with cleaned from missin values 'reviews.text' column content
    
    #[2]
    #Using 'preprocess' function remove all stop words and punctation marks from 'amazon_reviews_cleaned_df".
    #Add results of this operation into 'amazon_reviews_cleaned_df['processed.text']' what will create
    #new column 'processed.text' in 'amazon_reviews_cleaned_df
    
    #[3]
    #Create 'data' list with strings as an items cleaned from 'stop words' and 'punctation marks' from
    #'amazon_reviews_cleaned_df['processed.text]' object
    
    
    #[1]
    amazon_reviews_df = df_input[['reviews.text']]
    amazon_reviews_cleaned_df = amazon_reviews_df.dropna(subset=['reviews.text'])
    #[2]
    amazon_reviews_cleaned_df['processed.text'] = amazon_reviews_cleaned_df['reviews.text'].apply(preprocess)
    #[3]
    data = amazon_reviews_cleaned_df['processed.text'].values
    
    
    sentiments = []     #Create empty list 'sentiment' for storing 'polarity_socores'

    #For every 'item' in 'data' list
    for item in data:
        
        polarity_score = analyze_polarity(item)     #Call 'analyze_polarity' function which takes 'item' string as an argument
                                                    #and calculates 'polarity', writes result in 'polarity_score' object.

        #For different 'analyze_polarity' returns (-1, 0 or 1),
        #create ritht 'sentiment' object.
        #'positive' when 'polarity_score' = 1, 'negative' when 'polarity_score' = -1, 'neutral' when 'polarity_score' = 0 
        if polarity_score > 0:
            sentiment = 'positive'
    
        elif polarity_score < 0:
            sentiment = 'negative'
    
        else:
            sentiment = 'neutral'

        #Add result at the end of list 'sentiments'
        sentiments.append(sentiment)


    #When all 'sentiments' are recognized,
    #count how much we have all toegather 'positive, 'negative' and 'neutral' sentiments
    #and write received numbers in appropriate objects; 'positive_count', 'negative_count', 'neutral_count' 
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')   
    neutral_count = sentiments.count('neutral')

    #Calculate lenght number of 'sentiments' list and write in 'total' object
    total = len(sentiments)

    #Calculate percentage of every sentiment and write results in appropriate objects;
    #'positive_perc', 'negative_perc', 'neutral_perc'
    positive_perc = (positive_count / total) * 100
    negative_perc = (negative_count / total) * 100
    neutral_perc = (neutral_count / total) * 100

    #Display received results
    print(f"\n\nPositive percentage: {positive_perc:.2f}%")
    print(f"Negative percentage: {negative_perc:.2f}%")
    print(f"Neutral percentage: {neutral_perc:.2f}%\n\n")



#Start. Call 'sentiment_analysis' function with 'amazon_df' as an argument.
sentiment_analysis(amazon_df)