import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

class RedditSentimentAnalyzer:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize Reddit API client
        
        Parameters:
        - client_id: Your Reddit app's client ID
        - client_secret: Your Reddit app's client secret
        - user_agent: A unique identifier for your script
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Download NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        
        # Initialize stopwords and sentiment analyzer
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

    def collect_submissions(self, subreddit_name, query=None, limit=100):
        """
        Collect submissions from a subreddit
        
        Parameters:
        - subreddit_name: Name of the subreddit
        - query: Optional search term
        - limit: Maximum number of submissions to collect
        
        Returns:
        - DataFrame with submission data
        """
        subreddit = self.reddit.subreddit(subreddit_name)
        
        submissions_data = []
        
        # Collect submissions based on search or hot posts
        if query:
            submissions = subreddit.search(query, limit=limit)
        else:
            submissions = subreddit.hot(limit=limit)
        
        for submission in submissions:
            submissions_data.append({
                'id': submission.id,
                'title': submission.title,
                'text': submission.selftext,
                'created_at': pd.to_datetime(submission.created_utc, unit='s'),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'url': submission.url
            })
        
        return pd.DataFrame(submissions_data)

    def collect_comments(self, submission_ids, limit=100):
        """
        Collect comments for given submission IDs
        
        Parameters:
        - submission_ids: List of submission IDs
        - limit: Maximum comments per submission
        
        Returns:
        - DataFrame with comment data
        """
        comments_data = []
        
        for submission_id in submission_ids:
            submission = self.reddit.submission(id=submission_id)
            submission.comments.replace_more(limit=0)  # Flatten comment tree
            
            for comment in submission.comments.list()[:limit]:
                comments_data.append({
                    'submission_id': submission_id,
                    'comment_id': comment.id,
                    'text': comment.body,
                    'created_at': pd.to_datetime(comment.created_utc, unit='s'),
                    'score': comment.score
                })
        
        return pd.DataFrame(comments_data)

    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http\S+', '', text)
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            tokens = [token for token in tokens if token not in self.stop_words]
            
            # Rejoin tokens
            cleaned_text = ' '.join(tokens)
            
            return cleaned_text
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            return text

    def analyze_sentiment(self, text):
        """
        Perform multi-method sentiment analysis
        """
        try:
            # TextBlob Sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # VADER Sentiment
            vader_sentiment = self.sia.polarity_scores(text)['compound']
            
            # Categorize sentiment
            def categorize_sentiment(score):
                if score > 0.05:
                    return 'Positive'
                elif score < -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'
            
            return {
                'textblob_score': textblob_sentiment,
                'vader_score': vader_sentiment,
                'sentiment_category': categorize_sentiment(vader_sentiment)
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {
                'textblob_score': 0,
                'vader_score': 0,
                'sentiment_category': 'Neutral'
            }

    def generate_visualizations(self, df):
        """
        Create comprehensive sentiment visualizations
        """
        plt.figure(figsize=(20, 15))
        
        # Sentiment Distribution Pie Chart
        plt.subplot(2, 2, 1)
        sentiment_counts = df['sentiment_category'].value_counts()
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
        plt.title('Sentiment Distribution', fontsize=16)
        
        # Sentiment Trends Over Time
        plt.subplot(2, 2, 2)
        daily_sentiment = df.groupby(df['created_at'].dt.date)['vader_score'].mean()
        plt.plot(daily_sentiment.index, daily_sentiment.values)
        plt.title('Sentiment Trend Over Time', fontsize=16)
        plt.xticks(rotation=45)
        
        # Boxplot of Sentiment Scores
        plt.subplot(2, 2, 3)
        sns.boxplot(x='sentiment_category', y='vader_score', data=df)
        plt.title('Sentiment Score Distribution', fontsize=16)
        
        # Word Cloud
        plt.subplot(2, 2, 4)
        text = ' '.join(df['cleaned_text'])
        wordcloud = WordCloud(width=800, height=400, 
                               background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Most Frequent Words', fontsize=16)
        
        plt.tight_layout()
        plt.show()

    def generate_report(self, df):
        """
        Generate comprehensive sentiment analysis report
        """
        print("--- Sentiment Analysis Report ---")
        print("\nOverall Sentiment Distribution:")
        print(df['sentiment_category'].value_counts(normalize=True) * 100)
        
        print("\nDetailed Sentiment Statistics:")
        sentiment_stats = df.groupby('sentiment_category')['vader_score'].agg(['count', 'mean', 'std'])
        print(sentiment_stats)
        
def main():
    # Replace with your actual Reddit API credentials
    CLIENT_ID = 'your_client_id'
    CLIENT_SECRET = 'your_client_secret'
    USER_AGENT = 'your_user_agent'

    # Initialize Reddit Sentiment Analyzer
    reddit_analyzer = RedditSentimentAnalyzer(CLIENT_ID, CLIENT_SECRET, USER_AGENT)

    # Collect submissions from a subreddit
    subreddit_name = 'technology'  # Example subreddit
    submissions_df = reddit_analyzer.collect_submissions(subreddit_name, limit=200)

    # Preprocess text
    submissions_df['cleaned_text'] = submissions_df['text'].apply(reddit_analyzer.preprocess_text)

    # Perform sentiment analysis
    submissions_df['sentiment'] = submissions_df['cleaned_text'].apply(reddit_analyzer.analyze_sentiment)

    # Extract sentiment columns
    submissions_df['textblob_score'] = submissions_df['sentiment'].apply(lambda x: x['textblob_score'])
    submissions_df['vader_score'] = submissions_df['sentiment'].apply(lambda x: x['vader_score'])
    submissions_df['sentiment_category'] = submissions_df['sentiment'].apply(lambda x: x['sentiment_category'])

    # Generate visualizations
    reddit_analyzer.generate_visualizations(submissions_df)

    # Generate report
    reddit_analyzer.generate_report(submissions_df)

    # Optional: Save data to CSV
    submissions_df.to_csv('reddit_sentiment_analysis.csv', index=False)

if __name__ == "__main__":
    main()