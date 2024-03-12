import streamlit as st
from googleapiclient.discovery import build
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import numpy as np

# Set up YouTube API credentials
api_key = "AIzaSyD3loolbDpjOvFQdzHNmlOQtNH6211ibY4"
youtube = build('youtube', 'v3', developerKey=api_key)

# Streamlit app
def main():
    st.title("YouTube Video Sentiment Analysis")

    # Get YouTube video URL from the user
    youtube_url = st.text_input("Enter YouTube video URL:")

    if youtube_url:
        try:
            # Extract video ID from URL
            video_id = youtube_url.split('=')[-1]

            # Embed the YouTube video
            embed_code = generate_embed_code(video_id)
            st.markdown(embed_code, unsafe_allow_html=True)

            # Get video comments and likes
            comments, likes = get_comments_and_likes(video_id)

            # Perform sentiment analysis and scale by likes
            sentiment_scores = analyze_sentiment(comments, likes)

            # Display sentiment analysis results
            st.write("Sentiment Analysis Results:")
            # Plot distribution of sentiments
            plot_sentiment_distribution(sentiment_scores)

        except Exception as e:
            st.error(f"Error: {e}")

def get_comments_and_likes(video_id):
    # Call YouTube API to retrieve top 250 comments by relevance
    response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        order='relevance',
        maxResults=100
    ).execute()

    # Extract comments and likes
    comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
    likes = [item['snippet']['topLevelComment']['snippet']['likeCount'] for item in response['items']]

    return comments, likes

def analyze_sentiment(comments, likes):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = []

    for comment, like_count in zip(comments, likes):
        # Perform sentiment analysis
        vs = analyzer.polarity_scores(comment)
        compound_sentiment = vs['compound']

        # Scale compound sentiment by likes
        # If likes are 0, use a small value (e.g., 1) to prevent scaled sentiment from becoming 0
        scaled_sentiment = compound_sentiment * max(like_count, 1)

        # Categorize sentiment based on compound score
        if scaled_sentiment >= 10:
            sentiment_scores.append('Strongly Positive')
        elif 0.2 <= scaled_sentiment < 10:
            sentiment_scores.append('Positive')
        elif -0.2 <= scaled_sentiment < 0.2:
            sentiment_scores.append('Neutral')
        elif -10 <= scaled_sentiment < -0.2:
            sentiment_scores.append('Negative')
        else:
            sentiment_scores.append('Strongly Negative')

    return sentiment_scores

def generate_embed_code(video_id):
    # Generate embed code for the YouTube video
    embed_code = f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'
    return embed_code

def plot_sentiment_distribution(sentiment_scores):
    # Count occurrences of each sentiment category
    sentiment_counts = {
        'Strongly Positive': sentiment_scores.count('Strongly Positive'),
        'Positive': sentiment_scores.count('Positive'),
        'Neutral': sentiment_scores.count('Neutral'),
        'Negative': sentiment_scores.count('Negative'),
        'Strongly Negative': sentiment_scores.count('Strongly Negative')
    }

    # Plot distribution of sentiments
    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())

    fig, ax = plt.subplots()
    ax.bar(labels, counts, color=['green', 'lightgreen', 'gray', 'lightcoral', 'red'])

    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Comments by Sentiment')

    st.pyplot(fig)

if __name__ == "__main__":
    main()
