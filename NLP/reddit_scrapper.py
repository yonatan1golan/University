import praw
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time

from secrets_config import RedditSecretsConfig  # <-- Make sure you have this with your creds

nltk.download('vader_lexicon')


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=RedditSecretsConfig.client_id,
            client_secret=RedditSecretsConfig.client_secret,
            user_agent=RedditSecretsConfig.user_agent
        )


def fetch_posts(client, keyword, limit=100):
    posts = []
    for submission in client.reddit.subreddit('all').search(keyword.lower(), sort='new', time_filter='month', limit=limit):
        posts.append({
            'title': submission.title,
            'selftext': submission.selftext,
            'created_utc': submission.created_utc
        })
        time.sleep(0.5)  # avoid hitting API rate limit
    return pd.DataFrame(posts)


def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def build_graph(df, individuals):
    G = nx.Graph()
    G.add_node('Tesla')
    mention_data = defaultdict(lambda: {'count': 0, 'sentiments': []})

    for _, row in df.iterrows():
        text = f"{row['title']} {row['selftext']}".lower()
        for person in individuals:
            if person.lower() in text and 'tesla' in text:
                sentiment = analyze_sentiment(text)
                mention_data[person]['count'] += 1
                mention_data[person]['sentiments'].append(sentiment)

    for person, data in mention_data.items():
        if data['count'] == 0:
            continue
        G.add_node(person)
        avg_sentiment_score = sum([{'positive': 1, 'neutral': 0, 'negative': -1}[s] for s in data['sentiments']]) / len(data['sentiments'])
        color = 'green' if avg_sentiment_score > 0.2 else 'red' if avg_sentiment_score < -0.2 else 'orange'
        width = data['count']
        G.add_edge('Tesla', person, weight=width, color=color)

    return G


def draw_graph(G):
    pos = nx.spring_layout(G, seed=42)
    colors = [G[u][v]['color'] for u, v in G.edges()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, edge_color=colors, width=weights, node_color='skyblue', node_size=1500, font_size=10)
    plt.title("Reddit Co-Mention Sentiment Graph with Tesla")
    plt.show()


if __name__ == '__main__':
    # Step 1: Connect to Reddit
    client = RedditClient()

    # Step 2: Define individuals to track
    individuals = ['Elon Musk', 'Jim Cramer', 'Cathie Wood', 'Joe Biden', 'Donald Trump']

    # Step 3: Fetch posts
    all_posts = pd.DataFrame()
    for person in individuals:
        print(f"Fetching posts for: {person}")
        posts_df = fetch_posts(client, f"Tesla {person}", limit=50)
        all_posts = pd.concat([all_posts, posts_df], ignore_index=True)

    # Step 4: Build the graph
    G = build_graph(all_posts, individuals)

    # Step 5: Visualize the graph
    draw_graph(G)