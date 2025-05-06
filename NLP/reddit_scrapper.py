import praw
import matplotlib.pyplot as plt
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timezone
from secrets_config import RedditSecretsConfig
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


class CONGIF:
    INDIVIDUALS = [
        "Elon Musk", "Zach Kirkhorn", "Tom Zhu", "JB Straubel",
        "Joe Biden", "Pete Buttigieg", "Lina Khan",
        "Cathie Wood", "Jim Cramer", "Ross Gerber", "Gary Black", "Adam Jonas",
        "Chamath Palihapitiya", "Mark Spiegel", "Michael Burry"
    ]
    SUBREDDITS = ['investing', 'stocks'] #'wallstreetbets', 'economy', 'politics']
    # INTERESTING_PERIODS = {
    #     '2020-03-01': '2020-10-30',
    #     '2021-09-01': '2022-02-28',
    #     '2022-07-01': '2022-12-31',
    #     '2024-05-01': '2024-12-31',
    #     '2025-01-01': '2025-03-31'
    # }

    POST_LIMIT = 50
    COMMENT_LIMIT = 10
    SENTIMENT_THRESHOLD = 0.10

class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=RedditSecretsConfig.client_id,
            client_secret=RedditSecretsConfig.client_secret,
            user_agent=RedditSecretsConfig.user_agent
        )

    def fetch_top_posts(self, subreddit_name, keyword, post_limit=CONGIF.POST_LIMIT):
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []

        for post in subreddit.search(keyword, limit=post_limit, sort='top'):
            post_info = {
                'title': post.title,
                'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                'score': post.score,
                'num_comments': post.num_comments,
                'comments': self._fetch_top_comments(post, CONGIF.COMMENT_LIMIT),
            }
            posts.append(post_info)
        return posts

    def _fetch_top_comments(self, post, comment_limit):
        post.comments.replace_more(limit=0)
        top_comments = post.comments[:comment_limit]
        return [
            {
                'author': str(comment.author),
                'body': comment.body,
                'score': comment.score
            }
            for comment in top_comments if comment.body not in ['[deleted]', '[removed]']
        ]


class Processor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def _analyze_sentiment(self, text):
        score = self.sia.polarity_scores(text)['compound']
        sentiment = 'positive' if score >= CONGIF.SENTIMENT_THRESHOLD else 'negative' if score <= -CONGIF.SENTIMENT_THRESHOLD else 'neutral'
        return score, sentiment

    def _clean_text(self, text):
        # remove emojis, special characters, etc.
        pass

    def process_posts(self, posts):
        for post in posts:
            score, label = self._analyze_sentiment(post['title'])
            post['sentiment_score'] = score
            post['sentiment_label'] = label
            for comment in post['comments']:
                score, label = self._analyze_sentiment(comment['body'])
                comment['sentiment_score'] = score
                comment['sentiment_label'] = label


class GraphMaker:
    def __init__(self):
        self.graph = nx.Graph()
        self.entities = ["Tesla"] + CONGIF.INDIVIDUALS
        self.min_width = 0.5
        self.max_width = 10.0
        self.custom_cmap = LinearSegmentedColormap.from_list(
            "custom_red_gray_green",
            ["darkred", "gray", "darkgreen"],
            N=256
        )

    def _find_mentions(self, text):
        mentions = []
        lowered = text.lower()
        for entity in self.entities:
            if entity.lower() in lowered:
                mentions.append(entity)
        return mentions

    def build_graph(self, posts):
        for post in posts:
            mentions = self._find_mentions(post['title'])
            self._add_edges(mentions, post['score'], post['sentiment_score'])

            for comment in post['comments']:
                comment_mentions = self._find_mentions(comment['body'])
                self._add_edges(comment_mentions, comment['score'], comment['sentiment_score'])

    def _add_edges(self, mentions, score, sentiment_score):
        for i in range(len(mentions)):
            for j in range(i + 1, len(mentions)):
                a, b = mentions[i], mentions[j]
                if self.graph.has_edge(a, b):
                    self.graph[a][b]['weight'] += score
                    self.graph[a][b]['sentiments'].append(sentiment_score)
                else:
                    self.graph.add_edge(a, b, weight=score, sentiments=[sentiment_score])

    def finalize_graph(self):
        weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        if weights:
            min_w = min(weights)
            max_w = max(weights)
        else:
            min_w = max_w = 1

        for u, v, data in self.graph.edges(data=True):
            # normalize weight
            if max_w != min_w:
                norm_weight = self.min_width + (data['weight'] - min_w) / (max_w - min_w) * (self.max_width - self.min_width)
            else:
                norm_weight = (self.max_width + self.min_width) / 2

            avg_sentiment = sum(data['sentiments']) / len(data['sentiments'])

            data['normalized_weight'] = norm_weight
            data['color'] = self._sentiment_to_color(avg_sentiment)

    def _sentiment_to_color(self, score):
        norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
        rgba = self.custom_cmap(norm(score)) 
        return rgba

    def visualize(self):
        pos = nx.spring_layout(self.graph, seed=42)

        edge_colors = [data['color'] for _, _, data in self.graph.edges(data=True)]
        edge_weights = [data['normalized_weight'] for _, _, data in self.graph.edges(data=True)]

        nx.draw(self.graph, pos, with_labels=True,
                edge_color=edge_colors,
                width=edge_weights,
                node_color='lightblue',
                font_size=8)

        plt.title("Tesla & Influencers Co-Mention Sentiment Graph")
        plt.show()


if __name__ == '__main__':
    client = RedditClient()
    processor = Processor()
    graphmaker = GraphMaker()
    all_posts = []

    for individual in CONGIF.INDIVIDUALS:
        for subreddit in CONGIF.SUBREDDITS:
            posts = client.fetch_top_posts(subreddit, individual)
            processor.process_posts(posts)
            all_posts.extend(posts)

    graphmaker.build_graph(all_posts)
    graphmaker.finalize_graph()
    graphmaker.visualize()
