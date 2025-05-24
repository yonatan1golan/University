# reddit
import praw

# nlp
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, models
from wordcloud import WordCloud

# graph
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# config
from secrets_config import RedditSecretsConfig

# system
from datetime import datetime, timezone
from collections import defaultdict
import pandas as pd
from pytz import UTC
import ast
import re
import emoji
import unicodedata

class CONFIG:
    INDIVIDUALS = [
        # "Elon Musk",           # CEO of Tesla. Founder, product architect, and central to all decisions at Tesla.
        # "Jeff Bezos",          # Amazon founder. Business and space rival to Musk (Blue Origin vs. SpaceX); often compared to Musk.
        # "Larry Page",          # Google co-founder. Longtime friend of Musk; Google/Waymo competes with Tesla in autonomous driving.
        # "Donald Trump",        # U.S. President. Influences public opinion and policy; has made comments on Tesla and Musk.
        # "Mark Zuckerberg",     # Meta CEO. Competes with Musk in social media; has made comments on Tesla and Musk.
        # "Joe Biden",           # U.S. President. Initially dismissive of Tesla in EV discussions; later acknowledged its EV leadership.
        # "Cathie Wood",         # CEO of ARK Invest. Major Tesla bull and investor; forecasts extremely high valuations for Tesla.
        # "Jim Cramer",          # CNBC host. Publicly flip-flopped on Tesla; currently supportive but controversial in Tesla circles.
        # "Chamath Palihapitiya",# VC and SPAC investor. Public Tesla bull and Musk supporter; promoted Tesla on media.
        # "Michael Burry",       # Famed for The Big Short. Publicly shorted Tesla; skeptical of valuation.
        
        "Gavin Newsom",        # Governor of California. Tesla's home state; has made comments on Tesla and Musk.
        "Alexandria Ocasio-Cortez", # U.S. Congresswoman. Criticized Musk and Tesla on social issues; represents a younger, progressive demographic.
        "Pete Buttigieg",      # U.S. Secretary of Transportation. Has commented on Tesla's role in EV adoption and infrastructure.
        "Bernie Sanders",      # U.S. Senator. Criticized Musk for wealth and influence; represents a progressive viewpoint on wealth inequality.
    ]

    SUBREDDITS = [
        'TeslaMotors',         # Main Tesla discussion hub
        'TeslaInvestorsClub',  # Tesla investment focused  
        'wallstreetbets',      # Retail trading community
        'investing',           # General investment discussions
        'electricvehicles',    # General EV discussions
        'technology',          # General tech discussions
        'politics',            # U.S. political discussions
        'RealTesla',           # Critical Tesla perspectives
        'elonmusk'             # Elon Musk specific
    ]

    ALIASES = {
        "Elon Musk": ["Elon", "Musk", "EM", "ElonMusk", "SpaceX", "X.com", "Tesla CEO"],
        "Jeff Bezos": ["Bezos", "Jeff", "Amazon founder", "Blue Origin", "JB"],
        "Mark Zuckerberg": ["Zuck", "Zuckerberg", "Meta CEO", "Facebook"],
        "Larry Page": ["Larry", "Google co-founder", "Alphabet"],
        "Donald Trump": ["Trump", "Donald", "POTUS 45", "45th President", "The Donald"],
        "Joe Biden": ["Biden", "President Biden", "Joe"],
        "Cathie Wood": ["Cathie", "ARK", "ARK Invest", "Cathie W", "ARKK"],
        "Jim Cramer": ["Cramer", "Mad Money", "Jim", "CNBC host"],
        "Chamath Palihapitiya": ["Chamath", "Chamath P", "Social Capital", "SPAC King"],
        "Michael Burry": ["Burry", "The Big Short", "Dr. Burry", "Scion Capital"],
        "Gavin Newsom": ["Newsom", "Governor Newsom", "CA Governor"],
        "Alexandria Ocasio-Cortez": ["AOC", "Ocasio-Cortez", "Congresswoman AOC"],
        "Pete Buttigieg": ["Buttigieg", "Mayor Pete", "Transportation Secretary"],
        "Bernie Sanders": ["Bernie", "Senator Sanders", "Sanders"],
    }

    COMPARATIVE_COMPANIES = [
        "Rivian", "NIO", "Lucid", "BYD", "Ford", "GM", "Apple", "Meta", "Palantir"
    ]

    TIME_PERIODS = {
        "covid_rebound": ("2020-03-01", "2020-10-31"),
        "pre_peak": ("2021-01-01", "2021-03-31"),
        "post_twitter": ("2022-10-01", "2022-12-31"),
        "modern_period": ("2023-01-01", "2024-12-31")
    }

    POST_LIMIT = 50
    COMMENT_LIMIT = 20
    SENTIMENT_THRESHOLD = 0.10


class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=RedditSecretsConfig.client_id,
            client_secret=RedditSecretsConfig.client_secret,
            user_agent=RedditSecretsConfig.user_agent
        )
        self.all_posts = []

    def _fetch_top_posts(self, subreddit_name, keyword, post_limit=CONFIG.POST_LIMIT):
        subreddit = self.reddit.subreddit(subreddit_name)
        posts = []

        for post in subreddit.search(keyword, limit=post_limit, sort='top'):
            post_info = {
                'title': post.title,
                'created_utc': datetime.fromtimestamp(post.created_utc, tz=timezone.utc),
                'score': post.score,
                'num_comments': post.num_comments,
                'comments': self._fetch_top_comments(post, CONFIG.COMMENT_LIMIT),
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

    def fetch_all_posts(self):
        for individual in CONFIG.INDIVIDUALS:
            print(f"Fetching posts for {individual}...")
            for subreddit in CONFIG.SUBREDDITS:
                posts = client._fetch_top_posts(subreddit, individual)
                self.all_posts.extend(posts)
            pd.DataFrame(posts).to_csv(f"NLP/posts_data/{individual.lower().replace(' ', '_')}.csv", index=False)

    def get_all_posts(self):
        """ returns all posts from all individuals, for the saved local files """
        to_return = []
        for individual in CONFIG.INDIVIDUALS:
            posts = pd.read_csv(f"NLP/posts_data/{individual.lower().replace(' ', '_')}.csv")
            to_return.extend(posts.to_dict(orient='records'))
        for post in to_return:
            post['comments'] = ast.literal_eval(post['comments'])
        return to_return
    
class Processor:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def _analyze_sentiment(self, text):
        score = self.sia.polarity_scores(text)['compound']
        sentiment = 'positive' if score >= CONFIG.SENTIMENT_THRESHOLD else 'negative' if score <= -CONFIG.SENTIMENT_THRESHOLD else 'neutral'
        return score, sentiment

    def normalize_text(self, text):
        text = emoji.demojize(text, delimiters=("", ""))  # removes colons around names (e.g. ":smile:" -> "smile")
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)  # control characters
        text = re.sub(r'[^\w\s\.,!?\'"\-]', '', text) # anything else strange
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _assign_post_periods(self, posts):
        copied = posts.copy()
        for post in copied:
            post['periods'] = []
            for period_name, (start_str, end_str) in CONFIG.TIME_PERIODS.items():
                start = pd.to_datetime(start_str).tz_localize(UTC)
                end = pd.to_datetime(end_str).tz_localize(UTC)
                created_utc = pd.to_datetime(post['created_utc'])
                if start <= created_utc <= end:
                    post['periods'].append(period_name)
                    break
                elif created_utc < start:
                    post['periods'].append('early_stage')
                    break
                elif created_utc > end:
                    post['periods'].append('2025_and_beyond')
                    break
        return copied

    def process_posts(self, posts):
        with_periods = self._assign_post_periods(posts)
        for post in with_periods:
            post['title'] = self.normalize_text(post['title'])
            score, label = self._analyze_sentiment(post['title'])
            post['sentiment_score'] = score
            post['sentiment_label'] = label
            for comment in post['comments']:
                comment['body'] = self.normalize_text(comment['body'])
                score, label = self._analyze_sentiment(comment['body'])
                comment['sentiment_score'] = score
                comment['sentiment_label'] = label
        return with_periods
    
class GraphMaker:
    def __init__(self):
        self.graph = nx.Graph()
        self.entities = ["Tesla"] + CONFIG.INDIVIDUALS
        self.min_width = 0.5
        self.max_width = 10.0
        self.custom_cmap = LinearSegmentedColormap.from_list(
            "custom_red_gray_green",
            ["#f80509", "#deb603", "#0bc746"],
            N=256
        )
        self.node_sentiment = defaultdict(list)
        self.competitor_mentions = defaultdict(int)

    def _find_mentions(self, text):
        mentions = set()
        lowered = text.lower()
        for entity in self.entities:
            aliases = [entity] + CONFIG.ALIASES.get(entity, [])
            for alias in aliases:
                if alias.lower() in lowered:
                    mentions.add(entity)

        for comp in CONFIG.COMPARATIVE_COMPANIES:
            if comp.lower() in lowered:
                mentions.add(comp)
                self.competitor_mentions[comp] += 1

        return list(mentions)

    def build_graph(self, posts):
        for post in posts:
            mentions = self._find_mentions(post['title'])
            self._add_edges(mentions, post['score'], post['sentiment_score'])
            for mention in mentions:
                self.node_sentiment[mention].append(post['sentiment_score'])
            for comment in post['comments']:
                comment_mentions = self._find_mentions(comment['body'])
                self._add_edges(comment_mentions, comment['score'], comment['sentiment_score'])
                for mention in comment_mentions:
                    self.node_sentiment[mention].append(comment['sentiment_score'])

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
        min_w = min(weights) if weights else 1
        max_w = max(weights) if weights else 1

        for u, v, data in self.graph.edges(data=True):
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

    def print_neighborhood(self):
        for node in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node))
            print(f"Neighbors of {node}:")
            for neighbor in neighbors:
                weight = self.graph[node][neighbor]['weight']
                sentiment = sum(self.graph[node][neighbor]['sentiments']) / len(self.graph[node][neighbor]['sentiments'])
                print(f"  {neighbor}: weight={weight}, sentiment={sentiment}")
            print("\n")

    def present_info(self):
        print("📊 Graph Overview:\n")

        print("🔹 Basic Stats:")
        print(f"  Nodes: {len(self.graph.nodes)}")
        print(f"  Edges: {len(self.graph.edges)}")
        print(f"  Is Connected: {self.is_graph_connected()}")
        print(f"  Density: {self.get_graph_density():.4f}")
        print(f"  Diameter: {self.get_graph_diameter()}")
        print(f"  Average Degree: {self.get_graph_average_degree():.2f}")
        print(f"  Avg Clustering Coefficient: {self.get_graph_average_clustering():.4f}")
        print(f"  Avg Shortest Path Length: {self.get_graph_average_shortest_path_length():.2f}")
        print(f"  Avg Node Sentiment: {self.get_graph_average_node_sentiment():.2f}")

        print("\n🔹 Structural Analysis:")
        cut_vertices = self.get_cut_vertexes()
        bridges = self.get_bridges()
        print(f"  Cut Vertices (Articulation Points): {cut_vertices if cut_vertices else 'None'}")
        print(f"  Bridges (Critical Edges): {bridges if bridges else 'None'}")

        print("\n🔹 Community Detection:")
        communities = self.get_communities()
        print(f"  Number of Communities: {len(communities)}")
        community_sizes = [len(c) for c in communities]
        print(f"  Community Sizes: {community_sizes}")

        print("\n🔹 Centrality Measures (Top 3 Nodes per Metric):")
        centrality = self.get_centrality_measures()

        def top_n(dictionary, n=3):
            return sorted(dictionary.items(), key=lambda x: x[1], reverse=True)[:n]

        for metric, values in centrality.items():
            top_nodes = top_n(values)
            top_str = ", ".join(f"{node} ({score:.2f})" for node, score in top_nodes)
            print(f"  {metric.capitalize()}: {top_str}")

    def get_cut_vertexes(self):
        return list(nx.articulation_points(self.graph))

    def get_bridges(self):
        return list(nx.bridges(self.graph))

    def get_communities(self):
        return list(nx.algorithms.community.greedy_modularity_communities(self.graph))

    def get_community_structure(self, verbose=True):
        communities = self.get_communities()
        community_structure = {i: list(community) for i, community in enumerate(communities)}
        
        if verbose:
            print("\n🔸 Community Structure:")
            for i, members in community_structure.items():
                print(f"  Community {i + 1} ({len(members)} members): {', '.join(members)}")
        
        return community_structure

    def get_centrality_measures(self):
        return {
            "degree": nx.degree_centrality(self.graph),
            "betweenness": nx.betweenness_centrality(self.graph),
            "closeness": nx.closeness_centrality(self.graph)
        }

    def get_graph_diameter(self):
        if nx.is_connected(self.graph):
            return nx.diameter(self.graph)
        else:
            return max(
                max(lengths.values())
                for node, lengths in nx.single_source_shortest_path_length(self.graph).items()
            )

    def get_graph_density(self):
        return nx.density(self.graph)

    def get_graph_average_clustering(self):
        return nx.average_clustering(self.graph)

    def get_graph_average_shortest_path_length(self):
        if nx.is_connected(self.graph):
            return nx.average_shortest_path_length(self.graph)
        else:
            return float('inf')

    def get_graph_average_degree(self):
        return sum(dict(self.graph.degree()).values()) / len(self.graph.nodes) if self.graph.nodes else 0

    def get_graph_average_node_sentiment(self):
        sentiments = [sum(scores) / len(scores) for scores in self.node_sentiment.values() if scores]
        return sum(sentiments) / len(sentiments) if sentiments else 0

    def is_graph_connected(self):
        return nx.is_connected(self.graph)

    def visualize(self):
        plt.figure(figsize=(18, 14))

        tesla_node = ["Tesla"]
        influencer_nodes = [n for n in self.graph.nodes if n in CONFIG.INDIVIDUALS]
        competitor_nodes = [n for n in self.graph.nodes if n in CONFIG.COMPARATIVE_COMPANIES]
        other_nodes = [n for n in self.graph.nodes if n not in tesla_node + influencer_nodes + competitor_nodes]

        shells = [tesla_node, influencer_nodes, competitor_nodes + other_nodes]
        pos = nx.shell_layout(self.graph, shells)

        edge_colors = [data['color'] for _, _, data in self.graph.edges(data=True)]
        edge_weights = [data['normalized_weight'] for _, _, data in self.graph.edges(data=True)]

        node_colors = []
        for node in self.graph.nodes:
            if node == "Tesla":
                node_colors.append("blue")
            elif node in CONFIG.INDIVIDUALS:
                node_colors.append("gray")
            elif node in CONFIG.COMPARATIVE_COMPANIES:
                node_colors.append("green")
            else:
                node_colors.append("lightgray")

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=1300, alpha=0.9)
        nx.draw_networkx_labels(self.graph, pos, font_size=10)
        nx.draw_networkx_edges(self.graph, pos, edge_color=edge_colors, width=edge_weights, alpha=0.6)

        plt.title("Tesla & Influencers Co-Mention Sentiment Graph", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class TextAnalysis:
    def __init__(self, posts):
        self.posts = posts
        self.averaged_scores = {}  # initialize to be set later

    def _color_func(self, word, *args, **kwargs):
        score = self.averaged_scores.get(word.lower(), 0)
        rgba = GraphMaker()._sentiment_to_color(score)
        r, g, b, _ = [int(c * 255) for c in rgba]
        return f"rgb({r}, {g}, {b})"

    def generate_word_cloud(self):
        from collections import defaultdict
        word_scores = defaultdict(list)
        all_text = ""

        for post in self.posts:
            for word in post['title'].split():
                word_scores[word.lower()].append(post.get('sentiment_score', 0))
                all_text += word + " "

            for comment in post['comments']:
                for word in comment['body'].split():
                    word_scores[word.lower()].append(comment.get('sentiment_score', 0))
                    all_text += word + " "

        self.averaged_scores = {word: sum(scores) / len(scores) for word, scores in word_scores.items()}

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud.recolor(color_func=self._color_func), interpolation='bilinear')
        plt.axis("off")
        plt.title("Word Cloud Colored by Sentiment")
        plt.show()

    def run_lda(self):
        texts = [[word for word in comment['body'].lower().split() if word.isalpha()] for post in self.posts for comment in post['comments']]
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        lda = models.LdaModel(corpus, id2word=dictionary, num_topics=5, passes=15)
        topics = lda.print_topics(num_words=5)
        print("\nLDA Topics:")
        for topic in topics:
            print(topic)

if __name__ == '__main__':
    client = RedditClient()
    processor = Processor()
    graphmaker = GraphMaker()
    all_posts = []

    client.fetch_all_posts() # uncomment to fetch new posts
    # all_posts = client.get_all_posts()
    # processed_posts = processor.process_posts(all_posts)

    # graphmaker.build_graph(processed_posts)
    # # graphmaker.print_neighborhood()
    # graphmaker.present_info()
    # graphmaker.get_community_structure()
    # graphmaker.finalize_graph()
    # graphmaker.visualize()

    # text_analysis = TextAnalysis(processed_posts)
    # text_analysis.generate_word_cloud()
    # text_analysis.run_lda()