import praw

from secrets_config import RedditSecretsConfig

class RedditClient:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=RedditSecretsConfig.client_id,
            client_secret=RedditSecretsConfig.client_secret,
            user_agent=RedditSecretsConfig.user_agent
        )

if __name__ == '__main__':
    client = RedditClient()
    subreddit = client.reddit.subreddit('learnpython')
    top_posts = subreddit.top(limit=10)
    for post in top_posts:
        print(f"Title: {post.title}")
        print(f"Score: {post.score}")
        print(f"URL: {post.url}")
        print(f"Author: {post.author}")
        print(f"Created: {post.created_utc}")
        print("-" * 40)