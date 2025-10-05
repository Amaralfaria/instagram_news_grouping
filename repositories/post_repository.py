from infra.db import get_database

db = get_database()
posts_collection = db["posts"]

def get_all_posts():
    return list(posts_collection.find()) 

def get_n_posts(n: int):
    return list(posts_collection.find().limit(n)) 