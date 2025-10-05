from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_database():
    connection_string = os.getenv("MONGO_CONNECTION_STRING")
    database = os.getenv("MONGO_DATABASE_NAME")
    client = MongoClient(connection_string)
    return client[database]