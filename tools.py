from pymongo import MongoClient

import os
import logging

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_mongodb.retrievers import MongoDBAtlasHybridSearchRetriever


# Setup AWS and Bedrock client
def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))


def create_embeddings(client):
    return BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=client)


# Initialize everything
bedrock_client = get_bedrock_client()
bedrock_embeddings = create_embeddings(bedrock_client)


class Tools:
    """
    Tools for agents to use, such as MongoDB search and web search.
    """

    def __init__(self, mongodb_uri=None, mongodb_db=None, mongodb_collection=None):
        self.mongodb_uri = mongodb_uri
        self.mongodb_db = mongodb_db
        self.mongodb_collection = mongodb_collection

        if mongodb_uri:
            self.mongo_client = MongoClient(mongodb_uri)
        else:
            self.mongo_client = None

    def search_mongodb(self, query):
        """
        Performs a hybrid search using MongoDB Atlas, combining both full-text and vector-based searches
        to retrieve relevant documents from multiple data sources.

        Args:
            query (str): The search query string to perform the hybrid search.

        Returns:
            str: A list of document contents retrieved by the hybrid search, with each document's content
                represented as a string.
        """

        if not self.mongo_client:
            raise ValueError("MongoDB client is not initialized.")

        logging.info("Connected to MongoDB...")

        # Use class properties for DB and Collection
        database = self.mongo_client[self.mongodb_db]
        collection = database[self.mongodb_collection]

        vector_store = MongoDBAtlasVectorSearch(
            text_key="About Place",
            embedding_key="details_embedding",
            index_name="vector_index",
            embedding=bedrock_embeddings,
            collection=collection,
        )

        retriever = MongoDBAtlasHybridSearchRetriever(
            vectorstore=vector_store,
            search_index_name="travel_text_search_index",
            top_k=10,
        )

        documents = retriever.invoke(query)
        return [doc.page_content for doc in documents]
