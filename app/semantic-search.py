from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

import getpass
import os

def getMamalPetsDoc():
    return [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        )
    ]   

def validateVoyageApi():
    if not os.environ.get("VOYAGE_API_KEY"):
        os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

def getNkaDocs():
    return PyPDFLoader("resources/semantic-search/nke-10k-2023.pdf").load()

def splitDocumentText(docs):
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    ).split_documents(docs)

def embedQuery(embeddings, split):
    return embeddings.embed_query(split.page_content)

def getVoyageEmbedding():
    return VoyageAIEmbeddings(model="voyage-3")

def getVectoryStoreFromEmbedding(embeddings):
    return InMemoryVectorStore(embeddings)

def indexDocs(vectore_store, splits):
    vectore_store.add_documents(documents=splits)

def similarity_search(vector_store, search):
    return vector_store.similarity_search_with_score(search)

def embedded_search(vectore_store, embeddings, search):
    embedding = embeddings.embed_query(search)
    return vectore_store.similarity_search_by_vector(embedding)

def printResults(type, results):
    print("#" * 20)
    print(f"#{type:^18}#")
    print("#" * 20)
    print("\nRESULTS")
    print("-------\n")
    print(results)

def printResultsWithScore(type, results):
    result, score = results[0]
    printResults(type, result)
    print(f"\nSCORE: {score}")

def semantic_search():
    validateVoyageApi()
    docs = getNkaDocs()
    splits = splitDocumentText(docs)
    embeddings = getVoyageEmbedding()
    store = getVectoryStoreFromEmbedding(embeddings)
    indexDocs(store, splits)
    user_text = input("What would you like to know about Nike?\n>")
    similarity_results = similarity_search(store, user_text)
    embedded_results = embedded_search(store, embeddings, user_text)

    printResultsWithScore("Similarity", similarity_results)
    printResults("Embedded", embedded_results[0])

semantic_search()
