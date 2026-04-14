from preprocessing.parser import process_row
from ingestion.csv_loader import load_csv


def run_pipeline(csv_file):
    # use csv module to read the csv file
    df = load_csv(csv_file)
    if df is None:
        return None
    # process each row using process_row function

    #run the chunking module to chunk the file content

    #run the embedding module to create embeddings for each chunk

    #store embeddings in a vector database like FAISS

    #return the doc statistics like number of documents processed, number of chunks created, etc.