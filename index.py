import pandas as pd
import faiss
import numpy as np

def create_index(input_pkl, output_faiss, output_csv):
    print("Creando índice FAISS...")
    df = pd.read_pickle(input_pkl)

    embeddings_matrix = np.stack(df['Embedding'].values)
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
    index.add(embeddings_matrix)

    faiss.write_index(index, output_faiss)
    df.to_csv(output_csv, index=False)
    print("Indexación completada. Índice guardado en:", output_faiss)

if __name__ == "__main__":
    create_index('data/preprocessed_courses.pkl', 'data/courses_index.faiss', 'data/indexed_courses.csv')
