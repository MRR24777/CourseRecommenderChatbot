import pandas as pd
from sentence_transformers import SentenceTransformer

def preprocess_data(input_csv, output_pkl):
    print("Cargando y preprocesando datos...")
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=['Description', 'Title', 'Link'])

    # Convertir descripciones a embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df['Embedding'] = df['Description'].apply(lambda x: model.encode(x))

    # Guardar datos procesados
    df.to_pickle(output_pkl)
    print("Datos preprocesados guardados en:", output_pkl)

if __name__ == "__main__":
    preprocess_data('data/Engineering Academy Courses.csv', 'data/preprocessed_courses.pkl')
