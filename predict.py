import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from chatbot import get_chatbot_response
from external_info import get_combined_info

def load_models():
    print("Cargando modelos...")
    df = pd.read_csv('data/indexed_courses.csv')
    index = faiss.read_index('data/courses_index.faiss')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = GPT2Tokenizer.from_pretrained("data/trained_gpt2")
    model = GPT2LMHeadModel.from_pretrained("data/trained_gpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    return df, index, embedding_model, generator

def recommend_courses(query, df, index, embedding_model, top_k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, top_k)
    return df.iloc[indices[0]][['Title', 'Description', 'Link']]

def generate_response(query):
    df, index, embedding_model, generator = load_models()
    cursos = recommend_courses(query, df, index, embedding_model)
    
    info_adicional = get_combined_info(query)
    mensaje = f"Información adicional sobre '{query}':\n{info_adicional}\n\n"
    mensaje += f"Los mejores cursos para '{query}' son:\n"
    
    for _, row in cursos.iterrows():
        mensaje += f"- {row['Title']}: {row['Description']} ({row['Link']})\n"
    
    # Asegurarse de que la longitud del mensaje no exceda los 1024 tokens
    max_length = 1024 - 50  # Reservar espacio para los nuevos tokens generados
    input_ids = generator.tokenizer.encode(mensaje, return_tensors='pt')
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]
        mensaje = generator.tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    response = generator(mensaje, max_new_tokens=50, num_return_sequences=1)
    return response[0]['generated_text']