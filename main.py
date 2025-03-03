from preprocess import preprocess_data
from index import create_index
from train_gpt2 import train_gpt2
from predict import generate_response

if __name__ == "__main__":
    print("Iniciando pipeline de recomendación de cursos...")
    
    # Paso 1: Preprocesar datos
    #preprocess_data('data/Engineering Academy Courses.csv', 'data/preprocessed_courses.pkl')
    
    # Paso 2: Indexar embeddings
    #create_index('data/preprocessed_courses.pkl', 'data/courses_index.faiss', 'data/indexed_courses.csv')
    
    # Paso 3: Entrenar GPT-2
    #train_gpt2('data/indexed_courses.csv', 'data/trained_gpt2')
    
    # Paso 4: Interacción continua con el chatbot
    while True:
        query = input("¿Qué quieres aprender? ")
        if query.lower() in ['salir', 'exit', 'quit']:
            print("Gracias por usar el recomendador de cursos. ¡Hasta luego!")
            break
        respuesta = generate_response(query)
        print("Recomendación de la IA:\n", respuesta)