from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import spacy

# Cargar el modelo de spaCy
nlp = spacy.load('en_core_web_sm')

# Crear y entrenar el chatbot
chatbot = ChatBot('CourseRecommender')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train('chatterbot.corpus.spanish')

def get_chatbot_response(query):
    response = chatbot.get_response(query)
    return response