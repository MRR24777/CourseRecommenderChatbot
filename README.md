# CourseRecommenderChatbot

This repository contains a ChatGPT/Copilot-style chatbot application that recommends courses based on user queries. The application uses pre-trained language models and information retrieval techniques to provide accurate and enriched course recommendations with additional information from various sources.

## Features
- **Course Recommendations**: Provides course recommendations with links and descriptions.
- **Additional Information**: Enriches responses with information from Wikipedia, Stack Exchange, ArXiv, and Open Library.
- **GPT-2 Training**: Trains a GPT-2 model with course descriptions to improve response accuracy.
- **Embedding Search**: Uses FAISS and Sentence Transformers to search for relevant courses.

## Main Files
- `chatbot.py`: Chatbot configuration and training.
- `external_info.py`: Fetching additional information from various sources.
- `index 1.py`: Creating the FAISS index from embeddings.
- `main 1.py`: Running the complete course recommendation pipeline.
- `predict 1.py`: Generating responses and course recommendations.
- `preprocess 1.py`: Preprocessing course data and generating embeddings.
- `train_gpt2 1.py`: Training the GPT-2 model with course descriptions.

## Execution Instructions
1. **Preprocess the data**: `python preprocess 1.py`
2. **Create the FAISS index**: `python index 1.py`
3. **Train the GPT-2 model**: `python train_gpt2 1.py`
4. **Run the complete pipeline**: `python main 1.py`

## Requirements
- Python 3.8+
- Libraries: transformers, torch, pandas, faiss, sentence-transformers, requests, nltk, certifi

## Contributions
Contributions are welcome! If you find any issues or have improvements, feel free to open an issue or submit a pull request.
