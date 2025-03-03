import requests
import certifi
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def get_external_info(topic):
    url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
    try:
        response = requests.get(url, verify=certifi.where())
        if response.status_code == 200:
            data = response.json()
            return data.get('extract', 'No se encontró información adicional.')
        else:
            synonyms = get_synonyms(topic)
            for synonym in synonyms:
                url = f"https://es.wikipedia.org/api/rest_v1/page/summary/{synonym.replace(' ', '_')}"
                response = requests.get(url, verify=certifi.where())
                if response.status_code == 200:
                    data = response.json()
                    return data.get('extract', 'No se encontró información adicional.')
            return 'No se encontró información adicional.'
    except requests.exceptions.SSLError as e:
        return f"Error de SSL: {e}"

def get_stack_exchange_info(topic):
    url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={topic}&site=stackoverflow"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['items']:
                return data['items'][0]['title'] + ": " + data['items'][0]['link']
            else:
                return 'No se encontró información adicional en Stack Exchange.'
        else:
            return 'No se encontró información adicional en Stack Exchange.'
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: {e}"

def get_arxiv_info(topic):
    url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results=1"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.text
            start_title = data.find('<title>') + len('<title>')
            end_title = data.find('</title>', start_title)
            title = data[start_title:end_title]
            
            start_link = data.find('<id>') + len('<id>')
            end_link = data.find('</id>', start_link)
            link = data[start_link:end_link]
            
            return f"{title}: {link}"
        else:
            return 'No se encontró información adicional en ArXiv.'
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: {e}"

def get_open_library_info(topic):
    url = f"https://openlibrary.org/search.json?q={topic}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['docs']:
                return data['docs'][0]['title'] + ": " + f"https://openlibrary.org{data['docs'][0]['key']}"
            else:
                return 'No se encontró información adicional en Open Library.'
        else:
            return 'No se encontró información adicional en Open Library.'
    except requests.exceptions.RequestException as e:
        return f"Error de conexión: {e}"

def get_combined_info(topic):
    wikipedia_info = get_external_info(topic)
    stack_exchange_info = get_stack_exchange_info(topic)
    arxiv_info = get_arxiv_info(topic)
    open_library_info = get_open_library_info(topic)
    
    combined_info = f"Wikipedia: {wikipedia_info}\n"
    combined_info += f"Stack Exchange: {stack_exchange_info}\n"
    combined_info += f"ArXiv: {arxiv_info}\n"
    combined_info += f"Open Library: {open_library_info}\n"
    
    return combined_info