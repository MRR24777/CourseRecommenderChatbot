import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

#Load the CSV file
current_path=Path(__file__).parent
print(current_path)
file_path= current_path/'Engineering Academy Courses.csv'
df=pd.read_csv(file_path)

#Show the first rows to understand the currente data structure
print(df.head())

# Data cleaning 

# Discard not neccesary data 

cols_to_keep= ['Course Title', 'Description', 'Discipline', 'Main Subject', 'Duration (min)', 'Proficiency Level']
df_cleaned=df[cols_to_keep]

#Delete duplicated rows

df_cleaned = df_cleaned.drop_duplicates()

# Fix typhofraphic errros
df_cleaned['Course Title'] = df_cleaned['Course Title'].str.lower()
df_cleaned['Description'] = df_cleaned['Description'].str.lower()
df_cleaned['Discipline'] = df_cleaned['Discipline'].str.lower()
df_cleaned['Main Subject'] = df_cleaned['Main Subject'].str.lower()
df_cleaned['Proficiency Level'] = df_cleaned['Proficiency Level'].str.lower()

#Rename the cols to ensure clarity
df_structured = df_cleaned.rename(columns={
    'Course Title': 'Course Title',
    'Description': 'Description',
    'Discipline': 'Category',
    'Duration (min)': 'Duration',
    'Proficiency Level': 'Difficulty Level'
})

#Save the data
# Step 4: Save the cleaned and structured data to a new CSV file
df_structured.to_csv(current_path/'Engineering_Academy_Courses_Structured.csv', index=False)

# Step 5: Preprocessing for the Model
# Tokenization (example using simple split, can be replaced with more advanced tokenization)
df_structured['Description_Tokenized'] = df_structured['Description'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Vectorization (example using TF-IDF)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df_structured['Description'].fillna(''))

# Save the tokenized and vectorized data to a new CSV file
tokenized_vectorized_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
tokenized_vectorized_df.to_csv(current_path/'Engineering_Academy_Courses_Tokenized_Vectorized.csv', index=False)

print("Data cleaned, structured, tokenized, and vectorized saved in 'Engineering_Academy_Courses_Tokenized_Vectorized.csv'")