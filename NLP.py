import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from groq import Groq

# Initialize Groq client with API key from environment variables
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Load old and new requirements from text files
with open('legacy_requirements.txt', 'r') as file:
    legacy_requirements = file.readlines()

with open('new_requirements.txt', 'r') as file:
    new_requirements = file.readlines()

# Strip any extra spaces and newlines
legacy_requirements = [req.strip() for req in legacy_requirements if req.strip()]
new_requirements = [req.strip() for req in new_requirements if req.strip()]

# Convert lists to DataFrame
legacy_requirements_df = pd.DataFrame(legacy_requirements, columns=['description'])
new_requirements_df = pd.DataFrame(new_requirements, columns=['description'])

# Combine into a single DataFrame
legacy_requirements_df['source'] = 'legacy'
new_requirements_df['source'] = 'new'
requirements = pd.concat([legacy_requirements_df, new_requirements_df], ignore_index=True)

# Load pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

# Define a function to remove stop words
def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

# Apply stop words removal
requirements['description'] = requirements['description'].apply(remove_stopwords)

# Vectorize requirement descriptions
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(requirements['description'])

# Cluster requirements to find similarities
num_clusters = 5  # Define number of clusters, adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
requirements['cluster'] = kmeans.fit_predict(X)

# Function to create meaningful combined sentences using Groq
def create_combined_sentence(legacy, new):
    prompt = f"Combine the following two requirements into one coherent and grammatically correct sentence, and only print the combined requirement without saying here is the combined requirement:\n\nLegacy Requirement: {legacy}\nNew Requirement: {new}\n\nCombined Sentence:"
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    combined_sentence = chat_completion.choices[0].message.content.strip()
    return combined_sentence

# Group requirements by clusters and sources
grouped_requirements = requirements.groupby(['cluster', 'source'])['description'].apply(list).unstack().fillna('')

# Integrate requirements by cluster
integrated_requirements = []
for cluster in grouped_requirements.index:
    legacy = grouped_requirements.loc[cluster, 'legacy']
    new = grouped_requirements.loc[cluster, 'new']
    if new:
        for l_req in legacy:
            for n_req in new:
                combined_sentence = create_combined_sentence(l_req, n_req)
                if combined_sentence:
                    integrated_requirements.append(f'"{combined_sentence.strip()}"')
        if len(legacy) == 0:
            for n_req in new:
                integrated_requirements.append(f'"{n_req.strip()}"')

# Ensure only 10 integrated requirements are printed
integrated_requirements = integrated_requirements[:10]

# Print integrated requirements
integrated_requirements_str = "\n".join(integrated_requirements)
print("\nIntegrated Requirements:\n", integrated_requirements_str)
