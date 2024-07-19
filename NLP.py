from flask import Flask, request, render_template, redirect
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
from groq import Groq

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__))
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Groq client with API key from environment variables
os.environ['GROQ_API_KEY'] = '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'
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load pre-trained NLP model
nlp = spacy.load('en_core_web_sm')

def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

def create_combined_sentence(legacy, new):
    prompt = f"Combine the following two requirements using air force terminology (aircraft, fueling, flight, aviators) into one coherent and grammatically correct sentence, make it short, and only print the combined requirement DO NOT SAY here is the combined requirement:\n\nLegacy Requirement: {legacy}\nNew Requirement: {new}\n\nCombined Sentence:"
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'new_file' not in request.files:
            return redirect(request.url)
        new_file = request.files['new_file']
        if new_file.filename == '':
            return redirect(request.url)
        if new_file:
            new_file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_file.filename)
            new_file.save(new_file_path)

            with open('legacy_requirements.txt', 'r') as file:
                legacy_requirements = file.readlines()

            with open(new_file_path, 'r') as file:
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

            # Apply stop words removal
            requirements['description'] = requirements['description'].apply(remove_stopwords)

            # Vectorization requirement descriptions
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(requirements['description'])

            # Cluster requirements to find similarities
            num_clusters = 5  # Define number of clusters, adjust as needed
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            requirements['cluster'] = kmeans.fit_predict(X)

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
                                integrated_requirements.append(combined_sentence.strip())
                    if len(legacy) == 0:
                        for n_req in new:
                            integrated_requirements.append(n_req.strip())

            # Ensure only 10 integrated requirements are printed
            integrated_requirements = integrated_requirements[:10]

            # Save integrated requirements to a CSV file
            output_df = pd.DataFrame(integrated_requirements, columns=['Integrated Requirements'])
            output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'integrated_requirements.csv')
            output_df.to_csv(output_csv_path, index=False)

    # Load integrated requirements from the CSV file
    integrated_requirements = []
    output_csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'integrated_requirements.csv')
    if os.path.exists(output_csv_path):
        integrated_requirements = pd.read_csv(output_csv_path)['Integrated Requirements'].tolist()

    return render_template('index.html', integrated_requirements=integrated_requirements)

if __name__ == '__main__':
    app.run(debug=True)
