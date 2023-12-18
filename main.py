import pandas as pd  # Add this line to import pandas
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data (replace this with your actual data)
founder_data = {
    'Founder': ['Founder1', 'Founder2', 'Founder3'],
    'Skills': ['Python, Java, Machine Learning', 'JavaScript, C++, SQL', 'Ruby, PHP, HTML'],
    'Experience': ['Marketing, Sales', 'Product Management', 'Finance, Operations'],
}

# Convert data to a pandas DataFrame
df_founders = pd.DataFrame(founder_data)

# Combine text-based features
df_founders['TextFeatures'] = df_founders['Skills'] + ' ' + df_founders['Experience']

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df_founders['TextFeatures'])

# Display the TF-IDF matrix
print(tfidf_matrix.toarray())
