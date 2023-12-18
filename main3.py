from faker import Faker
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import random

# Function to generate fake skills and experience
def generate_fake_skills_and_experience():
    skills_list = ['python', 'java', 'javaScript', 'ruby', 'c++', 'mobile app development', 'web development', 'machine learning', 'data science', 'cloud computing', 'devOps', 'database management', 'UI/UX design', 'full stack development']
    business_skills_list = ['business development', 'sales', 'marketing', 'product management', 'strategic planning', 'market research', 'financial modeling', 'operations management', 'project management', 'leadership', 'negotiation', 'communication skills', 'presentation skills', 'networking']
    industry_experience_list = ['software','financial services', 'healthcare', 'education', 'retail', 'manufacturing', 'real estate', 'construction', 'transportation', 'energy', 'media', 'entertainment', 'hospitality', 'telecommunications', 'agriculture', 'mining', 'government', 'non-profit', 'other']
    corporate_experience_list = [ 'finance', 'accounting', 'human resources', 'sales', 'marketing', 'operations', 'legal', 'information technology', 'customer service', 'research and development']
    

    skills = random.sample(skills_list + business_skills_list, k=random.randint(1, 5))or ['other']
    experience = random.sample( industry_experience_list + corporate_experience_list, k=random.randint(1, 3))or ['other']
    return skills, experience

# Create a fake data generator
fake = Faker()

# Generate synthetic data for founders
founders_data = {
    'Founder': [fake.name() for _ in range(1000)],
    'Skills': [],
    'Experience': [],
}

for _ in range(1000):
    skills, experience = generate_fake_skills_and_experience()
    founders_data['Skills'].append(skills)
    founders_data['Experience'].append(experience)


# Convert data to a pandas DataFrame
df_founders = pd.DataFrame(founders_data)

# Display the DataFrame
print(df_founders)

# Create a TF-IDF vectorizer for skills
skills_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
skills_tfidf_matrix = skills_tfidf_vectorizer.fit_transform(df_founders['Skills'].apply(lambda x: ' '.join(x if isinstance(x, list) else [])))

# Create a TF-IDF vectorizer for experience
experience_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
experience_tfidf_matrix = experience_tfidf_vectorizer.fit_transform(df_founders['Experience'].apply(lambda x: ' '.join(x if isinstance(x, list) else [])))

# Display the TF-IDF matrices
print("Skills TF-IDF matrix:")
print(skills_tfidf_matrix.toarray())

print("\nExperience TF-IDF matrix:")
print(experience_tfidf_matrix.toarray())

#####################################################
print("Co-founder section")
# Generate synthetic data for co-founders
co_founders_data = {
    'Co-Founder': [fake.name() for _ in range(2000)],
    'Skills': [],
    'Experience': [],
}

for _ in range(2000):
    co_skills, co_experience = generate_fake_skills_and_experience()
    co_founders_data['Skills'].append(co_skills)
    co_founders_data['Experience'].append(co_experience)

# Convert data to a pandas DataFrame
df_co_founders = pd.DataFrame(co_founders_data)
# Writing co-founder data to a CSV file
df_co_founders.to_csv('co_founders_data.csv', index=False)
# Reading co-founder data from CSV
co_founders_data_from_csv = pd.read_csv('co_founders_data.csv')


# Display the DataFrame
print(co_founders_data_from_csv)

# Create a TF-IDF vectorizer for skills
co_skills_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
co_skills_tfidf_matrix = co_skills_tfidf_vectorizer.fit_transform(co_founders_data_from_csv['Skills'].apply(lambda x: ' '.join(x if isinstance(x, list) else [])))

# Create a TF-IDF vectorizer for experience
co_experience_tfidf_vectorizer = TfidfVectorizer(stop_words='english')
co_experience_tfidf_matrix = co_experience_tfidf_vectorizer.fit_transform(co_founders_data_from_csv['Experience'].apply(lambda x: ' '.join(x if isinstance(x, list) else [])))

# Display the TF-IDF matrices
print("Skills TF-IDF matrix:")
print(co_skills_tfidf_matrix.toarray())

print("\nExperience TF-IDF matrix:")
print(co_experience_tfidf_matrix.toarray())

