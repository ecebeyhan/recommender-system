from faker import Faker
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier  # Import the classifier you want to use
from sklearn.metrics import accuracy_score, classification_report


skills_list = ['python', 'java', 'javascript', 'ruby', 'c++', 'mobileappdevelopment', 'webdevelopment', 'machinelearning', 'datascience', 'cloudcomputing', 'devops', 'databasemanagement', 'ui/uxdesign', 'fullstackdevelopment']
business_skills_list = ['businessdevelopment', 'sales', 'marketing', 'productmanagement', 'strategicplanning', 'marketresearch', 'financialmodeling', 'operationsmanagement', 'projectmanagement', 'leadership', 'negotiation', 'communicationskills', 'presentationskills', 'networking']
industry_experience_list = ['software','financialservices', 'healthcare', 'education', 'retail', 'manufacturing', 'realestate', 'construction', 'transportation', 'energy', 'media', 'entertainment', 'hospitality', 'telecommunications', 'agriculture', 'mining', 'government', 'nonprofit', 'other']
corporate_experience_list = [ 'finance', 'accounting', 'humanresources', 'sales', 'marketing', 'operations', 'legal', 'informationtechnology', 'customerservice', 'researchanddevelopment']
# Function to perform heuristic labeling
def heuristic_labeling(founder_info, recommended_cofounders):
    # Define heuristics based on your domain knowledge
    founder_info['tags'] = founder_info['Skills'] + ' ' + founder_info['Experience']
    skills = founder_info['tags'].lower()

    # Check for specific skills or experiences in founder_info
    if 'healthcare' in skills:
        return 'Healthcare_Founder'
    elif 'education' in skills:
        return 'Education_Founder'
    elif 'retail' in skills:
        return 'Retail_Founder'
    elif 'manufacturing' in skills:
        return 'Manufacturing_Founder'
    elif 'realestate' in skills:
        return 'RealEstate_Founder'
    elif 'construction' in skills:
        return 'Construction_Founder'
    elif 'transportation' in skills:
        return 'Transportation_Founder'
    elif 'energy' in skills:
        return 'Energy_Founder'
    elif 'media' in skills or 'entertainment' in skills:
        return 'Media_and_Entertainment_Founder'
    elif 'hospitality' in skills:
        return 'Hospitality_Founder'
    elif 'telecommunications' in skills:
        return 'Telecommunications_Founder'
    elif 'agriculture' in skills:
        return 'Agriculture_Founder'
    elif 'mining' in skills:
        return 'Mining_Founder'
    elif 'government' in skills:
        return 'Government_Founder'
    elif sum(skill in skills_list for skill in skills) >= 1 and sum(skill in business_skills_list for skill in skills) >= 1:
        return 'Tech_and_Business_Enthusiast_Founder'
    elif any(skill in skills for skill in skills_list) or 'informationtechnology' in skills:
        return 'Tech_Founder'
    elif any(skill in skills for skill in business_skills_list):
        return 'Business_Founder'
    

    # Check for specific skills or experiences in recommended_cofounders
    for _, cofounder_row in recommended_cofounders.iterrows():
        cofounder_skills = cofounder_row['tags'].lower()
        if 'healthcare' in cofounder_skills:
            return 'Healthcare_Founder'
        elif 'education' in cofounder_skills:
            return 'Education_Founder'
        elif 'retail' in cofounder_skills:
            return 'Retail_Founder'
        elif 'manufacturing' in cofounder_skills:
            return 'Manufacturing_Founder'
        elif 'realestate' in cofounder_skills:
            return 'RealEstate_Founder'
        elif 'construction' in cofounder_skills:
            return 'Construction_Founder'
        elif 'transportation' in cofounder_skills:
            return 'Transportation_Founder'
        elif 'energy' in cofounder_skills:
            return 'Energy_Founder'
        elif 'media' in cofounder_skills or 'entertainment' in cofounder_skills:
            return 'Media_and_Entertainment_Founder'
        elif 'hospitality' in cofounder_skills:
            return 'Hospitality_Founder'
        elif 'telecommunications' in cofounder_skills:
            return 'Telecommunications_Founder'
        elif 'agriculture' in cofounder_skills:
            return 'Agriculture_Founder'
        elif 'mining' in cofounder_skills:
            return 'Mining_Founder'
        elif 'government' in cofounder_skills:
            return 'Government_Founder'
        elif sum(skill in skills_list for skill in cofounder_skills) >= 1 and sum(skill in business_skills_list for skill in cofounder_skills) >= 1:
            return 'Tech_and_Business_Enthusiast_Founder'
        elif any(skill in cofounder_skills for skill in skills_list) or 'informationtechnology' in cofounder_skills:
            return 'Tech_Founder'
        elif any(skill in cofounder_skills for skill in business_skills_list):
            return 'Business_Founder'

    # Add more heuristics as needed
    # ...

    # If no specific heuristic conditions are met, return 'Other_Founder'
    return 'Other_Founder'

def recommend_cofounders(founder_info, co_founders_data, tfidf_vectorizer, tfidf_matrix_co_founders):
     # Concatenate 'Skills' and 'Experience' into a single column 'tags' for the founder
    founder_info['tags'] = founder_info['Skills'] + ' ' + founder_info['Experience']
    
    # Create a TF-IDF vector for the founder using the same vectorizer used for co-founders
    founder_tfidf_vector = tfidf_vectorizer.transform([founder_info['tags']])
    
    # Compute cosine similarity between the founder and all co-founders
    similarity_scores = cosine_similarity(founder_tfidf_vector, tfidf_matrix_co_founders)
    
    # Get indices of co-founders sorted by similarity score (descending order)
    co_founder_indices = similarity_scores.argsort()[0][::-1]
    
    # Get co-founder recommendations
    recommended_cofounders = co_founders_data.loc[co_founder_indices, ['Co-Founder', 'tags']]
    
    return recommended_cofounders

# # Function to generate fake skills and experience
# def generate_fake_skills_and_experience():
#     
#     # Ensure non-empty skills and experience
#     skills = ' '.join(random.sample(skills_list + business_skills_list, k=random.randint(1, 5))) or 'other'
#     experience = ' '.join(random.sample(industry_experience_list + corporate_experience_list, k=random.randint(1, 3))) or 'other'
#     # skills = random.sample(skills_list + business_skills_list, k=random.randint(1, 5)) or ['other']
#     # experience = random.sample( industry_experience_list + corporate_experience_list, k=random.randint(1, 3)) or ['other']
#     return skills, experience

# # Create a fake data generator
# fake = Faker()

#####################################################
# print("Co-founder section")

# # Generate synthetic data for co-founders
# co_founders_data = {
#     'Co-Founder': [fake.name() for _ in range(10000)],
#     'Skills': [],
#     'Experience': [],
# }

# for _ in range(10000):
#     co_skills, co_experience = generate_fake_skills_and_experience()
#     co_founders_data['Skills'].append(co_skills)
#     co_founders_data['Experience'].append(co_experience)

# # Convert data to a pandas DataFrame
# df_co_founders = pd.DataFrame(co_founders_data)
# # Writing co-founder data to a CSV file
# df_co_founders.to_csv('co_founders_data.csv', index=False)

# Reading co-founder data from CSV
train_data = pd.read_csv('train_data.csv')
eval_data = pd.read_csv('eval_data.csv')

# Shuffle co-founder data
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
eval_data = eval_data.sample(frac=1, random_state=42).reset_index(drop=True)

train_data['tags'] = train_data['Skills'] + ' ' + train_data['Experience']
last_version = train_data[['Co-Founder', 'tags']]

eval_data['tags'] = eval_data['Skills'] + ' ' + eval_data['Experience']
last_version_eval = eval_data[['Co-Founder', 'tags']]

# Create a TF-IDF vectorizer for co-founders
tfidf_vectorizer_co_founders = TfidfVectorizer(stop_words='english')
tfidf_vectorizer_co_founders2 = TfidfVectorizer(stop_words='english')
tfidf_matrix_co_founders = tfidf_vectorizer_co_founders.fit_transform(last_version['tags'])
tfidf_matrix_co_founders_eval = tfidf_vectorizer_co_founders2.fit_transform(last_version_eval['tags'])

# Example usage:
# Reading founder data from CSV
founders_data_from_csv = pd.read_csv('founders_data.csv')
# Assuming you have a specific founder's information, e.g., the second founder in the dataset
# Dynamic generation of founder information
founder_info = founders_data_from_csv.iloc[5]
print(founder_info)

# Call the recommend_cofounders function using the training data
recommendations_train = recommend_cofounders(founder_info, last_version, tfidf_vectorizer_co_founders, tfidf_matrix_co_founders)


# Apply heuristic labeling to the recommended cofounders in the training set
last_version['predicted_label'] = recommendations_train.apply(lambda row: heuristic_labeling(founder_info, row), axis=1)
print(recommendations_train.head())

# Train your machine learning model using the labeled training set
# Example using RandomForestClassifier; replace with your chosen classifier
clf = RandomForestClassifier(random_state=42)
X_train = tfidf_matrix_co_founders
y_train = last_version['predicted_label']
clf.fit(X_train, y_train)

# Call the recommend_cofounders function using the evaluation data
recommendations_eval = recommend_cofounders(founder_info, last_version_eval, tfidf_vectorizer_co_founders2, tfidf_matrix_co_founders_eval)

# Apply heuristic labeling to the recommended cofounders in the evaluation set
last_version_eval['predicted_label'] = recommendations_eval.apply(lambda row: heuristic_labeling(founder_info, row), axis=1)
print(recommendations_eval.head())

# Evaluate the model's performance on the labeled evaluation set
X_eval = tfidf_matrix_co_founders_eval
y_eval = last_version_eval['predicted_label']
y_pred = clf.predict(X_eval)

# Print classification report or other relevant evaluation metrics
print("Classification Report:")
print(classification_report(y_eval, y_pred))

