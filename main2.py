from faker import Faker
import pandas as pd
import random

# Function to generate fake skills
def generate_fake_skills():
    skills_list = ['Python', 'Java', 'JavaScript', 'Ruby', 'Machine Learning', 'Data Analysis', 'Marketing', 'Sales', 'Product Management', 'Finance']
    return random.sample(skills_list, k=random.randint(1, 5))

# Function to generate fake experience
def generate_fake_experience():
    experience_list = ['Startup', 'Tech Company', 'Finance Industry', 'Marketing Agency', 'Healthcare', 'E-commerce']
    return random.sample(experience_list, k=random.randint(1, 3))

fake = Faker()

# Generate synthetic data for founders
founders_data = {
    'Founder': [fake.name() for _ in range(1000)],
    'Skills': [generate_fake_skills() for _ in range(1000)],
    'Experience': [generate_fake_experience() for _ in range(1000)],
}

# Convert data to a pandas DataFrame
df_founders = pd.DataFrame(founders_data)

print(df_founders)
