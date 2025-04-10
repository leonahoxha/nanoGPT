
import pandas as pd
import numpy as np

df = pd.read_csv("data/salary_data/mock_salary_data.csv")

# Extract roles and salary ranges from training data
roles = df['Role'].unique().tolist()
salary_min = df['Salary'].min()
salary_max = df['Salary'].max()

# Sample 10 random persons from test set (simulate secret set)
test_df = df.sample(n=10, random_state=42)

print("🎲 Baseline: Random Role & Salary Guesses\n")
match_count = 0
for _, row in test_df.iterrows():
    name = row['Name']
    true_role = row['Role']
    true_salary = row['Salary']

    guessed_role = np.random.choice(roles)
    guessed_salary = np.random.randint(salary_min, salary_max + 1)

    guess = f"Name: {name}, Role: {guessed_role}, Salary: {guessed_salary}"
    truth = f"Name: {name}, Role: {true_role}, Salary: {true_salary}"
    print(f"{name} → random says: {guess}\n/ ground truth is: {truth}")
    print("-" * 100)

    if guessed_role == true_role and guessed_salary == true_salary:
        match_count += 1

print(f"✅ Exact Match Count: {match_count}/10")
