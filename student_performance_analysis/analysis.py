import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
data = pd.read_csv("dataset/student_data.csv")

# ===============================
#         DATA CLEANING
# ===============================

print("Initial Data Shape:", data.shape)
# 1️⃣ Handle Missing Names
data['name'] = data['name'].fillna("Unknown")

# 2️⃣ Clean Gender Column
data['gender'] = data['gender'].replace({
    'M': 'Male',
    'F': 'Female'
})
data['gender'] = data['gender'].fillna("Unknown")

# 3️⃣ Fix Department Spellings
data['department'] = data['department'].replace({
    'Scince': 'Science',
    'Commerse': 'Commerce'
})

# 4️⃣ Remove Duplicate Records
data.drop_duplicates(inplace=True)

# 5️⃣ Fix Invalid Numerical Values
data.loc[data['age'] < 0, 'age'] = data['age'].median()
data.loc[data['attendance'] > 100, 'attendance'] = 100
data.loc[data['hours_studied'] > 10, 'hours_studied'] = 10

# 6️⃣ Cap Subject Scores to 100
subjects = ['math_score', 'science_score', 'english_score']
for col in subjects:
    data.loc[data[col] > 100, col] = 100

# 7️⃣ Recalculate Total Score
data['total_score'] = data[['math_score', 'science_score', 'english_score']].sum(axis=1)

# Final Check
print("\nCleaned Data Shape:", data.shape)
print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

print("\nFirst 5 Cleaned Records:")
print(data.head())

# ===============================
#         DATA ANALYSIS
# ===============================

print("\n--- BASIC STATISTICS ---")
print(data[['math_score', 'science_score', 'english_score', 'total_score']].describe())

# 1️⃣ Average Score of Each Subject
print("\n--- AVERAGE SUBJECT SCORES ---")
print("Math Average:", data['math_score'].mean())
print("Science Average:", data['science_score'].mean())
print("English Average:", data['english_score'].mean())

# 2️⃣ Gender-wise Average Total Score
print("\n--- GENDER-WISE PERFORMANCE ---")
gender_avg = data.groupby('gender')['total_score'].mean()
print(gender_avg)

# 3️⃣ Department-wise Average Total Score
print("\n--- DEPARTMENT-WISE PERFORMANCE ---")
dept_avg = data.groupby('department')['total_score'].mean()
print(dept_avg)

# 4️⃣ Study Hours vs Total Score (Correlation)
print("\n--- STUDY HOURS VS TOTAL SCORE ---")
correlation = data['hours_studied'].corr(data['total_score'])
print("Correlation:", correlation)

# 5️⃣ Grade Distribution
print("\n--- GRADE DISTRIBUTION ---")
print(data['grade'].value_counts())

# ===============================
#        DATA VISUALIZATION
# ===============================

# 1️⃣ Grade Distribution (Bar Chart)
plt.figure(figsize=(6,4))
sns.countplot(x='grade', data=data, palette='Set2')
plt.title('Grade Distribution of Students')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.show()

# 2️⃣ Department-wise Average Total Score (Bar Chart)
plt.figure(figsize=(6,4))
dept_avg = data.groupby('department')['total_score'].mean().reset_index()
sns.barplot(x='department', y='total_score', data=dept_avg, palette='Set3')
plt.title('Department-wise Average Total Score')
plt.ylabel('Average Total Score')
plt.show()

# 3️⃣ Gender-wise Average Total Score (Bar Chart)
plt.figure(figsize=(6,4))
gender_avg = data.groupby('gender')['total_score'].mean().reset_index()
sns.barplot(x='gender', y='total_score', data=gender_avg, palette='Set1')
plt.title('Gender-wise Average Total Score')
plt.ylabel('Average Total Score')
plt.show()

# 4️⃣ Study Hours vs Total Score (Scatter Plot with Regression)
plt.figure(figsize=(6,4))
sns.regplot(x='hours_studied', y='total_score', data=data, scatter_kws={'s':50}, line_kws={'color':'red'})
plt.title('Study Hours vs Total Score')
plt.xlabel('Hours Studied')
plt.ylabel('Total Score')
plt.show()

# 5️⃣ Heatmap of Correlation Between Scores
plt.figure(figsize=(6,5))
score_corr = data[['math_score','science_score','english_score','total_score']].corr()
sns.heatmap(score_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Subject Scores')
plt.show()

#data.head()	Top 5 rows (default)
#data.tail()	Last 5 rows (default)
#data.sample(n)	Random n rows
#print(data)	All rows (small dataset)