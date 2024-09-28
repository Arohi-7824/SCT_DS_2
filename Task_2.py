import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from warnings import filterwarnings
filterwarnings(action='ignore')

train=pd.read_csv(r"C:\Users\arohi\OneDrive\Desktop\Task_2\train.csv")
print(train.head())
print(train.shape)

print(train.describe(include="all"))

# Fill missing 'Age' values with the mean
train['Age'].fillna(train['Age'].mean(), inplace=True)
train_cleaned = train.dropna(subset=['Cabin'])
print(train_cleaned.shape)


male_ind = len(train[train['Sex'] == 'male'])
print("No of Males in Titanic:",male_ind)

female_ind = len(train[train['Sex'] == 'female'])
print("No of Females in Titanic:",female_ind)

# Plot the count of survivors (0 = Did not survive, 1 = Survived)
sns.countplot(x=train['Survived'])
plt.title('Survived vs Did Not Survive')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

total_passengers = len(train)
# Calculate the death count (Survived == 0)
death_count = train['Survived'].value_counts().iloc[0]
death_percent = round((death_count / total_passengers) * 100)

print(f"Out of {total_passengers}, {death_count} people died in the accident, which is {death_percent}%.")

# Plot count of survived and not survived, with hue for gender
sns.countplot(x=train['Survived'], hue=train['Sex'])
plt.title('Survival Count by Gender')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Create a cross-tabulation for Sex and Survived, applying row-wise percentage calculation
survival_rate_by_gender = pd.crosstab(train['Sex'], train['Survived']).apply(lambda r: round((r / r.sum()) * 100, 1), axis=1)
print(survival_rate_by_gender)

# Create a count plot for 'Survived' with 'Pclass' as the hue
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# Create a histogram for the 'Age' column, dropping any NaN values
sns.histplot(train['Age'].dropna(), kde=False, color='darkred', bins=40)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

numeric_data = train.select_dtypes(include=['float64', 'int64'])
numeric_data = numeric_data.dropna()

#correlation matrix
correlation = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.show()

sns.pairplot(train)
plt.show()

sns.scatterplot(x='Age', y='Fare', hue='Pclass', data=train)
plt.title('Survival Status by Passenger ID and Class')
plt.xlabel('Passenger ID')
plt.ylabel('Survived')
plt.show()