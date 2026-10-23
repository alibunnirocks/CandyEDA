#import necessary libraries and load the dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv("candy-data.csv")

#1. explore data
print("First 5 rows of the dataframe:")
print(df.head(5))
print("Shape of the dataframe:")
print(df.shape)

#2. checking for missing values and duplicates
print("Missing values in each column:")
print(df.isnull().sum())
print("Number of unique values in each column:")
print(df.nunique())

#3. descriptive statistics
avgSugar = df['sugarpercent'].mean()
print("The average sugar percentage is: ",format(avgSugar, ".5f"))
avgWin = df['winpercent'].mean()
print("The average win percentage is: ",format(avgWin, ".5f"))

print("The range of sugar percentage is: ", df['sugarpercent'].max() - df['sugarpercent'].min())
print("The range of win percentage is: ", df['winpercent'].max() - df['winpercent'].min())
#part 4 is a scatter plot, we also did a scatterplot with a line of best fit.
#To see the plot with line of best fit, comment out part 4.
#4 visualizing data
sns.set(style="whitegrid")
#scatter plot of sugar percentage vs win percentage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sugarpercent', y='winpercent', data=df)
plt.title('Sugar Percentage vs Win Percentage')
plt.xlabel('Sugar Percentage')
plt.ylabel('Win Percentage')
plt.show()

#5. linear regression model
X = df[['sugarpercent']]
y = df['winpercent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 6. plot the scatter plot with the line of best fit
plt.figure(figsize=(10, 6))

sns.scatterplot(x='sugarpercent', y='winpercent', data=df)

# predict win percentages using the trained model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X)

# sort x-values so the line plots correctly
sorted_indices = np.argsort(X['sugarpercent'].values)

plt.plot(
    X['sugarpercent'].values[sorted_indices],
    y_pred[sorted_indices],
    color='red',
    label='Line of Best Fit'
)

plt.title('Sugar Percentage vs Win Percentage')
plt.xlabel('Sugar Percentage')
plt.ylabel('Win Percentage')
plt.legend()
plt.show()