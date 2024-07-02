import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

Hours=[2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7,4.8,3.8,6.9,7.8]
Scores=[21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30,54,35,76,86]

df = pd.DataFrame({
    'Hours': Hours,
    'Scores': Scores
})

X = df[['Hours']]
y = df['Scores']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #split data

model = LinearRegression()
model.fit(X_train, y_train) #create and train model

y_pred = model.predict(X_test)
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df)

hours_studied = pd.DataFrame({'Hours': [9.25]})
predicted_score = model.predict(hours_studied)
print(f'Predicted score for a student who studies 9.25 hours/day: {predicted_score[0]}')

plt.scatter(X, y, color='red',marker='x')
plt.plot(X, model.predict(X), color='blue')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.title('Hours Studied vs Percentage Score')
plt.show()