import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame({
    'Rating': ['High','Medium','Low','Medium'],
    'Delivery': ['Fast','Fast','Slow','Slow'],
    'Quality': ['Good','Average','Poor','Average'],
    'Sentiment': ['Positive','Positive','Negative','Negative']
})

le = LabelEncoder()

for column in data.columns:
    data[column] = le.fit_transform(data[column])

X = data[['Rating','Delivery','Quality']]
y = data['Sentiment']

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,y)

new = pd.DataFrame([[0,0,1]], columns=['Rating','Delivery','Quality'])
print("Prediction:", model.predict(new))

