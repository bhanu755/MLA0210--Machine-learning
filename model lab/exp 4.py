import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Step 1: Create Dataset
data = pd.DataFrame({
    'Rating': ['High','Medium','Low','Medium'],
    'Delivery': ['Fast','Fast','Slow','Slow'],
    'Quality': ['Good','Average','Poor','Average'],
    'Sentiment': ['Positive','Positive','Negative','Negative']
})

# Step 2: Encode Data
le = LabelEncoder()

for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Step 3: Split Features and Target
X = data[['Rating','Delivery','Quality']]
y = data['Sentiment']

# Step 4: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X, y)

# Step 5: Predict New Review
new = pd.DataFrame([[0,0,1]], columns=['Rating','Delivery','Quality'])
prediction = model.predict(new)

print("Prediction:", prediction)
