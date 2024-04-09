import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

# Load the data
data = pd.read_csv("dataset.csv")

# Preprocessing
# Drop rows with missing values
data.dropna(inplace=True)

# Encode categorical labels
label_encoder = LabelEncoder()
data['internalStatus'] = label_encoder.fit_transform(data['internalStatus'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['externalStatus'], data['internalStatus'], test_size=0.2, random_state=42)

# Feature extraction
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Model Development
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
