import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


np.random.seed(42)
# Load the dataset
data = {
    'age': np.random.randint(20, 80, 100),
    'income': np.random.randint(20000, 100000, 100),
    'education_years': np.random.randint(8, 20, 100),
    'buy': np.random.choice(['yes', 'no'], 100)
}
df = pd.DataFrame(data)

#Prepare features x and target y
x = df[['age', 'income', 'education_years']]
y = df['buy']

#Encode the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Create and train the decision tree classifier
clf = DecisionTreeClassifier(max_depth= 5, min_samples_leaf= 2, min_samples_split= 5,  random_state=42)

dt_model = clf.fit(x_train, y_train)

#Make predictions on the test set
y_pred = dt_model.predict(x_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
report = classification_report(y_test, y_pred)

# Example of making a prediction for a new customer
new_customer = np.array([[35, 60000, 16]])
prediction = dt_model.predict(new_customer)
print("\nPrediction for new customer:", label_encoder.inverse_transform(prediction)[0])

# Feature importance
feature_importance = pd.DataFrame({
    'feature': x.columns,
    'importance': dt_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='importance', ascending=False))


from dtreeviz.trees import dtreeviz

viz = dtreeviz(
    dt_model,
    x_train,
    y_train,
    target_name='buy',
    feature_names=list(x.columns),
    class_names=list(label_encoder.classes_)
)

viz.save("decision_tree_detailed.svg")