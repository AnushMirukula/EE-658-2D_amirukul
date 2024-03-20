#!/usr/bin/env python
# coding: utf-8

# In[10]:



#part 1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Handle null values
train_df = train_df.dropna()
test_df = test_df.dropna()

# Combine the datasets for label encoding
combined_df = pd.concat([train_df, test_df], axis=0)

# Print the shape of the combined dataset
print("Combined dataset shape:", combined_df.shape)

# Convert categorical variables to numerical format
label_encoder = LabelEncoder()
combined_df['Gender'] = label_encoder.fit_transform(combined_df['Gender'])
combined_df['Ever_Married'] = label_encoder.fit_transform(combined_df['Ever_Married'])
combined_df['Graduated'] = label_encoder.fit_transform(combined_df['Graduated'])
combined_df['Profession'] = label_encoder.fit_transform(combined_df['Profession'])
combined_df['Spending_Score'] = label_encoder.fit_transform(combined_df['Spending_Score'])
combined_df['Var_1'] = label_encoder.fit_transform(combined_df['Var_1'])

# Split the combined dataset back into train and test
train_df = combined_df[:len(train_df)].copy()  # Use .copy() to avoid the warning
test_df = combined_df[len(train_df):].copy()

# Print the shape of the training and testing datasets
print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

# Encode the target variable for the classification task in the training set
train_df.loc[:, 'Segmentation'] = label_encoder.fit_transform(train_df['Segmentation'])

# Print the shape of the training dataset after encoding
print("Training dataset shape after encoding:", train_df.shape)

# Split data into features and target variable
X_train = train_df.drop(['ID', 'Segmentation'], axis=1)
y_train = train_df['Segmentation']

X_test = test_df.drop(['ID', 'Segmentation'], axis=1)

# Print the shapes of feature matrices and target variable for training and testing
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)


# In[11]:


#part 2

import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Initialize weights and biases
input_size = X_train.shape[1]
hidden_size = 4  # You can adjust the number of hidden units
output_size = len(np.unique(y_train))

weights_input_hidden = np.random.randn(input_size, hidden_size)
biases_input_hidden = np.zeros((1, hidden_size))

weights_hidden_output = np.random.randn(hidden_size, output_size)
biases_hidden_output = np.zeros((1, output_size))

# Implement forward propagation
def forward_propagation(X):
    hidden_input = np.dot(X, weights_input_hidden) + biases_input_hidden
    hidden_output = sigmoid(hidden_input)
    
    output_input = np.dot(hidden_output, weights_hidden_output) + biases_hidden_output
    output_probabilities = sigmoid(output_input)
    
    return hidden_output, output_probabilities

# Apply forward propagation to the training data
hidden_output, output_probabilities = forward_propagation(X_train)

# Check the shapes of intermediate outputs
print("Hidden Output Shape:", hidden_output.shape)
print("Output Probabilities Shape:", output_probabilities.shape)


# In[15]:


#part 3


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Evaluate the MLPClassifier on the training data
accuracy_mlp = accuracy_score(y_train, y_pred_train)
confusion_matrix_mlp = confusion_matrix(y_train, y_pred_train)
classification_report_mlp = classification_report(y_train, y_pred_train)

# Print the results
print("MLPClassifier Results on Training Data:")
print("Accuracy:", accuracy_mlp)
print("Confusion Matrix:\n", confusion_matrix_mlp)
print("Classification Report:\n", classification_report_mlp)


# In[16]:


#[part4]
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Evaluate the Logistic Regression model on the training data
accuracy_logreg = accuracy_score(y_train, y_pred_train_logreg)
confusion_matrix_logreg = confusion_matrix(y_train, y_pred_train_logreg)
classification_report_logreg = classification_report(y_train, y_pred_train_logreg)

# Print the results
print("Logistic Regression Results on Training Data:")
print("Accuracy:", accuracy_logreg)
print("Confusion Matrix:\n", confusion_matrix_logreg)
print("Classification Report:\n", classification_report_logreg)


# In[17]:


# part 5
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Function to evaluate a model
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    confusion_mat = confusion_matrix(y, y_pred)
    classification_rep = classification_report(y, y_pred)
    
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", classification_rep)

# Evaluate the Neural Network from Scikit-learn
print("Neural Network from Scikit-learn:")
evaluate_model(mlp_classifier, X_train, y_train)

# Evaluate the Logistic Regression model
print("\nLogistic Regression Model:")
evaluate_model(logreg_model, X_train, y_train)


# In[ ]:




