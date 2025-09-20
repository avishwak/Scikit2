# Ensemle Learning

# Principal Component Analysis (PCA) and Pipeline

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
import seaborn as sns

# Importing the dataset
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print(df.head())
print(df.info())
print(df.describe())
print(df['target'].value_counts())

# Visualizing the target distribution
sns.countplot(x='target', data=df)
plt.title('Target Distribution')
plt.show()

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by each component: {explained_variance}')

# Training the Logistic Regression model on the Training set
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# Visualizing the Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()


# Using Pipeline to streamline the process
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression(random_state=0))
])

# Fitting the pipeline on the Training set
pipeline.fit(X_train, y_train)

# Predicting the Test set results using the pipeline
y_pred_pipeline = pipeline.predict(X_test)

# Making the Confusion Matrix for the pipeline
cm_pipeline = confusion_matrix(y_test, y_pred_pipeline)
print('Confusion Matrix (Pipeline):')
print(cm_pipeline)

# Visualizing the Confusion Matrix for the pipeline
sns.heatmap(cm_pipeline, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix (Pipeline)')
plt.show()  

