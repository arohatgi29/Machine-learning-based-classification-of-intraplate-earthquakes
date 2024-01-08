import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import warnings        # Allows the code to ignore uneccessary warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df=pd.read_csv('data_after_preprocessing.csv')
df.info()

#Define features and target
X = df[['Latitude','Longitude','Continent','Depth','Moment magnitude','Seismic moment']].values
Y = df[['Tectonic association']].values.reshape(X.shape[0])

# Scale data to have mean 0 and variance 1, which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)

names=['Rifts','Passive margins','Non rifted crust']
feature_names=['Latitude (degree)','Longitude (degree)','Continent (0-Africa, 1-Australia, 2-China, 3-Eurasia, 4-India, 5-North America, 6-South America)','Depth (km)','Moment magnitude','Seismic moment (N*m)']

plt.figure(figsize=(9,12))
plt.subplot(3,1,1)
for target, target_name in enumerate(names):
    X_plot = X[Y == target]
    plt.scatter(X_plot[:, 1], X_plot[:, 0], marker='o', s=X_plot[:,4]**3/5, label=target_name)
plt.xlabel(feature_names[1])
plt.ylabel(feature_names[0])
plt.legend();

ax=plt.subplot(3,1,2)
for target, target_name in enumerate(names):
    X_plot = X[Y == target]
    plt.scatter(X_plot[:, 2], X_plot[:, 3], marker='o', s=X_plot[:,4]**3/5, label=target_name)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.legend();
ax.invert_yaxis()

plt.subplot(3,1,3)
for target, target_name in enumerate(names):
    X_plot = X[Y == target]
    plt.scatter(X_plot[:, 4], X_plot[:, 5], marker='o', s=X_plot[:,4]**3/5, label=target_name)
plt.xlabel(feature_names[4])
plt.ylabel(feature_names[5])
plt.legend();
plt.yscale('log')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy on the test data: {accuracy:.2f}")

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


report = classification_report(Y_test, y_pred, target_names=[str(i) for i in range(3)])
print("Classification Report:\n", report)


conf_matrix = confusion_matrix(Y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


