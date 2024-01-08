import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Times New Roman']
plt.rcParams['axes.unicode_minus']=False
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

df=pd.read_csv('data_after_preprocessing.csv')
df.info()

#Define features and target
X = df[['Latitude','Longitude','Continent','Depth','Moment magnitude','Seismic moment']].values
y = df[['Tectonic association']].values.reshape(X.shape[0])

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Scale data to have mean 0 and variance 1, which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=2)
Y_test_f=np.argmax(Y_test, axis=1)# compared with Y_test_pred_f

n_features = X_scaled.shape[1]
n_classes = Y.shape[1]
print('\n','Number of features:',n_features)
print('\n','Number of classes:',n_classes)

names=['Rifts','Passive margins','Non rifted crust']
feature_names=['Latitude (degree)','Longitude (degree)','Continent (0-Africa, 1-Australia, 2-China, 3-Eurasia, 4-India, 5-North America, 6-South America)','Depth (km)','Moment magnitude','Seismic moment (N*m)']

plt.figure(figsize=(9,12))
plt.subplot(3,1,1)
for target, target_name in enumerate(names):
    X_plot = X[y == target]
    plt.scatter(X_plot[:, 1], X_plot[:, 0], marker='o', s=X_plot[:,4]**3/5, label=target_name)
plt.xlabel(feature_names[1])
plt.ylabel(feature_names[0])
plt.legend();

ax=plt.subplot(3,1,2)
for target, target_name in enumerate(names):
    X_plot = X[y == target]
    plt.scatter(X_plot[:, 2], X_plot[:, 3], marker='o', s=X_plot[:,4]**3/5, label=target_name)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.legend();
ax.invert_yaxis()

plt.subplot(3,1,3)
for target, target_name in enumerate(names):
    X_plot = X[y == target]
    plt.scatter(X_plot[:, 4], X_plot[:, 5], marker='o', s=X_plot[:,4]**3/5, label=target_name)
plt.xlabel(feature_names[4])
plt.ylabel(feature_names[5])
plt.legend();
plt.yscale('log')

plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.3)

def create_custom_model(input_dim, output_dim, nodes, n=1, name='model'):
    def create_model():
        # Create model
        model = Sequential(name=name)
        for i in range(n):
            model.add(Dense(nodes, input_dim=input_dim, activation='relu'))
        model.add(Dense(output_dim, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return model
    return create_model

models = [create_custom_model(n_features, n_classes, 10, i, 'model_{}'.format(i)) 
          for i in range(1, 4)]

for create_model in models:
    create_model().summary()

    history_dict = {}

# TensorBoard Callback
cb = TensorBoard()

for create_model in models:
    model = create_model()
    print('Model name:', model.name)
    history_callback = model.fit(X_train, Y_train,
                                 batch_size=16,
                                 epochs=100,
                                 verbose=0,
                                 validation_data=(X_test, Y_test),
                                 callbacks=[cb])
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    history_dict[model.name] = [history_callback, model]
    
    Y_test_pred=model.predict(X_test)
    Y_test_pred_f=np.argmax(Y_test_pred, axis=1)
    print('Accuracy:', accuracy_score(Y_test_f, Y_test_pred_f))
    print('Precision:', precision_score(Y_test_f, Y_test_pred_f, average='macro'))
    print('Recall:', recall_score(Y_test_f, Y_test_pred_f, average='macro'))
    print('F1-score:', f1_score(Y_test_f, Y_test_pred_f, average='macro'))
    
    report = classification_report(Y_test_f, Y_test_pred_f, target_names=[str(i) for i in range(3)])
    print("Classification Report:\n", report)
    
    conf_matrix = confusion_matrix(Y_test_f, Y_test_pred_f)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    print('\n')

    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 6))

for model_name in history_dict:
    val_accurady = history_dict[model_name][0].history['val_accuracy']
    val_loss = history_dict[model_name][0].history['val_loss']
    ax1.plot(val_accurady, label=model_name)
    ax2.plot(val_loss, label=model_name)
    
ax1.set_ylabel('Validation accuracy')
ax2.set_ylabel('Validation loss')
ax2.set_xlabel('Epochs')
ax1.legend()
ax2.legend();
