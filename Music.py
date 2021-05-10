# Importing the libraries
import pandas as pd

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# def baseline_model():
# Initializing Neural Network
model = Sequential()
# Adding the input layer and the first hidden layer
model.add(Dense(activation="relu", input_dim=54, units=38, kernel_initializer="he_uniform"))
# Adding the second hidden layer
model.add(Dense(activation="relu", units=38, kernel_initializer="he_uniform"))
# # Adding the third hidden layer
#model.add(Dense(activation="relu", units=38, kernel_initializer="he_uniform"))
# Adding the output layer
model.add(Dense(activation="softmax", units=6, kernel_initializer="he_uniform"))


# Compiling Neural Network
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
sc = StandardScaler()

# Importing the dataset
dataset = pd.read_csv('Dataset/Audio_features_train.csv')

# Get all the features starting from tempo
features = dataset.loc[:, 'tempo':]
features = features.values
labels = dataset.loc[:, 'label'].dropna().astype(int)
labels = to_categorical(labels)


# Fix naming here
def build_model():
    test_size = 0.333
    random_seed = 5

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_seed)

    X_train = sc.fit_transform(X_train)

    classifier = model

    # Fitting our model
    classifier.fit(X_train, y_train, batch_size=20, epochs=100)

    X_test = sc.fit_transform(X_test)

    results = model.evaluate(X_test, y_test, batch_size=20)

    print("Test accuracy:", results[1]*100)
    emotion_probabilities = model.predict(X_test)


    # print(emotion_probabilities.shape)
    predicted_emotions = []

    for i in range(1, len(emotion_probabilities)):
        for j in range(len(emotion_probabilities[0])):
            if emotion_probabilities[i][j] > 0.5:
                predicted_emotions.append(j)
    # print(predicted_emotions)
    # print(len(predicted_emotions))

# Predicting the Test set results
def predict_emotion():
    test_features = pd.read_csv('Dataset/Audio_features.csv')
    test_features = test_features.loc[:, 'tempo':]

    test_features = test_features.values
    test_features = sc.transform(test_features)
    #print(test_features)
    emotion_probabilities = model.predict(test_features)
    #print(emotion_probabilities)
    predicted_emotions = []

    for i in range(1, len(emotion_probabilities)):
        for j in range(len(emotion_probabilities[0])):
            if emotion_probabilities[i][j] > 0.5:
                predicted_emotions.append(j)

    return predicted_emotions
