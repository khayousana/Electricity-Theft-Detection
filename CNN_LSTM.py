import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, confusion_matrix, \
    precision_recall_fscore_support, roc_auc_score
from tensorflow.keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Flatten, Dense

# Définition des constantes
tf.random.set_seed(1234)
epochs_number = 150
test_set_size = 0.2
oversampling_flag = 1
oversampling_percentage = 0.2

# Fonction pour lire les données
def read_data():
    rawData = pd.read_csv('S:\Mes documents\Bureau\PFE MID\Les articles\ElectricityTheftDetection\SmartGridFraudDetection-master\SmartGridFraudDetection-master\data\preprocessedR90.csv')

    # Séparation des features et de la cible
    y = rawData[['FLAG']]
    X = rawData.drop(['FLAG', 'CONS_NO'], axis=1)

    # Réindexation des colonnes selon les dates
    X.columns = pd.to_datetime(X.columns)
    X = X.reindex(X.columns, axis=1)

    # Division des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y['FLAG'], test_size=test_set_size, random_state=0)

    # Suréchantillonnage de la classe minoritaire pour gérer le déséquilibre
    if oversampling_flag == 1:
        over = SMOTE(sampling_strategy=oversampling_percentage, random_state=0)
        X_train, y_train = over.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

# Fonction pour afficher les résultats
def results(y_test, prediction):
    print("Accuracy:", 100 * accuracy_score(y_test, prediction))
    print("RMSE:", mean_squared_error(y_test, prediction, squared=False))
    print("MAE:", mean_absolute_error(y_test, prediction))
    print("F1:", 100 * precision_recall_fscore_support(y_test, prediction)[2])
    print("AUC:", 100 * roc_auc_score(y_test, prediction))
    print("Matrice de confusion:\n", confusion_matrix(y_test, prediction), "\n")

# Fonction pour créer et entraîner le modèle
def build_and_train_model(X_train, X_test, y_train, y_test):
    print('Building and training model:')

    # Prétraitement des données pour les séquences temporelles
    X_train = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Création du modèle
    input_shape = (X_train.shape[1], X_train.shape[2])
    nb_classes = 1  # Binary classification

    input_layer = tf.keras.layers.Input(input_shape)
    lstm = tf.keras.layers.LSTM(8)(input_layer)
    lstm = tf.keras.layers.Dropout(0.2)(lstm)

    permute = tf.keras.layers.Permute((2, 1))(input_layer)
    conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(permute)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

    conv2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.Activation('relu')(conv2)

    conv3 = tf.keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.Activation('relu')(conv3)

    gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)

    concat = tf.keras.layers.concatenate([lstm, gap_layer])

    output_layer = tf.keras.layers.Dense(nb_classes, activation='sigmoid')(concat)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Entraînement du modèle
    history = model.fit(X_train, y_train, epochs=epochs_number, validation_split=0.1, shuffle=True, verbose=1)

    # Prédiction et évaluation
    predictions_proba = model.predict(X_test)
    predictions = (predictions_proba > 0.5).astype(int)
    model.summary()
    results(y_test, predictions)
    
    # Plotting
    plt.figure(figsize=(20, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# ----Main----
X_train, X_test, y_train, y_test = read_data()
build_and_train_model(X_train, X_test, y_train, y_test)
