

import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#upload images with labels
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            #oil spill
            if label == 1 and filename[0] == 'O' :
                images.append(img)
                labels.append(label)
            #non oil spill
            if label == 0 and filename[0] == 'N':
                images.append(img)
                labels.append(label)
    return images, labels



# #Cargar imágenes de entrenamiento
train_folder = "Spill_Data/Train"
oil_images, oil_labels = load_images_from_folder(os.path.join(train_folder), 1)
non_oil_images, non_oil_labels = load_images_from_folder(os.path.join(train_folder), 0)

print(len(oil_images))
print(len(non_oil_labels))





# # Combinar imágenes y etiquetas
# X = np.array(oil_images + non_oil_images)
# y = np.array(oil_labels + non_oil_labels)
#
# # Dividir los datos en conjuntos de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Entrenar el clasificador SVM
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train.reshape(len(X_train), -1), y_train)
#
# # Evaluar el modelo
# y_pred_train = clf.predict(X_train.reshape(len(X_train), -1))
# y_pred_test = clf.predict(X_test.reshape(len(X_test), -1))
# train_accuracy = accuracy_score(y_train, y_pred_train)
# test_accuracy = accuracy_score(y_test, y_pred_test)
#
# print("Precisión en el conjunto de entrenamiento:", train_accuracy)
# print("Precisión en el conjunto de prueba:", test_accuracy)
