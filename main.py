

import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as patches


#upload images with labels

def preprocess_image(image_path, target_size=(100, 100)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    return image

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = preprocess_image(os.path.join(folder, filename))
        # img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
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



def extract_features(images):
    features = []
    for image in images:
        hog_features = hog(image, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        features.append(hog_features)
    return np.array(features)
def plotImage(image):
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

################ PRE-PROCESS ####################################

# upload images with labels (1 -> oil spill , 0 -> non oill spill)
train_folder = "Spill_Data/Train"
oil_images, oil_labels = load_images_from_folder(os.path.join(train_folder),1)
non_oil_images, non_oil_labels = load_images_from_folder(os.path.join(train_folder), 0)

print("oil spill size: ",len(oil_images))
print("non oil spill size: ",len(non_oil_images))

# extract features

# img = oil_images[17]
# print(img)
# print("len: ", len(img))
# # plotImage(oil_images[0])
# print("_______________________________")
# fd,hog_image = hog(img, block_norm='L2-Hys', pixels_per_cell=(8, 8), cells_per_block=(2, 2),visualize=True)
# print(fd)
# print("len: ", len(fd))
# plotImage(hog_image)

# split dataset in training and test
all_images = oil_images + non_oil_images
all_labels = oil_labels + non_oil_labels
X = extract_features(all_images)

X_train, X_test, y_train, y_test = train_test_split(X, all_labels, test_size=0.2, random_state=42)

# Training Model
svm_classifier = svm.SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)

# Make preictions
y_pred = svm_classifier.predict(X_test)

# Classifier accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



