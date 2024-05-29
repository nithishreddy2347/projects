import cv2
import numpy as np
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from skimage import color, exposure
from skimage.transform import resize
from PIL import Image
import glob

def load_data(fruit, tipo, B, clase, testing, flag):
    label = []
    arr = []
    strr = f"FruitsDB/{fruit}/{tipo}/*"
    for file_ in glob.glob(strr):
        img = cv2.imread(file_)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if testing:
            if flag:
                yen_threshold = threshold_yen(img)
                bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
                gray_img = color.rgb2gray(bright)
                h = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                arr.append(h)
            else:
                gray_img = color.rgb2gray(img)
                h = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
                arr.append(h)
        else:
            arr.append(img)
        label.append(clase)

    return arr, label

def whole_train_data(tipo, B, flag):
    apples_data, apples_label = load_data('Apples', tipo, B, 0, 1, flag)
    mangoes_data, mangoes_label = load_data('Mangoes', tipo, B, 1, 1, flag)
    oranges_data, oranges_label = load_data('Oranges', tipo, B, 2, 1, flag)
    data = np.concatenate((apples_data, mangoes_data, oranges_data))
    labels = np.concatenate((apples_label, mangoes_label, oranges_label))
    return data, labels

def preprocessing(arr):
    arr_prep = []
    for i in range(np.shape(arr)[0]):
        img = arr[i]
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = gray_img
        img = resize(img, (72, 72), anti_aliasing=True)
        arr_prep.append(img)
    return arr_prep

def preprocessing_part_two(arr):
    arr_feature = []
    for i in range(np.shape(arr)[0]):
        img = cv2.cvtColor(arr[i], cv2.COLOR_BGR2GRAY)
        img = resize(img, (72, 72), anti_aliasing=True)
        
        if img.ndim == 2:
            ftr, _ = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            ftr = exposure.rescale_intensity(ftr, in_range=(0, 10))
            arr_feature.append(ftr)
    
    return arr_feature

def train_model_svm(data_train, labels_train):
    clf = svm.SVC()
    clf.fit(data_train, labels_train)
    return clf
def train_model_knn(data_train, labels_train, k):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(data_train, labels_train)
    return knn_clf

def get_precision(svm, knn, test_target, test):
    result_svm = svm.predict(test)
    result_knn = knn.predict(test)
    correct_svm = np.sum(result_svm == test_target)
    correct_knn = np.sum(result_knn == test_target)
    return (correct_svm * 100.0 / result_svm.size), (correct_knn * 100.0 / result_knn.size)

def run_svm_knn(flag, k):
    data_train, labels_train = whole_train_data('Train', 16, flag)
    data_test, labels_test = whole_train_data('Test', 16, flag)

    data_train = np.vstack(data_train)
    data_test = np.vstack(data_test)
    labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)

    data_train = np.float32(data_train)
    data_test = np.float32(data_test)

    svm_model = train_model_svm(data_train, labels_train)
    knn_model = train_model_knn(data_train, labels_train, k)

    classes = ["Apples", "Mangoes", "Oranges"]

    apples_data, apples_label = load_data('Apples', 'Test', 16, 0, 0, flag)
    mangoes_data, mangoes_label = load_data('Mangoes', 'Test', 16, 1, 0, flag)
    oranges_data, oranges_label = load_data('Oranges', 'Test', 16, 2, 0, flag)
    data = np.concatenate((apples_data, mangoes_data, oranges_data))

    accuracy_svm, accuracy_knn = get_precision(svm_model, knn_model, labels_test, data_test)

    ans_svm = [accuracy_svm, data, data_test, svm_model]
    ans_knn = [accuracy_knn, data_test, labels_test, knn_model]

    return ans_svm, ans_knn, classes

import matplotlib.pyplot as plt

def predict_fruit(svm_model, knn_model, flag, k):
    input_image_path = input("Enter the file path of the image: ")
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if flag:
        yen_threshold = threshold_yen(img)
        bright = rescale_intensity(img, (0, yen_threshold), (0, 255))
        gray_img = color.rgb2gray(bright)
        img_feature = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
    else:
        gray_img = color.rgb2gray(img)
        img_feature = hog(gray_img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

    img_feature = np.reshape(img_feature, (1, -1))[:, :30000].astype(np.float32)

    svm_result = svm_model.predict(img_feature)
    knn_result = knn_model.predict(img_feature)

    prediction_svm = classes[svm_result[0]]
    prediction_knn = classes[knn_result[0]]

    plt.imshow(img)
    plt.title(f"SVM Prediction: {prediction_svm}\nKNN Prediction: {prediction_knn}")
    plt.axis('off')
    plt.show()
import matplotlib.pyplot as plt

def plot_accuracy(iterations, accuracies_svm, accuracies_knn):
    plt.plot(iterations, accuracies_svm, label='SVM', marker='o')
    plt.plot(iterations, accuracies_knn, label='KNN', marker='o')

    plt.xlabel('Iterations')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Comparison: SVM vs KNN')
    plt.legend()
    plt.show()

# Example usage:
iterations = [1, 2, 3, 4, 5]  # Replace with your actual iteration values
accuracies_svm = [95, 96, 97, 98, 99]  # Replace with your actual SVM accuracies
accuracies_knn = [92, 94, 96, 97, 98]  # Replace with your actual KNN accuracies


if __name__ == '__main__':
    k_value = 11
    results_svm, results_knn, classes = run_svm_knn(False, k_value)

    print("Precision of SVM: {:.2f}%".format(results_svm[0]))
    print("Precision of KNN: {:.2f}%".format(results_knn[0]))

    predict_fruit(results_svm[3], results_knn[3], False, k_value)
    plot_accuracy(iterations, accuracies_svm, accuracies_knn)