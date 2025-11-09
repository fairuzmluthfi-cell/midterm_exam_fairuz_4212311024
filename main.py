# mempersiapkan/memanggil module
import matplotlib.pyplot as plt                 # module untuk nampilin gambar/grafik
import pandas as pd                             # module untuk mengelola file csv
import seaborn                                  # module untuk membuat grafik
from tqdm import tqdm
from skimage.feature import hog                 # module untuk memproses gambar ke HOG
from sklearn import datasets                    # module untuk mengelola datasets
from sklearn.model_selection import LeaveOneOut # module untuk memproses HOG ke LOOCV
from sklearn.svm import SVC               # module untuk klasifikasi LOOCV
from sklearn.metrics import confusion_matrix    # module untuk membuat matriks LOOCV
from sklearn.metrics import accuracy_score      # module untuk membuat nilai akurasi LOOCV
from sklearn.metrics import precision_score     # module untuk membuat nilai presisi LOOCV
from sklearn.metrics import f1_score            # module untuk membuat nilai F1 LOOCV

# mengambil data dari dataset csv
train_dataset = pd.read_csv('emnist-letters-train.csv', header=None)

# memisahkan image dan label dari emnist-letters-train.csv
train_images = train_dataset.iloc[:, 1:].values # code untuk mengambil array gambar
train_labels = train_dataset.iloc[:, 0].values  # code untuk mengambil array class
print(f"\njumlah gambar : {train_images.shape[0]}\n") # code untuk menampilkan jumlah dataset

# menyiapkan parameter yang akan digunakan untuk HOG
feature = hog(train_images[0].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), block_norm='L2') # mengambil 1 contoh gambar untuk diambil parameternya saja
n_dims = feature.shape[0]                       # mengambil dimensi dari train image[0]

# proses hog pada masing masing gambar train
n_samples_train = train_images.shape[0]         # mengambil jumlah dataset train_images
X_train, y_train = datasets.make_classification(n_samples=n_samples_train, n_features=n_dims) #membuat klasifikasi pada dataset train
for i in range(n_samples_train):
    X_train[i], _ = hog(train_images[i].reshape(28,28), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), visualize=True, block_norm='L2') # mengubah gambar ke parameter HOG
    y_train[i] = train_labels[i]                # mengambil nilai class pada masing masing gambar

    print(f"HOG Process : {i + 1}/{n_samples_train}", end="\r")
print(f"HOG Process : {n_samples_train}/{n_samples_train}")

# mengolah gambar dari HOG ke LOOCV
y_true = []                                     # menyimpan hasil proses LOOCV untuk nilai sebenarnya
y_pred = []                                     # menyimpan hasil proses LOOCV untuk nilai prediksi

cv = LeaveOneOut()                              # memanggil LOOCV
model = SVC(kernel="linear")                    # memanggil model SVM.SVC

for train_idx, test_idx in tqdm(cv.split(X_train), total=cv.get_n_splits(X_train), desc="Running LOOCV"):
    model.fit(X_train[train_idx], y_train[train_idx])
    pred = model.predict(X_train[test_idx])
    y_true.append(y_train[test_idx][0])
    y_pred.append(pred[0])

# menampilkan nilai akurasi
accuracy = accuracy_score(y_true, y_pred)       # menghitung nilai akurasi
print(f"\nAkurasi : {accuracy:.2f}")            # menampilkan akurasi

# menampilkan nilai presisi
precision = precision_score(y_true, y_pred, average='macro') # menghitung nilai presisi
print(f"Presisi : {precision:.2f}")           # menampilkan nilai presisi

# menampilkan nilai F1
f1 = f1_score(y_true, y_pred, average='macro')  # menghitung nilai F1
print(f"F1 : {f1:.2f}")                         # menampilkan nilai F1

# menampilkan matrix
cm = confusion_matrix(y_true, y_pred)           # mengubah nilai sesungguhnya dan prediksi ke matriks
alfabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'] # label untuk X dan y pada matriks

seaborn.heatmap(cm, annot=False, cmap='Greens', xticklabels=alfabet, yticklabels=alfabet) # membuat grafik matriks
plt.title('Confusion Matrix (LOOCV)')           # judul untuk matriks
plt.xlabel('Predicted Label')                   # label untuk X pada matriks
plt.ylabel('True Label')                        # label untuk y pada matriks
plt.show()                                      # menampilkan matriks