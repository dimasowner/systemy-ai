import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

input_file = r'c:\Users\dima\Desktop\studying\4 курс 2 семестр\сші\lab2\LR_2_task_1\income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 10000 

try:
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue

            data = line[:-1].split(', ')

            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            elif data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1
except FileNotFoundError:
    print(f"Помилка: Файл {input_file} не знайдено. Перевірте шлях!")
    exit()

X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape)
for i, item in enumerate(X[0]):
    if item.replace('.', '', 1).isdigit():
        X_encoded[:, i] = X[:, i]
    else:
        le = preprocessing.LabelEncoder()
        X_encoded[:, i] = le.fit_transform(X[:, i])
        label_encoder.append(le)

X_final = X_encoded[:, :-1].astype(int)
y_final = X_encoded[:, -1].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=5)

kernels = [
    {'name': 'Polynomial (degree 8)', 'kernel': 'poly', 'degree': 8},
    {'name': 'Gaussian (RBF)', 'kernel': 'rbf', 'degree': None},
    {'name': 'Sigmoid', 'kernel': 'sigmoid', 'degree': None}
]

print(f"{'Kernel Type':<25} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10}")
print("-" * 65)

for k in kernels:
    if k['kernel'] == 'poly':
        svc = SVC(kernel=k['kernel'], degree=k['degree'], random_state=0, max_iter=2000)
    else:
        svc = SVC(kernel=k['kernel'], random_state=0, max_iter=2000)
    
    classifier = OneVsOneClassifier(svc)
    classifier.fit(X_train, y_train)
    
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"{k['name']:<25} | {acc*100:>8.2f}% | {f1*100:>8.2f}% | {prec*100:>8.2f}%")

input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 
              'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = int(label_encoder[count].transform([input_data[i]])[0])
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
predicted_class = classifier.predict(input_data_encoded)
print("\n[Test Point Prediction with Sigmoid]:", label_encoder[-1].inverse_transform(predicted_class)[0])