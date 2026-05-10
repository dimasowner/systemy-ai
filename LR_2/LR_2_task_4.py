import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot

input_file = r'c:\Users\dima\Desktop\studying\4 курс 2 семестр\сші\lab2\LR_2_task_1\income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 5000 

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
    print(f"Помилка: Файл {input_file} не знайдено!")
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

X_final = X_encoded[:, :-1].astype(float)
y_final = X_encoded[:, -1].astype(int)

scaler = StandardScaler()
X_final = scaler.fit_transform(X_final)

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X_final, y_final, test_size=0.20, random_state=1
)

models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=1000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
print(f"{'Алгоритм':<8} | {'Accuracy (Mean)':<18} | {'Std Dev':<10}")
print("-" * 45)

for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name:<8} | {cv_results.mean():>15.4f}  | {cv_results.std():>8.4f}")

pyplot.boxplot(results, labels=names)
pyplot.title('Порівняння алгоритмів (Census Income)')
pyplot.ylabel('Точність (Accuracy)')
pyplot.grid(axis='y', linestyle='--', alpha=0.7)
pyplot.show()