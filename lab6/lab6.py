import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# =================================================================
# ЗАВДАННЯ 3 — Прогноз гри за погодними умовами (Варіант 3)
# =================================================================
print("=" * 55)
print("ЗАВДАННЯ 3 — Наївний Байєс: Прогноз гри (Weather)")
print("=" * 55)

# Відтворюємо таблицю з методичних рекомендацій
weather_data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}

df_weather = pd.DataFrame(weather_data)

# Кодування текстових даних у числа
le_weather = {}
X_weather = pd.DataFrame()
for col in ['Outlook', 'Humidity', 'Wind']:
    le = LabelEncoder()
    X_weather[col] = le.fit_transform(df_weather[col])
    le_weather[col] = le

le_play = LabelEncoder()
y_play = le_play.fit_transform(df_weather['Play'])

# Навчання моделі CategoricalNB
model_weather = CategoricalNB()
model_weather.fit(X_weather, y_play)

# Параметри для Варіанту 3: Sunny, High, Weak
test_weather = {'Outlook': 'Sunny', 'Humidity': 'High', 'Wind': 'Weak'}
test_row = pd.DataFrame([{col: le_weather[col].transform([val])[0] 
                          for col, val in test_weather.items()}])

# Прогноз
pred_play = le_play.inverse_transform(model_weather.predict(test_row))[0]
proba_play = model_weather.predict_proba(test_row)[0]

print(f"\nУмови: Outlook={test_weather['Outlook']}, Humidity={test_weather['Humidity']}, Wind={test_weather['Wind']}")
print(f"Ймовірності — No: {proba_play[0]:.4f}, Yes: {proba_play[1]:.4f}")
print(f"Результат: Чи відбудеться матч? -> {pred_play}")

# =================================================================
# ЗАВДАННЯ 4 — Байєсівський аналіз: квитки Renfe
# =================================================================
print("\n" + "=" * 55)
print("ЗАВДАННЯ 4 — Байєсівський аналіз: квитки Renfe")
print("=" * 55)

url = "https://raw.githubusercontent.com/susanli2016/MachineLearning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)

# Видалення пропусків у цінах, якщо вони є
df = df.dropna(subset=['price'])

print(f"\nРозмір датасету: {df.shape[0]} рядків")

# Створення цільової змінної: розбиття цін на 3 категорії
df['price_category'] = pd.cut(df['price'], bins=3, labels=['Низька', 'Середня', 'Висока'])

print(f"\nРозподіл цінових категорій:")
for cat, cnt in df['price_category'].value_counts().items():
    print(f"  {cat}: {cnt} ({cnt/len(df)*100:.1f}%)")

# Кодування ознак
features = ['train_type', 'train_class', 'fare']
le_dict = {}
X_renfe = pd.DataFrame()
for col in features:
    le = LabelEncoder()
    X_renfe[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

le_y = LabelEncoder()
y_renfe = le_y.fit_transform(df['price_category'].astype(str))

X_train, X_test, y_train, y_test = train_test_split(X_renfe, y_renfe, test_size=0.2, random_state=42)

# Навчання моделі
model_renfe = CategoricalNB()
model_renfe.fit(X_train, y_train)
y_pred = model_renfe.predict(X_test)

print(f"\nТочність моделі: {accuracy_score(y_test, y_pred):.4f}")
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred, zero_division=0))

# Візуалізація
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Завдання 4: Наївний Байєс — Квитки Renfe', fontsize=13, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_y.classes_, yticklabels=le_y.classes_, ax=axes[0])
axes[0].set_title('Матриця плутанини')

colors = ['#4CAF50', '#FFC107', '#F44336']
for cat, color in zip(['Низька', 'Середня', 'Висока'], colors):
    subset = df[df['price_category'] == cat]['price']
    axes[1].hist(subset, bins=20, alpha=0.65, label=cat, color=color)
axes[1].set_title('Розподіл цін за категоріями')
axes[1].legend()

plt.tight_layout()
plt.show()

# Демонстрація прогнозу
print("\n--- Демо прогнозу для нових квитків ---")
test_cases = [
    {'train_type': 'AVE',     'train_class': 'Preferente',  'fare': 'Flexible'},
    {'train_type': 'AVE',     'train_class': 'Turista',      'fare': 'Promo'}
]

for case in test_cases:
    row = pd.DataFrame([{col: le_dict[col].transform([val])[0] for col, val in case.items()}])
    pred_cat = le_y.inverse_transform(model_renfe.predict(row))[0]
    print(f"  Квиток: {case['train_type']}, {case['train_class']}, {case['fare']} -> Передбачена ціна: {pred_cat}")