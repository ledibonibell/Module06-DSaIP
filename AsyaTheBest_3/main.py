import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("dataset.csv", sep=',')
df = data.copy()
data.info()

# Очистка пропущенных значений
df.drop(['Sunshine', 'Evaporation'], axis=1, inplace=True)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
# categorical_cols = df.select_dtypes(include=['object', 'category']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
df.dropna(subset=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'], inplace=True)

print("\n");
df.info()

# Преобразование категориальных признаков
cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Целевая переменная
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Разделение на признаки и целевую переменную
X = df.drop(['Date', 'RainTomorrow'], axis=1)
y = df['RainTomorrow']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier(max_depth=5, random_state=17)
tree.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

tree_pred = tree.predict(X_test)
print(confusion_matrix(y_test,tree_pred))

# Параметры для GridSearchCV
tree_params = {
    'max_depth': range(1,11),
    'max_features': range(4,19)
}

# Настройка гиперпараметров
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)
print(tree_grid.fit(X_train, y_train))
print(tree_grid.best_params_)

from sklearn.ensemble import RandomForestClassifier

# Создание модели
forest = RandomForestClassifier(n_estimators=100, random_state=17, n_jobs=-1)

# Параметры для GridSearchCV
forest_params = {
    'max_depth': range(1,11),
    'max_features': range(4,19)
}

# Настройка гиперпараметров
forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=1)
print(forest_grid.fit(X_train, y_train))
print(forest_grid.best_params_)