import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
sns.set()

data = pd.read_csv("dataset.csv", sep=',')
datac = data.copy()
datac.info()

# Очистка значений, у которых порядка половины NaN
datac.drop(['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm'], axis=1, inplace=True)

numeric_cols = datac.select_dtypes(include=['float64', 'int64']).columns
# categorical_cols = datac.select_dtypes(include=['object', 'category']).columns

datac[numeric_cols] = datac[numeric_cols].fillna(datac[numeric_cols].mean())
datac.dropna(subset=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday', 'RainTomorrow'], inplace=True)
datac.info()

# Преобразование категориальных признаков
df = datac.copy()

# Label Encoding
cat_cols = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# One Hot Encoding
df = pd.get_dummies(data=df, columns=['RainToday'], drop_first=True, dtype='int8')

# Целевая переменная
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

df = df.drop(['MaxTemp', 'WindGustDir', 'Date', 'Location', 'WindDir3pm', 'Humidity3pm', 'WindSpeed3pm', 'Pressure3pm', 'Temp3pm'], axis=1)
X = df.drop(['RainTomorrow'], axis=1)
y = df['RainTomorrow']

X.head()

##################
##################

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix

# Оценим качетсво модели
# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=17, n_jobs=-1)

# Параметры для GridSearchCV
forest_params = {
    'max_depth': range(1,11),
    'max_features': range(4,10)
}

# Настройка гиперпараметров
forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=1)
print(forest_grid.fit(X_train, y_train))
forest_pred = forest_grid.predict(X_test)

print(forest_grid.best_params_)