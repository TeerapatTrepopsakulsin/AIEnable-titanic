import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.read_csv('../data/train_preprocessed.csv')

# Features
target = 'Survived'
features = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'HasPrefix', 'TicketNumber', 'TicketIsLine', 'TicketLength']
numeric = ['Age', 'SibSp', 'Parch', 'Fare', 'TicketNumber', 'TicketLength']
categorical = ['Pclass', 'Embarked', 'Sex', 'HasPrefix', 'TicketIsLine']

# Preprocessing
cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
num_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("cat", cat_tf, categorical),
        ("num", num_tf, numeric),
    ]
)


# Train/Test Split
X = df[features].copy()
y = df[target].copy()
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Model Construction
clf = LogisticRegression(max_iter=1000, n_jobs=None)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", clf)
])

pipe.fit(X_tr, y_tr)
pred = pipe.predict(X_te)
acc = accuracy_score(y_te, pred)
print(acc)
print(classification_report(y_te, pred))

joblib.dump(pipe, '../data/titanic_model.pkl')
print("Save!!")
