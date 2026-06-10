import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("StudentsPerformance.csv")

# ==========================
# FEATURE ENGINEERING
# ==========================
df['total_marks'] = (
    df['math score']
    + df['reading score']
    + df['writing score']
)

df['percentage'] = (
    df['total_marks'] / 300 * 100
)

df['grade'] = pd.cut(
    df['percentage'],
    bins=[0, 50, 65, 80, 100],
    labels=['Fail', 'Average', 'Good', 'Excellent']
)

# ==========================
# LABEL ENCODING
# ==========================
encoders = {}

cat_cols = [
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
]

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ==========================
# FEATURE & TARGET
# ==========================
X = df.drop(
    ['total_marks', 'percentage', 'grade'],
    axis=1
)

y = df['grade']

# ==========================
# TRAIN MODEL KNN
# ==========================
model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance'
)

model.fit(X, y)

# ==========================
# SIMPAN MODEL
# ==========================
joblib.dump(model, "model_knn.pkl")
joblib.dump(encoders, "encoders.pkl")

print("model_knn.pkl berhasil dibuat")
print("encoders.pkl berhasil dibuat")
