import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import joblib



df = pd.read_csv("hybrid_training_data.csv")

X = df[['hits', 'users', 'cmds']]

y = df['is_danger']



model = RandomForestClassifier(n_estimators=100)

model.fit(X, y)



joblib.dump(model, "mtd_brain.joblib")

print("Model 'mtd_brain.joblib' trained and saved!")
