from numpy.core.fromnumeric import size
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

penguin_df = pd.read_csv("penguins.csv")
print(penguin_df.head())
print(penguin_df.isnull().sum())
penguin_df.dropna(inplace=True)

output = penguin_df["species"]
print(f"Output variable:\n {output.head()}")
features = penguin_df.drop(["species", "year"], axis=1)
features = pd.get_dummies(features)
print(f"Feature variables:\n {features.head()}")
output, uniques = pd.factorize(output)
print(f"uniques:\n {uniques} ")

X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.2)

rfc = RandomForestClassifier(random_state=1)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

score = round(accuracy_score(y_pred, y_test), 2)
print(f"Accuracy of the model is: {score} ")

# The open() function creates two pickle files, 
# the pickle.dump() function writes our Python files to said files, 
# and the close() function closes the files.
# Syntax --> open(file, mode)
rf_pickle = open("random_forest_penguin.pickle", "wb") 
# wb indicates that the file is opened for writing in binary mode.
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

output_pickle = open("output_penguin.pickle", "wb")
pickle.dump(uniques, output_pickle)
output_pickle.close()

# Feature importance
fig, ax = plt.subplots()
ax = sns.barplot(rfc.feature_importances_, features.columns)
plt.title("Which features are the most important for prediction?")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
fig.savefig("feature_importance.png")