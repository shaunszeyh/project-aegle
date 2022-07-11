import pandas as pd
import numpy as np
from sklearn import tree, neighbors, svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Get the columns needed
df = pd.read_csv("healthcare-dataset-stroke-data-n.csv")
target = df["stroke"]
df_n = df[["age", "gender_n", "hypertension", "heart_disease", "ever_married_n", "work_type_n", "residence_type_n", "avg_glucose_level", "bmi", "smoking_status_n",]]

# Models to be used
models = [
    tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0), 
    neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform'),
    svm.SVC(),
    LogisticRegression(max_iter=5000),
    XGBClassifier(objective='binary:logistic', n_estimators=100000, max_depth=5, learning_rate=0.001, n_jobs=-1, tree_method='hist')
]

# Determine if SMOTE model results in correct application of oversampling
for model in models[:-1]: # Cannot do this for XGBClassifier, will cause terminal to crash :(
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under), ('model', model)]
    pipeline = Pipeline(steps=steps)
    # Evaluate pipeline
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(pipeline, df_n, target, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Mean ROC AUC: %.3f' % np.mean(scores))

# Oversample minority class and undersample majority to balance data (Dataset is heavily skewed towards patients with negative stroke results)
over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
X, y = pipeline.fit_resample(df_n, target)

# Find out which factors are more likely to affect one's chances of getting a stroke
classifier = SelectKBest(score_func=f_classif, k=5)
fits = classifier.fit(X, y)
classifier_output = pd.DataFrame(fits.scores_)
classifier_output = pd.concat([pd.DataFrame(X.columns), classifier_output], axis=1)
classifier_output.columns = ["Attribute", "Score"]
print(classifier_output.sort_values(by="Score", ascending=False))

# Create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

def get_accuracy(model): # run and return accuracy of a model
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    cm = confusion_matrix(y_test, prediction)
    cr = classification_report(y_test, prediction)
    return cr

# Trying with decision tree (Accuracy around 79%)
decision_tree_model = models[0]
print("Accuracy for Decision Tree:")
print(get_accuracy(decision_tree_model))

# Trying with K-Nearest Neighbours (Accuracy around 74%)
knn_model = models[1]
print("Accuracy for KNN:")
print(get_accuracy(knn_model))

# Trying with Support Vector Machines (Accuracy around 77%)
svm_model = models[2]
print("Accuracy for SVM:")
print(get_accuracy(svm_model))

# Trying with Logistic Regression (Accuracy around 78%)
regression = models[3]
print("Accuracy for Logistic Regression:")
print(get_accuracy(regression))

# Trying with XGBClassifier (Accuracy around 80%)
xgb = models[4]
print("Accuracy for XGBClassifier:")
print(get_accuracy(xgb))