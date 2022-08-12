import pandas as pd
from sklearn import tree, neighbors, svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
import pickle

# Get the columns needed
df = pd.read_csv("healthcare-dataset-stroke-data-n.csv")
target = df["stroke"]
df_n = df[["age", "gender_n", "hypertension", "heart_disease", "ever_married_n", "work_type_n", "residence_type_n", "avg_glucose_level", "bmi", "smoking_status_n",]]

# Oversample minority class and undersample majority to balance data (Dataset is heavily skewed towards patients with negative stroke results)
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(df_n, target)

# Find out which factors are more likely to affect one's chances of getting a stroke
classifier = SelectKBest(score_func=f_classif, k=5)
fits = classifier.fit(X, y)
classifier_output = pd.DataFrame(fits.scores_)
classifier_output = pd.concat([pd.DataFrame(X.columns), classifier_output], axis=1)
classifier_output.columns = ["Attribute", "Score"]
print(classifier_output.sort_values(by="Score", ascending=False))

# Create train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y, shuffle=True)

def get_accuracy(model): # run and return accuracy of a model
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    cm = confusion_matrix(y_test, prediction)
    cr = classification_report(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)
    return model, cr, auc

# Accuracy of Random Forest: 87%
# ROC AUC of Random Forest: 0.87
kf = KFold(n_splits=5, shuffle=True, random_state=4)
rf_model = RandomForestClassifier(random_state=42)
rf_param = {
    "n_estimators": [50, 100, 200, 500, 1000],
    "max_depth": [3, 4, 5, 8],
}
grid_rf = GridSearchCV(rf_model, rf_param, scoring='roc_auc', cv=kf, n_jobs=-1)

# Accuracy of Logistic Regression: 79%
# ROC AUC of Logistic Regression: 0.78
lr_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
lr_param = {
    "penalty": ["l1", "l2"], 
    "C": [0.01, 0.05, 0.1, 0.5, 1.0, 10.0, 15],
}
grid_lr = GridSearchCV(lr_model, lr_param, scoring='roc_auc', cv=5, n_jobs=-1)

# Accuracy of AdaBoost: 82%
# ROC AUC of AdaBoost: 0.82
ab_model = AdaBoostClassifier(random_state=42)
ab_param = {
    'n_estimators': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20],
    'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
    'algorithm': ["SAMME", "SAMME.R"],
}
grid_ab = GridSearchCV(ab_model, ab_param, scoring='roc_auc', cv=5, n_jobs=-1)

# Accuracy of XGBClassifier: 95%
# ROC AUC of XGBClassifier: 0.95
xgb_model = XGBClassifier(objective='binary:logistic', nthread=4, seed=42)
xgb_param = {
    'max_depth': range(2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05],
}
grid_xgb = GridSearchCV(xgb_model, xgb_param, scoring='roc_auc', cv=5, n_jobs=-1)

# Accuracy of Decision Trees: 87%
# ROC AUC of Decision Trees: 0.87
tree_model = tree.DecisionTreeClassifier(random_state=42)
tree_param = {
    'max_features': ['sqrt', 'log2'],
    'ccp_alpha': [0.1, 0.01, 0.001, 1.0],
    'max_depth': [5, 6, 7, 8, 9], 
    'criterion': ['gini', 'entropy'],
}
grid_tree = GridSearchCV(tree_model, tree_param, scoring='roc_auc', cv=5, n_jobs=-1)

# Accuracy of KNN: 90%
# ROC AUC of KNN: 0.88
knn_model = neighbors.KNeighborsClassifier()
knn_param = {
    'n_neighbors': range(3, 10),
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
}
grid_knn = GridSearchCV(knn_model, knn_param, scoring='roc_auc', cv=5, n_jobs=-1)

# Accuracy of SVM: 82%
# ROC AUC of SVM: 0.82
svm_model = svm.SVC(random_state=42)
svm_param = {
    'C': [0.2, 0.4, 0.6, 0.8, 1.0],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
}
grid_svm = GridSearchCV(svm_model, svm_param, scoring='roc_auc', cv=5, n_jobs=-1)

models = [
    grid_tree,
    grid_knn,
    grid_svm,
    grid_lr,
    grid_xgb,
    grid_rf,
    grid_ab,
]

model_names = [
    "Decision_Tree",
    "K-Nearest_Neighbours",
    "Support_Vector_Machines",
    "Logistic_Regression",
    "XGBClassifier",
    "Random_Forest",
    "AdaBoost",
]

for i in range(len(models)):
    model, acc, auc = get_accuracy(models[i])
    filename = "parameters/" + model_names[i] + ".sav"
    pickle.dump(model, open(filename, "wb"))
    print("Accuracy for " + model_names[i] + ":")
    print(acc)
    print("ROC AUC Score:", round(auc, 2))
