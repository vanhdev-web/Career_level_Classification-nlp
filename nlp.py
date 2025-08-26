import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose._column_transformer import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from lazypredict.Supervised import LazyClassifier
import re

data = pd.read_excel("../NLP/final_project.ods", dtype=str)
print(data.columns)

def filter_location(location):
    new_location = re.findall(r",\s\w{2}",location)
    if len(new_location) != 0 :
        return new_location[0][2:]
    else:
        return location

data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

# data split
target = "career_level"
x = data.drop(target, axis= 1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2, random_state= 100, stratify= y )


# data preprocessing
# ros = SMOTEN(random_state=0,k_neighbors=2, sampling_strategy={
#     "managing_director_small_medium_company":200,
#     "specialist": 400,
#     "director_business_unit_leader": 600,
#     "bereichsleiter": 1200
# })
# x_train, y_train = ros.fit_resample(x_train, y_train)



preprocessor = ColumnTransformer([
    ("title",TfidfVectorizer(stop_words="english", ngram_range= (1,1)),"title"),
    ("description",TfidfVectorizer(stop_words="english", ngram_range= (1,2), min_df=0.01, max_df=0.95),"description"),
    ("industry",TfidfVectorizer(stop_words="english", ngram_range= (1,1)),"industry"),
    ("location", OneHotEncoder(handle_unknown= "ignore"), ["location"]),
    ("function", OneHotEncoder(handle_unknown= "ignore"), ["function"]),
])

clf = Pipeline(steps=[
    ("preprocessor", preprocessor), #(6458, 7903)
    ("feature_selector", SelectKBest(chi2, k=300)),
    ("model",RandomForestClassifier())
])


params = {
    # "model__n_estimators" : [100, 200, 300],
    "model__criterion" : ["gini", "entropy", "log_loss"],

}
# f1-weighted
# result for only gridSearchCV  "model__criterion" : ["gini", "entropy", "log_loss"]
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.62      0.05      0.10       192
#          director_business_unit_leader       1.00      0.14      0.25        14
#                    manager_team_leader       0.64      0.66      0.65       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.78      0.94      0.85       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.73      1615
#                              macro avg       0.51      0.30      0.31      1615
#                           weighted avg       0.71      0.73      0.69      1615

# slectkbest: 800
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.67      0.08      0.15       192
#          director_business_unit_leader       1.00      0.14      0.25        14
#                    manager_team_leader       0.64      0.75      0.69       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.83      0.92      0.87       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.75      1615
#                              macro avg       0.52      0.32      0.33      1615
#                           weighted avg       0.75      0.75      0.72      1615

# # slectkbest: 300
#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.59      0.16      0.25       192
#          director_business_unit_leader       1.00      0.07      0.13        14
#                    manager_team_leader       0.65      0.74      0.69       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.83      0.92      0.87       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.75      1615
#                              macro avg       0.51      0.31      0.32      1615
#                           weighted avg       0.74      0.75      0.73      1615


grid_search = GridSearchCV(param_grid= params, estimator= clf, cv = 4, verbose= 2, scoring= "recall_weighted")
grid_search.fit(x_train, y_train)



print(grid_search.best_estimator_)

y_predict = grid_search.predict(x_test)
prediction = classification_report(y_test, y_predict)
print(prediction)

# recall-weighted

#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.66      0.17      0.27       192
#          director_business_unit_leader       1.00      0.14      0.25        14
#                    manager_team_leader       0.65      0.76      0.70       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.84      0.91      0.88       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.76      1615
#                              macro avg       0.53      0.33      0.35      1615
#                           weighted avg       0.76      0.76      0.74      1615

lazypredict = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
model, predictions = lazypredict.fit(x_train, x_test, y_train, y_test)

import pickle

# sau khi grid_search.fit(x_train, y_train) hoàn tất
# lưu mô hình tốt nhất vào file pickle
with open("career_model.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)



# nếu muốn load lại mô hình từ pickle để dùng predict
with open("career_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)