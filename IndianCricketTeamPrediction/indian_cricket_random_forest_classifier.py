import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
data=pd.read_csv("IndianCricketTeamPrediction/indian_cricket_players_500.csv")
print(data.head())
print("----drop the name and catches columns----")
df=data.drop(columns=['PLAYER','CATCHES'])
print(df.head())
# make X and y for model:
X=df.iloc[:,:10]
y=df.iloc[:,10]
# split data:
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# random forest model:
RandomForest_Classifier=RandomForestClassifier()
RandomForest_Classifier.fit(X_train,y_train)
y_pred=RandomForest_Classifier.predict(X_test)
# ----model evaluation----:
eval_score=RandomForest_Classifier.score(X_test,y_test)
print(f"Evaluation score ={eval_score}")
# clasification :
print(f"Classifiction reprt:\n{classification_report(y_test,y_pred)}")
# confusion martics
confu_martix=confusion_matrix(y_test,y_pred)
print(f"Confusion matrics:\n{confu_martix}")
# fearure importance:
feature=pd.DataFrame(RandomForest_Classifier.feature_importances_,index=X.columns)
print(feature)

# Hyper parameters:
print("With hyperpameter :")
rf2=RandomForestClassifier(
    n_estimators=1000,
    criterion='entropy',
    min_samples_split=9,
    max_depth=14,
    random_state=42
)
rf2.fit(X_train,y_train)
y_pred2=rf2.predict(X_test)
print(f"Classifiction reprt:\n{classification_report(y_test,y_pred2)}")
print(f"Confusion matrics:\n{confusion_matrix(y_test,y_pred2)}")

# ploting confusion matrix:
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

confu_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=confu_matrix,
    display_labels=["Not Selected", "Selected"]
)

disp.plot()
plt.show()

