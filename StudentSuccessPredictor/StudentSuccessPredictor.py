import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv("StudentSuccessPredictor/student_data_1000.csv")
print(df.head())
# -------data preprocessing------- :
# Handling missing data:
print(df.isnull().sum())
# Encoding categorical values:
label_encoder=LabelEncoder() # label enoder object
df['Internet']=label_encoder.fit_transform(df['Internet'])
df['Passed']=label_encoder.fit_transform(df['Passed'])
# Feature Scaling :
Standard_Scaler=StandardScaler() # standard scaler object
features=['StudyHours','Attendance','PastScore','SleepHours']
df_scaled=df.copy()
df_scaled[features]=Standard_Scaler.fit_transform(df[features])
print(df_scaled)
# detemine X and y for Regression:
# for classification:
X=df_scaled[['StudyHours','Attendance','PastScore','SleepHours','Internet']]
y=df_scaled['Passed']
# for linear regression:
X_reg = df_scaled[['StudyHours','Attendance','SleepHours','Internet']]
y_reg = df_scaled['PastScore']
# Split the data:
# for linear gression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
# for classification
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# -------Spervised learning-------
# Implementing linear regression predicting student score from ['StudyHours','Attendance','SleepHours','Internet'] ;
Linear_Regression=LinearRegression() # objecct
Linear_Regression.fit(X_train_reg,y_train_reg)
y_pred_score=Linear_Regression.predict(X_test_reg)
# Implementing Linear regression on the basis of study hours:
X_single = df[['StudyHours']]
y_single = df['PastScore']
LR_single = LinearRegression()
LR_single.fit(X_single, y_single)
pred_single = LR_single.predict(X_single)
# Implementing Logistic regression predic pass or fail:
Logistic_Regression=LogisticRegression() # object
Logistic_Regression.fit(X_train,y_train)
y_pred_logistic=Logistic_Regression.predict(X_test)
# Implementing KNN for pass or fail:
KNN=KNeighborsClassifier(n_neighbors=2) # object
KNN.fit(X_train,y_train)
y_pred_KNN=KNN.predict(X_test)
# Implementing Decision tree for pass or fail:
Decision_Tree=DecisionTreeClassifier() #object
Decision_Tree.fit(X_train,y_train)
y_pred_tree=Decision_Tree.predict(X_test)

# -------model evaluation-------:
# for Linear regression : Regression matrics
print("----evaluation for Linear regression")
print(f"Mean absolute error={mean_absolute_error(y_test_reg,y_pred_score)}")
print(f"Mean squared error ={mean_squared_error(y_test_reg,y_pred_score)}")
print(f"Root mean squared error={np.sqrt(mean_squared_error(y_test_reg,y_pred_score))}")
# for Logistic regression: Classification metrics
print("----evaluation for the Logistic regression----")
print(f"accuracy={accuracy_score(y_test,y_pred_logistic)}")
print(f"precision={precision_score(y_test,y_pred_logistic)}")
print(f"recall={recall_score(y_test,y_pred_logistic)}")
print(f"f1={f1_score(y_test,y_pred_logistic)}")
# classification report:
print("classification Reort :")
print(classification_report(y_test,y_pred_logistic)) # actual,prediction
# confusion metrics:
confu_martix_logistic=confusion_matrix(y_test,y_pred_logistic) # actual,prediction
# for KNN : Classification metrics
print("----evaluation for the KNN----")
print(f"accuracy={accuracy_score(y_test,y_pred_KNN)}")
print(f"precision={precision_score(y_test,y_pred_KNN)}")
print(f"recall={recall_score(y_test,y_pred_KNN)}")
print(f"f1={f1_score(y_test,y_pred_KNN)}")
# confusion metrics:
confu_martix_KNN=confusion_matrix(y_test,y_pred_KNN) # actual,prediction
# for Decision tree: Classification metrics
print("----evaluation for the Decission tree----")
print(f"accuracy={accuracy_score(y_test,y_pred_tree)}")
print(f"precision={precision_score(y_test,y_pred_tree)}")
print(f"recall={recall_score(y_test,y_pred_tree)}")
print(f"f1={f1_score(y_test,y_pred_tree)}")
# confusion metrics:
confu_martix_tree=confusion_matrix(y_test,y_pred_tree) # actual,prediction
# -------Grapgh-------
# for student marks disriburion
plt.figure(figsize=(6,4))
plt.hist(df['PastScore'],bins=10,color='skyblue',edgecolor='black')
plt.xlabel("Marks->")
plt.ylabel("Number of Student->")
plt.grid(True)
plt.show()
# Create 3D figure
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')

# 3D scatter plot
ax.scatter(
    df['StudyHours'],
    df['Attendance'],
    df['SleepHours'] # type: ignore
)
# Labels
ax.set_xlabel("Study Hours")
ax.set_ylabel("Attendance")
ax.set_zlabel("Sleep Hours")
ax.set_title("3D Plot of Study Hours, Attendance, and Sleep Hours")
plt.show()
# scater + regression line :
plt.figure(figsize=(10,7))
plt.scatter(X_single, y_single, color='blue', label='Actual Score')
plt.plot(X_single, pred_single, color='red', label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Past Score")
plt.legend()
plt.grid(True)
plt.show()
# -------confusion matrix-------
# for logistic regression
plt.figure(figsize=(5,4))
plt.imshow(confu_martix_logistic)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
# Class labels
classes = ['Fail', 'Pass']
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)
# Add numbers inside cells
for i in range(confu_martix_logistic.shape[0]):
    for j in range(confu_martix_logistic
.shape[1]):
        plt.text(j, i, confu_martix_logistic
    [i, j],
                 ha="center", va="center")
plt.tight_layout()
plt.show()
# for KNN
plt.figure(figsize=(5,4))
plt.imshow(confu_martix_KNN)
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
# Class labels
classes = ['Fail', 'Pass']
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)
# Add numbers inside cells
for i in range(confu_martix_KNN.shape[0]):
    for j in range(confu_martix_KNN
.shape[1]):
        plt.text(j, i, confu_martix_KNN
    [i, j],
                 ha="center", va="center")
plt.tight_layout()
plt.show()
# for logistic Decision tree
plt.figure(figsize=(5,4))
plt.imshow(confu_martix_tree)
plt.title("Confusion Matrix - Decision tree")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.colorbar()
# Class labels
classes = ['Fail', 'Pass']
plt.xticks([0,1], classes)
plt.yticks([0,1], classes)
# Add numbers inside cells
for i in range(confu_martix_tree.shape[0]):
    for j in range(confu_martix_tree
.shape[1]):
        plt.text(j, i, confu_martix_tree
    [i, j],
                 ha="center", va="center")
plt.tight_layout()
plt.show()

print("-------PREDICT YOUR RESULT-------")
try:
    study_hour = float(input("Enter study hour :"))
    attendence = float(input("Enter attendence :"))
    past_score = float(input("Enter past score :"))
    sleep_hours = float(input("Enter sleep hour :"))
    internet = int(input("Enter 1 for internet access else 0:"))

    user_df = pd.DataFrame([{
        'StudyHours': study_hour,
        'Attendance': attendence,
        'PastScore': past_score,
        'SleepHours': sleep_hours,
        'Internet': internet
    }])

    # scale only numeric features, in the same order as during fitting
    user_df[features] = Standard_Scaler.transform(user_df[features])

    # select features in the same order used for training X
    user_X = user_df[['StudyHours','Attendance','PastScore','SleepHours','Internet']]

    prediction = Logistic_Regression.predict(user_X)[0]
    result = "Pass" if prediction == 1 else "Fail"
    print(f"prediction based on inputs : {result}")

except Exception as e:
    print(e)
