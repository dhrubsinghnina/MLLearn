import tensorflow as tf
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# importing data
df=pd.read_csv("CustomerChurnForBusinesses/CustomerDataSet.csv")
df= df.drop("CustomerID", axis=1)
df=df.head(10000)
print(df)

# Preprocessing:

# Null:
print(df.isnull().sum())
print(f"Describe:\n{df.describe()}")
print(f"Shape:{df.shape}")

# Encoding catagorical values:
data=df.copy()
data=pd.get_dummies(df,columns=["Subscription Type"])
data=pd.get_dummies(data,columns=["Contract Length"])
data=data.replace({True:1,False:0})

Label_Encoder=LabelEncoder()
data["Gender"]=Label_Encoder.fit_transform(df["Gender"])
print("After encoding:")
print(data.head())

# defining X and y (KEEP X AS DATAFRAME)
X = data.drop("Churn", axis=1)
y = data["Churn"]


# spliting data:
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# feature scaling:
num_cols = ['Age','Tenure','Usage Frequency','Support Calls',
            'Payment Delay','Total Spend','Last Interaction']

scaler = StandardScaler()
x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
x_test[num_cols] = scaler.transform(x_test[num_cols])

# model training
reg = tf.keras.regularizers.l2(0.001) # type: ignore

model = Sequential([
    Dense(50, activation='relu', input_shape=(x_train.shape[1],), kernel_regularizer=reg),
    Dense(25, activation='relu', kernel_regularizer=reg),
    Dense(12, activation='relu', kernel_regularizer=reg),
    Dense(1, activation='sigmoid')
])

# opeimization define
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=10**(-3)),  # type: ignore # Adaptive moment estimation
    loss='binary_crossentropy', 
    metrics=['accuracy']
)

print(f"Model summary:\n{model.summary()}")
# model training:
history=model.fit(x_train,y_train,epochs=10,batch_size=500,validation_split=0.2,verbose=1)
print(f"Keys on history :{history.history.keys()}")

# Making prediction from testing data:
y_prob = model.predict(x_test)
y_pred = (y_prob > 0.5).astype(int)

# confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Customer Churn")
plt.show()

# ploting loss on trainging vs loss on validation
plt.figure()
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# ploting accuracy on training vs acccuracy on validation
plt.figure()
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# taking user input:
feature_names = X.columns
def predict_churn(model, scaler, feature_names):
    print("\nEnter customer details:")

    age = int(input("Age: "))
    gender = input("Gender (Male/Female): ")
    tenure = int(input("Tenure: "))
    usage = int(input("Usage Frequency: "))
    support = int(input("Support Calls: "))
    delay = int(input("Payment Delay: "))
    spend = float(input("Total Spend: "))
    last = int(input("Last Interaction (days): "))
    sub_type = input("Subscription Type (Basic/Standard/Premium): ")
    contract = input("Contract Length (Monthly/Quarterly/Annual): ")

    # Encode gender
    gender = 1 if gender.lower() == "male" else 0

    # Create base row
    user_dict = {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": delay,
        "Total Spend": spend,
        "Last Interaction": last,
        "Subscription Type_Basic": 0,
        "Subscription Type_Standard": 0,
        "Subscription Type_Premium": 0,
        "Contract Length_Monthly": 0,
        "Contract Length_Quarterly": 0,
        "Contract Length_Annual": 0
    }

    # Set correct one-hot value
    user_dict[f"Subscription Type_{sub_type}"] = 1
    user_dict[f"Contract Length_{contract}"] = 1

    # Convert to DataFrame
    user_df = pd.DataFrame([user_dict])
    user_df = user_df.reindex(columns=feature_names, fill_value=0)

    # Scale numerical columns
    num_cols = ['Age','Tenure','Usage Frequency','Support Calls',
                'Payment Delay','Total Spend','Last Interaction']
    
    user_df[num_cols] = scaler.transform(user_df[num_cols])

    # Prediction
    prob = model.predict(user_df)[0][0]
    pred = 1 if prob > 0.5 else 0

    print("\nPrediction:", "CHURN" if pred == 1 else "NOT CHURN")
    print("Churn probability:", round(prob, 3))
predict_churn(model, scaler, feature_names)