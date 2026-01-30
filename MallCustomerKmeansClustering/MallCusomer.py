
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("K-means/Mall_Customers.csv")
print(df.head())
print(f"shape of data :{df.shape}")
print(f"dataset info :{df.info}")
print(f"Summary stastics: {df.describe(include='all')}")
print(f"missing valuse :{df.isnull().sum()}")
print(f"data types \n{df.dtypes}")

# Analyzing:
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# ---- 1. Age Histogram ----
plt.subplot(2, 2, 1)
plt.hist(df['Age'], bins=10, color='C1', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')

# ---- 2. Gender Frequency ----
plt.subplot(2, 2, 2)
df['Gender'].value_counts().plot(kind='bar', edgecolor='black')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Frequency')

# ---- 3. Age vs Annual Income ----
plt.subplot(2, 2, 3)
plt.scatter(df['Age'], df['Annual Income'])
plt.xlabel('Age')
plt.ylabel('Annual Income')
plt.title('Age vs Annual Income')

# ---- 4. Gender vs Annual Income ----
plt.subplot(2, 2, 4)
plt.scatter(df['Gender'], df['Annual Income'])
plt.xlabel('Gender')
plt.ylabel('Annual Income')
plt.title('Gender vs Annual Income')

plt.tight_layout()
plt.show()


# Encoding categorical value:
Encoder=LabelEncoder()
df['Gender']=Encoder.fit_transform(df['Gender'])
print("After lable encoder:")
print(df.head())

# Deciding the feature:
X=df[['Gender','Age','Annual Income','Spending Score']]

# making model:
model=KMeans(n_clusters=4,random_state=42,n_init=10)
df['group']=model.fit_predict(X) # train and get output
print("After model fited:")
print(df)


# Ploting The clustring: 

inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# ---- Scatter plot ----
scatter = ax.scatter(
    df['Age'],
    df['Annual Income'],
    df['Spending Score'], # type: ignore
    c=df['group'],
    cmap='viridis',
    s=60,
    alpha=0.75
)

# ---- Plot centroids ----
centroids = model.cluster_centers_
ax.scatter(
    centroids[:, 1],  # Age
    centroids[:, 2],  # Annual Income
    centroids[:, 3],  # Spending Score # type: ignore
    c='red',
    s=200,
    marker='X',
    label='Centroids'
)

# ---- Labels ----
ax.set_xlabel('Age', labelpad=10)
ax.set_ylabel('Annual Income', labelpad=10)
ax.set_zlabel('Spending Score', labelpad=10)

# ---- View angle ----
ax.view_init(elev=25, azim=45)

# ---- Legend & colorbar ----
ax.legend()
plt.colorbar(scatter, ax=ax, shrink=0.6, label='Cluster')

plt.title('3D Customer Segmentation using K-Means', pad=20)
plt.tight_layout()
plt.show()

# Silhouette Score
from sklearn.metrics import silhouette_score
score = silhouette_score(X, df['group'])
print("Silhouette Score:", score)

# Compare with Different K Values
for k in [2, 3, 4, 5]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    print(k, silhouette_score(X, labels))
    
df.boxplot(column='Spending Score', by='group')
plt.title('Spending Score by Cluster')
plt.suptitle('')
plt.show()


# Interpreted the cluster:
cluster_sizes = df['group'].value_counts()
print(cluster_sizes)
cluster_summary = df.groupby('group')[['Age','Annual Income','Spending Score']].mean()
print(cluster_summary)

