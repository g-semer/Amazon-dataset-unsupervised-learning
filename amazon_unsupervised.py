import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


amazon_df = pd.read_csv(r"C:\Users\gseme\Downloads\amazon.csv")
print(amazon_df.shape)
print("amazon_df.head()")
print(amazon_df.head())
print("----------------------------------------")
print("amazon_df.info()")
print(amazon_df.info())
amazon_df['Volume'] = amazon_df['Volume'].str.replace(',', '').astype(float)
print("----------------------------------------")
print("amazon_df.isna().sum()")
print(amazon_df.isna().sum())
print("----------------------------------------")

amazon_date = amazon_df['Date']
amazon_nodate = amazon_df.drop('Date', axis=1)
scaler = StandardScaler()
amazon_scaled = scaler.fit_transform(amazon_nodate)
print("amazon_scaled")
print(amazon_scaled)
print("----------------------------------------")

kmeans = KMeans(n_clusters=2, n_init='auto', random_state=30)
labels = kmeans.fit_predict(amazon_scaled)
print("labels")
print(labels)
print("----------------------------------------")

amazon_df_out = amazon_df.copy()
target_list = [0 for j in range(amazon_df_out.shape[0])]
amazon_df_out['target'] = target_list
for i in range(amazon_df_out.shape[0]):
    if amazon_df['Open'].iloc[i] >= amazon_df['Close'].iloc[i]:
        amazon_df_out.loc[i, 'target'] = 0
    else:
        amazon_df_out.loc[i, 'target'] = 1

amazon_df_out['value'] = labels
print("amazon_df_out.head()")
print(amazon_df_out.head())
print("----------------------------------------")

comparison = [True if amazon_df_out['target'].iloc[j] == amazon_df_out['value'].iloc[j] else False for j in range(amazon_df_out.shape[0])]
print(f"Correct predictions: {comparison.count(True)}\nFalse predictions: {comparison.count(False)}")

plt.subplot(3, 2, 1)
plt.scatter(amazon_scaled[:, 0], amazon_scaled[:, 1], c=labels)
plt.xlabel("Open (scaled)")
plt.ylabel("High (scaled)")
plt.title("Open - High")

plt.subplot(3, 2, 2)
plt.scatter(amazon_scaled[:, 0], amazon_scaled[:, 2], c=labels)
plt.xlabel("Open (scaled)")
plt.ylabel("Low (scaled)")
plt.title("Open - Low")

plt.subplot(3, 2, 3)
plt.scatter(amazon_scaled[:, 0], amazon_scaled[:, 3], c=labels)
plt.xlabel("Open (scaled)")
plt.ylabel("Close (scaled)")
plt.title("Open - Close")

plt.subplot(3, 2, 4)
plt.scatter(amazon_scaled[:, 0], amazon_scaled[:, 4], c=labels)
plt.xlabel("Open (scaled)")
plt.ylabel("Volume (scaled)")
plt.title("Open - Volume")

plt.subplot(3, 2, 5)
plt.scatter(amazon_scaled[:, 3], amazon_scaled[:, 2], c=labels)
plt.xlabel("Close (scaled)")
plt.ylabel("Low (scaled)")
plt.title("Close - Low")

plt.subplot(3, 2, 6)
plt.scatter(amazon_scaled[:, 3], amazon_scaled[:, 4], c=labels)
plt.xlabel("Close (scaled)")
plt.ylabel("Volume (scaled)")
plt.title("Close - Volume")

plt.show()


