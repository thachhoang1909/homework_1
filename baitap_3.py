
# coding: utf-8

# # Bài tập 3: Face Dataset

# In[17]:


# import libray
from sklearn.cluster import KMeans, DBSCAN, spectral_clustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
#plt.gray()
get_ipython().magic('matplotlib inline')


# In[18]:


faces = fetch_lfw_people(min_faces_per_person=70)


# In[19]:


print("Number of images: ", faces.images.shape)
print("Number of classes: ",len(set(faces.target)))


# #### Import Dataset
#     - Dataset gồm 1288 ảnh 62 x 47.
#     - Đã được label với target gồm 5749 class. 
#     - Mỗi class gồm ảnh của một người. 
#     - Mỗi class gồm ít nhất 70 ảnh. 

# In[20]:


# create a fig to show image
fig = plt.figure(figsize=(8,5))

plt.title('Two samples in dataset')
plt.axis('off')
# for all 0-9 labels
for i in range(2):
    # initialize subplots in a grid 2x5 at i+1th position
    ax = fig.add_subplot(1, 2, 1+i)
    
    # display image
    ax.imshow(faces.images[i], cmap=plt.cm.binary)
    
    #don't show the axes
    plt.axis('off')

plt.show()


# ### Trích xuất đặc trưng
# 
#     - Sử dụng Local Binary Pattern. 
#     

# In[21]:


# import feature detector & descriptor library
from skimage.feature import local_binary_pattern


# In[22]:


feature = local_binary_pattern(faces.images[0], P=8, R=0.5)


# In[23]:


plt.matshow(feature)


# In[24]:


import numpy as np


# In[25]:


fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(1,1,1)
ax.hist(feature.reshape(-1), bins=list(range(257)))
plt.title('256-dimensional feature vector')
plt.show()


# ### Trích xuất feature từ mỗi ảnh

# In[26]:


def getLBP_feature(image):
    feature = local_binary_pattern(image, P=8, R=0.5)
    return np.histogram(feature, bins=list(range(257)))[0]

feature_LBP = list(map(getLBP_feature, faces.images))
feature_LBP = np.array(feature_LBP)


# <a id='baitap3_kmean'></a>
# ## 3.2 KMean Clustering

# In[27]:


# import library
from sklearn.cluster import KMeans


# In[28]:


model_kmean = KMeans(n_clusters=7)


# In[29]:


label_kmean = model_kmean.fit_predict(feature_LBP)


# ### Cross-table

# In[30]:


# import library
import pandas as pd


# In[31]:


df = pd.DataFrame({'label':label_kmean, 'True Label':faces.target})
ct = pd.crosstab(df['label'], df['True Label'])
print(ct.tail(10))


# - Nhận xét hiện tại: 
#     - Kết quả trả về của KMean không hiệu quả. Khó phân biệt label đúng cho mỗi vùng. 

# ### Cluster by Spectral Clustering

# In[44]:


# import library
from sklearn.cluster import spectral_clustering
from sklearn.metrics.pairwise import cosine_similarity 

graph = cosine_similarity(feature_LBP)
label_spectral = spectral_clustering(graph, n_clusters=7)


# - Cross-table 

# In[45]:


# create a DataFrame with labels and truth lables of digits data
df = pd.DataFrame({'labels': label_spectral, 'Truth_labels': faces.target})

# Create a cross-tablutation
ct = pd.crosstab(df['labels'], df['Truth_labels'])

print(ct)


# - Dựa trên Cross Table: 
#     - Các hình thuộc cùng một class được cluster dàn trải vào các vùng. Nên không xác định được vùng nào của class nào.
#     - Kết quả cluster cho ra thấp. 

# <a id='baitap3_dbscan'></a>
# ## 3.4 DBSCAN

# In[33]:


eps, min_samples = 0.015065,10

#import DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute')

label_dbscan = dbscan.fit_predict(feature_LBP)


# In[34]:


# print cross-table
df_dbscan = pd.DataFrame({'labels':label_dbscan,'Truth labels':faces.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# - Dựa trên Cross Table: 
#     - Có thể thấy số lượng điểm noise rất nhiều. 
#     - Phần nhiều hình được cluster vào chung một vùng dù khác class. 

# <a id='baitap3_agglomerative'></a>
# ## 3.5 Agglomerative Clustering

# In[36]:


# import library
from sklearn.cluster import AgglomerativeClustering

aggModel = AgglomerativeClustering(n_clusters=10)

label_agglomerative = aggModel.fit_predict(feature_LBP)


# In[37]:


# print cross-table
df_dbscan = pd.DataFrame({'labels':label_agglomerative,'Truth labels':faces.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# - Dựa trên Cross table: 
#     - Nhận thấy kết quả cluster không hiệu quả. 
#     - Các hình được cluster đồng đều các vùng. Không xác định được đâu là cluster đúng của một class. 
# - Dựa vào <a href='#baitap3_pca'>data visualization</a> phần nào đó (không chắc chắn, do giảm số chiều để visualize) cho thấy dữ liệu hiện tại được gom tụ thành một nhóm, mật độ cao ở trung tâm và hơi rời rạc ở rìa. Nên do đặc thù lan của DBSCAN không thể cluster tốt. 

# <a id='baitap3_visualize'></a>
# ## 3.6 Visualize kết quả của thuật toán phân lớp 

# <a id='baitap3_visualize_pca'></a>
# ### 3.6.1 PCA 

# In[41]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(feature_LBP)


# In[46]:


# create a fig to show image
fig = plt.figure(figsize=(11,10))

ax = fig.add_subplot(3, 2, 1)
ax.scatter(pca[:,0],pca[:,1], c=label_kmean)
ax.set_title('KMean')

ax = fig.add_subplot(3, 2, 2)
ax.scatter(pca[:,0],pca[:,1], c=label_spectral)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(3, 2, 3)
ax.scatter(pca[:,0],pca[:,1], c=faces.target)
ax.set_title('True Label')

ax = fig.add_subplot(3, 2, 5)
ax.scatter(pca[:,0],pca[:,1], c=label_agglomerative)
ax.set_title('Agglomerative Clustering')

ax = fig.add_subplot(3, 2, 6)
ax.scatter(pca[:,0],pca[:,1], c=label_dbscan)
ax.set_title('DBSCAN')

plt.show()


# - Từ hình trên cho thấy, khi sử dụng thuật toán PCA để giảm số chiều và visualize data gây khó khăn cho việc xác định vùng bằng mắt (True Label). 
# - Biểu đồ True Label cho thấy các class được trộn lẫn vào nhau, trong khi đó KMean và Agglomerative Clustering lại cho ra các vùng riêng biệt. Spectral clustering cho kết quả khả quan hơn, khi các vùng có trộn lẫn, có vẻ giống với True Label.

# <a id='baitap3_visualize_tsne'></a>
# ### 3.6.2 Visualize by T-SNE
# - T-SNE giảm số chiều của data về số liệu 2 chiều. 

# In[47]:


from sklearn.manifold import TSNE

TSNE_model = TSNE(learning_rate=100)
tnse = TSNE_model.fit_transform(faces.data)


# In[49]:


# create a fig to show image
fig = plt.figure(figsize=(11,11))

ax = fig.add_subplot(3, 2, 1)
ax.scatter(tnse[:,0],tnse[:,1], c=label_kmean)
ax.set_title('KMean')

ax = fig.add_subplot(3, 2, 2)
ax.scatter(tnse[:,0],tnse[:,1], c=label_spectral)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(3, 2, 3)
ax.scatter(tnse[:,0],tnse[:,1], c=faces.target)
ax.set_title('True Label')

ax = fig.add_subplot(3, 2, 5)
ax.scatter(tnse[:,0],tnse[:,1], c=label_agglomerative)
ax.set_title('Agglomerative Clustering')

ax = fig.add_subplot(3, 2, 6)
ax.scatter(tnse[:,0],tnse[:,1], c=label_dbscan)
ax.set_title('DBSCAN')

plt.show()


# <a id='baitap3_evaluate'></a>
# ## 3.7 Evaluate clustering algorithm

# - Sử dụng các hệ đo lường để đánh giá thuật toán: Homogeneity, Completeness, V-measure, Adjusted Random, Adjusted Mutual Information.

# In[51]:


from sklearn import metrics
def compareAlgorithm(algorithms, targetLabel, data):
    
    print('#Sample: %d\t#Class: %d\t#feature: %d'%(data.shape[0], len(set(label_kmean)),data.shape[1]))
    print(82*'_')
    print('init\t\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    for algorithm in algorithms:
        print('%-9s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
              % (algorithm,
                 metrics.homogeneity_score(targetLabel, algorithms[algorithm]),
                 metrics.completeness_score(targetLabel, algorithms[algorithm]),
                 metrics.v_measure_score(targetLabel, algorithms[algorithm]),
                 metrics.adjusted_rand_score(targetLabel, algorithms[algorithm]),
                 metrics.adjusted_mutual_info_score(targetLabel,  algorithms[algorithm]),
                 metrics.silhouette_score(data, algorithms[algorithm],
                                          metric='euclidean',
                                          sample_size=300)))
    print(82*'_')
compareAlgorithm({'KMean':label_kmean, 'Spectral':label_spectral,'DBSCAN':label_dbscan,'Agglomerative':label_agglomerative},                 faces.target, feature_LBP)

