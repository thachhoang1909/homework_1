
# coding: utf-8

# # Mục Lục
# ## <a href='#baitap1'>1. Bài tập 1</a>
# ## <a href='#baitap2'>2. Bài tập 2</a>
# ### <a href='#baitap2_kmean'>2.1 KMean</a>
# ### <a href='#baitap2_spectral'>2.2Spectral Clustering</a>
# ### <a href='#baitap2_dbscan'>2.3 DBSCAN</a>
# ### <a href='#baitap2_agglomerative'>2.4 Agglomerative Clustering</a>
# ### <a href='#baitap2_visualize'>2.5 Visualization</a>
# #### <a href='#baitap2_visualize_pca'>2.5.1 PCA</a>
# #### <a href='#baitap2_visualize_tsne'>2.5.2 T-SNE</a>
# ### <a href='#baitap2_evaluate'>2.6 Evaluation </a>
# 
# ## <a href='#baitap3'>Bài tập 3</a>
# ### <a href='#baitap3_feature'>3.1 Trích Xuất Đặc Trưng LBP</a>
# ### <a href='#baitap3_kmean'>3.2 KMean</a>
# ### <a href='#baitap3_spectral'>3.3 Spectral Clustering</a>
# ### <a href='#baitap3_dbscan'>3.4 DBSCAN</a>
# ### <a href='#baitap3_agglomerative'>3.5 Agglomerative Clustering</a>
# ### <a href='#baitap3_visualize'>3.6 Visualization</a>
# #### <a href='#baitap3_visualize_pca'>3.6.1 PCA</a>
# #### <a href='#baitap3_visualize_tsne'>3.6.2 T-SNE</a>
# ### <a href='#baitap3_evaluate'>3.7 Evaluation </a>
# 
# ## <a href='#baitap4'>Bài tập 4</a>
# ### <a href='#baitap4_feature'>4.1 Trích Xuất Đặc Trưng HoG </a>
# ### <a href='#baitap4_clustering'>4.2 Áp dụng thuật toán KMean, Spectral, Agglomerative Clustering</a>
# 
# ## <a href='#thamkhao'>5. Tham Khảo</a>

# <A>

# <A>

# <A>
#     

# <A>

# <A>

# ## Link Github: https://github.com/thachhoang1909/homework_1

# <a id='baitap1'></a>
# # Bài tập 1: KMeans trên bộ random 2 Gaussian

# - Import hàm KMean từ thư viện scikit-learn.
# - Hàm make_blobs tạo bộ dữ liệu ngẫu nhiên.

# In[203]:


# import library
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#plt.gray()
get_ipython().magic('matplotlib inline')


# - Phát sinh ngẫu nhiên bộ dữ liệu gồm 2 chiều (feature) và có 2 tâm (2 vùng).
# - X gồm 2 cột (feature), 150 dòng (sample).
# - Y là ID vùng của mỗi điểm mà nó thuộc về.

# In[204]:


X,Y = make_blobs(n_samples=200, n_features=2, centers=2)


# In[205]:


plt.scatter(X[:,0], X[:,1])
plt.show()


# - Bên trên là visualize bộ dữ liệu X, gồm 2 vùng phân biệt.
# - Tạo KMean model với k=2
# - Hàm KMeans.fit_predict chạy thuật toán KMean clustering trên bộ dữ liệu X và trả về label vùng của mỗi điểm. 

# In[206]:


model = KMeans(n_clusters=2)
label = model.fit_predict(X)


# ### Visualize kết quả clustering

# In[207]:


plt.scatter(X[:,0], X[:,1], c=label, alpha=0.5)

centroids = model.cluster_centers_
centroid_X = centroids[:,0]
centroid_Y = centroids[:,1]

plt.scatter(centroid_X, centroid_Y, marker='D', s=50)

plt.show()


# <a id='baitap2'></a>
# # 2. Bài tập 2: Hand-writen digits

# In[208]:


# Import library
from sklearn.cluster import KMeans
import sklearn.datasets as datasets
import pandas as pd


# In[209]:


# Load dataset
digits = datasets.load_digits()


# In[210]:


print('Dataset info: ', digits.images.shape)
# Import matplotlib to show data
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
#plt.gray()
plt.matshow(digits.images[0], cmap=plt.cm.binary)
plt.title('An image in Dataset')
plt.axis('off')


# - Dataset gồm:
#     - 1797 hình 8x8 đã được gán nhãn (digits.target).
#     - 10 lớp. 
# 

# <a id='baitap2_kmean'></a>
# ## 2.1 KMean Clustering

# In[211]:


# Apply KMeans clustering to data
number_clusters = 10
#create KMeans model
model = KMeans(n_clusters=number_clusters)

# fit model to data and predict on this data
label_kmean = model.fit_predict(digits.data)


# In[212]:


# create a DataFrame with labels and truth lables of digits data
df = pd.DataFrame({'labels': label_kmean, 'Truth_labels': digits.target})
# Create a cross table
ct = pd.crosstab(df['labels'], df['Truth_labels'])
print(ct)


# #### - Dựa trên cross table có thể thấy: 
#     - Có sự nhập nhằng khó xác định cluster của hình digit 1 và hình digit 8.
#     - Thông qua cluster center, có thể nhận ra 99 hình digit 1 được cho vào cluster của digit 8.
#     - Không có hình noise.

# 
# #### - Clustering Center

# In[213]:


# create a fig to show image
fig = plt.figure(figsize=(8,3))
plt.title('Cluster Center')
plt.axis('off')
# for all 0-9 labels
for i in range(10):
    # initialize subplots in a grid 2x5 at i+1th position
    ax = fig.add_subplot(2, 5, 1+i)
    
    # display image
    ax.imshow(model.cluster_centers_[i].reshape((8,8)), cmap=plt.cm.binary)
    #don't show the axes
    plt.axis('off')

plt.show()


# <a id='baitap2_spectral'></a>
# ## 2.2 Spectral Clustering

# In[214]:


# import library
from sklearn.cluster import spectral_clustering
from sklearn.metrics.pairwise import cosine_similarity 


# - Tính ma trận tương đồng giữa các sample
# - Áp dụng thuật toán spectral_clustering trên ma trận tương đồng vừa tính.

# In[215]:


graph = cosine_similarity(digits.data)
label_spectral = spectral_clustering(graph, n_clusters=10)


# #### Cross-table 

# In[216]:


# create a DataFrame with labels and truth lables of digits data
df = pd.DataFrame({'labels': label_spectral, 'Truth_labels': digits.target})

# Create a cross-tablutation
ct = pd.crosstab(df['labels'], df['Truth_labels'])

print(ct)


# #### - Dựa trên cross table có thể thấy: 
#     - Có sự nhập nhằng khó xác định cluster của hình digit 1 và hình digit 8.
#     - Thông qua cluster center, có thể nhận ra phần lớn hình digit 1 được cho vào cluster của digit 8.
#     - Không có hình noise.

# <a id='baitap2_dbscan'></a>
# ## 2.3 DBSCAN <a href='#3'>[3]</a>

# Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996). <a href='#3'>[3]</a>"A density-based
# algorithm for discovering clusters in large spatial databases with noise". Proceedings of
# the Second International Conference on Knowledge Discovery and Data Mining (KDD-
# 96). AAAI Press. pp. 226-231 

# - DBSCAN là thuật toán gom tụm dựa trên mật độ, hiệu quả với cơ sở dữ
# liệu lớn, có khả năng xử lý nhiễu.
# - Ý tưởng chính của thuật toán là vùng lân cận mỗi đối tượng trong một cụm
# có số đối tượng lớn hơn ngưỡng tối thiểu. Hình dạng vùng lân cận phụ
# thuộc vào hàm khoảng cách giữa các đối tượng (khoảng cách Manhattan,
# khoảng cách Euclidean).
# - Độ phức tạp thuật toán: tính toán (N*logN), dữ liệu (N^2).
# - Thuật toán cơ bản:
#     - Gồm 2 tham số: eps và minPts
#     - Từ một mẫu (nút) chưa được chọn, kiểm tra các điểm gần nhất, nếu
# số lượng các điểm này lớn hơn giá trị minPts thfi bắt đầu một nhóm
# mới. Nếu không sẽ đánh dấu là điểm nhiễu. Điểm nhiễu này vẫn có
# thể thuộc một nhóm khác, khi đó sẽ bỏ đánh dấu điểm nhiễu.
#     - Cứ thế mở rộng ra đến khi không thể tìm thêm điểm mới cho nhóm.

# In[217]:


eps, min_samples = 0.0595 , 10

#import DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute')

label_dbscan = dbscan.fit_predict(digits.data)



# #### Cross table

# In[218]:


# print cross-table
df_dbscan = pd.DataFrame({'labels':label_dbscan,'Truth labels':digits.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# #### - Dựa trên cross table có thể thấy: 
#     - Có sự nhập nhằng khó xác định cluster của hình digit 3 và hình digit 9.
#     - Thông qua cluster center, có thể nhận ra phần lớn hình digit 3 được cho vào cluster của digit 9.
#     - Có khá là nhiều hình không bị gán noise, không tìm được cluster. 

# <a id='baitap2_agglomerative'></a>
# ## 2.4 Agglomerative Clustering

# In[219]:


# import library
from sklearn.cluster import AgglomerativeClustering

aggModel = AgglomerativeClustering(n_clusters=10)

label_agglomerative = aggModel.fit_predict(digits.data)


# #### Cross table

# In[220]:


# print cross-table
df_dbscan = pd.DataFrame({'labels':label_agglomerative,'Truth labels':digits.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# #### - Dựa trên cross table có thể thấy: 
#     - Có sự nhập nhằng khó xác định cluster của hình digit 3 và hình digit 9.
#     - Thông qua cluster center, có thể nhận ra phần lớn hình digit 3 được cho vào cluster của digit 9.
#     - Không có hình noise.

# <a id='baitap2_visualize'></a>
# ## 2.5 Visualize kết quả của thuật toán phân lớp 

# <a id='baitap2_visualize_pca'></a>
# ### 2.5.1 PCA 

# - Sử dụng thuật toán PCA để giảm số chiều features.

# In[221]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit_transform(digits.data)


# In[222]:


# create a fig to show image
fig = plt.figure(figsize=(11,10))

ax = fig.add_subplot(3, 2, 1)
ax.scatter(pca[:,0],pca[:,1], c=label_kmean)
ax.set_title('KMean')

ax = fig.add_subplot(3, 2, 2)
ax.scatter(pca[:,0],pca[:,1], c=label_spectral)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(3, 2, 3)
ax.scatter(pca[:,0],pca[:,1], c=digits.target)
ax.set_title('True Label')

ax = fig.add_subplot(3, 2, 5)
ax.scatter(pca[:,0],pca[:,1], c=label_agglomerative)
ax.set_title('Agglomerative Clustering')

ax = fig.add_subplot(3, 2, 6)
ax.scatter(pca[:,0],pca[:,1], c=label_dbscan)
ax.set_title('DBSCAN')

plt.show()


# <a id='baitap2_visualize_tsne'></a>
# ### 2.5.2 Visualize by T-SNE
# - T-SNE giảm số chiều của data về số liệu 2 chiều. 

# In[225]:


from sklearn.manifold import TSNE

TSNE_model = TSNE(learning_rate=100)

tnse = TSNE_model.fit_transform(digits.data)


# In[226]:


# create a fig to show image
fig = plt.figure(figsize=(11,11))

ax = fig.add_subplot(3, 2, 1)
ax.scatter(tnse[:,0],tnse[:,1], c=label_kmean)
ax.set_title('KMean')

ax = fig.add_subplot(3, 2, 2)
ax.scatter(tnse[:,0],tnse[:,1], c=label_spectral)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(3, 2, 3)
ax.scatter(tnse[:,0],tnse[:,1], c=digits.target)
ax.set_title('True Label')

ax = fig.add_subplot(3, 2, 5)
ax.scatter(tnse[:,0],tnse[:,1], c=label_agglomerative)
ax.set_title('Agglomerative Clustering')

ax = fig.add_subplot(3, 2, 6)
ax.scatter(tnse[:,0],tnse[:,1], c=label_dbscan)
ax.set_title('DBSCAN')

plt.show()


# <a id='baitap2_evaluate'></a>
# ## 2.6 Evaluate clustering algorithm

# - Sử dụng các hệ đo lường để đánh giá thuật toán: Homogeneity, Completeness, V-measure, Adjusted Random, Adjusted Mutual Information.

# In[227]:


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


# In[228]:


compareAlgorithm({'KMean':label_kmean, 'Spectral':label_spectral,'DBSCAN':label_dbscan,'Agglomerative':label_agglomerative},                 digits.target, digits.data)


# ### Nhận xét:
# - Dựa trên bảng số liệu. Có thể thấy thuật toán Agglomerative cho ra kết quả chính xác hơn DBSCAN, Spectral, KMean với raw digit data.

# <a id='baitap3'></a>
# # 3. Bài tập 3: Face Dataset

# In[229]:


# import libray
from sklearn.cluster import KMeans, DBSCAN, spectral_clustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
#plt.gray()
get_ipython().magic('matplotlib inline')


# In[230]:


faces = fetch_lfw_people(min_faces_per_person=70)


# In[231]:


print("Number of images: ", faces.images.shape)
print("Number of classes: ",len(set(faces.target)))


# ### Dataset
#     - Dataset gồm 1288 ảnh 62 x 47.
#     - Đã được label với target gồm 5749 class. 
#     - Mỗi class gồm ảnh của một người. 
#     - Mỗi class gồm ít nhất 70 ảnh. 

# In[232]:


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


# <a id='baitap3_feature'></a>
# ## 3.1 Trích xuất đặc trưng
# 
#     - Sử dụng Local Binary Pattern.

# In[233]:


# import feature detector & descriptor library
from skimage.feature import local_binary_pattern

feature = local_binary_pattern(faces.images[0], P=8, R=0.5)
plt.matshow(feature, cmap=plt.cm.binary)


# In[234]:


import numpy as np

fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(1,1,1)
ax.hist(feature.reshape(-1), bins=list(range(257)))
plt.title('256-dimensional feature vector')
plt.show()


# ### Trích xuất feature từ mỗi ảnh

# In[235]:


def getLBP_feature(image):
    feature = local_binary_pattern(image, P=8, R=0.5)
    return np.histogram(feature, bins=list(range(257)))[0]


# In[236]:


feature_LBP = list(map(getLBP_feature, faces.images))
feature_LBP = np.array(feature_LBP)


# <a id='baitap3_kmean'></a>
# ## 3.2 KMean Clustering

# In[237]:


model_kmean = KMeans(n_clusters=7)
label_kmean = model_kmean.fit_predict(feature_LBP)

df = pd.DataFrame({'label':label_kmean, 'True Label':faces.target})
ct = pd.crosstab(df['label'], df['True Label'])
print(ct.tail(10))


# - Nhận xét hiện tại: 
#     - Kết quả trả về của KMean không hiệu quả. Khó phân biệt label đúng cho mỗi vùng. 

# <a id='baitap3_spectral'></a>
# ## 3.3 Spectral Clustering

# In[238]:


# import library
from sklearn.cluster import spectral_clustering
from sklearn.metrics.pairwise import cosine_similarity 

graph = cosine_similarity(feature_LBP)
label_spectral = spectral_clustering(graph, n_clusters=7)


# - Cross-table 

# In[239]:


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

# In[240]:


eps, min_samples = 0.015065,10

#import DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine', algorithm='brute')

label_dbscan = dbscan.fit_predict(feature_LBP)


# In[241]:


# print cross-table
df_dbscan = pd.DataFrame({'labels':label_dbscan,'Truth labels':faces.target})
ct_dbscan=pd.crosstab(df_dbscan['labels'],df_dbscan['Truth labels'])
print(ct_dbscan)


# - Dựa trên Cross Table: 
#     - Có thể thấy số lượng điểm noise rất nhiều. 
#     - Phần nhiều hình được cluster vào chung một vùng dù khác class. 

# <a id='baitap3_agglomerative'></a>
# ## 3.5 Agglomerative Clustering

# In[242]:


# import library
from sklearn.cluster import AgglomerativeClustering

aggModel = AgglomerativeClustering(n_clusters=10)

label_agglomerative = aggModel.fit_predict(feature_LBP)


# #### Cross table

# In[243]:


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

# In[244]:


pca = PCA(n_components=2).fit_transform(feature_LBP)


# In[245]:


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

# In[246]:


tnse = TSNE_model.fit_transform(faces.data)


# In[247]:


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

# In[250]:


compareAlgorithm({'KMean':label_kmean, 'Spectral':label_spectral,'DBSCAN':label_dbscan,'Agglomerative':label_agglomerative},                 faces.target, feature_LBP)


# <a id='baitap4'></a>
# # 4. Bài tập 4: Human Dataset

# - Mục tiêu: Gom nhóm hình có sự xuất hiện của con người và không có con người. 
# - Dataset: INRIA Person Dataset.
# 

# In[251]:


# import library
from scipy import misc
import glob

images = glob.glob('dataset/*.png')
# create a fig to show image
fig = plt.figure(figsize=(10,5))

plt.title('Some samples in dataset')
plt.axis('off')
# for all 0-9 labels
k= 0
for i in range(10):
    # initialize subplots in a grid 2x5 at i+1th position
    if(i==4):
        k+= 600
    image = misc.imread(images[i+k], mode='L')
    ax = fig.add_subplot(2, 5, 1+i) 
    # display image
    ax.imshow(image, cmap=plt.cm.binary)
    
    #don't show the axes
    plt.axis('off')
plt.show()


# - Dataset gồm 1044 64x47. Gồm 2 class: 558 hình person và 486 hình non-person

# - Tiến hành rút trích HOG feature từ mỗi hình và lưu lại.
# 

# <a id='baitap4_feature'></a>
# ## 4.1 Extract HoG Feature

# - Sử dụng hàm HoG của thư viện Skimage để thực hiện rút trích đặc trưng HoG từ mỗi ảnh. 

# In[258]:


from skimage.feature import hog
from skimage import data, color, exposure

target = [] # To get true lable
fds, hog_images = [],[]
for imagePath in images:
    
    if(imagePath[8]=='c'):
        target.append(1)
    else:
        target.append(0)
    
    image = misc.imread(imagePath, mode='L')
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualise=True)
    
    fds.append(fd)
    hog_images.append(hog_image)

# Convert fds, hog_image from list to numpy array
feature_HOG = np.array(fds)
hog_images = np.array(hog_images) 


# In[259]:


fig = plt.figure(figsize=(15,5))
plt.title('Two sample of HOG feature')
plt.axis('off')

image = misc.imread(images[0], mode='L')
ax = fig.add_subplot(1,4,1)
ax.imshow(image,cmap=plt.cm.binary)
ax.set_title('Orignal Image')
ax = fig.add_subplot(1,4,2)
ax.imshow(hog_images[0],cmap=plt.cm.binary)
ax.set_title('HOG Image')

image = misc.imread(images[800], mode='L')
ax = fig.add_subplot(1,4,3)
ax.imshow(image,cmap=plt.cm.binary)
ax.set_title('Orignal Image')
ax = fig.add_subplot(1,4,4)
ax.imshow(hog_images[0],cmap=plt.cm.binary)
ax.set_title('HOG Image')

plt.show()


# <a id='baitap4_clustering'></a>
# ## 4.2 Áp dụng thuật toán KMean, Spectral, Agglomerative Clustering

# ### KMean

# In[260]:


model_kmean = KMeans(n_clusters=2)
label_kmean = model_kmean.fit_predict(feature_HOG)

print("Cross Table")
print(50*'_')
df = pd.DataFrame({'label':label_kmean, 'True Label':target})
ct = pd.crosstab(df['label'], df['True Label'])
print(ct)


# ###  Spectral Clustering

# In[261]:


graph = cosine_similarity(feature_HOG)
label_spectral = spectral_clustering(graph, n_clusters=2)

print("Cross Table")
print(50*'_')
df = pd.DataFrame({'label':label_spectral, 'True Label':target})
ct = pd.crosstab(df['label'], df['True Label'])
print(ct)


# ### Agglomerative Clustering

# In[262]:


aggModel = AgglomerativeClustering(n_clusters=2)
label_agglomerative = aggModel.fit_predict(feature_HOG)

print("Cross Table")
print(50*'_')
df = pd.DataFrame({'label':label_agglomerative, 'True Label':target})
ct = pd.crosstab(df['label'], df['True Label'])
print(ct)


# - Nhận thấy cả 3 thuật toán đề không cho ra kết quả tốt. Khó xác định giữa 2 class.

# ### Visualize kết quả

# #### PCA

# In[263]:


pca = PCA(n_components=2).fit_transform(feature_HOG)


# In[264]:


# create a fig to show image
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(2, 2, 1)
ax.scatter(pca[:,0],pca[:,1], c=label_kmean)
ax.set_title('KMean')

ax = fig.add_subplot(2, 2, 2)
ax.scatter(pca[:,0],pca[:,1], c=label_spectral)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(2, 2, 3)
ax.scatter(pca[:,0],pca[:,1], c=label_agglomerative)
ax.set_title('Agglomerative')

ax = fig.add_subplot(2, 2, 4)
ax.scatter(pca[:,0],pca[:,1], c=target)
ax.set_title('True Label')

plt.show()


# ### T-SNE

# In[265]:


tnse = TSNE_model.fit_transform(feature_HOG)


# In[266]:


# create a fig to show image
fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(2, 2, 1)
ax.scatter(tnse[:,0],tnse[:,1], c=label_kmean)
ax.set_title('KMean')

ax = fig.add_subplot(2, 2, 2)
ax.scatter(tnse[:,0],tnse[:,1], c=label_spectral)
ax.set_title('Spectral Clustering')

ax = fig.add_subplot(2, 2, 3)
ax.scatter(tnse[:,0],tnse[:,1], c=label_agglomerative)
ax.set_title('Agglomerative')

ax = fig.add_subplot(2, 2, 4)
ax.scatter(tnse[:,0],tnse[:,1], c=target)
ax.set_title('True Label')

plt.show()


# ### Evaluate

# In[267]:


compareAlgorithm({'KMean':label_kmean, 'Spectral':label_spectral,'Agglomerative':label_agglomerative},                 target, feature_HOG)


# ### Nhận xét: 
# - Kết quả không tốt. Với dữ liệu khó có thể cluster thành 2 nhóm phân biệt person và non-person. 
# - Nguyên nhân: có thể do input đầu vào là hình grayscale, một số hình bị quá sáng, mất mát thông tin trong quá trình chuyển ảnh màu sang ảnh grayscale.
# 

# <a id='thamkhao'></a>
# # 5. Tham Khảo

# 1. <a href='http://scikit-learn.org/stable/modules/clustering.html'>http://scikit-learn.org/stable/modules/clustering.html</a>
# 2. <a href='http://pascal.inrialpes.fr/data/human/'>INRIA Person Dataset</a>
# 3. <a href='http://www.lsi.upc.edu/~bejar/amlt/material_art/DM%20clustring%20DBSCAN%20kdd-96.pdf'>Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996). "A density-based algorithm for discovering clusters in large spatial databases with noise".</a>
# 4. <a href='http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html'>A demo of K-Means clustering on the handwritten digits data</a>
