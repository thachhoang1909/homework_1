
# coding: utf-8

# # Bài tập 1: KMeans trên bộ random 2 Gaussian

# - Import hàm KMean từ thư viện scikit-learn
# - Hàm make_blobs tạo bộ dữ liệu ngẫu nhiên 

# In[1]:


# import library
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
plt.gray()
get_ipython().magic('matplotlib inline')


# - Phát sinh ngẫu nhiên bộ dữ liệu gồm 2 chiều (feature) và có 2 tâm (2 vùng).
# - X gồm 2 cột (feature), 150 dòng (sample).
# - Y là ID vùng của mỗi điểm mà nó thuộc về.

# In[2]:


X,Y = make_blobs(n_samples=200, n_features=2, centers=2)


# In[3]:


plt.scatter(X[:,0], X[:,1])
plt.show()


# - Bên trên là visualize bộ dữ liệu X, gồm 2 vùng phân biệt.
# - Tạo KMean model với k=2
# - Hàm KMeans.fit_predict chạy thuật toán KMean clustering trên bộ dữ liệu X và trả về label vùng của mỗi điểm. 

# In[4]:


model = KMeans(n_clusters=2)

label = model.fit_predict(X)


# - Visualize kết quả clustering

# In[10]:


plt.scatter(X[:,0], X[:,1], c=label, alpha=0.5)

centroids = model.cluster_centers_
centroid_X = centroids[:,0]
centroid_Y = centroids[:,1]

plt.scatter(centroid_X, centroid_Y, marker='D', s=50)

plt.show()

