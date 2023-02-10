import numpy as np
from sklearn.neighbors import KDTree
from scipy.cluster.vq import kmeans,vq
from metric import metric_of_clustering,metric_of_clustering_SNR
from skimage.filters import threshold_otsu


class VoxelBinarization:
    def __init__(self,dataset,p_to_index,size,quantile):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.size = size
        self.quantile = quantile

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        
        #if self.quantile == 'otsu':
        threshold = threshold_otsu(data_point[:,3])
        # else:
        #     threshold = np.percentile(data_point[:,3],self.quantile)
        point_pro = data_point[data_point[:,3]>threshold]
        point_cloud = point_pro.astype(np.int32)
        
        return point_cloud

class VoxelBasedCluster:
    def __init__(self,dataset,p_to_index,n_clusters,size):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.size = size
        self.n_clusters = n_clusters

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        ## set features for clustering
        features = data_point[:,3].reshape((-1,1))
        center = kmeans(features,self.n_clusters)[0]
        label = vq(features,center)[0]
        ## select prominent cluster
        al = np.argmax(center[:,0])
        index = np.where(label==al)[0]
        point_cloud = data_point[index]
        point_cloud = point_cloud.astype(np.int32)
        
        return point_cloud

class VoxelBasedCluster_Voxelfilter:
    def __init__(self,dataset,p_to_index,n_clusters,quantile,size):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.size = size
        self.n_clusters = n_clusters
        self.quantile = quantile

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        ## set features for clustering
        threshold = np.percentile(data_point[:,3],self.quantile)
        point_pro = data_point[data_point[:,3]>threshold]
        features = point_pro[:,3].reshape((-1,1))
        center = kmeans(features,self.n_clusters)[0]
        label = vq(features,center)[0]
        ## select prominent cluster
        al = np.argmax(center[:,0])
        index = np.where(label==al)[0]
        point_cloud = point_pro[index]
        point_cloud = point_cloud.astype(np.int32)
        
        return point_cloud

class DensityBasedCluster:
    def __init__(self,dataset,p_to_index,n_clusters,quantile,radius,size):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.quantile = quantile
        self.radius = radius
        self.size = size
        self.n_clusters = n_clusters

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        
        ## calculate neightbour density
        threshold = np.percentile(data_point[:,3],self.quantile)
        point_pro = data_point[data_point[:,3]>threshold]
        kdtree = KDTree(point_pro[:,:3])
        density = kdtree.query_radius(data_point[:,:3],self.radius)
        densities = np.array([len(den) for den in density])
        densities = densities/(np.max(densities))
        
        ## set features for clustering
        features = np.stack((densities,data_point[:,3]),1)
        center = kmeans(features,self.n_clusters)[0]
        label = vq(features,center)[0]

        ## select prominent cluster
        al = np.argmax(center[:,0])
        index = np.where(label==al)[0]
        point_cloud = data_point[index]
        point_cloud = point_cloud.astype(np.int32)
        
        return point_cloud

class DensityBasedCluster_VoxelFilter:
    def __init__(self,dataset,p_to_index,n_clusters,quantile,radius,size):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.quantile = quantile
        self.radius = radius
        self.size = size
        self.n_clusters = n_clusters

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        
        ## calculate neightbour density
        threshold = np.percentile(data_point[:,3],self.quantile)
        point_pro = data_point[data_point[:,3]>threshold]
        kdtree = KDTree(point_pro[:,:3])
        density = kdtree.query_radius(point_pro[:,:3],self.radius)
        densities = np.array([len(den) for den in density])
        densities = densities/(np.max(densities))
        
        ## set features for clustering
        features = np.stack((densities,point_pro[:,3]),1)
        center = kmeans(features,self.n_clusters)[0]
        label = vq(features,center)[0]

        ## select prominent cluster
        al = np.argmax(center[:,0])
        index = np.where(label==al)[0]
        point_cloud = point_pro[index]
        point_cloud = point_cloud.astype(np.int32)
        
        return point_cloud

class DensityBasedCluster_DenFilter:
    def __init__(self,dataset,p_to_index,n_clusters,quantile,radius,size,den_thrshld):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.quantile = quantile
        self.radius = radius
        self.size = size
        self.n_clusters = n_clusters
        self.den_thrshld = den_thrshld

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        
        ## calculate neightbour density
        threshold = np.percentile(data_point[:,3],self.quantile)
        point_pro = data_point[data_point[:,3]>threshold]
        kdtree = KDTree(point_pro[:,:3])
        density = kdtree.query_radius(data_point[:,:3],self.radius)
        densities = np.array([len(den) for den in density])
        densities = densities/(np.max(densities))
        
        ## density filter
        densities_pro = densities[densities>self.den_thrshld]
        data_point_pro = data_point[densities>self.den_thrshld]

        ## set features for clustering
        features = np.stack((densities_pro,data_point_pro[:,3]),1)
        center = kmeans(features,self.n_clusters)[0]
        label = vq(features,center)[0]

        ## select prominent cluster
        al = np.argmax(center[:,0])
        index = np.where(label==al)[0]
        point_cloud = data_point_pro[index]
        point_cloud = point_cloud.astype(np.int32)
        
        return point_cloud

class DensityBasedCluster_BothFilter:
    def __init__(self,dataset,p_to_index,n_clusters,quantile,radius,size,den_thrshld):
        self.dataset = dataset
        self.p_to_index = p_to_index
        self.quantile = quantile
        self.radius = radius
        self.size = size
        self.n_clusters = n_clusters
        self.den_thrshld = den_thrshld

    def clustering(self,img3d):
        data = img3d
        ## flatten 3D image --> x,y,z,voxel
        data_p = data.reshape((-1,1))
        data_p.shape
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(self.size*self.size)
            p2 = (pi-self.size*self.size*p1)//self.size
            p3 = pi-self.size*self.size*p1-self.size*p2
            data_point[i,:] = [p1,p2,p3,data_p[i]]
        
        ## calculate neightbour density
        threshold = np.percentile(data_point[:,3],self.quantile)
        point_pro = data_point[data_point[:,3]>threshold]# voxel filter
        kdtree = KDTree(point_pro[:,:3])
        density = kdtree.query_radius(point_pro[:,:3],self.radius)
        densities = np.array([len(den) for den in density])
        densities = densities/(np.max(densities))

        ## density filter
        densities_pro = densities[densities>self.den_thrshld]
        data_point_pro = point_pro[densities>self.den_thrshld]
        
        ## set features for clustering
        features = np.stack((densities_pro,data_point_pro[:,3]),1)
        center = kmeans(features,self.n_clusters)[0]
        label = vq(features,center)[0]

        ## select prominent cluster
        al = np.argmax(center[:,0])
        index = np.where(label==al)[0]
        point_cloud = data_point_pro[index]
        point_cloud = point_cloud.astype(np.int32)
        
        return point_cloud



