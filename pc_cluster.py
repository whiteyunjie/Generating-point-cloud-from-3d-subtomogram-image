import numpy as np
from sklearn.neighbors import KDTree
from scipy.cluster.vq import kmeans,vq
from metric import metric_of_clustering,metric_of_clustering_SNR

def cluster_v(img,clsname,p_to_index,size=64):
    cls = p_to_index[clsname]
    data = img
    #size=24
    data_p = data.reshape((-1,1))
    data_p.shape
    data_point = np.zeros((len(data_p),4))
    for i in range(len(data_p)):
        pi = i
        p1 = pi//(size*size)
        p2 = (pi-size*size*p1)//size
        p3 = pi-size*size*p1-size*p2
        data_point[i,:] = [p1,p2,p3,data_p[i]]
    features = data_point[:,3].reshape((-1,1))
    center = kmeans(features,5)[0]
    label = vq(features,center)[0]
    al = np.argmax(center[:,0])
    index = np.where(label==al)[0]
    point = data_point[index]
    point = point.astype(np.int32)
    psa = point
    
    # pnew = {0:[],1:[]}
    # check = {i:0 for i in range(2)}
    # for pt in psa:
    #     clsidx = int(seg[pt[0],pt[1],pt[2]])
    #     if clsidx == 1:
    #         pnew[1].append(pt.tolist())
    #     else:
    #         pnew[0].append(pt.tolist())
    #     check[clsidx] += 1
    
    # r_clu_gt,r_clu_bg = metric_of_clustering_SNR(check,seg)
    return psa

def cluster_v_d(img,quantile,clsname,p_to_index,radius=3,size=64):
    ##set threshold with voxels and consider density
    cls = p_to_index[clsname]
    data = img
    #size=24
    data_p = data.reshape((-1,1))
    data_p.shape
    data_point = np.zeros((len(data_p),4))
    for i in range(len(data_p)):
        pi = i
        p1 = pi//(size*size)
        p2 = (pi-size*size*p1)//size
        p3 = pi-size*size*p1-size*p2
        data_point[i,:] = [p1,p2,p3,data_p[i]]
    ## set threshold
    threshold = np.percentile(data_point[:,3],quantile)
    point_pro = data_point[data_point[:,3]>threshold]
    kdtree = KDTree(point_pro[:,:3])
    density = kdtree.query_radius(data_point[:,:3],radius)
    densities = np.array([len(den) for den in density])
    densities = densities/(np.max(densities))

    ## filter
#     densities_pro = densities[densities>den_thrshld]
#     data_point_pro = data_point[densities>den_thrshld]
    
    features = np.stack((densities,data_point[:,3]),1)
    #cluster
    center = kmeans(features,3)[0]
    label = vq(features,center)[0]
    al = np.argmax(center[:,0])
    index = np.where(label==al)[0]
    point = data_point[index]
    point = point.astype(np.int)
    
    psa = point
    # pnew = {0:[],1:[]}
    # check = {i:0 for i in range(2)}
    # for pt in psa:
    #     clsidx = int(seg[pt[0],pt[1],pt[2]])
    #     if clsidx == 1:
    #         pnew[1].append(pt.tolist())
    #     else:
    #         pnew[0].append(pt.tolist())
    #     check[clsidx] += 1
    
    # #print(check)
    # r_clu_gt,r_clu_bg = metric_of_clustering_SNR(check,seg)
    return psa

def cluster_v_d_filter(img,quantile,den_thrshld,clsname,p_to_index,radius=3,size=64):
    data = img
    #size=24
    # 权重肯定要大些
    quantile = 80
    radius = 5
    den_thrshld = 0.3
    size = 64
    data_p = data.reshape((-1,1))
    data_p.shape
    data_point = np.zeros((len(data_p),4))
    for i in range(len(data_p)):
        pi = i
        p1 = pi//(size*size)
        p2 = (pi-size*size*p1)//size
        p3 = pi-size*size*p1-size*p2
        data_point[i,:] = [p1,p2,p3,data_p[i]]
    ## set threshold
    threshold = np.percentile(data_point[:,3],quantile)
    point_pro = data_point[data_point[:,3]>threshold]
    #print(data_point[:5])
    #print(point_pro[:5])
    kdtree = KDTree(point_pro[:,:3])
    density = kdtree.query_radius(data_point[:,:3],radius)
    densities = np.array([len(den) for den in density])
    densities = densities/(np.max(densities))

    ## filter
    densities_pro = densities[densities>den_thrshld]
    data_point_pro = data_point[densities>den_thrshld]
    #print(np.sum(densities>den_thrshld))
    #print(density[:5])
    #print(data_point_pro.shape)

    features = np.stack((densities_pro,data_point_pro[:,3]),1)
    #cluster
    center = kmeans(features,3)[0]
    label = vq(features,center)[0]
    al = np.argmax(center[:,0])
    index = np.where(label==al)[0]
    point = data_point_pro[index]
    point = point.astype(np.int)
    #print(len(index))

    psa = point
    return psa