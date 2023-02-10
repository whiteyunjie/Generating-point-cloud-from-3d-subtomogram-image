import mrcfile
import os
import numpy as np


def read_data(path):
    mrc = mrcfile.open(path,mode='r',permissive=True)
    a = mrc.data
    assert a.shape[0] > 0
    a = a.astype(np.float32) 
    # shrec2020，扁平状的，做tranpose好一点
    #a = a.transpose([2,1,0])
    
    return a

def pc_normalization(img,reverse=False):
    img_pro = (img-np.min(img))/(np.max(img)-np.min(img))
    if reverse:
        img_pro = 1-img_pro
    return img_pro

def cut_from_whole_map_pad(whole_map, c, siz, default_val):
    vb = np.full((int(siz),int(siz),int(siz)),default_val) #set non-enclosed regions as default value
    siz_h = np.ceil( siz / 2.0 )

    start = c - siz_h;      start.astype(int)
    end = start + siz;      end.astype(int)

    start_vb = start.copy()
    start[np.where(start < 0)] = 0
    end[np.where(end > whole_map.shape)] = np.array(whole_map.shape)[end > whole_map.shape]

    se = np.zeros( (3,2), dtype=np.int16)
    se[:,0] = start
    se[:,1] = end

    se_vb = se.copy()

    se_vb[:,0] = se_vb[:,0] - start_vb
    se_vb[:,1] = se_vb[:,1] - start_vb
    #print(se_vb)
    #print(whole_map)
    vb[se_vb[0,0]:se_vb[0,1], se_vb[1,0]:se_vb[1,1], se_vb[2,0]:se_vb[2,1]] = whole_map[se[0,0]:se[0,1], se[1,0]:se[1,1], se[2,0]:se[2,1]]

    return vb

def write_data(data, path, overwrite=False):
    assert data.ndim == 3  # only for 3D array

    data = data.astype(np.float32)
    #data = data.transpose([2,1,0])        # this is according to tomominer.image.vol.eman2_util.numpy2em
    with mrcfile.new(path, overwrite=overwrite) as m:
        m.set_data(data)

def voxel_reshape(img,alpha=1,voxelvalue=False):
    data = img
    size = data.shape
    data_p = data.reshape((-1,1))
    data_p.shape
    if voxelvalue:
        data_point = np.zeros((len(data_p),4))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(size[1]*size[2])
            p2 = (pi-size[1]*size[2]*p1)//size[2]
            p3 = pi-size[1]*size[2]*p1-size[2]*p2
            data_point[i,:] = [p1,p2,p3,alpha*data_p[i]]
    else:
        data_point = np.zeros((len(data_p),3))
        for i in range(len(data_p)):
            pi = i
            p1 = pi//(size[1]*size[2])
            p2 = (pi-size[1]*size[2]*p1)//size[2]
            p3 = pi-size[1]*size[2]*p1-size[2]*p2
            data_point[i,:] = [p1,p2,p3]
    return data_point
