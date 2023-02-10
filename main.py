import numpy as np
import pandas as pd
import mrcfile
import scipy.ndimage as sn
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import kmeans,vq
#from aitom.filter.gaussian import smooth
import os
from tqdm import tqdm
import json
from sklearn.neighbors import KDTree
from data_process import read_data,pc_normalization,write_data,voxel_reshape
#from pc_cluster import cluster_v,cluster_v_d,cluster_v_d_filter
from den_based_cluster import *
from metric import metric_of_clustering,metric_of_clustering_SNR
import argparse

def pc_norm(img):
    img = (img - np.min(img))/(np.max(img)-np.min(img))
    return img

parser = argparse.ArgumentParser(description='parameters')
## data
parser.add_argument('--dataset',type=str,default='SNR',choices=['SNR','Qiang'],help='subtomogram dataset')
parser.add_argument('--noise_level',type=str,default='none')
parser.add_argument('--data_dir',type=str,help='direction of dataset')

## method parameter
parser.add_argument('--method',type=str,default='baseline',
                    choices=['binary_otsu','baseline','baseline2','den_based','voxel_filter','den_filter','both_filter'])
parser.add_argument('--n_clusters',type=int,default=3,help='number of clusters')
parser.add_argument('--quantile',type=int,default=50,help='voxel quantile for filtering points when calculating density')
parser.add_argument('--den_thrshld',type=float,help='threshold of density for filtering')
parser.add_argument('--radius',type=int,default=3,help='neighbour radius for calculating density')
parser.add_argument('--mode_thrshld',type=float,default=1,help='threshold of modes for fitering')

## generation parameter
parser.add_argument('--stage',type=str,default='metric',help='metirc: measure with ground truth, test:only generate pc, no ground truth')
parser.add_argument('--save_dir',type=str,help='save path of generated data')
parser.add_argument('--n_points',type=int,default=0,help='whether fix the number of points, 0 means no limitaions')

args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'SNR':
        particle = ['background','1bxn','1f1b','1yg6','covid']
        size = 64
        clsnameidx = 0
    if args.dataset == 'Qiang':
        particle = ['background','ribo','26S','TRiC']
        size = 24
        clsnameidx = 2
    p_to_index = {particle[i]:i for i in range(len(particle))}


    ## 可视化先不考虑


    ## voxel data : subtomogram dir
    datalist = []
    if args.noise_level != 'none':
        for datadir in os.listdir(f'{args.data_dir}/img_snr{args.noise_level}'):
            datalist.append(datadir.split('.')[0])
    else:
        for datadir in os.listdir(f'{args.data_dir}/img'):
            datalist.append(datadir.split('.')[0])
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.save_dir = os.path.join(args.save_dir,f'pc_data_{args.method}')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    pc_save_dir = os.path.join(args.save_dir,'pc_data')
    if not os.path.exists(pc_save_dir):
        os.makedirs(pc_save_dir)
    
    ## cluster method
    if args.method == 'binary_otsu':
        clusterer = VoxelBinarization(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    size=size,
                                    quantile=args.quantile)
    if args.method == 'baseline':
        clusterer = VoxelBasedCluster(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    n_clusters=args.n_clusters,
                                    size=size)
    if args.method == 'baseline2':
        clusterer = VoxelBasedCluster_Voxelfilter(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    n_clusters=args.n_clusters,
                                    quantile=args.quantile,
                                    size=size)
    if args.method == 'den_based':
        clusterer = DensityBasedCluster(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    n_clusters=args.n_clusters,
                                    quantile=args.quantile,
                                    radius=args.radius,
                                    size=size)
    if args.method == 'voxel_filter':
        clusterer = DensityBasedCluster_VoxelFilter(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    n_clusters=args.n_clusters,
                                    quantile=args.quantile,
                                    radius=args.radius,
                                    size=size)
    if args.method == 'den_filter':
        clusterer = DensityBasedCluster_DenFilter(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    n_clusters=args.n_clusters,
                                    quantile=args.quantile,
                                    radius=args.radius,
                                    size=size,
                                    den_thrshld=args.den_thrshld)
    if args.method == 'both_filter':
        clusterer = DensityBasedCluster_BothFilter(dataset=args.dataset,
                                    p_to_index=p_to_index,
                                    n_clusters=args.n_clusters,
                                    quantile=args.quantile,
                                    radius=args.radius,
                                    size=size,
                                    den_thrshld=args.den_thrshld)          

    
    # record cluster infomation
    clu_info_dir = os.path.join(args.save_dir,f'cluster_info_{args.method}.csv')
    
    if not os.path.exists(clu_info_dir):
        data_csv = open(clu_info_dir,'w')
        data_csv.write('pid,particle_name,particle_idx,total_points,particle_points,bg_points,R_clu_gt,R_clu_bg,R_other_clu\n')
        if args.noise_level != 'none':
            datapath = os.path.join(args.data_dir,f'img_snr{args.noise_level}')
            labelpath = os.path.join(args.data_dir,f'seg_snr{args.noise_level}')
        else:
            datapath = os.path.join(args.data_dir,f'img')
            labelpath = os.path.join(args.data_dir,f'seg')
        #print(datapath)
        for pid in tqdm(datalist):
            clsname = pid.split('_')[clsnameidx]
            cls = p_to_index[clsname]

            imgpath = os.path.join(datapath,f'{pid}.mrc')
            img = read_data(imgpath)
            #print(img[0,0,0])
            if args.dataset =='Qiang': img = pc_normalization(img,True)
            else: img = pc_norm(img)
            #print(pid)
            #print(img[0,0,0])
            

            ## generate point cloud
            cluster_pc = clusterer.clustering(img)
            
            if args.stage == 'metric':
                segpath = os.path.join(labelpath,f'{pid}_seg.mrc')
                seg = read_data(segpath)
                if args.dataset == 'SNR':
                    pnew = {0:[],1:[]}
                    check = {i:0 for i in range(2)}
                    for pt in cluster_pc:
                        clsidx = int(seg[pt[0],pt[1],pt[2]])
                        if clsidx == 1:
                            pnew[1].append(pt.tolist())
                        else:
                            pnew[0].append(pt.tolist())
                        check[clsidx] += 1
                    print(check)
                    r_clu_gt,r_clu_bg = metric_of_clustering_SNR(check,seg)
                    r_other_clu = 0
                    #print(r_clu_gt)
                    data_csv.write(f'{pid},{clsname},{cls},{cluster_pc.shape[0]},{check[1]},{check[0]},{r_clu_gt},{r_clu_bg},{r_other_clu}\n')

                else:
                    pnew = {0:[],cls:[]}
                    check = {i:0 for i in range(4)}
                    for pt in cluster_pc:
                        clsidx = int(seg[pt[0],pt[1],pt[2]])
                        if clsidx == cls:
                            pnew[cls].append(pt.tolist())
                        else:
                            pnew[0].append(pt.tolist())
                        check[clsidx] += 1
                    print(check)
                    r_clu_gt,r_clu_bg,r_other_clu = metric_of_clustering(cls,check,seg)
                    data_csv.write(f'{pid},{clsname},{cls},{cluster_pc.shape[0]},{check[cls]},{check[0]},{r_clu_gt},{r_clu_bg},{r_other_clu}\n')
                ptsdir = os.path.join(pc_save_dir,pid+'.json')
                with open(ptsdir,'w') as f:
                    json.dump(pnew,f)
            else:
                pnew = np.save(os.path.join(pc_save_dir,pid+'_pred.npy'))
        data_csv.close()
    ## summary
    summary = []
    data = pd.read_csv(clu_info_dir)
    for i in range(1,len(particle)):
        p = particle[i]
        df = data[data['particle_name']==p]
        df_ny = df.values
        print(df_ny)
        summary.append([p] + np.mean(df_ny[:,3:],0).tolist())
    ss = pd.DataFrame(summary,columns=['particle_name','total_points','particle_points','bg_points','r_clu_gt','r_clu_bg','r_other_clu'])
    summary_csv_dir = os.path.join(args.save_dir,f'cluster_info_{args.method}_summary.csv')
    ss.to_csv(summary_csv_dir)


    


    

        







