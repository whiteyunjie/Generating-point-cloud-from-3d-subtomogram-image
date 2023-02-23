# Generating-point-cloud-from-3d-subtomogram-image
A clustering based unsupervised method for generating point cloud from 3d subtomogram image



### usage
<img src=https://github.com/whiteyunjie/Generating-point-cloud-from-3d-subtomogram-image/blob/main/image/framework.jpg width=70% />


#### Generating point cloud data from 3d subtomogram image

```python
    python main.py --method den_filter --dataset SNR --data_dir DATA_DIR --quantile 50
                   --den_thrshld 0.3 --radius 5 --save_dir SAVE_DIR
                   --n_clusters 3 --noise_level 001
```
* ```method```: method name of generating point cloud from 3d subtomogram image
* ```dataset```: SNR(simluated dataset)/Qiang(read dataset), other datasets is also allowed.
* ```data_dir```: root path of 3d subtomogram input
* ```save_dir```: save path of generated point cloud data
* ```noise_level```: SNR noise level when use simulated dataset (choose from 01,005,003,001, representing SNR of 0.1,0.05,0.03,0.01 respectively)

detailed parameters of the generation method

* ```quantile```: quantile of voxel values for constructing point cloud with inconsistent density(0-100)
* ```den_thrshld```: density threshold of density filter
* ```radius```:radius of neighbour region (0-1)
* ```n_clusters```: number of clusters of clusterer module (3-5 is better)

Result directory form:
```
save_dir/pts_snr{noise_level}/
    pc_data_{method}
        pc_data
            1bxn_0.json
            1bxn_1.json
            ...
        cluster_info_{method}.csv
        cluster_info_{method}_summary.csv          
```
cluster_info_{method}* .csv contains metric of each generated point cloud data. 

