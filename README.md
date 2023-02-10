# Generating-point-cloud-from-3d-subtomogram-iamge
A clustering based unsupervised method for generating point cloud from 3d subtomogram iamge

![截屏2023-02-10 10 14 54](https://user-images.githubusercontent.com/49239327/217983198-708a48ed-6b81-4dfe-9a1b-1491e8589927.jpg)

### usage
```python
    python main.py --method den_filter --dataset SNR --data_dir DATA_DIR --quantile 50 --den_thrshld 0.3 --radius 5 --save_dir SAVE_DIR
    --n_clusters 3 --noise_level 001
```
