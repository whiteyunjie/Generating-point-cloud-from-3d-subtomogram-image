import numpy as np

def metric_of_clustering_SNR(check,seg):
    count_all = 0
    for cat in check:
        count_all += check[cat]
    label_idx = 1
    count_gt = np.argwhere(seg==label_idx).shape[0]
    count_cluster = check[label_idx]

    r_clu_gt = count_cluster/count_gt
    r_clu_bg = count_cluster/check[0]
    #r_other_clu = (count_all-check[0]-count_cluster)/count_cluster

    #print(f'label cluster ratio: {r_clu_gt:.6f}')
    #print(f'count ratio(label:bg): {r_clu_bg:.6f}')
    #print(f'count ratio(other labels:label): {r_other_clu:.6f}')
    return r_clu_gt,r_clu_bg

# function
def metric_of_clustering(label_idx,check,seg):
    count_all = 0
    for cat in check:
        count_all += check[cat]
    labelidx = label_idx
    count_gt = np.argwhere(seg==label_idx).shape[0]
    count_cluster = check[label_idx]

    if count_cluster == 0: return 0,0,-1
    r_clu_gt = count_cluster/count_gt
    r_clu_bg = count_cluster/check[0]
    r_other_clu = (count_all-check[0]-count_cluster)/count_cluster

    #print(f'label cluster ratio: {r_clu_gt:.6f}')
    #print(f'count ratio(label:bg): {r_clu_bg:.6f}')
    #print(f'count ratio(other labels:label): {r_other_clu:.6f}')
    return r_clu_gt,r_clu_bg,r_other_clu