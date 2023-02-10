import numpy as np
import os
import json
from tqdm import tqdm
import argparse
from data_process import read_data,write_data

imgroot = '../pc_data/SNR/JPEGImages_64'
segroot = '../pc_data/SNR/Annotations_64'
labelroot = '../pc_data/SNR/json_label'

snr_dict = {'001':0.01,'003':0.03,'005':0.5,'01':0.1}

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--noise_level',type=str,default='001')
args = parser.parse_args()

if __name__ == '__main__':
    os.makedirs(f'../pc_data/SNR/img_snr{args.noise_level}',exist_ok=True)
    os.makedirs(f'../pc_data/SNR/seg_snr{args.noise_level}',exist_ok=True)
    for i in tqdm(range(4000)):
        img = read_data(os.path.join(imgroot,f'{i}.mrc'))
        seg = read_data(os.path.join(segroot,f'{i}.mrc'))
        
        with open(os.path.join(labelroot,f'target{i}.json'),'r') as f:
            label = json.load(f)
        name = label['name']
        
        # add noise
        ## old noise
        # SNR = 1
        # img_power  = np.mean(img**2)
        # noise_var = img_power*SNR
        # noise = np.random.normal(0,np.sqrt(noise_var),img.shape)
        # img_noise = img + noise

        ## new noise
        SNR = snr_dict[args.noise_level]
        img_power  = np.mean(img**2)
        noise_var = img_power/(10*SNR)

        signal_power = (1/len(img.reshape(-1,)))*np.sum(img**2)
        noise_power = signal_power/(10**(SNR/10))
        noise = noise_power*np.random.normal(0,np.sqrt(noise_var),img.shape)
        img_noise = img + noise
        
        if not os.path.exists(f'../pc_data/SNR/img_snr{args.noise_level}/{name}_{i}.mrc'):
            write_data(img_noise,f'../pc_data/SNR/img_snr{args.noise_level}/{name}_{i}.mrc')
        if not os.path.exists(f'../pc_data/SNR/seg_snr{args.noise_level}/{name}_{i}_seg.mrc'):
            write_data(seg,f'../pc_data/SNR/seg_snr{args.noise_level}/{name}_{i}_seg.mrc')