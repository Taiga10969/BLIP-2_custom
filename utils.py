import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt



def attention_rollout(attention_weight, normalization_method='min-max'): 
    #input>attention_weight　np.shape() : (L,H,N,N)
    #L:層数, H:ヘッド数, N:パッチ数+cls_token

    #ヘッド方向に平均
    mean_head = np.mean(attention_weight, axis=1) # shape:(L,N,N)

    #N*Nの単位行列を加算
    mean_head = mean_head + np.eye(mean_head.shape[1])

    #正規化
    mean_head = mean_head / mean_head.sum(axis=(1,2))[:, np.newaxis, np.newaxis]

    #層方向に乗算
    v = mean_head[-1]
    for n in range(1,len(mean_head)):
        v = np.matmul(v, mean_head[-1 - n])

    #クラストークンと各パッチトークン間っとのAttention Weightから，
    #入力画像サイズまで正規化しながらリサイズしてAttention Mapを生成
    mask = v[0, 1:].reshape(16,16)
    mask = normalize_attention_map(mask, normalization_method)
    attention_map = cv2.resize(mask, (224, 224))[np.newaxis]

    return attention_map[0]
  
  
def normalize_attention_map(attention_map, normalization_method='min-max'):
    if normalization_method == 'min-max':
        min_attention = attention_map.min()
        max_attention = attention_map.max()
        normalized_attention = (attention_map - min_attention) / (max_attention - min_attention)
    elif normalization_method == 'max':
        normalized_attention = attention_map / attention_map.max()
    elif normalization_method == 'z-score':
        mean_attention = attention_map.mean()
        std_attention = attention_map.std()
        normalized_attention = (attention_map - mean_attention) / std_attention
    elif normalization_method == 'softmax':
        exp_attention = np.exp(attention_map)
        normalized_attention = exp_attention / np.sum(exp_attention)
    else:
        raise ValueError("Invalid normalization method")

    return normalized_attention


def heatmap_to_resize_and_rgb(heatmap):
    # カラーマップに変換
    colormap = plt.get_cmap('jet')
    colored_heatmap = colormap(heatmap)
    # RGBに変換
    rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    return rgb_image
