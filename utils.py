import torch
import numpy as np



def attention_rollout(attention_weight): 
    #input>attention_weight　np.shape() : (L,H,N,N)
    #L:層数, H:ヘッド数, N:パッチ数+cls_token

    #ヘッド方向に平均
    attention_weight = np.mean(attention_weight, axis=1) # shape:(L,N,N)

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
    mask = v[0, 1:].reshape(14,14)
    attention_map = cv2.resize(mask / mask.max(), (ori_img.shape[2], ori_img.shape[3]))[np.newaxis]

    return attention_map
  
  
