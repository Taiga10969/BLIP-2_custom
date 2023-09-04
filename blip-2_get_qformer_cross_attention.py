import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

from utils import normalize_attention_map, heatmap_to_resize_and_rgb


import cv2

def show_vis_processor(vis_processor_output, save_name='image.png'):
    data = vis_processor_output.cpu().numpy()
    data = (data - data.min()) / (data.max() - data.min())
    data = np.transpose(data[0], (1, 2, 0))
    plt.imshow(data)
    plt.axis('off')  # 軸を表示しない
    plt.savefig(save_name)
    plt.close()

# check GPU usage
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count()==0: print('Use 1 GPU')
else: print(f'Use {torch.cuda.device_count()} GPUs')

model_pth  ='/taiga/moonshot/blip2_moonshot/pth/blip2_t5_pretrain_flant5xl_ALL.pth'

# モデルの読み込み
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5',
    model_type='pretrain_flant5xl', #pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    is_eval=True,
    #device=device
)


model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))

model.to(device)

# マーライオンの画像 (論文の再現実験)===============================================================
print('=====image1=====')
# 画像の読み込み
raw_image1 = Image.open('/taiga/moonshot/blip2_moonshot/images/merlion.png').convert('RGB')
image1 = vis_processors['eval'](raw_image1).unsqueeze(0).to(device)
#print('vis_processors output : ', image)
print('vis_processors output.shape : ', image1.shape)

prompt1 = "Question: which city is this? Answer:"
answer1 = 'singapore'

raw_image2 = Image.open('/taiga/moonshot/blip2_moonshot/images/flant5.png').convert('RGB')
image2 = vis_processors['eval'](raw_image2).unsqueeze(0).to(device)

prompt2 = "Question: what type of figure in this image? Answer:"
answer2 = 'line graph'

input_datas = {
    "image": torch.cat([image1, image2], dim=0),
    "text_input": [prompt1, prompt2],
    "text_output": [answer1, answer2]
}

output = model.forward_custom(input_datas)

print('output[qformer_cross_attn].shape : ', output['qformer_cross_attn'][0].shape) #[batch, block, attention_weight, attention_weight) : torch.Size([2, 12, 32, 257])

batch = 0
img_ = np.transpose(image1.cpu().numpy()[0], (1,2,0))
normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
img_ = (normalized_data * 255).astype(np.uint8)



# batch:0, block:11(last) に対して，Attention Weightの可視化を行う
attns = output['qformer_cross_attn'][0]
matrix_to_visualize = attns[batch][11].cpu().detach().numpy()

plt.imshow(matrix_to_visualize, cmap='viridis')
#plt.colorbar()
plt.title('BLIP-2 Q-Former last block Attention Weight')
plt.xlabel('path feature from Image Encoder')
plt.ylabel('Q-Former Queries')
plt.savefig('get_qformer_cross_attention/last_AttentionWeight.png')
plt.close()



# batch:0, block:11(last), Queries:0 に対してAttention Weightの可視化を行う
# batch:0 に対応した画像を読み込む

q = 0  #query(0:15)

attns = output['qformer_cross_attn'][0]
matrix_to_visualize = attns[batch][11][q].cpu().detach().numpy()

attention_map = np.resize(matrix_to_visualize, (16, 16))
attention_map = normalize_attention_map(attention_map, normalization_method='min-max')

attention_map = cv2.resize(attention_map, (224, 224))
attention_map_rgb = heatmap_to_resize_and_rgb(attention_map)

attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)

plt.imshow(attention_map, cmap='viridis')
plt.title(f'BLIP-2 Q-Former Query:{q}')
plt.savefig(f'get_qformer_cross_attention/last_block_AttentionMap_query{q}.png')
plt.close()


'''
# 16枚の画像を横に並べるためのsubplotの設定
fig, axs = plt.subplots(1, 16, figsize=(16, 2))

for q in range(16):
    attns = output['qformer_cross_attn'][0]
    matrix_to_visualize = attns[batch][11][q].cpu().detach().numpy()

    attention_map = np.resize(matrix_to_visualize, (16, 16))
    attention_map = normalize_attention_map(attention_map, normalization_method='min-max')

    attention_map = cv2.resize(attention_map, (224, 224))
    attention_map_rgb = heatmap_to_resize_and_rgb(attention_map)

    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)

    axs[q].imshow(attention_map, cmap='viridis')
    axs[q].axis('off')
    axs[q].set_title(f'Query {q}')

# タイトルを設定
fig.suptitle('BLIP-2 Q-Former Queries 0-15', fontsize=16)

# 16枚の画像を1つの画像として保存
plt.tight_layout()
plt.savefig('get_qformer_cross_attention/test.png', transparent=True)
plt.close()
'''


for b in range(12):
    # 16枚の画像を横に並べるためのsubplotの設定
    fig, axs = plt.subplots(1, 16, figsize=(16, 2))
    for q in range(16):
        attns = output['qformer_cross_attn'][0]
        matrix_to_visualize = attns[batch][b][q].cpu().detach().numpy()

        attention_map = np.resize(matrix_to_visualize, (16, 16))
        attention_map = normalize_attention_map(attention_map, normalization_method='min-max')

        attention_map = cv2.resize(attention_map, (224, 224))
        attention_map_rgb = heatmap_to_resize_and_rgb(attention_map)

        attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)

        axs[q].imshow(attention_map, cmap='viridis')
        axs[q].axis('off')
        axs[q].set_title(f'Query {q}')

    # タイトルを設定
    fig.suptitle('BLIP-2 Q-Former Queries 0-15', fontsize=16)

    # 16枚の画像を1つの画像として保存
    plt.tight_layout()
    plt.savefig(f'get_qformer_cross_attention/block_{b}_Attention_Maps.png', transparent=True)
    plt.close()



# 16枚の画像を横に並べるためのsubplotの設定 block方向に平均

attns = output['qformer_cross_attn'][0]
attns = attns[batch].cpu().detach().numpy()
#print(np.shape(attns)) #(12, 32, 257)
attns = np.mean(attns, axis=0)
#print(np.shape(attns)) #(32, 257)

'''
fig, axs = plt.subplots(1, 32, figsize=(32, 2))
for q in range(32):
    matrix_to_visualize = attns[q]
    #print(np.shape(matrix_to_visualize)) #(257,)
    attention_map = np.resize(matrix_to_visualize, (16, 16))
    attention_map = normalize_attention_map(attention_map, normalization_method='min-max')
    attention_map = cv2.resize(attention_map, (224, 224))
    attention_map_rgb = heatmap_to_resize_and_rgb(attention_map)
    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)
    axs[q].imshow(attention_map, cmap='viridis')
    axs[q].axis('off')
    axs[q].set_title(f'Query {q}')
# タイトルを設定
fig.suptitle('BLIP-2 Q-Former Queries 0-15', fontsize=16)
# 16枚の画像を1つの画像として保存
plt.tight_layout()
plt.savefig(f'get_qformer_cross_attention/block_means.png', transparent=True)
plt.close()
'''

# 16枚の画像を2行に表示するために、1行あたり8枚ずつに分割
fig, axs = plt.subplots(2, 8, figsize=(16, 4))
for q in range(16):
    matrix_to_visualize = attns[q]
    attention_map = np.resize(matrix_to_visualize, (16, 16))
    attention_map = normalize_attention_map(attention_map, normalization_method='min-max')
    attention_map = cv2.resize(attention_map, (224, 224))
    attention_map_rgb = heatmap_to_resize_and_rgb(attention_map)
    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)
    
    # サブプロットのインデックスを計算
    row_index = q // 8
    col_index = q % 8
    
    axs[row_index, col_index].imshow(attention_map, cmap='viridis')
    axs[row_index, col_index].axis('off')
    axs[row_index, col_index].set_title(f'Query {q}')

# タイトルを設定
fig.suptitle('BLIP-2 Q-Former Queries 0-15', fontsize=16)
# 画像を1つの画像として保存
plt.tight_layout()
plt.savefig('get_qformer_cross_attention/block_means.png', transparent=True)
plt.savefig('get_qformer_cross_attention/block_means.svg', transparent=True)
plt.close()
