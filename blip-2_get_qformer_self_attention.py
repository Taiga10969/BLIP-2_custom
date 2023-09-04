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
#print(output)

#def visualize_keys(dictionary):
#    keys = dictionary.keys()
#    for key in keys:
#        print(key)
#
#visualize_keys(output)

print('output[attention(self-attention)].shape : ', output['qformer_self_attn'][0].shape) #[batch, block, query, query) : torch.Size([2, 12, 32, 32])


self_attn = output['qformer_self_attn'][0].cpu().detach().numpy()

batch = 0
block = 11

attn = self_attn[batch][block]
plt.imshow(attn, cmap='viridis')
#plt.colorbar()
plt.title('BLIP-2 Q-Former last block self-Attention')
plt.xlabel('Q-Former Querie')
plt.ylabel('Q-Former Queries')
plt.savefig('get_qformer_self_attention/last_block_AttentionWeight.png')
plt.close()
