import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

import cv2

from utils import attention_rollout, normalize_attention_map

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

model_pth  ='/taiga/moonshot/BLIP2_custom/blip2_t5_pretrain_flant5xl_ALL.pth'

# モデルの読み込み
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5_custom',
    model_type='pretrain_flant5xl', #pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    is_eval=True,
    #device=device
)

# model samarry
#summary(model)
#print(dir(model))
#print(vis_processors['eval'])


## モデル内の階層名を表示
#for name, module in model.named_children():
#    print(name)

model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))
print(f'My log : load_state_dict << {model_pth}')

model.to(dtype=torch.float32, device=device)

#===============================================================

# 画像の読み込み

# 画像の読み込み
raw_image0 = Image.open('/taiga/moonshot/BLIP2_custom/images/merlion.png').convert('RGB')
image0 = vis_processors['eval'](raw_image0).unsqueeze(0).to(device)
#print('vis_processors output : ', image)
print('vis_processors output.shape : ', image0.shape)

prompt0 = "Question: which city is this? Answer:"
answer0 = 'singapore'

raw_image1 = Image.open('/taiga/moonshot/BLIP2_custom/images/my_dog.jpg').convert('RGB')
image1 = vis_processors['eval'](raw_image1).unsqueeze(0).to(device)
prompt1 = "Question: What does this image show? Answer:"
answer1 = 'dog'


raw_image2 = Image.open('/taiga/moonshot/BLIP2_custom/images/flant5.png').convert('RGB')
image2 = vis_processors['eval'](raw_image2).unsqueeze(0).to(device)

raw_image3 = Image.open('/taiga/moonshot/BLIP2_custom/images/sample_figure1.png').convert('RGB')
image3 = vis_processors['eval'](raw_image3).unsqueeze(0).to(device)

prompt2 = "Question: what type of figure in this image? Answer:"
answer2 = 'line graph'

input_datas = {
    "image": torch.cat([image0, image1, image2, image3], dim=0),
    "text_input": [prompt0, prompt1, prompt2, prompt2],
    "text_output": [answer0, answer1, answer2, answer2]
}

output = model.forward_test(input_datas)

print('loss : ', output['loss'])
#print('output[text] : ', output['text'])
print('output[text][logits].shape : ', output['text']['logits'].shape)



#image_embeds, image_atts, attn = model.imege_embeds_test_forward(input_datas)
attn = model.get_attention(input_datas)

attn = np.transpose(attn,(1,0,2,3,4))

#print('image_embeds : ', image_embeds.shape)
#print('image_atts : ', image_atts.shape)
print('attn.shape : ', np.shape(attn))



def heatmap_to_rgb(heatmap):
    # カラーマップに変換
    colormap = plt.get_cmap('jet')
    colored_heatmap = colormap(heatmap)
    # RGBに変換
    rgb_image = (colored_heatmap[:, :, :3] * 255).astype(np.uint8)
    return rgb_image

#===============================================================


fig, axes = plt.subplots(1, 4, figsize=(40, 12))
for batch_num in range(4):
    cls = 0
    normalization_method = 'min-max'  # 正規化手法'min-max', 'z-score', 'softmax'
    if batch_num == 0:
        image = image0
    if batch_num == 1:
        image = image1
    if batch_num == 2:
        image = image2
    if batch_num == 3:
        image = image3
    img_ = np.transpose(image.cpu().numpy()[0], (1,2,0))

    normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
    img_ = (normalized_data * 255).astype(np.uint8)

    attention_map = attn[batch_num]

    attention_map = attention_rollout(attention_map, normalization_method)

    row = 0
    col = batch_num
    attention_map_rgb = heatmap_to_rgb(attention_map)
    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)
    axes[col].imshow(attention_map)
    axes[col].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    #axes[row, col].set_title('Attention', fontsize=30)
plt.tight_layout()
plt.savefig('get_attention/attention_maps_subplot.png', transparent=True)
plt.savefig('get_attention/attention_maps_subplot.svg', transparent=True)
plt.show()
