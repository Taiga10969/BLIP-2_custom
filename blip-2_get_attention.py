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

model_pth  ='/taiga/moonshot/blip2_moonshot/pth/blip2_t5_pretrain_flant5xl_ALL.pth'

# モデルの読み込み
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5',
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
raw_image0 = Image.open('/taiga/moonshot/blip2_moonshot/images/merlion.png').convert('RGB')
image0 = vis_processors['eval'](raw_image0).unsqueeze(0).to(device)
#print('vis_processors output : ', image)
print('vis_processors output.shape : ', image0.shape)

prompt0 = "Question: which city is this? Answer:"
answer0 = 'singapore'

raw_image1 = Image.open('/taiga/moonshot/blip2_moonshot/images/my_dog.jpg').convert('RGB')
#raw_image1 = Image.open('/taiga/moonshot/blip2_moonshot/images/learning_rate_scheduler.png').convert('RGB')
image1 = vis_processors['eval'](raw_image1).unsqueeze(0).to(device)
prompt1 = "Question: What does this image show? Answer:"
answer1 = 'dog'


raw_image2 = Image.open('/taiga/moonshot/blip2_moonshot/images/flant5.png').convert('RGB')
image2 = vis_processors['eval'](raw_image2).unsqueeze(0).to(device)

#raw_image3 = Image.open('/taiga/moonshot/blip2_moonshot/images/sample_figure1.png').convert('RGB')
raw_image3 = Image.open('/taiga/moonshot/blip2_moonshot/images/learning_rate_scheduler.png').convert('RGB')
image3 = vis_processors['eval'](raw_image3).unsqueeze(0).to(device)

prompt2 = "Question: What title of figure in this image? Answer:"
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
attn = model.get_image_encoder_attention(input_datas)

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
plt.savefig('get_image_encoder_attention/attention_maps_subplot.png', transparent=True)
plt.savefig('get_image_encoder_attention/attention_maps_subplot.svg', transparent=True)
plt.show()







## 可視化するデータの番号
#batch_num=2
#cls = 0
#normalization_method = 'min-max'  # 正規化手法'min-max', 'z-score', 'softmax'
#img_ = np.transpose(image2.cpu().numpy()[0], (1,2,0))
#
#normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
#img_ = (normalized_data * 255).astype(np.uint8)
#
#attention_map = attn[batch_num]
#
#attention_map = attention_rollout(attention_map, normalization_method)
#
#print('attention_map.shape : ', attention_map.shape)
#
#attention_map_rgb = heatmap_to_rgb(attention_map)
#attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)
#plt.imshow(attention_map)
#plt.savefig('get_image_encoder_attention/attention_map_test.png')
#plt.close()



'''
for head in range(16):
    attention = attn[batch_num][head][cls][1:] # 2枚目の画像の0ヘッド目のclsトークンとclsトークンを除いたもの
    print(np.shape(attention))

    attention_map = np.reshape(attention, (16, 16))
    print('attention_map', np.shape(attention_map))

    # 画像の正規化
    #print(image2.cpu().numpy().shape)
    normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
    img_ = (normalized_data * 255).astype(np.uint8)

    attention_map = normalize_attention_map(attention_map, normalization_method)

    # 原画像と重ね合わせて表示
    attention_map = attention_map.astype(np.float32)
    print('test : ', np.shape(attention_map))
    attention_map = cv2.resize(attention_map, (224,224), interpolation=cv2.INTER_LINEAR) # バイリニア補間
    print('test : ', np.shape(attention_map))
    attention_map_rgb = heatmap_to_rgb(attention_map)
    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)
    plt.imshow(attention_map)
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    #plt.colorbar()  # カラーバーの表示
    #plt.title(split_pre_text[pos])
    plt.savefig('get_image_encoder_attention/attention_map_test_head['+ str(head) +'].png', transparent=True)
    plt.savefig('get_image_encoder_attention/attention_map_test_head['+ str(head) +'].svg', transparent=True)
    plt.close()


# ヘッドごとの注意マップを収集するリストを初期化
attention_maps = []

for head in range(16):
    attention = attn[batch_num][head][cls][1:]  # 2枚目の画像の0ヘッド目のclsトークンとclsトークンを除いたもの
    attention_map = np.reshape(attention, (16, 16))

    attention_map = normalize_attention_map(attention_map, normalization_method)
    
    # 重み付けなどの処理
    attention_map = attention_map.astype(np.float32)
    attention_map = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_LINEAR)
    attention_maps.append(attention_map)

# リスト内の注意マップを平均化
average_attention_map = np.mean(attention_maps, axis=0)

# 画像の正規化と表示
normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
img_ = (normalized_data * 255).astype(np.uint8)

average_attention_map_rgb = heatmap_to_rgb(average_attention_map)
attention_map_combined = cv2.addWeighted(img_, 0.5, average_attention_map_rgb, 0.5, 0)

plt.imshow(attention_map_combined)
plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
# plt.colorbar()  # カラーバーの表示
# plt.title(split_pre_text[pos])
plt.savefig('get_image_encoder_attention/average_attention_map.png', transparent=True)
plt.savefig('get_image_encoder_attention/average_attention_map.svg', transparent=True)
plt.close()
#このコードは、各ヘッドの注意マップを収集し、それらを平均化して1枚の画像に結合し、最終的な注意マップを表示して保存します。ご注意くださいが、 heatmap_to_rgb 関数がどのように実装されているかによっては、正しく動作しない場合がありますので、必要に応じて適切なコードを追加してください。





fig, axes = plt.subplots(2, 8, figsize=(40, 12))

for head in range(16):
    attention = attn[batch_num][head][cls][1:]

    attention_map = np.reshape(attention, (16, 16))

    normalized_data = (img_ - img_.min()) / (img_.max() - img_.min())
    img_ = (normalized_data * 255).astype(np.uint8)

    attention_map = normalize_attention_map(attention_map, normalization_method)
    attention_map = attention_map.astype(np.float32)
    attention_map = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_LINEAR)
    attention_map_rgb = heatmap_to_rgb(attention_map)
    attention_map = cv2.addWeighted(img_, 0.5, attention_map_rgb, 0.5, 0)

    row = head // 8
    col = head % 8
    axes[row, col].imshow(attention_map)
    axes[row, col].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False, bottom=False, left=False, right=False, top=False)
    axes[row, col].set_title('Attention Head ' + str(head), fontsize=30)

plt.tight_layout()
plt.savefig('get_image_encoder_attention/attention_maps_subplot.png', transparent=True)
plt.savefig('get_image_encoder_attention/attention_maps_subplot.svg', transparent=True)
plt.show()

'''
