import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
from torchinfo import summary
import matplotlib.pyplot as plt
import numpy as np

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
summary(model)
#print(dir(model))
#print(vis_processors['eval'])


# モデル内の階層名を表示
for name, module in model.named_children():
    print(name)

model.load_state_dict(torch.load(model_pth, map_location=torch.device('cpu')))

model.to(device)

# マーライオンの画像 (論文の再現実験)===============================================================
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

# 画像の読み込み
raw_image3 = Image.open('/taiga/moonshot/blip2_moonshot/images/flant5.png').convert('RGB')
image3 = vis_processors['eval'](raw_image3).unsqueeze(0).to(device)

prompt3 = "Abstract: The cost of vision-and-language pre-training has become increasingly prohibitive due to end-toend training of large-scale models. This paper proposes BLIP-2, a generic and efficient pretraining strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pretrained in two stages. The first stage bootstraps vision-language representation learning from a frozen image encoder. The second stage bootstraps vision-to-language generative learning from a frozen language model. BLIP-2 achieves state-of-the-art performance on various visionlanguage tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model’s emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions. This figure caption is :"
answer3 = "Effect of vision-language representation learning on vision-to-language generative learning. Without representation learning, the Q-Former fails the bridge the modality gap, leading to significantly lower performance on zero-shot VQA."

#input_datas = {
#    "image": torch.cat([image1, image2], dim=0),
#    "text_input": [prompt1, prompt2],
#    "text_output": [answer1, answer2]
#}

# add long prompt (add abstract)
input_datas = {
    "image": torch.cat([image1, image2, image3], dim=0),
    "text_input": [prompt1, prompt2, prompt3],
    "text_output": [answer1, answer2, prompt3]
}

output = model.forward_custom(input_datas)
#output = model.forward(input_datas)

print('loss : ', output['loss'])
#print('output[text] : ', output['text'])
print('output[outputs][logits].shape : ', output['outputs']['logits'].shape)


#print('output[qformer_cross_attn].shape : ', output['qformer_cross_attn'][0].shape) #[batch, block, attention_weight, attention_weight)
#print('output[qformer_cross_attn].shape : ', output['qformer_cross_attn'])
