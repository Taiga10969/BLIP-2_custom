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

# モデルの読み込み
model, vis_processors, _ = load_model_and_preprocess(
    name='blip2_t5_custom',
    model_type='pretrain_flant5xl', #pretrain_flant5xl, caption_coco_flant5xl, pretrain_flant5xxl
    is_eval=True,
    device=device
)

# model samarry
summary(model)
print(dir(model))
print(vis_processors['eval'])

# マーライオンの画像 (論文の再現実験)===============================================================
print('=====image1=====')
# 画像の読み込み
raw_image = Image.open('/taiga/moonshot/BLIP2_custom/images/merlion.png').convert('RGB')
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
print('vis_processors output : ', image)
print('vis_processors output.shape : ', image.shape)

show_vis_processor(vis_processor_output=image)

# 推論1 (promptなし)
print('prompt : - ')
output = model.generate({'image':image})
print('output : ', output)

# 推論2 (promptあり)
prompt = "Question: which city is this? Answer:"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)

# 推論3 (promptあり，過去の複数の会話)
# prepare context prompt
context = [
    ("which city is this?", "singapore"),
    ("why?", "it has a statue of a merlion"),
]
question = "where is the name merlion coming from?"
template = "Question: {} Answer: {}."
# context, question, templateからpromptを作成
prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question + " Answer:"
print('promot : ', prompt)
output = model.generate({"image": image,"prompt": prompt})
print('output : ', output)

# 論文図1 (グラフ図) ============================================================================
print('=====image2=====')
# 画像の読み込み
raw_image = Image.open('/taiga/moonshot/BLIP2_custom/images/flant5.png').convert('RGB')
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
#print('vis_processors output : ', image)
#print('vis_processors output.shape : ', image.shape)

show_vis_processor(vis_processor_output=image, save_name='figure1.png')

# 推論1 (promptなし)
print('prompt : - ')
output = model.generate({'image':image})
print('output : ', output)

# 推論2 (promptあり)
prompt = "Question: what type of figure in this image? Answer:"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)

# 推論3 (promptあり)
prompt = "Please create a caption for this figure"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)

# 論文図2 (アーキテクチャ図) ============================================================================
print('=====image3=====')
# 画像の読み込み
raw_image = Image.open('/taiga/moonshot/BLIP2_custom/images/stage2.png').convert('RGB')
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
#print('vis_processors output : ', image)
#print('vis_processors output.shape : ', image.shape)

show_vis_processor(vis_processor_output=image, save_name='figure2.png')

# 推論1 (promptなし)
print('prompt : - ')
output = model.generate({'image':image})
print('output : ', output)

# 推論2 (promptあり)
prompt = "Question: what type of figure in this image? Answer:"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)

# 推論3 (promptあり)
prompt = "Please create a caption for this figure"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)

# 論文図3 (subfigureを含む画像) ============================================================================
print('=====image4=====')
# 画像の読み込み
raw_image = Image.open('/taiga/moonshot/BLIP2_custom/images/sample_figure1.png').convert('RGB')
image = vis_processors['eval'](raw_image).unsqueeze(0).to(device)
#print('vis_processors output : ', image)
#print('vis_processors output.shape : ', image.shape)

show_vis_processor(vis_processor_output=image, save_name='figure3.png')

# 推論1 (promptなし)
print('prompt : - ')
output = model.generate({'image':image})
print('output : ', output)

# 推論2 (promptあり)
prompt = "Question: what type of figure in this image? Answer:"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)

# 推論3 (promptあり)
prompt = "Are subfigures included in this image? Answer:"
print('prompt : ', prompt)
output = model.generate({"image": image, "prompt": prompt})
print('output : ', output)
