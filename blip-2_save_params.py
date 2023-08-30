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
    #device=device
)

# model samarry
summary(model)
#print(dir(model))
#print(vis_processors['eval'])

# パラメータを保存
torch.save(model.state_dict(), "blip2_t5_pretrain_flant5xl_ALL.pth")

# モデル内の階層名を表示
for name, module in model.named_children():
    print(name)

# visual_encoder 以外の部分のパラメータを抽出
params_to_save = {
    "ln_vision": model.ln_vision.state_dict(),
    "Qformer": model.Qformer.state_dict(),
    "t5_model": model.t5_model.state_dict(),
    "t5_proj": model.t5_proj.state_dict()
}

# パラメータを保存
torch.save(params_to_save, "blip2_t5_pretrain_flant5xl_Unexpected_ViT.pth")
