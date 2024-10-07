import json
import numpy as np
from tqdm import tqdm



total_data = "data/download/llava-laion-cc-sbu-558k/chat.json"
# train_data_path = "data/download/llava-v1.5-instruct/train_val/llava_v1_5_train.json"
# val_data_path = "data/download/llava-v1.5-instruct/train_val/llava_v1_5_val.json"

# ## randomly choose 10% of the data
with open(total_data, "r") as f:
    data = json.load(f)
# ## image: coco, gqa, ocr_vqa, textvqa, vg
# ## for val data, we randomly choose 5K from each image type
# images_type = ["coco", "gqa", "ocr_vqa", "textvqa", "vg"]
# coco_data = []
# gqa_data = []
# ocr_vqa_data = []
# textvqa_data = []
# vg_data = []

# val_size = 5000
# train_data = []
# val_data = []
# ## split the data into train and val
# ## randomly select 5K from each type and put them into val_data
# ## for training data, put the rest into train_data
# for d in tqdm(data):
#     if "image" not in d:
#         continue
#     if d["image"].startswith("coco"):
#         coco_data.append(d)
#     elif d["image"].startswith("gqa"):
#         gqa_data.append(d)
#     elif d["image"].startswith("ocr_vqa"):
#         ocr_vqa_data.append(d)
#     elif d["image"].startswith("textvqa"):
#         textvqa_data.append(d)
#     elif d["image"].startswith("vg"):
#         vg_data.append(d)

# np.random.shuffle(coco_data)
# np.random.shuffle(gqa_data)
# np.random.shuffle(ocr_vqa_data)
# np.random.shuffle(textvqa_data)
# np.random.shuffle(vg_data)

# val_data.extend(coco_data[:val_size])
# val_data.extend(gqa_data[:val_size])
# val_data.extend(ocr_vqa_data[:val_size])
# val_data.extend(textvqa_data[:val_size])
# val_data.extend(vg_data[:val_size])
# ## shuffle the val_data
# np.random.shuffle(val_data)
# print(len(val_data))

# train_data.extend(coco_data[val_size:])
# train_data.extend(gqa_data[val_size:])
# train_data.extend(ocr_vqa_data[val_size:])
# train_data.extend(textvqa_data[val_size:])
# train_data.extend(vg_data[val_size:])
# ## shuffle the train_data
# np.random.shuffle(train_data)
# print(len(train_data))

# with open(train_data_path, "w") as f:
#     json.dump(train_data, f)

# with open(val_data_path, "w") as f:
#     json.dump(val_data, f)






    

data = np.array(data)
print(len(data))
np.random.shuffle(data)

ratio = 0.05

selected_data = data[:int(len(data) * ratio)]

with open("data/download/llava-laion-cc-sbu-558k/chat_5_percent.json", "w") as f:
    json.dump(selected_data.tolist(), f)



## train: 599610 
## for dataset pruning: we set alpha = 0.2, 0.4, 0.6, 0.8
## 0.2: 119922, 0.4: 239844, 0.6: 359766, 0.8: 479688

# data = data[:119922]

# with open("data/download/llava-v1.5-instruct/llava_v1_5_mix665k_20_percent.json", "w") as f:
#     json.dump(data.tolist(), f)



