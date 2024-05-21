from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.dataloader import create_dataloaders
from src.lora_model import LORAEngine
from src.influence import IFEngine

import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import pickle

import warnings
warnings.filterwarnings("ignore")

## set the seed
np.random.seed(42)
import torch
torch.manual_seed(42)





model_name_or_path="roberta-large"
noise_ratio=0.2
batch_size=32
target_modules=["value","query"]
device="cuda"
num_epochs=15


task = 'cola'
lr = 6e-5

dataloader_outputs = create_dataloaders(model_name_or_path=model_name_or_path,
                                            task=task,
                                            noise_ratio=noise_ratio,
                                            batch_size=batch_size)
train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn=dataloader_outputs

lora_engine = LORAEngine(model_name_or_path=model_name_or_path,
                            target_modules=target_modules,
                            train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            device=device,
                            num_epochs=num_epochs,
                            lr=lr,
                            lora=True,
                            low_rank=16, 
                            task=task)


lora_engine.build_LORA_model()
lora_engine.train_LORA_model()

## get the train_grad and val_grad from fine-tuned model
tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)



influence_engine = IFEngine()
influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict, noise_index)

influence_engine.compute_hvps(compute_accurate=False)
influence_engine.compute_IF()

n_train=influence_engine.n_train
true_label=np.zeros(n_train)
true_label[noise_index]=1

method_dict={
            'identity': 'TracIN',
            'datainf': 'DataInf',
            'iterative': 'HyperINF',
            'LiSSA': 'LiSSA'
            }

plt.figure(figsize=(5,4))

for method in influence_engine.IF_dict:
    detection_rate_list=[]
    low_quality_to_high_quality=np.argsort(influence_engine.IF_dict[method])[::-1]
    for ind in range(1, len(low_quality_to_high_quality)+1):
        # detected_samples: the samples that are detected as noise 
        detected_samples = set(low_quality_to_high_quality[:ind]).intersection(noise_index)
        detection_rate = 100*len(detected_samples)/len(noise_index)
        detection_rate_list.append(detection_rate)
    

    plt.plot(100*np.arange(len(low_quality_to_high_quality))/n_train,
            detection_rate_list,
            #marker='s',
            label=method_dict[method])
    

## plot random detection rate from (0,0) to (100,100)
plt.plot([0, 100], [0, 100], linestyle='--', color='black', label='Random')
## plot perfect detection rate from (0,0) to (20,100), (20,100) to (100,100)
plt.plot([0, 20], [0, 100], linestyle='--', color='red', label='Perfect')
plt.plot([20, 100], [100, 100], linestyle='--', color='red')
plt.legend(fontsize=15)

plt.savefig(f"{task}_r=32_detection_rate.pdf", bbox_inches='tight')


