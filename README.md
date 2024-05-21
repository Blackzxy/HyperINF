# HyperINF
This repo contains two main parts for paper HyperINF:
1. Mislabeled Data Detection
2. Data Selection for LLM and VLM

## Mislabeled Data Detection
In this task, we use the code of [DataInf](https://github.com/ykwon0407/DataInf) and compare our HyperINF with the privided baselines. Our implementation of HyperINF can be found in `HyperINF/Mislabeled_Data_Detection/src/influence.py`. 

You can run `python Mislabeled_Data_Detection/test.py` to see the performance comparision among all the methods. We also provide the results of detection on `COLA` dataset as an example in the image `HyperINF/cola_r=16_detection_rate.pdf`.


## Data Selection for LLM and VLM
We ultilize the code of [Prismatic-VLM](https://github.com/TRI-ML/prismatic-vlms/tree/main), please clone that repo first, and follow the instruction of it to build the environment and download the dataset (it could be large for VLM, so it will take a while to download).

For Data selection for LLM, we choose the datasets which are available in HuggingFace: [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa), [PIQA](https://github.com/ybisk/ybisk.github.io/tree/master/piqa/data), [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa) and [CommonSenseQA](https://huggingface.co/datasets/tau/commonsense_qa).

We modify the original codes to support only LLM and data selection for both LLM and VLM. We provide the modified codes in `LLM_VLM_Finetune/scripts` and `LLM_VLM_Finetune/prismatic`. You can add and replace the files in the original repo with the files in our repo.

1. The file `data_selection.py` is used for **Data Selection for VLM**, it will compute the gradients of val dataset and the influence score of each training data points, then sort the data points according to the influence score and save them. You can change the `stage` in the config to `data-pruning_llm` for using the language model's last layer to compute the influence score or `data-pruning_projector` for using the projector to compute the influence score.

2. The file `pretrain_llm.py` is used for **Data Selection and Finetuing for LLM**. It is also modified from the original codes for supporting the data selection for LLM. You can change the `stage` in the config to `llama-dataset-pruning` for using the language model's last layer to compute the influence score and select top-k data points based on the influence score, or `llm-finetune` to finetune the LLM in each dataset.
