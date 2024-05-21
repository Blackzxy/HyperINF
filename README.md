# HyperINF
This repo contains two main parts for paper HyperINF:
1. Mislabeled Data Detection
2. Data Selection for LLM and VLM

## Mislabeled Data Detection
In this task, we use the code of [DataInf](https://github.com/ykwon0407/DataInf) and compare our HyperINF with the privided baselines. Our implementation of HyperINF can be found in `HyperINF/Mislabeled_Data_Detection/src/influence.py`. 

You can run `python Mislabeled_Data_Detection/test.py` to see the performance comparision among all the methods. We also provide the results of detection on `COLA` dataset as an example in the image.


## Data Selection for LLM and VLM
We ultilize the code of [Prismatic-VLM](https://github.com/TRI-ML/prismatic-vlms/tree/main), please follow the instruction of it to build the environment and download the dataset (it could be large for VLM, so it will take a while to download).

For Data selection for LLM, we choose the datasets which are available in HuggingFace: [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa), [PIQA](https://github.com/ybisk/ybisk.github.io/tree/master/piqa/data), [LohiQA](https://huggingface.co/datasets/lucasmccabe/logiqa) and [CommonSenseQA](https://huggingface.co/datasets/tau/commonsense_qa).

We modify the original codes to support only LLM and data selection for both LLM and VLM. 
