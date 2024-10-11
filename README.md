# What Do VLMs NOTICE? A Mechanistic Interpretability Pipeline for Noise-free Text-Image Corruption and Evaluation

## Description
We introduce NOTICE, the first Noise-free Text-Image Corruption and Evaluation pipeline for mechanistic interpretability in VLMs. NOTICE incorporates a Semantic Minimal Pairs (SMP) framework for image corruption and Symmetric Token Replacement (STR) for text. This approach enables semantically meaningful causal mediation analysis for both modalities, providing a robust method for analyzing multimodal integration within models like BLIP. Our experiments on the SVO-Probes, MIT-States, and Facial Expression Recognition datasets reveal crucial insights into VLM decision-making, identifying the significant role of middle-layer cross-attention heads. Further, we uncover a set of ''universal cross-attention heads'' that consistently contribute across tasks and modalities, each performing distinct functions such as implicit image segmentation, object inhibition, and outlier inhibition. The figure below demonstrated our framework:

![datasets](https://github.com/wrudman/NOTICE/assets/35315239/5926c790-4bf6-4ad4-809b-c43bfaf12c5f)

## Requirements
Python 3.9.16

PyTorch Version: 2.2.0+cu121

To install requirements:

```setup
pip install -r requirements.txt
```
## Preprocessing
This paper uses three datasets (SVO-Probes, MIT-States, Facial Expressions). The preprocessing steps are described in the preprocessing folder. For each dataset, run the following: 

```
1_preprocessing_{task_name}.ipynb
```
```
python 2_evaluate_BLIP.py --task_name
```
```
3_select_final_samples_{task_name}.ipynb
```

## Activation Patching

After preprocessing, you can find the image and text patching code in `BLIP_patching.py`, and you can use the shell script `BLIP_patching.sh`, to run it. In addition, we provide a start-to-finish tutorial that shows how to obtain heatmaps shown in the paper: `BLIP_patching_tutorial.ipynb` and  `circuit_analysis.ipynb`. 

Code to perform activation patching on Llava is in the Llava directory. Patching code is largely the same as BLIP, however, we make a couple key changes. Given the size of Llava, we only patch text tokens in the prompt and do not patch image tokens for module-wise patching. For attention-head patching, we only patch the last token of the correct answer. Patching one attention head for all layers takes approximately 9 hours on one 3090-GPU and 6 hours on one A100.  

