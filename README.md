## TVE: Learning Meta-attribution for Transferable Vision Explainer

### Research Motivation

Explainable machine learning significantly improves the transparency of deep neural networks (DNN).
However, existing work is constrained to explaining the behavior of individual model predictions, and lacks the ability to transfer the explanation across various models and tasks.
This limitation results in explaining various tasks being time- and resource-consuming.
To address this problem, we develop a DNN-based transferable vision explainer, named TVE, pre-trained on large-scale image datasets, and leverage its transferability to explain various vision models for downstream tasks.


### Research Challenge

The development of transferable explainers introduces two non-trivial two challenges:

**CH1:** Without task-specific exposure during the pre-training, how to ensure the universal effectiveness of explainer for various downstream tasks? 

**CH2:** How to adapt the explainer to a specific  task without fine-tuning on the task-specific data?


### Dependency
````angular2html
torch 
torchvision
pillow
transformers
datasets
accelerate
````

### Leverage our Pre-trained Explainer

Run our pre-trained explainer to generate heatmaps for explaining ViT classification models
````angular2html
python3 heatmap_demo.py
````

The heatmap of local explanation will be
<div align=center>
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/image/image-1-7-784.png">
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/explanation/heatmap-image-1-7-784.png.png">
<br>
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/image/image-0-9-416.png">
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/explanation/heatmap-image-0-9-416.png.png">
<br>
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/image/image-1-0-541.png">
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/explanation/heatmap-image-1-0-541.png.png">
<br>
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/image/image-1-30-542.png">
<img width="150" height="150" src="https://anonymous.4open.science/r/LETA-D997/output/explanation/heatmap-image-1-30-542.png.png">
</div>


