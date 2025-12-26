Data Imbalance in Diabetes Dataset
==================================

The dataset is imbalanced because the number of *Healthy*, *Pre-Diabetic*, and *Diabetic* cases are not equally distributed.

Why this is important:
- Biased models
- Poor recall for minority classes
- Poor generalization

Solutions implemented:
- Weighted F1-score
- Stratified split
- Balanced class weights in model
