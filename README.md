# Animal_Detection

Image Object Detection.

---

# Project Details:

Task:			Object Detection

Model Architecture:  XceptionFPN

mAP(50:0.05:95):		74

Inference Speed:	65 ms

Total Speed:	106ms

Notes:

total speed means the inferencing time + decoding prediction time

---

# Outlines:

- Loading and processing data
- Visualizing Data
- Choosing anchor boxes aspect ratio
- Writing TFRecords
- Split a small data set for tuning
- Loading data into the pipeline
- Model Building
  - Choosing the best model architecture
  - Train
  - Evaluation
  - Testing
    - Comparing model prediction on testing data with its labels
    - Testing the model using images downloaded from the internet
    - Testing the model using videos downloaded from the internet

---

# Model prediction compared with truth

![](My_Model/prediction/3.png)

---

# Folders Description:

- MY_Model: contains the model architecture tuning history and final architecture training history.
  -------------------------------------------------------------------------------------------------
- workspace: contains helper function I have used in the main notebook.

---
