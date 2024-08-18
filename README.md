# Pneumonia Chest X-ray Classification [(Collab Link)](https://colab.research.google.com/drive/1TqQegFXJHg4EBemBEaI0B4DTHbvuualH?usp=sharing)

## Introduction
* **Convolutional Neural Network (CNN)** model developed in PyTorch to perform **binary classification** on X-ray jpeg images from [this Kaggle Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).
* After opening the notebook in Collab, please go to `File>Save as copy in Google Drive` to experiment with the code after reading the **Data Handing** section below.
* **Saved Models** (CNN_v1_03, CNN_v2, CNN_v3_02) and their **Train and Test history** (Train Test History.zip) can be downloaded and uploaded to Google Drive for personal usage.

---

## Features
### Data Handling 
* Code requires **.zip file** containing the **Pneumonia Dataset** to be uploaded to **Google Drive** linked to collab notebook.
* .zip file is automatically extracted to `drive\My Drive\ML Data Sets` while removing any corrupted images.
* Train, test and validation set as `torch.datasets` are created.
* Train, test and validation data loaders are created.
* Model test and train history can be saved to Google Drive. 
* Saved `model state_dict` and train and test history can be loaded to "resume progress."

### CNN Model
* The [TinyVGG](https://poloclub.github.io/cnn-explainer/) model architecture inspires the CNN model.
* Techniques such as **Learning Rate Scheduling, RAdam warmup**, and hyperparameter tuning are used to enhance model performance.
* **Regularization** in the form of Dropout layers is present to avoid overfitting.
* **Image augmentation** performed using `torchvision.transforms.v2`.
* Custom training, testing and **Early Stopping + Checkpointing** features.

---

## Evaluation Criteria
* Model evaluated on the following metrics:
  * Test loss, Test Accuracy | Test and Train history plotted
  * Confusion matrix | Sensitivity and Specificity
  * AUROC

### Results
#### CNN_v1 ( file: `CNN_v1_03.pth` )
* 512 x 512 input resolution of X-ray images
* Total Training Time: **~1hr 45mins** for **11 epochs** (T4 GPU: Google Collab)
  ![image](https://github.com/nirvan840/Pneumonia-Chest-X-ray-Classification/assets/56934010/6b11bec0-262e-43d3-b276-50ff5861fe70)

#### CNN_v2 ( file: `CNN_v2.pth` )
* 256 x 256 input resolution of X-ray images
* Total Training Time: **~1hr 20mins** for **12 epochs** (T4 GPU: Google Collab)
  ![image](https://github.com/nirvan840/Pneumonia-Chest-X-ray-Classification/assets/56934010/05d51093-92fe-49b4-97c2-af49a2f27c9e)

#### CNN_v3 ( file: `CNN_v3_02.pth` )
* 128 x 128 input resolution of X-ray images
* Total Training Time: **~1hr** for **13 epochs** (T4 GPU: Google Collab)
  ![image](https://github.com/nirvan840/Pneumonia-Chest-X-ray-Classification/assets/56934010/e5ff1b7f-0715-4d75-bfba-e84fb06c3003)

### Best Performance
* CNN_v2 has best overall performance & training time
  * Training time:  ~1hr 20mins ( T4 GPU Google Collab )
  * Test accuracy: 88.12% ( 600+ test X-rays ) | Sensitivity: 92.05% | AUROC: 0.947

---

## Conclusion
* Loading, Saving and Resuming progress features developed for a model saved in .pth format.
* Through thorough testing, certain image augmentations were determined "best suited" for the dataset:
  * RandomHorizontalFlip
  * RandomRotation
  * Normalization (Better and faster convergence)
* Usage of Adam optimiser from 1st epoch was seen disadvantageous.
  * Adam requires a "warmup" phase
  * RAdam used to solve this problem
  * Leading to better convergence
* Input resolutions:
  * 512 x 512 - No noticeable advantage
  * 256 x 256 - Significant decrease in training time | Marginally better performance, wrt to 512 x 512
  * 128 x 128 - No noticeable advantage over 256 x 256
  * **Following results are in a scenario ( < 15 epochs ) where early stopping was initiated due to no improvement in test loss for 5 epochs or more, 128 x 128 might have a noticable reduction in training time for higher number of epochs**.

---

## Future Plans
* Further optimise CNN model using BatchNormalization layers.
* Use CNN model to generate feature map for few-shot learning. 

