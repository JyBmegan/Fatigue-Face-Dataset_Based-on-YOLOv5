### Non-contact Psychological Fatigue Assessment using Computer Vision

This repository holds the code and data for our project on assessing psychological fatigue, which we submitted to the "Hou Can Cup" challenge. Our goal was to build a tool that could spot mental fatigue without using subjective surveys or cumbersome physical sensors, relying only on computer vision.

We tried out three different ways to sort fatigue into three levels: **Awake**, **Moderately Fatigued**, and **Severely Fatigued**.

1.  **YOLOv5 Model**: A deep learning model that looks at single pictures (frames from a video) to spot fatigue.
2.  **Random Forest (PERCLOS + Blink Rate)**: A simpler machine learning model that analyzes video directly using well-known eye-tracking metrics.
3.  **Random Forest (Multi-indicator)**: Another video-based model where we tested out some extra behavioral features.


### Why We Did This

Mental fatigue is a real problem that affects everything from driving safety to how well you can study. The usual ways to measure it aren't great. Questionnaires can be unreliable because it's just how someone feels at that moment, and using sensors like EEG is a hassle and pretty much impossible outside of a lab.

We thought using a simple camera to analyze facial expressions and eye movements could be a better way forward. We noticed that most existing work didn't have data on Chinese populations, usually just sorted people into "tired" or "not tired," and often wasn't backed by proper psychological standards. We wanted to try and fix those issues.

### The dataset: Fatigue-Face-Dataset

We created this dataset by recording people before and after they completed a task designed to wear them out mentally.

* **Participants**: 39 volunteers (33 female, 6 male), with an average age of about 23.
* **How Fatigue Was Induced**: We used a 40-minute computer task (a modified SART) that requires constant focus, which is known to cause mental fatigue.
* **Labeling Method**:
    * We used the Swedish Occupational Fatigue Inventory-25 (SOFI-C) as our main tool.
    * To set clear boundaries between fatigue levels, we used the Karolinska Sleepiness Scale (KSS) to find the best cut-off scores.
    * **The Three Levels**:
        * **Awake**: SOFI-C score < 74
        * **Moderate Fatigue**: SOFI-C score from 74 to 139
        * **Severe Fatigue**: SOFI-C score of 140 or more
* **Data Structure**: We converted the 78 videos into still frames. To make sure our model wasn't just memorizing faces, we split the data so that one person's images would only appear in either the training or the testing set, never both.
    * **Training Set**: 7,918 images from 31 people.
    * **Validation Set**: 2,286 images from the remaining 8 people.

### Download the Dataset

**Images**: https://drive.google.com/drive/folders/1rqwfbrj8Nvu76CV0XmQ6mrHOp6x2gLja?usp=sharing 
**Raw Label**: https://drive.google.com/drive/folders/1PZ-UIrUja2R1Ei6dK14Xc8lkOEj8jyTo?usp=share_link
**Raw Videos**: https://drive.google.com/drive/folders/1RU2QeS1u-Ntll0SLTLb5yj92IE54eVok?usp=share_link

### Training

#### 1. YOLOv5 (Image-Based Detection)

For this method, we used the popular YOLOv5 model to find a face in a picture and classify its fatigue level at the same time.

* **Training Command**:
    ```shell
    !python train.py \
        --img 640 \
        --batch-size 16 \
        --epochs 50 \
        --data ../2_yolo_dataset/dataset.yaml \
        --weights yolov5s.pt \
        --cfg models/yolov5s.yaml
    ```

#### 2. RandomForest with PERCLOS & Blink Rate (Video-Based)

Here, we processed the videos directly to pull out a couple of key numbers related to eye behavior, then fed those into a Random Forest model. You can find the code for this in the `RandomForest_PERCLOS` folder.

* **Features Used**:
    * **Blink Frequency**: Simply how many times a person blinks in a minute.
    * **PERCLOS (Percent Eye Closure)**: This is a well-researched metric that measures how much time the eyes are mostly closed. It's a strong sign of drowsiness.

#### 3. RandomForest with Multi-indicators (Experimental Video-Based)

This was a second version of the Random Forest model where we tried adding more detailed behavioral features to see if it would help. The code is in the `RandomForest(Multi-indicators)` folder.

* **Extra Features**: On top of the basics, we added:
    * **Average Blink Duration**: Tired people tend to blink more slowly.
    * **Max Closure Duration**: This looks for long eye closures that could be "microsleeps".
    * **Blink Rate Variability**: When you're tired, your blinking can become less regular.

### Results and What We Found

The biggest takeaway is that all our models were thrown off by the fact that our dataset was imbalanced. We had way more examples of people being "Awake" than "Severely Fatigued." This made it really hard for the models to learn how to spot the more tired states.

#### Model 1: YOLOv5

The model was fast and could run in real-time, but its accuracy was very uneven.

| Category | Instances | Precision | Recall (R) | AP@0.5 |
| :--- | :--- | :--- | :--- | :--- |
| Awake | 1395 | 0.42 | **1.0** | 0.611 |
| Moderate Fatigue | 736 | **0.957** | 0.18 | 0.681 |
| Severe Fatigue | 154 | 0.0134 | 0.00649 | 0.11 |
| **All** | **2285** | **0.463** | **0.395** | **0.467** |


#### Model 2: RandomForest (PERCLOS + Blink Rate)

**Overall Accuracy**: 46.15%

| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Awake | 0.62 | 0.62 | 0.62 | 8 |
| Moderate Fatigue | 0.33 | 0.25 | 0.29 | 4 |
| Severe Fatigue | 0.00 | 0.00 | 0.00 | 1 |
| **Weighted Avg** | **0.49** | **0.46** | **0.47** | **13** |


#### Model 3: RandomForest (Multi-indicator)

**Overall Accuracy**: 38.46%

| Category | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| Awake | 0.50 | 0.62 | 0.56 | 8 |
| Moderate Fatigue | 0.00 | 0.00 | 0.00 | 4 |
| Severe Fatigue | 0.00 | 0.00 | 0.00 | 1 |
| **Weighted Avg** | **0.31** | **0.38** | **0.34** | **13** |


### Acknowledgements

This work was done for the "厚粲杯" National Collegiate Psychological and Cognitive Intelligence Assessment Challenge. A big thanks to the event organizers and to all the students who participated in our study.
