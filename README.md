# Food Taste Similarity Prediction using Image Embeddings

## Methodology Overview

My methodology involves predicting the taste similarity of food triplets based on image data. Here's a concise description of the approach I followed:

### Data Preparation:
- I began by importing necessary libraries for numerical operations, data manipulation, and deep learning. This includes tools like pandas for data handling, torchvision for computer vision tasks, and torch for building and training neural networks.
- I loaded my dataset containing 10,000 images along with their corresponding labels. These images represent various food items.
- The dataset was split into training, validation, and test sets using a standard practice of 80-20 split for training-validation.

### Dataset Customization:
- I defined a custom dataset class named ImageTriplesSet to handle loading and preprocessing of image triplets. Each triplet consists of three images representing different food items.
- Image transformations such as resizing, cropping, and normalization were applied to preprocess the images. These transformations ensure that the images are in a suitable format for feeding into a deep learning model.

### Model Selection and Training:
- I chose a pre-trained VGG16 model for feature extraction. VGG16 is a popular convolutional neural network architecture known for its effectiveness in image recognition tasks.
- The selected model was loaded and transferred to Kaggles GPU P100
- A custom triplet margin loss function was defined to optimize the model for learning embeddings that capture taste similarity.
- Training was performed using stochastic gradient descent (SGD) optimization with momentum and weight decay. The model was trained for a specified number of epochs.

### Evaluation:
- During training, I monitored training loss and periodically printed the progress.
- After training, the model was evaluated on the validation set to calculate the F1 score, which serves as a measure of performance.
- The F1 score indicates the model's ability to predict taste similarity based on image embeddings.

### Prediction:
- Finally, the trained model was used to make predictions on the test set. Each triplet was passed through the model, and the predicted labels (indicating taste similarity) were stored.
- The elapsed time for prediction was recorded.

### Visualization and Output:
- I visualized the training loss over time to assess the training progress and convergence.
- Predicted labels for the test triplets were saved to an output file for further analysis or submission.

Overall, my methodology involved leveraging a pre-trained deep learning model (VGG16) to learn embeddings from food images and using these embeddings to predict taste similarity between food triplets. The process encompassed data preparation, model training, evaluation, and prediction, with a focus on leveraging transfer learning and deep learning techniques for image-based similarity analysis.
