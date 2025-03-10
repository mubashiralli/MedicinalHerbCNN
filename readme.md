# Deep Learning CNN Model for ChineseHerbs Dataset

The `Chinese Herb Dataset (CNH-98)` is a collection designed to aid in the classification and recognition of `Chinese medicinal herbs` through image-based machine learning models. This dataset includes thousands of images labeled across various classes of Chinese herbs, each corresponding to a particular herb or plant used in traditional Chinese medicine (TCM). The dataset is typically used for training deep learning models, such as Convolutional Neural Networks (CNNs), for image classification tasks, and is especially valuable for botanical researchers, TCM practitioners, and developers working on applications involving medicinal plants.

## Setup Environment

**Explore CUDA:**

if target system has GPU enabled graphics card, please refer to these support to train model quicker

- https://anaconda.org/anaconda/cudnn | https://developer.nvidia.com/cudnn
- https://developer.nvidia.com/cuda-gpus

1. **Clone the repository:**

   ```bash
   git clone git@github.com:mubashiralli/MedicinalHerbCNN.git #SSH
   cd MedicinalHerbCNN
   ```

2. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

   **Ananconda Support:**

   ```bash
   conda create -n tf-gpu python==3.8 -y
   conda activate tf-gpu
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Run Jupyter Notebook

1. **Start Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```
   ### or
   ```bash
   code notebook
   ``` 
   ### requires  `ipykernel` to run jupyter-notebook file in Visual Studio code

2. **Open the notebook:**
   Navigate to the Jupyter Notebook interface in your web browser and open the `ChineseHerbs_CNN.ipynb` file.

## Deep Learning CNN Model

The Jupyter notebook `ChineseHerbs_CNN.ipynb` contains the following sections:

1. **Data Loading and Preprocessing:**

   - Load the ChineseHerbs Dataset.
   - Preprocess the data (e.g., normalization, resizing).

2. **Model Architecture:**

   - Define the CNN model architecture using frameworks like TensorFlow or PyTorch.

3. **Training the Model:**

   - Compile the model.
   - Train the model on the training dataset.
   - Validate the model on the validation dataset.

4. **Evaluation:**

   - Evaluate the model performance on the test dataset.
   - Generate metrics such as accuracy, precision, recall, and F1-score.

5. **Prediction:**
   - Use the trained model to make predictions on new data.

## Dataset

The ChineseHerbs Dataset should be placed in the `./` directory. Ensure the dataset is properly structured for loading and preprocessing in the notebook.

## Requirements

- Python 3.x
- TensorFlow
- Jupyter Notebook
- Other dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
