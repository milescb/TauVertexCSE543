# Estimating the Decay Vertex of Leptonically Decaying Tau Leptons with Deep Learning

## Setting up the environment

1. **Create the Conda Environment**:
    ```sh
    conda create --name myenv
    ```

2. **Activate the Conda Environment**:
    ```sh
    conda activate myenv
    ```

3. **Install Dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Data

For convenience, the dataset used for training is included at `data/data_large.root`. This sample contains ~1.7 million events to train on. 

## Running the code

To train simply run the training script. This saves a trained model to the selected output directory. Then, to test, run the testing script over the trained model. 