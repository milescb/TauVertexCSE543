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

For convenience, a small dataset used for training is included at `data/data.root`. To download a sample with ~1.7 million events run:

```
curl -o data/data_large.root https://cernbox.cern.ch/remote.php/dav/public-files/3w2v4biLSGyi9BL/data_large.root
```

## Running the code

To train simply run the training script:

```
python train.py -out output -epoch 50 -bs 128 -lr 0.0004 --hidden-size 128 --n-gaussians 3
```

This saves a trained model to the selected output directory. Then, to test, run the testing script over the trained model:

```
python test.py --model-path output/model.pth --output-dir plots
```

A pre-trained network on the large data-set is saved in `output/model_mdn.pth`. To get plots from this immediately, run 

```
python test.py --model-path output/model_mdn.pth
```

Note: the large dataset must be used for testing this model out of the box. 