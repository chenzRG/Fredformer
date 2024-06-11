# Fredformer

This code is related to the paper "Fredformer: Frequency Debiased Transformer for Time Series Forecasting" (KDD2024).

# Dependencies
Fredformer is built based on PyTorch.
You can install PyTorch following the instructions in [PyTorch](https://pytorch.org/get-started/locally/). For example:

```bash
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
After ensuring that PyTorch is installed correctly, you can install other dependencies via:

```bash
pip install -r requirements.txt
```

# About data:
We have prepared a dataset for this example: Weather. The CSV file is in the dataset folder.

# Training. 
For Fredfromer:
The scripts for our are in the directory ```./scripts/Fredformer```.
You can run the following command, then open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/Fredformer/weather.sh
 ```
Log files will be generated and updated in  ```./logs/``` during training.


All baselines can be found at https://github.com/thuml/Time-Series-Library.
