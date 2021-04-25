# Mem2Img 
- Black Hat Asia 2021
    - Mem2Img: Memory-Resident Malware Detection via Convolution Neural Network
    - https://www.blackhat.com/asia-21/briefings/schedule/#memimg-memory-resident-malware-detection-via-convolution-neural-network-22262
## Installation
1. Clone this repository on your machine
    ```bash
    # Clone the repository on "master" branch
    $ git clone -b master https://github.com/AragornTseng/Mem2Img
    ```
2. Install the required packages via the following command
    ```bash
    # Run the command at the root of the repository
    $ pip3 install -r requirements.txt
    ```

## Execution

1. Put the resident-malware memory blocks dataset into the directory [`"1_raw_memory_blocks"\`]("1_raw_memory_blocks"/)

2. Open the PowerShell and run `split_data.ps1` (take a few minutes)
    * Run [`split_data.ps1`](split_data.ps1)
        ```bash
        # Make sure your current directory is correct
        PS> .\split_data.ps1
        ```
    * If succeed, you will see the following files (folders) in folder [`2_memory_blocks\`](2_memory_blocks/)
        * `Train` - For training
        * `Testing` - For testing

3. Run [`to_png.py`](to_png.py) (take a few minutes)
    ```bash
    # Make sure your current directory is correct
    PS> python3 to_png.py 2_memory_blocks(input_dir) 3_png(output_dir) 
    ```
    * If succeed, you will see the the training and testing datasets in folder [`3_png\`](3_png/)

3. Run [`augmentation.py`](augmentation.py) (take a few minutes)
    ```bash
    # Make sure your current directory is correct
    PS> python3 augmentation.py 3_png(intput_dir) 
    ```
    * If succeed, you will see the the augmentation png in folder [`3_png\`](3_png/)

4. Run [`train.py`](train.py) (take a few minutes)
    ```bash
    # Make sure your current directory is correct
    PS> python3 train.py 3_png\Train(training_dir) 3_png\Testing(testing_dir) 
    ```
    - generate training model and output the testing score

## Testing

1. Run [`test.py`](test.py) (take a few minutes)
    ```bash
    # Make sure your current directory is correct
    PS> python3 test.py (testing image path) 
    ```
    - ouput the predicted class and the nearest five points in training datasets
- Model file is loacated on goole drive
    - https://drive.google.com/drive/folders/1S_lEU3lMiU5wwY8eBNB6OrwgHJogZBU9?usp=sharing
## Visualization

- The code for visualization is in [`visualization.ipynb`](visualization.ipynb)
    - confusion matrix
    - t-sne
    - saliency map
    - grand-cam 