# Mem2Img 


## Installation


## Execution

1. Put the resident-malware memory blocks dataset into the directory [`"1_memory_blocks"\`]("1_memory_blocks"/)

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


## Testing

1. Run [`test.py`](test.py) (take a few minutes)
    ```bash
    # Make sure your current directory is correct
    PS> python3 test.py (testing image path) 
    ```

## Visualization

- The code for visualization is in [`visualization.ipynb`](visualization.ipynb)