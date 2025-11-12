# Unified MEL: A Unified Framework for Multimodal Entity Linking with Large Language Models

This repository is the official implementation for the paper titled "Unified MEL: A Unified Framework for Multimodal Entity Linking with Large Language Models".

<p align="center">
  <img src="framework.png" alt="unimel" width="640">
</p>

## Usage

#### Step 1: Install and set up environment

```python
pip install -r requirements.txt
conda create -n unified-mel python==3.8.18
conda activate unified-mel
```



#### Step 2: Dataset

WikiDiverse from https://github.com/wangxw5/wikiDiverse.

#### Step 3: Train

If you want to train a new checkpoint, please refer to peft (https://github.com/huggingface/peft) or swift (https://github.com/modelscope/swift).



#### Step 4: Run

```
cd Unified-MEL
bash run.sh 0 wikidiverse  # for wikidiverse
```



## Code Structure

```python
├─code
│  │  main.py
│  │  
│  └─untils
│      │  dataset.py
│      │  functions.py
│      │  
│      └─__pycache__
│              dataset.cpython-38.pyc
│              functions.cpython-38.pyc
│              
└─config
        wikidiverse.yaml
│  framework.png
│  README.md
│  requirements.txt
│  run.sh
```

## Results

#### Main results

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
</head>
<body>
<table>
  <thead>
    <tr>
      <td>DWE+</td>
      <td>72.5</td>
      <td>97.3</td>
      <td>98.8</td>
      <td>99.6</td>
      <td>72.8</td>
      <td>97.5</td>
      <td>98.9</td>
      <td>99.7</td>
      <td>51.2</td>
      <td>91.0</td>
      <td>96.3</td>
      <td>98.9</td>
    </tr>
    <tr>
      <td>UniMEL (ours)</td>
      <td>94.8</td>
      <td>97.9</td>
      <td>98.3</td>
      <td>98.8</td>
      <td>94.1</td>
      <td>97.2</td>
      <td>98.4</td>
      <td>98.9</td>
      <td>92.9</td>
      <td>97.0</td>
      <td>99.5</td>
      <td>99.8</td>
    </tr>
  </tbody>
</table>
</body>
</html>

