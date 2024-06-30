# image-captioning in your language
This project is to explore RNN and Transformer architectures through the image captioning task. Moreover, it becomes more interesting when approached in your language.

## Project Overview
### Objective
The main goal is to comprehend the architecture of the encoder-decoder model based on RNN and transformer networks, as well as the fundamental pipeline in image captioning. Most of the project refer to [this repository](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/tree/master).

### Dataset
The project utilizes the COCO2017 dataset, employing the train/val datasets and annotations available from the provided [link](https://cocodataset.org/#download).

## Translation and Preprocessing
### Translation
The [Facebook/NLLB-200-Distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) translation model from Hugging Face is employed to translate English into other desired languages. To obtain a dataset with captions in your language, execute the following command:
```
python captions_translator.py
```

### Data Preprocessing
For efficient data loading, image values are stored in `.hdf5` format, and corresponding captions for each image are gathered from the original annotations. To preprocess the data, execute the following command:
```
python create_input_files.py
```

## Model Architectures
### LSTM
Over view of the LSTM-based encoder-decoder model.
![image](docs/rnn_encoder_decoder.png)

### Transformer
Describe the Transformer architecture used for image captioning.

## Training
### Training Setup
Provide details on the training setup, including hardware, software, and dependencies.

### Fine-Tuning on Translated Dataset
Explain the process of fine-tuning the models on the translated dataset.

## Evaluation
### Metrics
List the evaluation metrics used to assess model performance.

### Results
Present the results of the models, including any tables or graphs.

## Installation
Provide a step-by-step guide on how to install the project and its dependencies.

## Usage
### Running the Code
Instructions on how to run the training and evaluation scripts.

### Inference
Explain how to use the trained models for captioning new images.

## Experiments and Findings
### Comparative Analysis
Compare the performance of LSTM and Transformer models.


## Contributing
Provide guidelines for contributing to the project.

## Acknowledgements
Acknowledge any contributors, advisors, or sources of inspiration.

## References
List any references or resources used in the project.