# Tutorial: image-captioning in your language
This project is to play with RNN and Transformer architectures through the image captioning task. Moreover, it becomes more interesting when approached in your language. In this project, I use Vietnamese as translated language.

## Instalation
```
conda env create -f environment.yml
```

## Dataset
The project utilizes the COCO2017 dataset, employing the train/val datasets and annotations available from the provided [link](https://cocodataset.org/#download).

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
`Soon be added`
## Demo
Demo of captioning model can be seen [here](https://github.com/danhugo/image-captioning/blob/main/notebook.ipynb]).
## Training
```
python train.py
```

## Evaluation
Evaluation metrics: BLEU
```
python eval.py
```
