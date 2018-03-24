# Simple CNN for Fashion MNIST
A simple architecture is used to classification task of Fashion MNIST dataset. This model achieves ~90% accuracy.

```
usage: cnn.py [-h] [-bs BATCH_SIZE] [-nbw NUMBER_WORKER] [-ep EPOCHES]
              [-mn MODEL_NAME] [-lr LEARNING_RATE]
              {train,test,demo}

positional arguments:
  {train,test,demo}     Different mode of script: train|test|demo

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size will used in training phase.
  -nbw NUMBER_WORKER, --number_worker NUMBER_WORKER
                        number of worker used on prepare data.
  -ep EPOCHES, --epoches EPOCHES
                        number of epoches to run.
  -mn MODEL_NAME, --model_name MODEL_NAME
                        name of model without extension, the extension is
                        added automatically.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate of training.
                        
```
