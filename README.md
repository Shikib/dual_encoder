## Introduction

This code is an implementation of the Dual-Encoder LSTM introduced in [The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems](https://arxiv.org/abs/1506.08909).

The data can be found [here](https://drive.google.com/file/d/0B_bZck-ksdkpVEtVc1R6Y01HMWM/view) and must be placed in the `data/` directory.

This project has some (WIP, and therefore messy) improvements on the model introduced in the paper. I've experimented with an attention mechanism and [CNN-based encoders](https://arxiv.org/abs/1408.5882). I intend to clean all of this up soon and add a variety of other improvements, in the **very near future**.

## Training

Simply run `python3 train.py`. Edit the hyperparemters at the the top of the file.

Pre-requisites are PyTorch, CUDA, numpy and NLTK.

## Inference

The `predict.py` file contains two methods of interest for inference, using the pre-trained model provided in the repo. To run inference with an alternate model, replace the code on line 7 of the file.

`predict_val`, given a context (sequence of messages delimited by particular tokens) and a reply (single message) determines the likelihood of the reply following the context.

`predict`, given a string (be it a message or a sequence) provides the output of the encoder for the string. This output can be utilizes as a mesasge embedding for various purposes.

## Contact

Feel free to contact me at mehrishikib@gmail.com if you have any questions regarding this implementation.
