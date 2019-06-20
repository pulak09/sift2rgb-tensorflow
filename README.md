# sift2rgb-tensorflow

This is a Tensorflow implementation of SIFT feature descriptors to RGB image convertion as described in the following paper

Synthetic View Generation for AbsolutePose Regression and Image Synthesis, P. Purkait, C. Zhao and C. Zach, BMVC 2019
http://bmvc2018.org/contents/papers/0221.pdf

The code is adapted from https://github.com/affinelayer/pix2pix-tensorflow
For any query email to pulak.isi@gmail.com

## Usage 

It requires .sift and .rgb file-pairs to train and evaluate the network. The user needs to specify .SIFT and .RGB file names as described in file dataset_new_train.txt

To train the network:

    python main.py --phase train 

To evaluate:

    python main.py --test train 

