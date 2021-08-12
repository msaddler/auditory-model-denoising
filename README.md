<a name="top"></a>

# Speech Denoising with Auditory Models ([arXiv](https://arxiv.org/abs/2011.10706), [sound examples](https://mcdermottlab.mit.edu/deep_feature_denoising/demo.html)
This is a TensorFlow implementation of our [Speech Denoising with Auditory Models](https://arxiv.org/abs/2011.10706).

Contact: [Mark Saddler](mailto:msaddler@mit.edu) or [Andrew Francl](mailto:francl@mit.edu)

<a name="citation"></a>
## Citation
If you use our code for research, please cite our paper:
Mark R. Saddler*, Andrew Francl*, Jenelle Feather, Kaizhi Qian, Yang Zhang, Josh H. McDermott. Deep Network Perceptual Losses for Speech Denoising. [arXiv:2011.10706](https://arxiv.org/abs/2011.10706). 2021.

### License
The source code is published under the MIT license. See [LICENSE](./LICENSE.md) for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

### Requirements
In order to speed setup and aid reproducibility we provide a Singularity container. This container holds all the libraries and dependencies needed to run the code and allows you to work in the same environment as was originally used. Please see the [Singularity Documentation](https://sylabs.io/guides/3.8/user-guide/) for more details.

Download Singularity image: [tensorflow-1.13.0-denoising.simg](https://drive.google.com/file/d/1KFGMJnuX4l1KRQRRVnzbjXE6bjHA7Tjm/view?usp=sharing)

### Pre-trained Models
We provide checkpoints for our trained models to allow users to quickly test our transformation on their own audio. These are also necessary to run our demonstration jupyter notebook.

Download models directory: [checkpoints for audio_transforms and recognition_networks](https://drive.google.com/drive/folders/1HmXSCVOKQCq7G62rs9KE_jsvVO0UqclC?usp=sharing)

### Quick start

We provide a jupyter notebook that walk-through usage of the model during testing and provides examples detailing how to denoise audio using our model. 

Go to [Demo Notebook](./DEMO.ipynb).


