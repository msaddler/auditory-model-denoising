<a name="top"></a>
## Speech Denoising with Auditory Models ([arXiv](https://arxiv.org/abs/2011.10706), [audio examples](https://mcdermottlab.mit.edu/denoising/demo.html))
This is a TensorFlow implementation of our [Speech Denoising with Auditory Models](https://arxiv.org/abs/2011.10706).

Contact: [Mark Saddler](mailto:msaddler@mit.edu) or [Andrew Francl](mailto:francl@mit.edu)

<a name="citation"></a>
### Citation
If you use our code for research, please cite our paper:
Mark R. Saddler\*, Andrew Francl\*, Jenelle Feather, Kaizhi Qian, Yang Zhang, Josh H. McDermott (2021). Speech Denoising with Auditory Models. *Proc. Interspeech* 2021, 2681-2685. [arXiv:2011.10706](https://arxiv.org/abs/2011.10706).

<a name="license"></a>
### License
The source code is published under the MIT license. See [LICENSE](./LICENSE.md) for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

<a name="requirements"></a>
### Requirements
In order to speed setup and aid reproducibility we provide a Singularity container. This container holds all the libraries and dependencies needed to run the code and allows you to work in the same environment as was originally used. Please see the [Singularity Documentation](https://sylabs.io/guides/3.8/user-guide/) for more details. Download Singularity image: [tensorflow-1.13.0-denoising.simg](https://drive.google.com/file/d/1KFGMJnuX4l1KRQRRVnzbjXE6bjHA7Tjm/view?usp=sharing).

<a name="models"></a>
### Trained Models
We provide model checkpoints for all of our trained audio transforms and deep feature recognition networks. Users must download the audio transform checkpoints to evaluate our denoising algorithms on their own audio. Both sets of checkpoints must be downloaded to run our [DEMO Jupyter notebook](./DEMO.ipynb). Download the entire `auditory-model-denoising/models` directory [here](https://drive.google.com/drive/folders/1HmXSCVOKQCq7G62rs9KE_jsvVO0UqclC?usp=sharing):
- Recognition network checkpoints: [auditory-model-denoising/models/recognition_networks](https://drive.google.com/file/d/1v9dKlRCnMP7X9v5IFcg4H0U1bXottgDo/view?usp=sharing)
- Auditory transform checkpoints: [auditory-model-denoising/models/audio_transforms](https://drive.google.com/file/d/1L21NqxN-nVSzpY9CtjtH-1zlKPhEQkow/view?usp=sharing)

<a name="demo"></a>
### Quick Start
We provide a [Jupyter notebook](./DEMO.ipynb) that (1) demos how to run our trained denoising models and (2) provides examples of how to compute the auditory model losses used to train the models. A [second notebook](./DEMO_train_audio_transform.ipynb) demos how to train a new audio transform using the auditory model losses.
