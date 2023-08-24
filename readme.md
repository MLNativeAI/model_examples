
# MLnativeAI Model Examples

This repo contains complete examples on how to build production-ready ML model Docker images. We use these mostly for internal testing of [MLnative Platform](https://mlnative.com) and on the [Playground](https://playground.mlnative.com)

Most models here come from [HuggingFace](https://huggingface.co/). 

In order to build a new model version, all you have to do is set a new tag in `develop.yaml`. 

The build process for each model pre-downloads the model to speed up the startup time of the container.

