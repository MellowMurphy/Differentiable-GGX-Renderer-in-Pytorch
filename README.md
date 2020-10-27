# Differentiable-GGX-Renderer-in-Pytorch
This is a translation of the Tensorflow version of the differentiable renderer in Flexible SVBRDF Capture with a Multi-Image Deep Network by Valentin Deschaintre et al. 2019.
Their project is located at https://team.inria.fr/graphdeco/projects/multi-materials/

For now, I have only translated the bare minimum required to use the renderer. More will be added later.
To test out results from random noise, simply run `test_renderer.py` and both the diffuse and specular rendering results will be saved.
I have made sure that rendering results from the original TF version and the Pytorch version match exactly, so no need to worry about that.

Also included is the translation table I have been keeping up with during the task. It is no means vast, but includes many often used functions.
It is located at `TensorFlow-PyTorch Translator.pdf`.
