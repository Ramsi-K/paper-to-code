# VQGAN Implementation

## Summary

This repository contains an implementation of the Vector Quantized Variational Autoencoder (VQGAN) model as described in the paper [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2102.07074) by Esser et al. The VQGAN model combines the power of transformers with vector quantization to generate high-resolution images with fine details.

The VQGAN model introduces a novel approach to image synthesis by combining transformers with vector quantization. It utilizes a hierarchical VQ-VAE architecture to encode input images into discrete latent codes, which are then decoded by a transformer-based generator to produce high-resolution images. This approach allows for efficient training and generation of diverse and realistic images across various datasets.

## Discussion

The Vector Quantized Variational Autoencoder (VQ-VAE) is a novel approach in generative modeling that combines the expressive power of VAEs with the scalability and efficiency of VQ methods. In the paper "Neural Discrete Representation Learning," van den Oord et al. introduce the VQ-VAE, proposing a hierarchical VAE architecture with a discrete latent space. Unlike traditional continuous latent space models, the VQ-VAE employs discrete latent variables, each representing a discrete entity or "codebook vector" in the data distribution. This discrete representation enables efficient encoding and decoding of data while preserving semantic information.

The VQ-VAE architecture consists of an encoder network that maps input data to discrete latent codes, a codebook that stores the discrete code vectors, and a decoder network that reconstructs the input data from the encoded latent codes. During training, the encoder learns to map input data to the nearest codebook vectors, while the decoder learns to reconstruct the original data from the quantized latent codes. By introducing a discretization bottleneck in the latent space, the VQ-VAE encourages the model to learn meaningful representations of the input data, facilitating efficient generation and manipulation of high-dimensional data.

One key advantage of the VQ-VAE is its ability to learn disentangled and interpretable representations of complex data distributions. By imposing a discrete structure on the latent space, the model can capture discrete factors of variation in the data, such as object identities or semantic attributes. This disentanglement property enables intuitive manipulation of data attributes and facilitates tasks such as image generation, style transfer, and content manipulation. Additionally, the hierarchical nature of the VQ-VAE allows for scalable representation learning, making it well-suited for large-scale generative modeling tasks.

Overall, the VQ-VAE presents a promising approach to representation learning and generative modeling, offering a principled framework for learning discrete latent representations of complex data distributions. By combining the advantages of VAEs and discrete latent variable models, the VQ-VAE provides a versatile tool for tasks such as image generation, data compression, and unsupervised learning. Further research and experimentation with the VQ-VAE architecture are likely to yield insights into its capabilities and potential applications in various domains.

## Methodology

To convert the VQGAN paper to code, the following modules were implemented:

- `codebook.py`: Codebook management for vector quantization.
- `config.py`: Configuration settings for the model.
- `decoder.py`: Decoder architecture for generating images.
- `discriminator.py`: Discriminator architecture for training the model.
- `encoder.py`: Encoder architecture for encoding input images.
- `helper.py`: Helper functions for training and evaluation.
- `lpips.py`: Perceptual similarity loss for training.
- `mingpt.py`: Modified GPT architecture for generating images.
- `training_transformer.py`: Training loop for transformer-based model.
- `training_vqgan.py`: Training loop for VQGAN model.
- `transformer.py`: Transformer architecture for generating images.
- `utils.py`: Utility functions for data loading and preprocessing.
- `vqgan.py`: Main script for model initialization and training.

## Results and Future Plans

Due to computational constraints, the results of training on a local machine with limited batch size and image size were inconclusive. However, the implemented code functions correctly, and there are plans to revisit the project for cloud-based training in the future.

## References

1. [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2102.07074)
2. [Neural Discrete Representation Learning. arXiv preprint arXiv:1711.00937.](https://arxiv.org/abs/1711.00937)
3. [Official repo](https://github.com/CompVis/taming-transformers)
4. [VQ-GAN | PyTorch Implementation](https://youtu.be/_Br5WRwUz_U?si=coxcPxOJHvIvLx2w)
5. [All Things VQGAN](https://youtu.be/Q0YPkEbaOIY?si=uDlogQC4YifbTcca)


```
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@article{DBLP:journals/corr/abs-2102-07074,
  author       = {Yifan Jiang and
                  Shiyu Chang and
                  Zhangyang Wang},
  title        = {TransGAN: Two Transformers Can Make One Strong {GAN}},
  journal      = {CoRR},
  volume       = {abs/2102.07074},
  year         = {2021},
  url          = {https://arxiv.org/abs/2102.07074},
  eprinttype    = {arXiv},
  eprint       = {2102.07074},
  timestamp    = {Thu, 28 Apr 2022 16:17:29 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2102-07074.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

<!-- 2. 
3. [Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv preprint arXiv:1312.6114.](https://arxiv.org/abs/1312.6114)
4. [Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., & Abbeel, P. (2016). Infogan: Interpretable representation learning by information maximizing generative adversarial nets. In Advances in Neural Information Processing Systems (pp. 2172-2180).](https://papers.nips.cc/paper/2016/hash/057cbaa2e6f6d0f3ed5760aeba2e8b43-Abstract.html)
5. [Hjelm, R. D., Fedus, W., Lavoie-Marchildon, S., Grewal, K., Bachman, P., Trischler, A., & Bengio, Y. (2018). Learning deep representations by mutual information estimation and maximization. arXiv preprint arXiv:1808.06670.](https://arxiv.org/abs/1808.06670)
6. [Razavi, A., & van den Oord, A. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. arXiv preprint arXiv:1906.00446.](https://arxiv.org/abs/1906.00446)
7. [Esser, P., Rombach, R., & Ommer, B. (2018). A variational U-Net for conditional appearance and shape generation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 8857-8866).](https://openaccess.thecvf.com/content_cvpr_2018/html/Esser_A_Variational_U-Net_CVPR_2018_paper.html) -->
