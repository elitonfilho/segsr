# segsr
SRGAN-based Image SR improved by semantic segmentation

SegSR focus on improving image super-resolution by using a segmentation module capable of verifying the performance of the generator module.
The inference performance captured by the segmentation module behaves as a a criterion to the generator module, impproving thus its performance.

## Supported architectures
- Generator:
    - [SRGAN](https://arxiv.org/abs/1609.04802)
- Discriminator:
    - VGG
- Segmentation module:
    - [UNet](https://arxiv.org/pdf/1505.04597.pdf)
    - [HRNet](https://arxiv.org/abs/1908.07919)



## Acknowledgement

Our codebase was heavily influented by the following repositories:

https://github.com/leftthomas/SRGAN

https://github.com/xinntao/BasicSR

https://github.com/stefanopini/simple-HRNet