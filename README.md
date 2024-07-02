# Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach

# LR-HR pairs generated from RodoSol-ALPR

The **High-Resolution (HR)** images used in our experiments were generated as follows. For each image from the chosen datasets, we first cropped the LP region using the annotations provided by the authors. We then used the same annotations to rectify each LP image, making it more horizontal, tightly bounded, and easier to recognize. The rectified image serves as the HR image.

We generated **Low-Resolution (LR)** versions of each HR image by simulating the effects of an optical system with lower resolution. This was achieved by iteratively applying random Gaussian noise to each HR image until we reached the desired degradation level for a given LR image (i.e., SSIM < 0.1). We padded the LR and HR images to maintain the aspect ratio before resizing.

Here are some HR-LR image pairs created from the [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset:
<img src="./Media/image2.png" width="600"/>

### How to obtain the RodoSol-SR and PKU-SR datasets

As we are not the creators of the [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset, we have decided to grant access to the images we have generated from these datasets upon request, subject to the signing of a licensing agreement. In essence, the RodoSol-SR dataset is released for academic research only and is free to researchers from educational or research institutes for **non-commercial purposes**.

To be able to download the dataset, please read [**this license agreement**](./Media/license-agreement.pdf) carefully, fill it out and send it back to the second author ([rblsantos@inf.ufpr.br](mailto:rblsantos@inf.ufpr.br)) (who also manages access to the [RodoSol-ALPR](https://github.com/raysonlaroca/rodosol-alpr-dataset) dataset). **Your e-mail must be sent from a valid university account** (.edu, .ac or similar).

In general, you will receive a download link within 3-5 business days. Failure to follow the instructions may result in no response.

# Usage

## Testing

## Training From Scratch

# Citation

If you use our code or datasets in your research, please cite:
* V. Nascimento, R. Laroca, R. O. Ribeiro, W. R. Schwartz, D. Menotti, “Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach,” in Conference on Graphics, Patterns and Images (SIBGRAPI), pp. 1-8, Sept. 2024.

```
@article{nascimento2024enhancing,
  title = {Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach},
  author = {V. {Nascimento} and R. {Laroca} and R. O. {Ribeiro} and W. R. {Schwartz} and D. {Menotti}},
  year = {2024},
  journal = {Conference on Graphics, Patterns and Images (SIBGRAPI)},
  volume = {},
  number = {},
  pages = {1-8},
  doi = {},
  issn = {},
}
```

You may also be interested in our [previous work](https://github.com/valfride/lpr-rsr-ext/):
* V. Nascimento, R. Laroca, J. A. Lambert, W. R. Schwartz, D. Menotti, “Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers,” in *Computers & Graphics*, vol. 113, pp. 69-76, 2023. [[Science Direct]](https://doi.org/10.1016/j.cag.2023.05.005) [[arXiv]](https://arxiv.org/abs/2305.17313)

## Related publications

A list of all our papers on ALPR can be seen [here](https://scholar.google.com/scholar?hl=pt-BR&as_sdt=0%2C5&as_ylo=2018&q=allintitle%3A+plate+OR+license+OR+vehicle+author%3A%22David+Menotti%22&btnG=).

## Contact

Please contact Valfride Nascimento ([vwnascimento@inf.ufpr.br](mailto:vwnascimento@inf.ufpr.br)) with questions or comments.
