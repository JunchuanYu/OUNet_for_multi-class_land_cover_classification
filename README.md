
<font size='5'>**Research on the optimization of multi-class land cover classification using deep learning with multispectral imagery**</font>


Yichuan Li, [Junchuan Yu*](https://github.com/JunchuanYu), Ming Wang, Minying Xie, Laidian Xi, Yunxuan Pang, and Changhong Hou

*corresponding author

## Updates
* **[2024.3.24]** Paper submission.
* **[2024.4.23]** Paper revision
* **[2024.4.24]** Code release


## Dataset
* The Worldview3 data used in this study was released during the Dstl Satellite Imagery Feature Detection competition, provided by the Defense Science and Technology Laboratory of the United Kingdom. the original data can be download through this link: [kaggle dtsl dataset](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data) 
* The training data used for the optimal model can be downloaded via this link: [training dataset](https://drive.google.com/file/d/1Bgc09QiLkD6Q_yeJIUWT_dX4-qwYJ-45/view?usp=sharing)
* The panoramic data used for testing can be downloaded via this link: [testing dataset](https://drive.google.com/file/d/1PXazNEqBFySvvvPYxQVijx_9RdprFtDw/view?usp=sharing)

## Code

<div align="center">
    <img src="https://dunazo.oss-cn-beijing.aliyuncs.com/blog/network_revise3-04.jpg" width="600"  alt="描述文字">
</div>

<div align="center">

Structure of the proposed Model
</div>


Click the links below to download the checkpoint for the corresponding model type.

- [Baseline model]: [Google drive](https://drive.google.com/file/d/1zRdvWVNoyKhaHXjm3h85jTPgLgsLSjVy/view?usp=sharing)
  + Save the file in your download directory: `/checkpoint/baseline.hdf5`

- [OUNet model]: [Google drive](https://pan.baidu.com/s/19zsbA38Wk_qTBNoDp29F9w?pwd=prif)
  + Save the file in your download directory: `/checkpoint/OUNet.hdf5`

+ The supporting library information of the code is shown below:

<div align="center">

| Package                   | Version |
|:-------------------------:|:-------:|
| GDAL                      | 3.4.3   |
| h5py                      | 3.1.0   |
| matplotlib                | 3.7.2   |
| numpy                     | 1.22.1  |
| tensorflow                | 2.10.0  |
| pandas                    | 1.5.0   |
| sklearn                   | 1.1.2   |

</div>

## Acknowledgement
*  [HeyWhale Community](https://www.heywhale.com)  Provided the computational platform for this work.
+ [kaggle](https://www.kaggle.com)  Provided the dataset  for this work.

If you find this study helpful, please star this repo [OUNet_for_multi-class_land_cover_classification](https://github.com/JunchuanYu/OUNet_for_multi-class_land_cover_classification), and cite using this BibTeX:

```bibtex
@article{
  title={TResearch on the optimization of multi-class land cover classification using deep learning with multispectral imagery},
  author={Lichuan Li, Junchuan Yu*, Ming Wang, Ming Ying, Laidian Xi, Yunxuan Pang, and Changhong Hou}
  year={2024}
}
```

