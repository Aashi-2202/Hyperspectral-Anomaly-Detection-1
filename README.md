Hyperspectral Anomaly Detection with WLSDL
Paper and Code about my research on hyperspectral anomaly detection

This repository builds upon the existing work on Hyperspectral Anomaly Detection via Background and Potential Anomaly Dictionaries Construction (IEEE TGRS 2018
) and extends it by implementing the Weighted Low-rank and Structured Dictionary Learning (WLSDL) algorithm for improved anomaly detection performance.

The WLSDL algorithm is integrated with the original framework to further enhance anomaly detection results in hyperspectral images.

📖 Introduction

Hyperspectral anomaly detection aims to locate unusual pixels that differ significantly from the background.

The original work included Joint Sparse Representation and Low Rank and Sparse Representation (LRSR).

In this extension, we incorporate the WLSDL method, which improves background suppression and anomaly enhancement.

Sample Output using WLSDL:

📂 Contents

demo.py → Main demo file for running experiments

HyperProTool.py → Basic hyperspectral image processing utilities

dic_constr.py → Dictionary construction for background & anomalies

LRSR.py → Low-rank and sparse representation with ALM

ROC_AUC.py → Functions for ROC and AUC calculation

result_show.py → Visualization of experimental results

WLSDL.py → Implementation of Weighted Low-rank and Structured Dictionary Learning

⚙️ Requirements

Python 3.x

Numpy, Scipy, Matplotlib

Tested on Windows 10 (Intel Core i7)

📊 Example Dataset

The dataset used is the San Diego AVIRIS hyperspectral dataset, originally from:

Xu et al., “Anomaly detection in hyperspectral images based on low-rank and sparse representation”, IEEE TGRS 2015.

Spatial resolution: 3.5m/pixel, with 224 spectral bands (370–2510 nm).

🙌 Credits

This work is an extension of:

Ning Huyan, Xiangrong Zhang, Huiyu Zhou, and Licheng Jiao,
“Hyperspectral Anomaly Detection via Background and Potential Anomaly Dictionaries Construction”,
IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 4, pp. 2263–2276, 2018.

@article{huyan2018hyperspectral,
  title={Hyperspectral Anomaly Detection via Background and Potential Anomaly Dictionaries Construction},
  author={Huyan, Ning and Zhang, Xiangrong and Zhou, Huiyu and Jiao, Licheng},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={57},
  number={4},
  pages={2263--2276},
  year={2018},
  publisher={IEEE}
}


Dataset reference:

@article{xu2015anomaly,
  title={Anomaly detection in hyperspectral images based on low-rank and sparse representation},
  author={Xu, Yang and Wu, Zebin and Li, Jun and Plaza, Antonio and Wei, Zhihui},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={54},
  number={4},
  pages={1990--2000},
  year={2015},
  publisher={IEEE}
}

🚀 Future Work

Explore integration with deep learning methods for end-to-end anomaly detection.

Benchmark WLSDL against other recent anomaly detection algorithms.
