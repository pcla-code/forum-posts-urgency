# Overview

This repository contains supplementary materials for the following conference paper:

Valdemar Švábenský, Ryan S. Baker, Andrés Zambrano, Yishan Zou, and Stefan Slater.\
**Towards Generalizable Detection of Urgency of Discussion Forum Posts.**\
In Proceedings of the 16th International Conference on Educational Data Mining (EDM 2023).

# Contents of the repository

## Data

* `All_Courses_REDACTED_CODED.csv`: The training and cross-validation data set used for this work. For details, please refer to Sections 3.1--3.3 of the paper.
* `Stanford.csv`: The test data set used for this work. For details, please refer to Section 3.6 of the paper. Citation of this data set:

Agrawal, A. and Paepcke, A., 2014. The Stanford MOOC Posts Dataset.\
URL: http://infolab.stanford.edu/~paepcke/stanfordMOOCForumPostsSet.tar.gz. \
Original page available on URL: https://web.archive.org/web/20220908024430/https://datastage.stanford.edu/StanfordMoocPosts/. 

## Code

* `main.py`: Python code responsible for data pre-processing and model training. For details, please refer to Sections 3.4--3.5 of the paper.
* `process-Stanford.py`: Python code responsible for preparing the test set for processing. For details, please refer to Section 3.6 of the paper.

## Other documents and files

* `Coding-Instructions.pdf`: 
* `LICENSE`: MIT license for the Python code.
* `README.md`: This file.

# How to cite

If you use or build upon the materials, please use the BibTeX entry below to cite the original work.

```
@inproceedings{Svabensky2023towards,
    author    = {\v{S}v\'{a}bensk\'{y}, Valdemar and Baker, Ryan S. and Zambrano, Andr\'{e}s and Zou, Yishan and Slater, Stefan},
    title     = {{Towards Generalizable Detection of Urgency of Discussion Forum Posts}},
    booktitle = {Proceedings of the 16th International Conference on Educational Data Mining},
    series    = {EDM 2023},
    year      = {2023}
}
```
