# Overview

This repository contains supplementary materials for the following conference paper:

Valdemar Švábenský, Ryan S. Baker, Andrés Zambrano, Yishan Zou, and Stefan Slater.\
**Towards Generalizable Detection of Urgency of Discussion Forum Posts.**\
In Proceedings of the 16th International Conference on Educational Data Mining (EDM 2023).\
https://educationaldatamining.org/EDM2023/proceedings/2023.EDM-short-papers.29/2023.EDM-short-papers.29.pdf

# Contents of the repository

## Data

* `data-per-course`: A folder with the data set divided per the 9 courses and split into two parts for the two coders.
* `All_Courses_REDACTED_CODED.csv`: The final training and cross-validation data set used for this work, resulting from merging all files from `data-per-course` into a single file. For details, please refer to Sections 3.1–3.3 of the paper.
* `Stanford.csv`: The test data set used for this work. For details, please refer to Section 3.6 of the paper. Citation of this data set:

Agrawal, A. and Paepcke, A., 2014. The Stanford MOOC Posts Dataset.\
URL: http://infolab.stanford.edu/~paepcke/stanfordMOOCForumPostsSet.tar.gz. \
Original page available on URL: https://web.archive.org/web/20220908024430/https://datastage.stanford.edu/StanfordMoocPosts/. 

This version of the data is slightly different than what was used in the paper due to fixing a minor redaction error. Therefore, the results of running the analysis on this data will differ from the results reported in the paper.

## Code

* `main.py`: Python code responsible for data pre-processing and model training. For details, please refer to Sections 3.4–3.5 of the paper.
* `process-Stanford.py`: Python code responsible for computing the numbers in Table 1 of the paper for the test set.

## Other documents and files

* `Coding-Instructions.pdf`: Protocol used for training the coders of the data set.
* `LICENSE`: MIT license for the Python code.
* `README.md`: This file.

# How to cite

If you use or build upon the materials, please use the BibTeX entry below to cite the original paper (not only this web link).

```
@inproceedings{Svabensky2023towards,
    author    = {\v{S}v\'{a}bensk\'{y}, Valdemar and Baker, Ryan S. and Zambrano, Andr\'{e}s and Zou, Yishan and Slater, Stefan},
    title     = {{Towards Generalizable Detection of Urgency of Discussion Forum Posts}},
    booktitle = {Proceedings of the 16th International Conference on Educational Data Mining},
    series    = {EDM '23},
    editor    = {Mingyu Feng and Tanja Käser and Partha Talukdar},
    location  = {Bengaluru, India},
    publisher = {International Educational Data Mining Society},
    month     = {07},
    year      = {2023},
    pages     = {302--309},
    numpages  = {8},
    isbn      = {978-1-7336736-4-8},
    url       = {https://educationaldatamining.org/EDM2023/proceedings/2023.EDM-short-papers.29/2023.EDM-short-papers.29.pdf},
    doi       = {10.5281/zenodo.8115790},
}
```
