# SCEMILA - README

Welcome to the Github repository supplementing the publication "Explainable AI identifies diagnostic cells of genetic AML subtypes." (Hehr M, Sadafi A, Matek C, Lienemann P, Pohlkamp C, et al. (2023) PLOS Digital Health 2(3): e0000187. https://doi.org/10.1371/journal.pdig.0000187). 

## Table of contents
1. Description

2. Getting started

    2.1 Data

    2.2 Dependencies   
    
    2.3 Code setup
    
    2.4 Execution

    2.5 Analysis

3. Authors
4. Acknowledgements
5. License


# 1. Description
## About
This Repository contains both the machine learning algorithm and the necessary functions to analyze and plot the figures published in the paper "Explainable AI identifies diagnostic cells of genetic AML subtypes." (Hehr M, Sadafi A, Matek C, Lienemann P, Pohlkamp C, et al. (2023) PLOS Digital Health 2(3): e0000187. https://doi.org/10.1371/journal.pdig.0000187).

## Contact
For questions and issues regarding the code, feel free to contact [Matthias Hehr](https://www.linkedin.com/in/matthias-hehr/). Otherwise, please reach out to the corresponding authors.  

# 2. Getting started

## 2.1 Data
To reproduce results, download the data and unzip it. The publication of our dataset is currently in progress, the data will be available at [The Cancer Imaging Archive (TCIA):](https://www.cancerimagingarchive.net/) https://doi.org/10.7937/6ppe-4020


## 2.2 Dependencies
The pipeline and corresponding analysis requires a python environment with various packages. The [requirements file](requirements.txt) will be of help to build a functioning python environment. 

## 2.3 Code setup
Once the library is built and the dataset is downloaded, adjust the paths for the dataset and output folder in the file [run_pipeline.py](ml_pipeline/run_pipeline.py). 
Locate your dataset and create an output folder to store the results, afterwards change the lines 

```python
# 1: Setup. Source Folder is parent folder for both mll_data_master and the /data folder
TARGET_FOLDER = 'result_directory'       # results will be stored here
SOURCE_FOLDER = 'data_directory'         # path to dataset
```  
Once the algorithm has been trained, the paths have to be adjusted similarly in the [analysis notebook](analysis/analysis_notebook.ipynb).

## 2.4 Execution
To start the pipeline, navigate to the folder [ml_pipeline](ml_pipeline) and load your environment. Train the algorithm for one fold by executing:

```
python3 run_pipeline.py --result_folder=result_folder_1
```
This will create a new folder in your directory `TARGET_FOLDER` called `result_folder_1` which will contain all relevant data generated during training, validation and testing. Important: the argument `--result_folder` has to be set, otherwise the script will not run. While the algorithm is configured to run with the same parameters used in the paper, many arguments can be manipulated, altering the training process:
|Argument|Description|Possible input|Default|
|---|---|---|---|
|`--fold`|Change this parameter to rotate through different folds of cross validation.For 5-fold cross validation (default), simply launch the code five times, every time with a different value for `--fold` in range of [0,1,2,3,4]|Integer, suggested: [0,1,2,3,4]|0|
|`--lr`|Learning rate|Float|0.00005|
|`--ep`|Maximum epochs to train for, until training stops|Integer|150|
|`--es`|Early stopping. Amount of epochs to keep training, while no improvement on the validation loss is made. |Integer|20|
|`--multi_att`|Enable multiple attention values for each image (one value per image, per possible class) as suggested in our paper.|Integer (0=False, 1=True)|1|
|`--prefix`|The prefix defines the set of features that should be loaded. If an own method for feature extraction is generated and the features are saved in the dataset folder, change this value to make use of the newly generated features.|String|fnl34_|
|`--filter_diff`|Filter out patients based on the amount of myeloblasts derived from human blood smear differential count (data stored the dataset master csv file). Value represents % of cells, patients with less (<) myeloblasts are filtered out.|Integer|20|
|`--filter_mediocre_quality`|Filter out patients with borderline acceptable sample quality. This data is derived from cytologist assessment of the digitized samples. |Integer (0=False, 1=True)|0|
|`--bootstrap_idx`|For further experiments: Integer value, which is responsible for dropping out a specific patient from the dataset. Setting this to -1 deactivates the mechanism. |Integer|-1|
|`--save_model`|Deactivate model saving to save storage (e.g. if only accuracy is relevant). The model file is required to generate the occlusion maps, so deactivating will prevent generating the data used for the occlusion maps. |Integer (0=False, 1=True)|1|

## 2.5 Analysis
To analyze the data generated and take a look at various visualizations, use the [analysis notebook](analysis/analysis_notebook.ipynb) and adjust the corresponding paths as mentioned in 2.3 (Code Setup).

The notebook is designed to simplify analysis of the results generated with the pipeline, by automated plotting of most of the figures published in the paper. These figures are then exported directly into the [output folder](analysis/output).
The last sections of our notebook require large amounts of RAM (we recommend 32GB), otherwise the pythonkernel might crash. 

# 3. Authors
Major contributions were made by the following people:

Matthias Hehr<sup>1,2,3</sup>, Ario Sadafi<sup>1,2,4</sup>, Christian Matek<sup>1,2,3</sup>, Peter Lienemann<sup>1,3</sup>, Christian Pohlkamp<sup>5</sup>, Torsten Haferlach<sup>5</sup>, Karsten Spiekermann<sup>3,6,7</sup> and Carsten Marr<sup>1,2,*</sup>

<sup>1</sup>Institute of AI for Health, Helmholtz Zentrum München – German Research Center for Environmental Health, Neuherberg, Germany
<sup>2</sup>Institute of Computational Biology, Helmholtz Zentrum München – German Research Center for Environmental Health, Neuherberg, Germany
<sup>3</sup>Laboratory of Leukemia Diagnostics, Department of Medicine III, University Hospital, LMU Munich, Munich, Germany
<sup>4</sup>Computer Aided Medical Procedures, Technical University of Munich, Munich, Germany
<sup>5</sup>Munich Leukemia Laboratory, Munich, Germany
<sup>6</sup>German Cancer Consortium (DKTK), Heidelberg, Germany
<sup>7</sup>German Cancer Research Center (DKFZ), Heidelberg, Germany
<sup>*</sup>Corresponding author: carsten.marr@helmholtz-muenchen.de



# 4. Acknowledgements
M.H. was supported by a José-Carreras-DGHO-Promotionsstipendium. C.M. has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (Grant agreement No. 866411)

# 5. License
[See the license](LICENSE). If you use this code, please cite our original paper:

Hehr M, Sadafi A, Matek C, Lienemann P, Pohlkamp C, et al. (2023) Explainable AI identifies diagnostic cells of genetic AML subtypes. PLOS Digital Health 2(3): e0000187. https://doi.org/10.1371/journal.pdig.0000187
