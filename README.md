# SDSS DR14 classification of stars galaxies and quasars
Classification of stars, galaxies and quasars from the Sloan Digital Sky Survey (SDSS) data release 14 (DR14)

## Project Overview
The overall aim for this project is to use both a classical, non-neural network approach and a neural network (NN) approach to classify stars, galaxies and quasars from SDSS DR14 and to then compare the two approaches.
We will be employing the use of machine learning (ML) techniques for this problem, as astronomical object classification is a difficult task within astronomy. The use of traditional ML and NN approaches can offer a quick and effective solution to these classification problems.

The project is divided into 3 components:
- **Q1: Traditional ML techniques**
- **Q2: Neural Network approach and comparison with traditional ML**
- **Q3: How different factors affect the performance of a NN**

**Q1**: Here we look at a traditional ML technique, explore how well it performs at this classification task and how we could potentially make it even more accurate. For this traditional approach we will investigate decision trees.
**Q2**: Then we move on to looking at a NN approach. We'll make use of the PyTorch library to build our NN, evaluate how it performs and then compare it to our decision tree.
**Q3**: Finally we'll look at how different factors can impact how our NN peforms, for example how does the amount of training data available to our model affect it's performance?

## Objectives
The main objectives for this project are:
- **Comparison of techniques**: How well does each approach perform? do traditional ML techniques still have their place in classification problems?
- **Neural Network performance**: Explore how parameters can affect the performance of a NN
- **Research**: Adress a chosen question on how NNs perform: How do choices about data such as i) amount of training data and ii) balance of
classes in a classification problem affect the performance of a neural network?

## The dataset
Within this project, we will be exploring the fourteenth data release from SDSS. This contains 10,000 records of astronomical objects, each obejct being assigned a class of `STAR`, `GALAXY` or `QSO` (quasar). DR14 contains obsvervations from the July 2016 observing campagin. The dataset contains 17 feature columns which describe things such as the targeted object's id, right ascension and declination (ra and dec), magnitude through the telescope's different filters, redshift etc. In particular, the magnitudes are a key feature for classifcation.  

The dataset can be accessed from [here](https://live-sdss4org-dr14.pantheonsite.io/).  

The dataset consists of the following columns:

| **Column**      | **Description**                                      | 
|------------------|------------------------------------------------------|
| `objid`         | Object identifier                                   |
| `ra`            | Right ascension                                     |
| `dec`           | Declination                                         |
| `u`             | Magnitues in ultraviolet                          |
| `g`             | Magnitude in the green light wavelength range                          |
| `r`             | Magnitude in the red light wavelength range                           |
| `i`             | Magnitude in the near infrared                          |
| `z`             | Magnitude in the infrared                          |
| `run`           | Run number of the observation                      |
| `rerun`         | Rerun number for calibration                       |
| `camcol`        | Camera column number                               |
| `field`         | Field number in the run                            |
| `specobjid`     | Spectroscopic object identifier                    |
| `class`         | Classification of the object   |
| `redshift`      | Redshift of the object                             |
| `plate`         | Spectroscopic plate number                         |
| `mjd`           | Modified Julian Date of the observation            |
| `fiberid`       | Fiber ID for the spectroscopic observation         |

The first few rows of the data:

| **objid**      | **ra**        | **dec**       | **u**     | **g**     | **r**     | **i**     | **z**     | **run** | **rerun** | **camcol** | **field** | **specobjid** | **class** | **redshift** | **plate** | **mjd**  | **fiberid** |
|----------------|---------------|---------------|-----------|-----------|-----------|-----------|-----------|---------|-----------|------------|-----------|---------------|-----------|--------------|-----------|----------|------------|
| 1.24E+18      | 183.5313257   | 0.08969303    | 19.47406  | 17.0424   | 15.94699  | 15.50342  | 15.22531  | 752     | 301       | 4          | 267       | 3.72E+18      | STAR      | -8.96E-06    | 3306      | 54922    | 491        |
| 1.24E+18      | 183.5983705   | 0.135285032   | 18.6628   | 17.21449  | 16.67637  | 16.48922  | 16.3915   | 752     | 301       | 4          | 267       | 3.64E+17      | STAR      | -5.49E-05    | 323       | 51615    | 541        |
| 1.24E+18      | 183.6802074   | 0.126185092   | 19.38298  | 18.19169  | 17.47428  | 17.08732  | 16.80125  | 752     | 301       | 4          | 268       | 3.23E+17      | GALAXY    | 0.1231112    | 287       | 52023    | 513        |
| 1.24E+18      | 183.8705294   | 0.049910685   | 17.76536  | 16.60272  | 16.16116  | 15.98233  | 15.90438  | 752     | 301       | 4          | 269       | 3.72E+18      | STAR      | -0.000110616 | 3306      | 54922    | 510        |
| 1.24E+18      | 183.8832883   | 0.102556752   | 17.55025  | 16.26342  | 16.43869  | 16.55492  | 16.61326  | 752     | 301       | 4          | 269       | 3.72E+18      | STAR      | 0.000590357  | 3306      | 54922    | 512        |

This dataset was chosen because it is a strong candidate for showing the strenghts and weaknesses of both traditional and NN approaches, as well as showing how different features from the dataset can impact the performance of both models.

## Motivations for each approach
### Decision tree
With all of the advanced ML techniques that we have today, then what is the point in using a traditional ML technique? To put it simply classical ML techniques still have their place within classification problems, especially if the dataset isn't too large or complex, and can be less computationally expensive. Furthermore, they are relatively easy for beginners to learn and imploment, which is why in Q1 we will be starting with a simple decision tree to tackle a classification problem. The aim is to guide beginners through the structure of a decision tree and how it interprets data, and then to show them how it can be improved.

### Neural networks
Neural networks are incredibly powerful tools in today's AI and ML techniques. This is because of they way that they mimic the way that the human brain works, allowing them to perform complex coputations and recognise patterns in data. We choose a NN approach for Q2 as this will allow us to compare how they perform compared to the traditional approaches, but also to understand the architecture behind them, hence why we use the PyTorch library rather than one like Tensorflow. We also explore what can affect the performance of a NN, and how we can overcome these challenges in Q3.

## Dependencies

- PyTorch version: 2.5.1+cu121
- Scikit-learn version: 1.6.0
- Seaborn version: 0.13.2
- Pandas version: 2.2.2
- Matplotlib version: 3.10.0
- NumPy version: 1.26.4

## Usage
To clone the repository use the following command:
`git clone https://github.com/DanTass02/SDSS-classification.git`  

Then make sure to pip install all of the libraries listed in `dependencies.txt`, using the command `!pip install (required library here)`

## Licence
This project is under the GNU General Public License v3.0, see `LICENCE` for more info

## Acknowledgements
I would like to thank the following for their assistance throughout the project:  
ChatGPT AI
- Provider: OpenAI
- Useage: ChatGPT was used to develop ideas and to debug code

Gemini AI
- Provider: Google
- Usage: Debugging code and for fetching versions of python libraries

## Contact

Dan Tassie  
dantass002@gmail.com  
MPhys Physics, Astrophysics & Cosmology  
University of Portsmouth
