# Machine Learning for Pulsar Glitch Prediction

This project applies modern machine learning techniques to classify **glitching vs. non-glitching pulsars** using the [ATNF Pulsar Glitch Catalogue](https://www.atnf.csiro.au/research/pulsar/psrcat/glitchTbl.html).

## Background: Pulsar Glitches

**Pulsar glitches** are sudden, rare spin-up events in neutron stars where their rotational frequency abruptly increases before gradually relaxing back. They are thought to be caused by:

- **Superfluid vortex unpinning** in the neutron star’s interior

- **Starquakes** due to stress release in the rigid crust

These events are rare — **only a small fraction of known pulsars glitch**, and even glitching pulsars may only show a handful of glitches over decades of observation.

Glitch occurrence has been linked to pulsar properties such as:

- **Age and spin period** (young, rapidly rotating pulsars are more glitch-prone)

- **Magnetic field strength**

- **Spin-down rate**

Because glitch events are scarce compared to normal pulsar behavior, the dataset is highly imbalanced, motivating the use of class-imbalance-aware ML techniques.
---

##  Project Highlights
- **Data Preprocessing**: Missing-value imputation, scaling, stratified sampling.  
- **Baseline Models**: Logistic Regression & Random Forest.  
- **Class Imbalance Techniques**:  
  - Class weighting  
  - Threshold tuning  
  - SMOTE oversampling  
  - XGBoost with `scale_pos_weight`  
- **Neural Networks**: Neural Networks (PyTorch): Implemented a fully-connected feed-forward architecture with (optional)dropout regularization and class imbalance handling using BCEWithLogitsLoss(pos_weight=...). Compared NN performance against classical ML models.
- **Evaluation Metrics**: Precision, Recall, F1-score.   

---

##  Repository Structure
- `notebooks/` :  
  - `01_EDA_and_Baseline_Models.ipynb` → Exploratory data analysis, feature preprocessing, baseline ML, imbalance handling.  
  - `02_EDA_and_Neural_Networks.ipynb` → PyTorch NN implementation with imbalance-aware loss functions.  
- `requirements.txt` : Python dependencies.  

---

## Usage
Launch Jupyter Notebook and open any notebook :
- Start with 01_EDA_and_Baseline_Models.ipynb for preprocessing and classical ML.

- Then explore 02_EDA_and_Neural_Networks.ipynb for the PyTorch implementation.

or alternatively upload these to google colab. 


## Limitations

- Predicting pulsar glitches is inherently challenging due to their rarity and the small number of observed events. Even after applying imbalance-handling techniques, model performance remains modest:

- In this project, Best performance so far: F2 ≈ 0.40, with precision ≈ 0.42 and recall ≈ 0.38 for the glitching class.

- This reflects both the scarcity of glitch events and the weak/noisy correlations with pulsar parameters.

- The current results should therefore be seen as a proof-of-concept rather than a deployable predictive tool.

# Future Work 
To improve upon these first results, promising directions include:

- More rigorous hyperparameter tuning.

- Bayesian Neural Networks: Capturing predictive uncertainty in glitch classification

- Richer datasets: Extending features with timing, multi-wavelength, or observational metadata


## Reference 

- **ATNF Pulsar Glitch Catalogue**: https://www.atnf.csiro.au/people/pulsar/psrcat/glitchTbl.html

- **psrqpy: A Python interface for querying the ATNF pulsar catalogue**  
  - Pitkin, M. (2018). *Journal of Open Source Software*, 3(22), 538.  
  - DOI: [10.21105/joss.00538](https://doi.org/10.21105/joss.00538)  