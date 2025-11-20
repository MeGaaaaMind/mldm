# Wildfire detection and severity prediction

## Overview
- Tabular modeling notebook predicts wildfire size (in hectares) from environmental attributes.
- Image modeling notebook classifies aerial/ground images into `fire`, `smoke`, or `non fire`.
- `data` folder hosts the tabular dataset and description of each attributes.

## Repository Layout
- `code/Tabular_code.ipynb` — end-to-end regression workflow (EDA, cleaning, feature engineering, multiple models, evaluation plots).
- `code/image_code.ipynb` — TensorFlow CNN pipeline for 3-class image classification with augmentation, checkpoints, and evaluation utilities.
- `code/model file link` — Google Drive URL for the saved CNN weights (`my_forest_fire_model3.keras`).
- `data/FW_Veg_Rem_Combined.csv` — ~55k labeled wildfire incidents used by the tabular notebook.
- `data/Wildfire_att_description.txt` — Description for every column in the CSV.


## Environment & Dependencies
- Python 3.8+ with Jupyter / VS Code / Colab.
- Core libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow>=2.12`, `Pillow`.

## Running the Notebooks
### Tabular Regression (`code/Tabular_code.ipynb`)
1. Run in Google Colab and upload the notebook file.
2. Run cells sequentially:
   - Exploratory analysis and visualizations.
   - Data cleaning (median imputation for weather gaps, dropping unused columns).
   - Train/test split with log-transformed target (`fire_size`).
   - Model training (Linear Regression, Decision Tree with `GridSearchCV`, Random Forest with `RandomizedSearchCV`, SVR) and comparison plots.
3. Inspect outputs; best R² ≈ 0.72 with Random Forest using top features such as `remoteness`, `longitude`, and lightning-caused fires.

### Image Classification (`code/image_code.ipynb`)
1. Run in Google Colab (GPU recommended) and mount Drive.
2. Verify the dataset paths and optional checkpoint folder under `MyDrive`.
3. Execute preprocessing cells to validate images and create augmented `ImageDataGenerator` loaders.
4. Skip the CNN training module and proceed directly to the module code 
5. Load the correct pre-trained model and evaluate metrics; accuracy ≈ 95–96%, confusion matrix, classification report, and qualitative prediction grids.


## Results 
- **Tabular Random Forest:** RMSE ≈ 1.38 (log scale), R² ≈ 0.72 on test data; remoteness and location dominate importance.
- **CNN Classifier:** Test accuracy ≈ 95.7%, balanced precision/recall for all three classes.

