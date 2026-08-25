from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 1.0
SMOTE_K_NEIGHBORS = 5

# Every categorical column produced by preprocess.py's OrdinalEncoder /
# LabelEncoder steps -- these must NOT be numerically interpolated by
# SMOTE, which is why we use SMOTENC (majority-vote on these columns)
CATEGORICAL_COLS = [
    'Exercise Habits',
    'Alcohol Consumption',
    'Stress Level',
    'Sugar Consumption',
    'Gender',
    'Smoking',
    'Family Heart Disease',
    'Diabetes',
    'High Blood Pressure',
    'Low HDL Cholesterol',
    'High LDL Cholesterol',
]


def create_smote():
    """
    Return a shared SMOTENC sampler configured with categorical column names.
    The actual column positions will be resolved automatically by SMOTENC
    when fit_resample() is called inside the pipeline.
    """
    return SMOTENC(
        categorical_features=CATEGORICAL_COLS,   # pass names instead of indices
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )


if __name__ == '__main__':
    print("Shared SMOTENC configuration loaded.")
    print(f"  sampling_strategy = {SMOTE_SAMPLING_STRATEGY}")
    print(f"  k_neighbors       = {SMOTE_K_NEIGHBORS}")
    print(f"  random_state      = {RANDOM_STATE}")
    print(f"  categorical_cols  = {CATEGORICAL_COLS}")
