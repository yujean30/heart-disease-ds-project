from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 42
SMOTE_SAMPLING_STRATEGY = 1.0
SMOTE_K_NEIGHBORS = 5

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

def create_smote(X):
    """
    Create the shared SMOTENC sampler with column index positions.
    """
    categorical_indices = [
        X.columns.get_loc(col)
        for col in CATEGORICAL_COLS
        if col in X.columns
    ]

    return SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )

if __name__ == '__main__':
    print("Shared SMOTENC configuration loaded.")