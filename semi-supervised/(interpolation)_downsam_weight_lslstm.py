# -*- coding: utf-8 -*-


import pandas as pd
'''
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')
'''

data21 = pd.read_csv('simuuser.csv')

sams_df = pd.read_csv('samHEuser.csv')

# Load the CSV file into a DataFrame
data = pd.read_csv('mazeHEuser.csv')

data21

#data21 = pd.read_csv('training_eye_data.csv')

#data21.describe()

print(data.columns)

selected_columns=['fms','Left_Openness','Right_Openness', 'Left_Diameter',
       'Right_Diameter', 'Left_PupilSensor_X',
       'Left_PupilSensor_Y', 'Right_PupilSensor_X',
       'Right_PupilSensor_Y', 'Left_GazeDir_X', 'Left_GazeDir_Y',
       'Left_GazeDir_Z', 'Right_GazeDir_X', 'Right_GazeDir_Y',
       'Right_GazeDir_Z','HR','EDA','fms_3class','Participant']
data = data[selected_columns]

data
print('maze data')

data

print(data21.columns)

data21.describe()

rename_map = {
    'fms': 'fms',  # Matches available column
    'left_eye_openness': 'Left_Openness',
    'right_eye_openness': 'Right_Openness',
    'leftpupildiameter': 'Left_Diameter',
    'rightpupildiameter': 'Right_Diameter',
    'leftpupilposinsensorx': 'Left_PupilSensor_X',
    'leftpupilposinsensory': 'Left_PupilSensor_Y',  # Matches available column
    'rightpupilposinsensorx': 'Right_PupilSensor_X',
    'rightpupilposinsensory': 'Right_PupilSensor_Y',  # Matches available column
    'nrmsrlefteyegazedirx': 'Left_GazeDir_X',
    'nrmsrlefteyegazediry': 'Left_GazeDir_Y',
    'nrmsrlefteyegazedirz': 'Left_GazeDir_Z',
    'nrmsrrighteyegazedirx': 'Right_GazeDir_X',
    'nrmsrrighteyegazediry': 'Right_GazeDir_Y',
    'nrmsrrighteyegazedirz': 'Right_GazeDir_Z',
    'hr': 'HR',
    'eda': 'EDA',
    'participant':'Participant'
}

# Rename the columns in your DataFrame
data21 = data21.rename(columns=rename_map)

# Z-Score method for multiple columns
def remove_outliers_zscore(df, columns, threshold=3):
    for col in columns:
        z_scores = (df[col] - df[col].mean()) / df[col].std()
        df = df[abs(z_scores) < threshold]
    return df

# Apply to multiple columns
columns_to_check = ['Left_Openness','Right_Openness', 'Left_Diameter',
       'Right_Diameter', 'Left_PupilSensor_X',
       'Left_PupilSensor_Y', 'Right_PupilSensor_X',
       'Right_PupilSensor_Y', 'Left_GazeDir_X', 'Left_GazeDir_Y',
       'Left_GazeDir_Z', 'Right_GazeDir_X', 'Right_GazeDir_Y',
       'Right_GazeDir_Z']  # Specify the columns to check for outliers
data21 = remove_outliers_zscore(data21, columns_to_check, threshold=3)

print("DataFrame after removing outliers:")
print(data21)

quantiles = data21['fms'].quantile([0.25, 0.75])
print('Quantiles for fms:', quantiles)

# Classify the 'fms' values based on quantiles
def quantile_fms(value, quantiles):
    if value <= quantiles[0.25]:
        return 0  # Low
    elif value <= quantiles[0.75]:
        return 1  # Medium
    else:
        return 2  # High

data21['fms_3class'] = data21['fms'].apply(lambda x: quantile_fms(x, quantiles))

print("\nData with 'fms_3class' classification:")

data21

print("simulation21")
counts = data21['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

selected_columns=['fms','Left_Openness','Right_Openness', 'Left_Diameter',
       'Right_Diameter', 'Left_PupilSensor_X',
       'Left_PupilSensor_Y', 'Right_PupilSensor_X',
       'Right_PupilSensor_Y', 'Left_GazeDir_X', 'Left_GazeDir_Y',
       'Left_GazeDir_Z', 'Right_GazeDir_X', 'Right_GazeDir_Y',
       'Right_GazeDir_Z','HR','EDA','fms_3class','Participant']
data21 = data21[selected_columns]


print('simulation data')

data21

print(sams_df)

sams_df['Participant'] = sams_df['folder'].str.extract(r'Eye head and other features\\(\d+)')

sams_df.select_dtypes(exclude=['int']).columns

sams_df=sams_df.drop(columns=['folder'])

#from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df_no_outliers=data

print(df_no_outliers.columns)

print("Maze data:")
df_no_outliers=data
counts = df_no_outliers['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

print("simulation data:")
d2_no_outliers=data21
counts = d2_no_outliers['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

"""2 dataset df_scaled, d2_scaled"""

#sams_df = pd.read_csv('sams.csv')

'''
X3 = sams_df.drop(columns=['fms','fms_3class'])  # Features
y3 = sams_df['fms_3class']

# Apply MinMaxScaler to the features
scaler = MinMaxScaler()
X_scaled3 = scaler.fit_transform(X3)

# Convert back to a DataFrame (optional)
X_scaled_sams = pd.DataFrame(X_scaled3, columns=X3.columns)

# Combine the scaled features with the target column
sams_scaled = pd.concat([X_scaled_sams, y3.reset_index(drop=True)], axis=1)
'''

sams_df

print("sams_df")

counts = sams_df['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Combine all datasets for fitting the scaler (excluding "fms", "fms_3class", and "Participant")
combined_data = pd.concat([
    df_no_outliers.drop(columns=['fms', 'fms_3class', 'Participant']),
    d2_no_outliers.drop(columns=['fms', 'fms_3class', 'Participant']),
    sams_df.drop(columns=['fms', 'fms_3class', 'Participant'])
])

# Initialize and fit MinMaxScaler on combined data
scaler = MinMaxScaler()
scaler.fit(combined_data)

# Scale df_no_outliers (excluding "Participant" from scaling)
X = df_no_outliers.drop(columns=['fms', 'fms_3class', 'Participant'])  # Exclude "Participant" from scaling
y = df_no_outliers['fms_3class']
participant = df_no_outliers['Participant']  # Keep "Participant" column
X_scaled = scaler.transform(X)  # Transform using pre-fitted scaler
df_scaled = pd.concat([participant.reset_index(drop=True),  # Keep "Participant"
                       pd.DataFrame(X_scaled, columns=X.columns),
                       y.reset_index(drop=True)], axis=1)

# Scale d2_no_outliers (excluding "Participant" from scaling)
X2 = d2_no_outliers.drop(columns=['fms', 'fms_3class', 'Participant'])
y2 = d2_no_outliers['fms_3class']
participant2 = d2_no_outliers['Participant']
X_scaled2 = scaler.transform(X2)  # Transform using the same scaler
d2_scaled = pd.concat([participant2.reset_index(drop=True),
                       pd.DataFrame(X_scaled2, columns=X2.columns),
                       y2.reset_index(drop=True)], axis=1)

# Scale sams_df (excluding "Participant" from scaling)
X3 = sams_df.drop(columns=['fms', 'fms_3class', 'Participant'])
y3 = sams_df['fms_3class']
participant3 = sams_df['Participant']
X_scaled3 = scaler.transform(X3)  # Transform using the same scaler
sams_scaled = pd.concat([participant3.reset_index(drop=True),
                         pd.DataFrame(X_scaled3, columns=X3.columns),
                         y3.reset_index(drop=True)], axis=1)

print(df_scaled.isnull().sum())

print(d2_scaled.isnull().sum())

sams_scaled = sams_scaled.dropna()

print(sams_scaled.isnull().sum())

print(sams_df.isnull().sum())

sams_scaled

sams_scaled = sams_scaled.iloc[::35]

sams_scaled

import tensorflow as tf
from numpy import concatenate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN, tcn_full_summary
from scipy.sparse import csr_matrix
import numpy as np

# Check for GPU availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs available: {len(gpus)}")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU found. Using CPU.")

def split_dataset_by_participants(df, label_column='fms_3class', test_size=0.2, random_state=42):
        # Get unique participants
        participants = df['Participant'].unique()

        # Split participants into train (80%) and test (20%)
        train_participants, test_participants = train_test_split(
            participants, test_size=test_size, random_state=random_state
        )

        # Create train and test datasets based on participant groups
        df_train = df[df['Participant'].isin(train_participants)]
        df_test = df[df['Participant'].isin(test_participants)]
        print(df_train.count(),df_train['Participant'].unique(),df_train['fms_3class'].value_counts())
        print(df_test.count(),df_test['Participant'].unique(),df_test['fms_3class'].value_counts())
        # Separate features (X) and labels (y)
        X_train, y_train = df_train.drop(columns=['Participant', label_column]), df_train[label_column]
        X_test, y_test = df_test.drop(columns=['Participant', label_column]), df_test[label_column]

        return X_train, X_test, y_train, y_test

sams_scaled['Participant'] = sams_scaled['Participant'].astype(int)

import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
df=df_scaled.copy()
# Get unique participants
from sklearn.decomposition import PCA
import seaborn as sns

pca = PCA(n_components=2)
#df_pca = pca.fit_transform(df.drop(columns=['Participant', 'fms_3class']))
df_pca = pca.fit_transform(df.drop(columns=['fms_3class']))
df['Participant'] = df_pca[:, 0]
df['Left_Diameter'] = df_pca[:, 1]

sns.scatterplot(x=df['Participant'], y=df['EDA'], hue=df['fms_3class'])
plt.title("Participant Clustering Before and After Augmentation")
plt.show()

df=sams_scaled.copy()
sns.scatterplot(x=df.index, y=df['EDA'])
plt.xlabel("Index")
plt.ylabel("Feature Value")
plt.title("Scatter Plot of Feature Over Index")
plt.show()

sns.scatterplot(x=df.index, y=df['EDA'], hue=df['Participant'], palette='viridis')

sns.scatterplot(x=df.index, y=df['EDA'], hue=df['fms_3class'], palette='viridis')

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
sns.countplot(data=sams_scaled, x='Participant', hue='fms_3class')
plt.xticks(rotation=90)
plt.title("Class Distribution per Participant")
plt.show()

print("Maze data:")
counts = df_scaled['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

print("simulation data:")

counts = d2_scaled['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

print("sams_df")

counts = sams_scaled['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

def balance_minority_classes(df, label_column='fms_3class', samples_per_class={}):
    """
    Generates and inserts synthetic samples for multiple minority classes using interpolation.

    Args:
        df (pd.DataFrame): The input dataset.
        label_column (str): The target class column.
        samples_per_class (dict): Dictionary specifying how many synthetic samples to generate per class.
                                  Example: {1: 3, 2: 2} → Generate 3 samples for class 1, 2 samples for class 2.

    Returns:
        pd.DataFrame: New dataset with synthetic samples inserted in between original rows.
    """
    df
    # Ensure we only modify minority classes that exist in the dataset
    available_classes = df[label_column].unique()
    samples_per_class = {cls: count for cls, count in samples_per_class.items() if cls in available_classes}

    # Convert DataFrame to a list of rows for modification
    df_list = df.to_dict('records')  # Convert DataFrame to list of dictionaries (to allow inserting rows)

    for minority_class, n_samples_needed in samples_per_class.items():
        minority_df = df[df[label_column] == minority_class]  # Filter only minority class rows

        if len(minority_df) < 2:
            print(f"Not enough data in class {minority_class} to interpolate.")
            continue  # Skip classes that have too few instances

        for _ in range(n_samples_needed):
            # Randomly pick an index within the minority class (excluding first/last row)
            idx = np.random.randint(1, len(minority_df) - 1)
            row_before = minority_df.iloc[idx - 1]
            row_after = minority_df.iloc[idx + 1]

            # Compute interpolated row (mean of the two rows)
            new_sample = (row_before[:-1] + row_after[:-1]) / 2
            new_sample[label_column] = minority_class  # Assign class label

            # Convert to dictionary format for insertion
            new_sample_dict = new_sample.to_dict()

            # Find index of row_after in the main DataFrame list
            insert_idx = next(i for i, row in enumerate(df_list) if row[label_column] == row_after[label_column])

            # Insert new row in between
            df_list.insert(insert_idx, new_sample_dict)

    # Convert back to DataFrame
    df_balanced = pd.DataFrame(df_list)

    return df_balanced

# User-defined samples to generate for each class
samples_to_add1 = {1: 1500, 2: 2000}  # Add 2 samples for class 1, 3 for class 2
samples_to_add2 = {1: 2, 2: 2000}
samples_to_add3 = {1: 1500, 2: 1500}
# Balance the dataset
df_scaled= balance_minority_classes(df_scaled, label_column='fms_3class', samples_per_class=samples_to_add1)
d2_scaled= balance_minority_classes(d2_scaled, label_column='fms_3class', samples_per_class=samples_to_add2)
sams_scaled= balance_minority_classes(sams_scaled, label_column='fms_3class', samples_per_class=samples_to_add3)

print("Maze data:")
counts = df_scaled['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

print("simulation data:")

counts = d2_scaled['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

print("sams_df")

counts = sams_scaled['fms_3class'].value_counts()
for f, count in counts.items():
    print(f" {f} has datapoints: {count}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, InputLayer, Bidirectional
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tcn import TCN
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_tcn_with_confidence(df1, df2, df3, true_labels_df3, timesteps=16, confidence_threshold=0.80):
    """
    Train a TCN model using labeled data and label propagation with class priors.
    """

    #  **Function to Split Dataset by Participants**
    '''
    def split_dataset_by_participants(df, label_column='fms_3class', test_size=0.15, random_state=42):
        participants = df['Participant'].unique()
        train_participants, test_participants = train_test_split(
            participants, test_size=test_size, random_state=random_state
        )
        df_train = df[df['Participant'].isin(train_participants)]
        df_test = df[df['Participant'].isin(test_participants)]
        return df_train, df_test

    #  **Split Data into Labeled, Unlabeled, and Test Sets**
    df1, testdf1 = split_dataset_by_participants(df1)
    df2, testdf2 = split_dataset_by_participants(df2)
    df3, testdf3 = split_dataset_by_participants(df3)
    '''
    #split random insted of participants
    df1, testdf1 = train_test_split(df1, test_size=0.20, random_state=42, stratify=df1['fms_3class'])
    df2, testdf2 = train_test_split(df2, test_size=0.20, random_state=42, stratify=df2['fms_3class'])
    df3, testdf3 = train_test_split(df3, test_size=0.20, random_state=42, stratify=df3['fms_3class'])




    df1_labeled, df1_unlabeled = train_test_split(df1, test_size=0.95, random_state=42, stratify=df1['fms_3class'])
    df2_labeled, df2_unlabeled = train_test_split(df2, test_size=0.95, random_state=42, stratify=df2['fms_3class'])
    df3_labeled, df3_unlabeled = train_test_split(df3, test_size=0.95, random_state=42, stratify=df3['fms_3class'])

    dfs = [df1_labeled, df1_unlabeled, df2_labeled, df2_unlabeled, df3_labeled, df3_unlabeled, testdf1, testdf2, testdf3]
    for df in dfs:
        df.drop(columns=['Participant'], inplace=True)

    #  **Prepare Test Data**
    combined_df = pd.concat([testdf1, testdf2, testdf3], axis=0).reset_index(drop=True)
    target_column = "fms_3class"
    X_test = combined_df.drop(columns=[target_column]).to_numpy()
    y_test = combined_df[target_column].to_numpy()

    #  **Prepare Labeled and Unlabeled Data**
    def extract_features_labels(df_labeled, df_unlabeled):
        X_labeled = df_labeled.drop(columns=['fms_3class']).values
        y_labeled = df_labeled['fms_3class'].values
        X_unlabeled = df_unlabeled.drop(columns=['fms_3class']).values
        y_unlabeled = df_unlabeled['fms_3class'].values  # True labels (for evaluation)
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled

    X_labeled_df1, y_labeled_df1, X_unlabeled_df1, y_unlabeled_df1 = extract_features_labels(df1_labeled, df1_unlabeled)
    X_labeled_df2, y_labeled_df2, X_unlabeled_df2, y_unlabeled_df2 = extract_features_labels(df2_labeled, df2_unlabeled)
    X_labeled_df3, y_labeled_df3, X_unlabeled_df3, y_unlabeled_df3 = extract_features_labels(df3_labeled, df3_unlabeled)

    #  **Combine Data**
    X_labeled = np.concatenate((X_labeled_df1, X_labeled_df2, X_labeled_df3), axis=0)
    y_labeled = np.concatenate((y_labeled_df1, y_labeled_df2, y_labeled_df3), axis=0)

    X_unlabeled = np.concatenate((X_unlabeled_df1, X_unlabeled_df2, X_unlabeled_df3), axis=0)
    y_unlabeled = np.concatenate((y_unlabeled_df1, y_unlabeled_df2, y_unlabeled_df3), axis=0)

    y_total = np.concatenate((y_labeled, y_unlabeled), axis=0)

    #  **Combine Labeled and Unlabeled Data for Training**
    nolabel = [-1 for _ in range(len(X_unlabeled))]
    X_train_mixed = np.concatenate((X_labeled, X_unlabeled), axis=0)
    y_train_mixed = np.concatenate((y_labeled, nolabel), axis=0)

    #  **Compute Class Priors**
    #unique_classes, class_counts = np.unique(y_labeled, return_counts=True)
    #class_prior = class_counts / np.sum(class_counts)  # Normalize to sum = 1
    class_weightslp = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labeled),
    y=y_labeled
    )

    print(class_weightslp)

    #  **Apply Label Spreading with Class Priors**
    #label_propagation = LabelPropagation(kernel='knn', n_neighbors=15, max_iter=2000)
    label_propagation = LabelSpreading(kernel='knn', n_neighbors=15, max_iter=1800)
    #label_propagation.class_prior = class_prior
    label_propagation.class_weight = dict(zip(np.unique(y_labeled), class_weightslp))
    label_propagation.fit(X_train_mixed, y_train_mixed)
    '''
    from sklearn.metrics.pairwise import rbf_kernel
    from scipy.sparse import csr_matrix


    def sparse_rbf_kernel(X, Y=None, gamma=0.01, sparsity_threshold=1e-3):
        """
        Custom Sparse RBF Kernel function for LabelSpreading.

        Parameters:
        - X: Input feature matrix.
        - Y: Optional second feature matrix (for pairwise computation).
        - gamma: RBF kernel hyperparameter.
        - sparsity_threshold: Minimum similarity value to retain in sparse representation.

        Returns:
        - Sparse RBF kernel matrix.
        """
        # Compute RBF similarity
        similarity_matrix = rbf_kernel(X, Y, gamma=gamma)

        # Apply sparsity threshold (set low values to 0)
        similarity_matrix[similarity_matrix < sparsity_threshold] = 0

        # Convert to sparse format to save memory
        return csr_matrix(similarity_matrix)

    from sklearn.semi_supervised import LabelSpreading

    #  Apply Label Spreading with Custom Sparse RBF Kernel
    label_propagation = LabelSpreading(kernel=sparse_rbf_kernel, max_iter=2000)
    label_propagation.fit(X_train_mixed, y_train_mixed)


    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.metrics.pairwise import polynomial_kernel

    # Define a function that computes the polynomial kernel
    def custom_polynomial_kernel(X, Y=None):
        return polynomial_kernel(X, Y, degree=3, coef0=1)


    #  Compute Cosine Similarity Graph
    #  Define a custom cosine similarity kernel function
    def cosine_kernel(X, Y=None):
        return cosine_similarity(X, Y)

    #  Apply Label Spreading with Custom Kernel
    label_propagation = LabelPropagation(kernel=cosine_kernel, max_iter=2000)
    label_propagation.fit(X_train_mixed, y_train_mixed)
    '''
    #  **Get Pseudo-Labels & Confidence Scores**
    transduced_labels = label_propagation.transduction_
    confidence_scores = np.max(label_propagation.label_distributions_, axis=1)

    y_pred_unlabeled = label_propagation.predict(X_unlabeled)
    print("\nLabel Spreading Classification Report (Unlabeled Data):")
    print(classification_report(y_unlabeled, y_pred_unlabeled))
    accuracy = accuracy_score(y_unlabeled, y_pred_unlabeled)
    print(f"Label Spreading Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_unlabeled, y_pred_unlabeled)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_unlabeled), yticklabels=np.unique(y_unlabeled))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix of Label Spreading (Unlabeled Data)')
    plt.show()




    #  **Filter High-Confidence Pseudo-Labels**
    high_confidence_indices = np.where(confidence_scores > confidence_threshold)[0]
    X_train_high_conf = X_train_mixed[high_confidence_indices]
    y_train_high_conf = transduced_labels[high_confidence_indices]



    #  **Compute Class Weights**
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_high_conf),
        y=y_train_high_conf
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(class_weights)
    #  **Prepare Data for TCN**


    #  **Modify Data Shapes for LSTM**
    num_features = X_labeled.shape[1]  # Total number of features

    X_train_high_conf = X_train_high_conf.reshape(-1, timesteps, num_features // timesteps)  # Reshape for LSTM
    X_test = X_test.reshape(-1, timesteps, num_features // timesteps)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_high_conf, y_train_high_conf, test_size=0.2, random_state=42, stratify=y_train_high_conf
    )
    input_shape = (timesteps, num_features // timesteps)
    #  **Define LSTM Model**
    model = Sequential([
        Bidirectional(LSTM(128, activation='tanh', input_shape=input_shape, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64, activation='tanh', return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32, activation='tanh', return_sequences=False),
        Dropout(0.3),

        Dense(64, activation='relu', kernel_regularizer='l2'),
        Dense(3, activation='softmax')  # 🔹 3-class classification
    ])

    #  **Compile LSTM Model**
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    #  **Train Model**
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    #  **Evaluate on Test Set**
    yhat_probs = model.predict(X_test)
    yhat_classes = yhat_probs.argmax(axis=1)


    return model, history, y_test, yhat_classes, yhat_probs, testdf1, testdf2, testdf3, label_propagation
true_labels_df3 = sams_scaled['fms_3class'].values
model, history, y_test_labeled, yhat_classes, yhat_probs,testdf1,testdf2,testdf3,label_propagation = train_tcn_with_confidence(df_scaled, d2_scaled, sams_scaled, true_labels_df3)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve

from sklearn.metrics import accuracy_score, classification_report

# Compute Test Accuracy
test_accuracy = accuracy_score(y_test_labeled, yhat_classes)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate Classification Report
class_report = classification_report(y_test_labeled, yhat_classes)
print("\nClassification Report:")
print(class_report)

X_test2 = testdf1.drop(columns=['fms_3class']).to_numpy()
y_test2 = testdf1['fms_3class'].to_numpy()

X_test2 = X_test2.reshape(-1, 16, X_test2.shape[1] // 16)
yhat_probs1 = model.predict(X_test2)
yhat_classes1 = yhat_probs1.argmax(axis=1)
# Compute Test Accuracy
test_accuracy = accuracy_score(y_test2, yhat_classes1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate Classification Report
class_report = classification_report(y_test2, yhat_classes1)
print("\nClassification Report:")
print(class_report)
# Confusion Matrix
cm = confusion_matrix(y_test2, yhat_classes1)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()



X_test2 = testdf2.drop(columns=['fms_3class']).to_numpy()
y_test2 = testdf2['fms_3class'].to_numpy()

X_test2 = X_test2.reshape(-1, 16, X_test2.shape[1] // 16)
yhat_probs1 = model.predict(X_test2)
yhat_classes1 = yhat_probs1.argmax(axis=1)
# Compute Test Accuracy
test_accuracy = accuracy_score(y_test2, yhat_classes1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate Classification Report
class_report = classification_report(y_test2, yhat_classes1)
print("\nClassification Report:")
print(class_report)

# Confusion Matrix
cm = confusion_matrix(y_test2, yhat_classes1)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

X_test2 = testdf3.drop(columns=['fms_3class']).to_numpy()
y_test2 = testdf3['fms_3class'].to_numpy()

X_test2 = X_test2.reshape(-1, 16, X_test2.shape[1] // 16)
yhat_probs1 = model.predict(X_test2)
yhat_classes1 = yhat_probs1.argmax(axis=1)
# Compute Test Accuracy
test_accuracy = accuracy_score(y_test2, yhat_classes1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Generate Classification Report
class_report = classification_report(y_test2, yhat_classes1)
print("\nClassification Report:")
print(class_report)

# Confusion Matrix
cm = confusion_matrix(y_test2, yhat_classes1)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2', 'Class 3'], yticklabels=['Class 1', 'Class 2', 'Class 3'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()



