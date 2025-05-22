# Import required libraries for data processing, machine learning, and visualization
import mne  # For EEG/ECG signal processing
import numpy as np  # For numerical operations
import pandas as pd  # For data handling
from sklearn.preprocessing import StandardScaler  # For data normalization
from sklearn.model_selection import train_test_split  # For splitting data
import tensorflow as tf  # For building and training CNN
from tensorflow.keras.models import Sequential  # For sequential model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout  # CNN layers
from tensorflow.keras.optimizers import Adam  # Optimizer for training
import matplotlib.pyplot as plt  # For plotting training history

class ECG:
    """Class to handle ECG data preprocessing and CNN input preparation."""
    def __init__(self, Baseline, Stimuli, SamplingRate):
        """Initialize ECG object with baseline and stimuli data.
        
        Args:
            Baseline: List of 18 baseline trials, each (15616, 2) NumPy array
            Stimuli: List of 18 stimuli trials, each (X, 2) NumPy array
            SamplingRate: Sampling frequency (e.g., 256 Hz)
        """
        # Create MNE Info object for two ECG channels
        self.info = mne.create_info(ch_names=['ECG_Left', 'ECG_Right'], 
                                   sfreq=SamplingRate, 
                                   ch_types=['ecg', 'ecg'])
        
        # Process Baseline: Convert 18 trials to MNE RawArray objects
        self.Baseline = []
        for i, trial in enumerate(Baseline):
            try:
                trial_data = np.array(trial).T  # Transpose to (2, 15616)
                if trial_data.shape[0] != 2 or trial_data.size == 0:
                    print(f"Warning: Invalid Baseline trial {i+1}, shape: {trial_data.shape}")
                    continue
                self.Baseline.append(mne.io.RawArray(trial_data, self.info))
            except Exception as e:
                print(f"Error creating RawArray for Baseline trial {i+1}: {e}")
        
        # Process Stimuli: Convert 18 trials to MNE RawArray objects
        self.Stimuli = []
        for i, trial in enumerate(Stimuli):
            try:
                trial_data = np.array(trial).T  # Transpose to (2, X)
                if trial_data.shape[0] != 2 or trial_data.size == 0:
                    print(f"Warning: Invalid Stimuli trial {i+1}, shape: {trial_data.shape}")
                    continue
                self.Stimuli.append(mne.io.RawArray(trial_data, self.info))
            except Exception as e:
                print(f"Error creating RawArray for Stimuli trial {i+1}: {e}")
    
    def preprocess(self):
        """Preprocess Baseline and Stimuli trials with filtering and normalization."""
        for i, raw in enumerate(self.Baseline + self.Stimuli):
            try:
                # Step 1: Quality Check - Ensure signal is not flat or empty
                data = raw.get_data()
                if data.size == 0 or np.std(data) < 1e-6:
                    print(f"Warning: Flat or empty ECG signal in trial {i+1}.")
                    continue
                
                # Step 2: Baseline Wander Removal - High-pass filter
                raw.filter(l_freq=0.5, h_freq=None, method='fir', picks='ecg')
                
                # Step 3: Powerline Noise Removal - Notch filter
                raw.notch_filter(freqs=50, picks='ecg')  # Adjust to 60 Hz if needed
                
                # Step 4: Band-Pass Filtering - Retain ECG-relevant frequencies
                raw.filter(l_freq=0.5, h_freq=40, method='fir', picks='ecg')
                
                # Step 5: Normalize ECG signal per channel
                data = raw.get_data(picks='ecg')  # Shape: (2, n_samples)
                scaler = StandardScaler()
                data = scaler.fit_transform(data.T).T  # Z-score normalize
                raw._data = data  # Update RawArray with normalized data
            except Exception as e:
                print(f"Error processing trial {i+1}: {e}")
    
    def get_cnn_input(self, fixed_length=15616):
        """Prepare ECG data for CNN with fixed length and channels-last format.
        
        Args:
            fixed_length: Number of samples per trial (default: 15616, 61s at 256 Hz)
        
        Returns:
            NumPy array of shape (n_trials, fixed_length, 2)
        """
        cnn_data = []
        for i, raw in enumerate(self.Stimuli):
            try:
                data = raw.get_data(picks='ecg')  # Shape: (2, X)
                # Truncate or pad to fixed_length
                if data.shape[1] > fixed_length:
                    data = data[:, :fixed_length]
                elif data.shape[1] < fixed_length:
                    data = np.pad(data, ((0, 0), (0, fixed_length - data.shape[1])), mode='constant')
                # Transpose to (fixed_length, 2) for channels-last
                data = data.T  # Shape: (fixed_length, 2)
                cnn_data.append(data)
            except Exception as e:
                print(f"Error preparing CNN input for stimulus {i+1}: {e}")
        return np.array(cnn_data)  # Shape: (n_trials, fixed_length, 2)

class Subject:
    """Class to manage subject data and ECG processing."""
    def __init__(self, age, gender, ECGBaseline, ECGStimuli, ECGSamplingRate, Valence, Arousal, Dominance):
        """Initialize Subject with demographic info, ECG data, and emotion scores.
        
        Args:
            age: Subject age
            gender: Subject gender
            ECGBaseline: Baseline ECG data
            ECGStimuli: Stimuli ECG data
            ECGSamplingRate: Sampling frequency
            Valence: List of 18 valence scores
            Arousal: List of 18 arousal scores
            Dominance: List of 18 dominance scores
        """
        self.Age = age
        self.Gender = gender
        self.ECG = ECG(ECGBaseline, ECGStimuli, ECGSamplingRate)
        self.Valence = np.array(Valence)
        self.Arousal = np.array(Arousal)
        self.Dominance = np.array(Dominance)
    
    def preprocess_ecg(self):
        """Preprocess ECG data for this subject."""
        self.ECG.preprocess()
    
    def get_cnn_input(self, fixed_length=15616):
        """Get CNN input data and emotion labels.
        
        Args:
            fixed_length: Number of samples per trial
        
        Returns:
            cnn_data: ECG data, shape (n_trials, fixed_length, 2)
            labels: Emotion scores, shape (n_trials, 3)
        """
        cnn_data = self.ECG.get_cnn_input(fixed_length)
        labels = np.stack([self.Valence, self.Arousal, self.Dominance], axis=-1)
        return cnn_data, labels

def build_cnn_model(input_shape=(15616, 2)):
    """Build a 1D CNN model for emotion prediction.
    
    Args:
        input_shape: Shape of input data (time_steps, channels)
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        # Input layer to define shape
        Input(shape=input_shape),
        # Conv1D to extract temporal features
        Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
        # Pooling to reduce dimensionality
        MaxPooling1D(pool_size=2),
        # Second Conv1D for deeper feature extraction
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        # Flatten for dense layers
        Flatten(),
        # Dense layers for classification
        Dense(128, activation='relu'),
        Dropout(0.5),  # Regularization to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.5),
        # Output layer for Valence, Arousal, Dominance
        Dense(3)
    ])
    # Compile with MSE loss for regression
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Main processing and training loop
try:
    # Initialize lists to collect data from all subjects
    all_data = []
    all_labels = []
    ecgfreq = data['DREAMER']['ECG_SamplingRate']  # Expected: 256 Hz
    fixed_length = 15616  # 61 seconds at 256 Hz
    
    # Process each subject in the DREAMER dataset
    for k in range(data['DREAMER']['noOfSubjects']):
        # Extract subject data
        trial = data['DREAMER']['Data'][k]
        age = trial['Age']
        gender = trial['Gender']
        ecgb = trial['ECG']['baseline']  # 18 trials, each (15616, 2)
        ecgs = trial['ECG']['stimuli']   # 18 trials, each (X, 2)
        val = trial['ScoreValence']
        aro = trial['ScoreArousal']
        dom = trial['ScoreDominance']
        
        # Create and preprocess Subject object
        sub = Subject(age, gender, ecgb, ecgs, ecgfreq, val, aro, dom)
        sub.preprocess_ecg()
        cnn_data, labels = sub.get_cnn_input(fixed_length)
        
        # Filter out invalid trials (e.g., NaN labels)
        valid_mask = ~np.isnan(labels).any(axis=1)
        if np.sum(valid_mask) > 0:
            all_data.append(cnn_data[valid_mask])
            all_labels.append(labels[valid_mask])
        else:
            print(f"Warning: No valid trials for subject {k+1}")
    
    # Combine data across subjects
    if not all_data:
        raise ValueError("No valid data collected from any subject.")
    
    X = np.concatenate(all_data, axis=0)  # Shape: (n_samples, 15616, 2)
    y = np.concatenate(all_labels, axis=0)  # Shape: (n_samples, 3)
    
    # Split into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Print shapes for debugging
    print("X_train shape:", X_train.shape)  # Expected: (n_samples, 15616, 2)
    print("y_train shape:", y_train.shape)  # Expected: (n_samples, 3)
    
    # Build and train CNN model
    model = build_cnn_model(input_shape=(fixed_length, 2))
    history = model.fit(X_train, y_train, 
                       validation_data=(X_test, y_test), 
                       epochs=20, 
                       batch_size=32, 
                       verbose=1)
    
    # Evaluate model on test set
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test MSE: {test_loss:.4f}, Test MAE: {test_mae:.4f}")
    
    # Compute Pearson correlation for each emotion
    y_pred = model.predict(X_test)
    for i, emotion in enumerate(['Valence', 'Arousal', 'Dominance']):
        corr = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        print(f"Pearson correlation for {emotion}: {corr:.4f}")
    
    # Save the trained model
    model.save('emotion_cnn_model.h5')
    
    # Plot and save training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cnn_training_history.png')
    plt.close()
    
except Exception as e:
    print(f"Error in dataset processing or training: {e}")f
