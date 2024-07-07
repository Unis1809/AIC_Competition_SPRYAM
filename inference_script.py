import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import subprocess
import os

# Define paths
compressed_weights_path = 'd:/data/best_model.weights.7z'
decompressed_weights_path = 'd:/data/best_model.weights.h5'
test_data_path = 'd:/data/test_features.pkl'
output_csv_path = 'predicted_transcripts.csv'

# Decompress the model weights
if not os.path.exists(decompressed_weights_path):
    subprocess.run(['7z', 'x', compressed_weights_path, '-o' + os.path.dirname(decompressed_weights_path)])

# Define your model architecture (same as during training)
def build_model(input_dim, output_dim, rnn_units=256, use_attention=True):
    input_data = Input(name='input', shape=(None, input_dim))
    
    # Convolutional layer
    x = Conv1D(filters=256, kernel_size=13, strides=1, padding='same', activation='relu')(input_data)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)
    
    # Bidirectional GRU layers with layer normalization and dropout
    x = Bidirectional(GRU(rnn_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Bidirectional(GRU(rnn_units, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Apply attention if enabled
    if use_attention:
        x = Attention()([x, x])
    
    # TimeDistributed Dense layer for output
    y_pred = TimeDistributed(Dense(output_dim, activation='softmax'))(x)
    
    model = Model(inputs=input_data, outputs=y_pred)
    return model

# Load the model architecture
input_dim = X_test.shape[-1]
output_dim = vocab_size  # Ensure this matches the vocabulary size used during training

model = build_model(input_dim, output_dim)
model.load_weights(decompressed_weights_path)

# Load test data from the .pkl file
test_df = pd.read_pickle(test_data_path)

# Assuming your test data has features and transcripts columns
X_test = pad_sequences(test_df['features'].tolist(), maxlen=max_feature_length, padding='post', dtype='float32')

# Make predictions on the test set
predictions = model.predict(X_test)

# Decode predictions using your tokenizer
decoded_predictions = decode_predictions(predictions, tokenizer)

# Add decoded predictions to the test dataframe
test_df['predicted_transcript'] = decoded_predictions

# Save the dataframe to a CSV file
test_df[['audio', 'predicted_transcript']].to_csv(output_csv_path, index=False)

print(f"Predictions saved to: {output_csv_path}")
