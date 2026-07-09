import pandas as pd
import h5py
import numpy as np

def convert_csv_to_h5(csv_file, h5_file):
    print(f"Attempting to convert {csv_file}...")
    
    # We use header=None because we are manually defining the columns after skipping the first row.
    # We use sep=None with engine='python' to auto-detect delimiters (comma, tab, semicolon).
    df = pd.read_csv(
        csv_file, 
        skiprows=1, 
        header=None,
        sep=None,
        encoding='latin1',
        engine='python', 
        on_bad_lines='skip'
    ) 
    
    print("CSV loaded successfully!")
    print(f"Detected {df.shape[1]} columns. Column names: {df.columns.tolist()}")
    
    if df.shape[1] < 2:
        print("ERROR: Less than 2 columns detected. Please check the CSV structure.")
        return

    print("Mapping columns...")
    
    # Ensure we are using the first two columns regardless of their names
    pos_values = df.iloc[:, 0].values # Use first column
    values = df.iloc[:, 1].values     # Use second column
    
    # Create 3D coordinates [pos, pos, pos] using numpy broadcasting
    coords = np.column_stack([pos_values, pos_values, pos_values])
    
    # Save to HDF5
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('/geometry/coordinates', data=coords)
        f.create_dataset('/results/force', data=values)
    
    print(f"✅ Conversion complete: {h5_file} created.")

# Run it
convert_csv_to_h5('your_data.csv', 'nist_data.h5')