import json
import h5py
import numpy as np
import os

def create_mock_data():
    print("Creating mock SysML JSON map...")
    sysml_map = {
        "digital_twin": "demo_structure_01",
        "components": {
            "primary_beam": {
                "type": "structural_member",
                "material": "steel",
                "data_path": "/simulations/load_case_A/primary_beam/stress"
            },
            "connection_plate": {
                "type": "joint",
                "material": "aluminum",
                "data_path": "/simulations/load_case_A/connection_plate/temperature"
            }
        }
    }
    
    with open("mock_sysml.json", "w") as f:
        json.dump(sysml_map, f, indent=4)

    print("Creating mock HDF5 dataset...")
    # Create an HDF5 file mimicking simulation outputs
    with h5py.File("demo_data.h5", "w") as f:
        # Create a group structure
        beam_group = f.create_group("/simulations/load_case_A/primary_beam")
        plate_group = f.create_group("/simulations/load_case_A/connection_plate")
        
        # Populate with some fake numpy arrays (e.g., stress values and temperatures)
        beam_group.create_dataset("stress", data=np.array([120.5, 145.2, 310.8, 405.1, 290.4]))
        plate_group.create_dataset("temperature", data=np.array([72.1, 74.5, 88.0, 95.5, 91.2]))

    print("✅ Mock data generated successfully! You now have mock_sysml.json and demo_data.h5")

if __name__ == "__main__":
    create_mock_data()