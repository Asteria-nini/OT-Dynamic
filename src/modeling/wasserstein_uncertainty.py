import numpy as np
import matplotlib.pyplot as plt
from data_processing import data_loader
from visualization import scheduling_visualisation

def get_durations_from_batches(patients_batches, select_keys):
    """
    Extract duration samples from selected patient batches
    """
    durations_all = []
    for key in sorted(select_keys, key=data_loader.batch_key_sort_key):
        df = patients_batches.get(key, None)
        if df is not None and not df.empty and 'Duration' in df.columns:
            durations = df['Duration'].dropna().values
            durations_all.append(durations)
        else:
            print(f"Warning: Batch {key} missing data or 'Duration' column, skipped.")
    if durations_all:
        return np.concatenate(durations_all)
    else:
        return np.array([])


def build_uncertainty_set(samples, radius):
    """
    Build uncertainty set structure based on samples and radius
    """
    return {
        'samples': samples,
        'radius': radius,
        'sample_size': len(samples)
    }


def compute_delta_duration(train_durations, radius, factor=1.0):
    """
    Compute delta_duration using IQR and radius parameter
    """
    if len(train_durations) == 0:
        print("Warning: Empty training samples, using delta_duration=0")
        return 0

    q75, q25 = np.percentile(train_durations, [75, 25])
    iqr = q75 - q25
    delta = factor * radius * iqr
    delta_ceiled = int(np.ceil(delta))
    print(f"Computed delta_duration={delta_ceiled} (radius={radius}, IQR={iqr:.2f})")
    return delta_ceiled


if __name__ == '__main__':
    # Data directory and batch settings
    base_path = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'
    batch_count = 20

    # Load all data
    time_slots, patients_batches, doctors_batches, operating_rooms = data_loader.load_all_data(base_path, batch_count)

    # Split batches into training and validation sets
    all_batch_keys = sorted(patients_batches.keys(), key=data_loader.batch_key_sort_key)
    training_keys = [k for k in all_batch_keys if 1 <= data_loader.batch_key_sort_key(k) <= 16]  # 1_11 ~ 1_16
    validation_keys = [k for k in all_batch_keys if 17 <= data_loader.batch_key_sort_key(k) <= 20]  # 1_17 ~ 1_20

    print(f"Training batches: {len(training_keys)}, Validation batches: {len(validation_keys)}")

    # Extract duration samples from training batches
    train_durations = get_durations_from_batches(patients_batches, training_keys)
    print(f"Total training duration samples: {len(train_durations)}")

    if len(train_durations) > 0:
        print(f"Training samples - Mean: {train_durations.mean():.2f}, Std: {train_durations.std():.2f}")

        # Plot uncertainty set visualization
        radius_example = 1.5
        fig = scheduling_visualisation.plot_wasserstein_uncertainty_set(train_durations, epsilon=radius_example)
        save_path = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\output\uncertainty_set.svg"
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to: {save_path}")
    else:
        print("Warning: No training duration samples available for plotting.")

    # Build uncertainty set
    uncertainty_set = build_uncertainty_set(train_durations, radius_example)
    print(f"Uncertainty set info: samples={uncertainty_set['sample_size']}, radius={uncertainty_set['radius']}")

    # Validation data duration samples
    val_durations = get_durations_from_batches(patients_batches, validation_keys)
    print(f"Validation duration samples: {len(val_durations)}")