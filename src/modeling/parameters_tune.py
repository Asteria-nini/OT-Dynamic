import matplotlib.pyplot as plt
from data_processing import data_loader
from modeling import wasserstein_uncertainty, wasserstein_sinkhorn
from visualization import report
import numpy as np


def get_durations_from_batches(patients_batches, select_keys):
    """Extract duration samples from selected batches"""
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


def calculate_schedule_metrics(patients, assignment_indices, doctor_ids, room_ids, final_time_slots, ts_idx,
                               cost_matrix):
    """Calculate scheduling performance metrics"""
    unassigned = [i for i, idx in enumerate(assignment_indices) if idx == -1]
    n_unassigned = len(unassigned)

    # Generate schedule details
    schedule, assigned_indices = report.generate_schedule(
        patients, assignment_indices,
        [(d, r, t) for d in range(len(doctor_ids))
         for r in range(len(room_ids))
         for t in range(len(final_time_slots))],
        ts_idx, doctor_ids, room_ids, final_time_slots)

    report_df = report.generate_report(schedule, patients, assigned_indices)

    # Calculate total cost
    total_overtime = report_df['OvertimeBlocks'].sum() if 'OvertimeBlocks' in report_df.columns else 0
    total_rooms = report_df['OpenedRooms'].sum() if 'OpenedRooms' in report_df.columns else 0

    cost_overtime = total_overtime * 100  # overtime cost per block
    cost_rooms = total_rooms * 50  # room opening cost

    total_cost = cost_overtime + cost_rooms

    return {
        'total_cost': total_cost,
        'unassigned_count': n_unassigned,
        'overtime_blocks': total_overtime,
        'opened_rooms': total_rooms,
        'assigned_count': len(patients) - n_unassigned
    }


def run_single_batch(batch_key, train_durations, epsilon, kappa, validation_keys,
                     base_path, batch_count, output_dir):
    """Run scheduling for a single batch with given parameters"""

    time_slots, patients_batches, doctors_batches, operating_rooms = data_loader.load_all_data(base_path, batch_count)

    patients = patients_batches.get(batch_key, None)
    doctors = doctors_batches.get(batch_key, None)

    if patients is None or patients.empty:
        print(f"Batch {batch_key} has empty patient data, skipped.")
        return None
    if doctors is None or doctors.empty:
        print(f"Batch {batch_key} has empty doctor data, skipped.")
        return None

    # Calculate delta duration
    delta_duration = wasserstein_uncertainty.compute_dynamic_delta_duration(train_durations, epsilon, kappa)

    # Preprocess data
    ts_idx, doctor_availability, room_availability, ts_list = data_loader.preprocess_data(
        time_slots, patients, doctors, operating_rooms)
    if ts_idx is None:
        print(f"Batch {batch_key} preprocessing failed, skipped.")
        return None

    doctor_ids = list(doctor_availability.keys())
    room_ids = list(room_availability.keys())

    # Run scheduling with overtime
    cost_matrix = wasserstein_sinkhorn.build_cost_matrix(patients, ts_idx, doctor_availability, room_availability,
                                                         ts_list,
                                                         delta_duration=delta_duration)
    assignment_matrix, assignment_indices, final_time_slots = wasserstein_sinkhorn.run_scheduling_with_overtime(
        patients, ts_idx, doctor_availability, room_availability, ts_list,
        doctor_ids, room_ids,
        delta_duration=delta_duration,
        max_overtime_blocks=20,
        overtime_step=2,
        reg=0.1,
        numItermax=1000)

    if assignment_indices is None:
        print(f"Batch {batch_key} solving failed.")
        return None

    # Calculate metrics
    metrics = calculate_schedule_metrics(patients, assignment_indices, doctor_ids, room_ids, final_time_slots, ts_idx,
                                         cost_matrix)
    return metrics


def tune_parameters(base_path, batch_count, output_dir,
                    training_keys, validation_keys,
                    epsilon_list, kappa_list):
    """Grid search for optimal epsilon and kappa parameters"""

    # Load all training samples
    time_slots, patients_batches, doctors_batches, operating_rooms = data_loader.load_all_data(base_path, batch_count)
    train_durations = get_durations_from_batches(patients_batches, training_keys)

    results = []
    total_runs = len(epsilon_list) * len(kappa_list)
    run_count = 0

    for epsilon in epsilon_list:
        for kappa in kappa_list:
            run_count += 1
            print(f"\n=== Run {run_count} / {total_runs} : epsilon={epsilon}, kappa={kappa} ===")

            total_cost = 0
            total_unassigned = 0
            valid_batches = 0

            for batch_key in validation_keys:
                metrics = run_single_batch(batch_key, train_durations, epsilon, kappa,
                                           validation_keys, base_path, batch_count, output_dir)
                if metrics is None:
                    continue

                total_cost += metrics['total_cost']
                total_unassigned += metrics['unassigned_count']
                valid_batches += 1

            if valid_batches == 0:
                print("Warning: No valid results for all validation batches, skipping parameter combination.")
                continue

            avg_cost = total_cost / valid_batches
            avg_unassigned = total_unassigned / valid_batches

            # Combined score (heavily penalize unassigned patients)
            score = avg_cost + 1000 * avg_unassigned

            print(f"[Parameters] epsilon={epsilon}, kappa={kappa}, "
                  f"avg_cost={avg_cost:.2f}, avg_unassigned={avg_unassigned:.2f}, score={score:.2f}")

            results.append({
                'epsilon': epsilon,
                'kappa': kappa,
                'avg_cost': avg_cost,
                'avg_unassigned': avg_unassigned,
                'score': score
            })

    # Find best parameters
    if not results:
        print("Error: No valid tuning results!")
        return None

    best_result = min(results, key=lambda x: x['score'])
    print(f"\nTuning completed, best parameters: epsilon={best_result['epsilon']}, kappa={best_result['kappa']}, "
          f"score={best_result['score']:.2f}")
    return best_result, results


def plot_tuning_results(results, epsilon_list, kappa_list, output_dir):
    """Plot tuning results as heatmaps and surface plots"""

    # Convert to matrices for plotting
    cost_matrix = np.full((len(kappa_list), len(epsilon_list)), np.nan)
    unassigned_matrix = np.full_like(cost_matrix, np.nan)
    score_matrix = np.full_like(cost_matrix, np.nan)

    # Fill matrices
    kappa_to_idx = {v: i for i, v in enumerate(kappa_list)}
    epsilon_to_idx = {v: i for i, v in enumerate(epsilon_list)}

    for res in results:
        k_idx = kappa_to_idx.get(res['kappa'], None)
        e_idx = epsilon_to_idx.get(res['epsilon'], None)
        if k_idx is not None and e_idx is not None:
            cost_matrix[k_idx, e_idx] = res['avg_cost']
            unassigned_matrix[k_idx, e_idx] = res['avg_unassigned']
            score_matrix[k_idx, e_idx] = res['score']

    # Plot heatmap function
    def plot_heatmap(matrix, title, cmap='viridis', filename=None):
        plt.figure(figsize=(8, 6))
        im = plt.imshow(matrix, origin='lower', cmap=cmap, aspect='auto',
                        extent=[epsilon_list[0], epsilon_list[-1], kappa_list[0], kappa_list[-1]])
        plt.colorbar(im)
        plt.title(title)
        plt.xlabel('epsilon')
        plt.ylabel('kappa')
        plt.xticks(epsilon_list)
        plt.yticks(kappa_list)
        plt.grid(False)
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"{title} heatmap saved to: {filename}")
        plt.show()

    # Generate heatmaps
    # plot_heatmap(cost_matrix, 'Average Total Cost', filename=f'{output_dir}/tuning_cost_heatmap.png')
    # plot_heatmap(unassigned_matrix, 'Average Unassigned Patients', filename=f'{output_dir}/tuning_unassigned_heatmap.png')
    # plot_heatmap(score_matrix, 'Overall Score', filename=f'{output_dir}/tuning_score_heatmap.png')

    # Optional 3D surface plots
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        X, Y = np.meshgrid(epsilon_list, kappa_list)

        def plot_3d_surface(Z, title, filename=None):
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor='k')
            ax.set_xlabel('epsilon')
            ax.set_ylabel('kappa')
            ax.set_zlabel(title)
            ax.set_title(title)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"{title} 3D surface saved to: {filename}")
            plt.show()

        # plot_3d_surface(cost_matrix, 'Average Total Cost (3D)', filename=f'{output_dir}/tuning_cost_surface.png')
        # plot_3d_surface(unassigned_matrix, 'Average Unassigned Patients (3D)', filename=f'{output_dir}/tuning_unassigned_surface.png')
        # plot_3d_surface(score_matrix, 'Overall Score (3D)', filename=f'{output_dir}/tuning_score_surface.png')

    except ImportError:
        print("3D plotting libraries not available, skipping 3D surface plots.")


def plot_metric_trends(results, epsilon_list, kappa_list, output_dir, fixed_param='kappa'):
    """Plot metric trends with one parameter fixed"""

    from collections import defaultdict
    metric_dict = {
        'avg_cost': defaultdict(dict),
        'avg_unassigned': defaultdict(dict),
        'score': defaultdict(dict)
    }

    for res in results:
        e = res['epsilon']
        k = res['kappa']
        if fixed_param == 'kappa':
            metric_dict['avg_cost'][k][e] = res['avg_cost']
            metric_dict['avg_unassigned'][k][e] = res['avg_unassigned']
            metric_dict['score'][k][e] = res['score']
        elif fixed_param == 'epsilon':
            metric_dict['avg_cost'][e][k] = res['avg_cost']
            metric_dict['avg_unassigned'][e][k] = res['avg_unassigned']
            metric_dict['score'][e][k] = res['score']
        else:
            raise ValueError("fixed_param must be 'kappa' or 'epsilon'")

    # Variable lists
    var_list = epsilon_list if fixed_param == 'kappa' else kappa_list
    fixed_list = kappa_list if fixed_param == 'kappa' else epsilon_list

    # Plot each metric
    for metric_name, metric_data in metric_dict.items():
        plt.figure(figsize=(8, 6))
        for fixed_val in fixed_list:
            y_values = []
            for var_val in var_list:
                y = metric_data.get(fixed_val, {}).get(var_val, None)
                if y is None:
                    y_values.append(float('nan'))
                else:
                    y_values.append(y)
            label = f"{fixed_param}={fixed_val}"
            plt.plot(var_list, y_values, marker='o', label=label)

        plt.title(f"{metric_name} vs {'epsilon' if fixed_param == 'kappa' else 'kappa'} (fixed {fixed_param})")
        plt.xlabel('epsilon' if fixed_param == 'kappa' else 'kappa')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        filename = f"{output_dir}/tuning_{metric_name}_trends_fixed_{fixed_param}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"{metric_name} trend plot (fixed {fixed_param}) saved to: {filename}")
        plt.show()


if __name__ == "__main__":
    import os
    import pandas as pd

    # Basic setup
    base_path = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'
    batch_count = 20
    output_dir = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\output"
    os.makedirs(output_dir, exist_ok=True)

    # Load and split batches
    _, patients_batches, _, _ = data_loader.load_all_data(base_path, batch_count)
    all_batch_keys = sorted(patients_batches.keys(), key=data_loader.batch_key_sort_key)
    print(f"All batch keys: {all_batch_keys}")

    training_keys = [k for k in all_batch_keys if 1 <= data_loader.batch_key_sort_key(k) <= 16]
    validation_keys = [k for k in all_batch_keys if data_loader.batch_key_sort_key(k) >= 17]
    print(f"Training batches: {training_keys}")
    print(f"Validation batches: {validation_keys}")

    # Parameter grid
    epsilon_list = np.arange(0.5, 2.1, 0.5)  # 0.5, 1.0, 1.5, 2.0
    kappa_list = np.round(np.arange(0.1, 1.4, 0.4), 1)  # 0.1, 0.5, 0.9, 1.3

    # Single batch validation example
    single_batch_key = '1_17'

    # Run parameter tuning
    best_params, all_results = tune_parameters(
        base_path=base_path,
        batch_count=batch_count,
        output_dir=output_dir,
        training_keys=training_keys,
        validation_keys=[single_batch_key],  # or use validation_keys for all batches
        epsilon_list=epsilon_list,
        kappa_list=kappa_list)

    # Save and visualize results
    if all_results:
        # Save to Excel
        df_results = pd.DataFrame(all_results)
        excel_path = os.path.join(output_dir, "parameter_tuning_results.xlsx")
        df_results.to_excel(excel_path, index=False)
        print(f"Tuning results saved to: {excel_path}")

        # Generate plots
        plot_tuning_results(all_results, epsilon_list, kappa_list, output_dir)

        # Generate trend plots
        plot_metric_trends(all_results, epsilon_list, kappa_list, output_dir, fixed_param='kappa')
        plot_metric_trends(all_results, epsilon_list, kappa_list, output_dir, fixed_param='epsilon')

    else:
        print("No valid tuning results obtained.")