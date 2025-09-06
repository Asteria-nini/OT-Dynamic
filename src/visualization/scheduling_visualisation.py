import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_wasserstein_uncertainty_set(samples, epsilon=1.0):
    # Plot Wasserstein uncertainty set diagram with empirical distribution and uncertainty range
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import gamma

    bins = 30
    counts, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10,6))
    # Empirical distribution histogram (blue)
    plt.bar(bin_centers, counts, width=bin_edges[1]-bin_edges[0], alpha=0.6,
            color='deepskyblue', label='Empirical Distribution')

    # Empirical mean and Wasserstein uncertainty range
    mean_val = samples.mean()
    lower_bound = mean_val - epsilon
    upper_bound = mean_val + epsilon

    plt.axvline(mean_val, color='navy', linestyle='--', label='Empirical Mean')
    plt.axvspan(lower_bound, upper_bound, color='lightcoral', alpha=0.3,
                label=f'Wasserstein Uncertainty Set (ε={epsilon})')

    # Example true distribution using Gamma curve
    x = np.linspace(bin_edges[0], bin_edges[-1], 300)
    shape_param = 3.0
    scale_param = mean_val / shape_param
    hypothetical_density = gamma.pdf(x, a=shape_param, scale=scale_param)
    plt.plot(x, hypothetical_density, color='darkred', linestyle=':', lw=2,
             label='One Possible True Distribution')

    plt.xlabel('Surgery Duration')
    plt.ylabel('Probability Density')
    plt.title('Wasserstein Uncertainty Set Illustration')
    plt.legend()
    plt.grid(True)
    fig = plt.gcf()

    return fig


def plot_gantt_chart(schedule, time_slots_list,
                     title="Surgical Scheduling Gantt Chart (Basic)"):
    # Generate and display surgical scheduling Gantt chart using Matplotlib
    if not schedule:
        print("Empty schedule list, cannot generate Gantt chart")
        return

    print("Generating Gantt chart...")

    rooms = sorted(set(item['room_id'] for item in schedule if item.get('room_id') is not None))
    if not rooms:
        print("No valid room information in schedule")
        return
    room_to_y = {room: i for i, room in enumerate(rooms)}

    time_slot_to_index = {slot: idx for idx, slot in enumerate(time_slots_list)}

    tasks = []
    for item in schedule:
        start_time = item.get('start_time')
        duration = item.get('duration', 0)
        room_id = item.get('room_id')

        start_index = time_slot_to_index.get(start_time, -1)
        if start_index == -1 or duration <= 0 or room_id not in room_to_y:
            print(f"Invalid task (Patient: {item.get('patient_id', '?')}, Start: {start_time}, Duration: {duration}, Room: {room_id}), skipping")
            continue

        bar_label = f"{item.get('patient_id', '?')}\n({item.get('doctor_id', '?')})"

        tasks.append({
            'y': room_to_y[room_id],
            'start': start_index,
            'duration': duration,
            'label': bar_label
        })

    if not tasks:
        print("No valid tasks for Gantt chart")
        return

    fig_width = max(15, len(time_slots_list) * 0.3)
    fig_height = max(6, len(rooms) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    colors = list(mcolors.TABLEAU_COLORS.values())
    num_colors = len(colors)

    for i, task in enumerate(tasks):
        color = colors[i % num_colors]
        ax.barh(task['y'], task['duration'], left=task['start'], height=0.6,
                align='center', color=color, edgecolor='black', alpha=0.85)
        ax.text(task['start'] + task['duration'] / 2, task['y'], task['label'],
                ha='center', va='center', fontsize=9, weight='bold', color='black')

    ax.set_xlabel("Time_slots", fontsize=16)
    ax.set_ylabel("Room", fontsize=16)
    ax.set_title(title, fontsize=20, weight='bold')

    ax.set_yticks(list(room_to_y.values()))
    ax.set_yticklabels(list(room_to_y.keys()), fontsize=12)
    ax.invert_yaxis()

    max_label_num = 25
    tick_step = max(1, len(time_slots_list) // max_label_num)
    x_ticks = np.arange(0, len(time_slots_list), tick_step)
    x_labels = [time_slots_list[i] for i in x_ticks]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=12)
    ax.set_xlim(0, len(time_slots_list))

    ax.grid(axis='x', linestyle=':', linewidth=0.6, color='gray')
    ax.grid(axis='y', linestyle=':', linewidth=0.6, color='gray')

    plt.tight_layout()
    plt.show()
    return fig


def plot_simple_duration_comparison(schedule_with_actual, max_patients=25,
                                   batch_key=None, output_dir=None):
    # Create expected vs actual surgery duration comparison plots
    print(f"Creating duration comparison plots (max {max_patients} patients)")

    if not schedule_with_actual:
        print("Empty schedule list")
        return None, None

    total_patients = len(schedule_with_actual)
    if total_patients > max_patients:
        selected_data = schedule_with_actual[:max_patients]
        print(f"Showing {max_patients} of {total_patients} patients")
    else:
        selected_data = schedule_with_actual
        print(f"Showing all {total_patients} patients")

    patient_ids = [item.get('patient_id', '') for item in selected_data]
    expected_durations = [item.get('duration', 0) for item in selected_data]
    actual_durations = [item.get('actual_duration', 0) for item in selected_data]

    # Bar chart
    fig_bar = plt.figure(figsize=(11, 6))
    bar_width = 0.3
    x = np.arange(len(patient_ids))

    plt.bar(x - bar_width / 2, expected_durations, bar_width, label='Expected Duration', color='#1f77b4')
    plt.bar(x + bar_width / 2, actual_durations, bar_width, label='Actual Duration', color='#ff7f0e')

    plt.title('Patient Surgery Duration Comparison' if not batch_key else f'Patient Surgery Duration Comparison - Batch {batch_key}')
    plt.xlabel('Patient ID')
    plt.ylabel('Duration')
    if len(patient_ids) > 20:
        plt.xticks(x, patient_ids, rotation=45, ha='right', fontsize=8)
    else:
        plt.xticks(x, patient_ids)

    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Scatter plot
    fig_scatter = plt.figure(figsize=(8, 6))

    all_expected = [item.get('duration', 0) for item in schedule_with_actual]
    all_actual = [item.get('actual_duration', 0) for item in schedule_with_actual]

    plt.scatter(all_expected, all_actual, color='#1f77b4', alpha=0.7)
    max_val = max(max(all_expected), max(all_actual))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Ideal Case (No Adjustment)')

    plt.title('Expected vs Actual Duration Scatter Plot' if not batch_key else f'Expected vs Actual Duration Scatter Plot - Batch {batch_key}')
    plt.xlabel('Expected Duration')
    plt.ylabel('Actual Duration')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    plt.tight_layout()

    # Save charts
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            bar_path = os.path.join(output_dir, f"{batch_key}_duration_comparison_bar.svg" if batch_key else "duration_comparison_bar.png")
            scatter_path = os.path.join(output_dir, f"{batch_key}_duration_comparison_scatter.svg" if batch_key else "duration_comparison_scatter.png")
            fig_bar.savefig(bar_path, dpi=300)
            fig_scatter.savefig(scatter_path, dpi=300)
            print(f"Charts saved to:\n  Bar chart => {bar_path}\n  Scatter plot => {scatter_path}")
        except Exception as e:
            print(f"Failed to save charts: {e}")

    return fig_bar, fig_scatter


def plot_adjusted_gantt_chart(adjusted_schedule, time_slots_list, title="Surgical Scheduling Gantt Chart (Adjusted)"):
    # Generate Gantt chart for adjusted schedule
    if not adjusted_schedule:
        print("Empty adjusted schedule list")
        return

    print("Generating adjusted Gantt chart...")

    # Get and sort all operating rooms
    rooms = sorted(list(set(item['room_id'] for item in adjusted_schedule)))
    if not rooms:
        print("No valid room information in adjusted schedule")
        return

    room_to_y = {room: i for i, room in enumerate(rooms)}
    time_slot_to_index = {slot: idx for idx, slot in enumerate(time_slots_list)}

    # Prepare task data
    tasks = []
    for item in adjusted_schedule:
        start_time = item['start_time']
        duration = item.get('actual_duration', item['duration'])
        room_id = item['room_id']
        start_index = time_slot_to_index.get(start_time, -1)

        if start_index == -1 or duration <= 0 or room_id is None:
            print(f"Invalid schedule item (Patient {item.get('patient_id', 'Unknown')}), skipping")
            continue

        # Status information for color coding
        status = ""
        if 'adjustment_type' in item:
            if item['adjustment_type'] == 'Rescheduled':
                status = "Rescheduled"
            elif item['adjustment_type'] == 'Conflict' and item.get('has_conflict', False):
                status = "Conflict"
            elif item['adjustment_type'] == '无需调整':
                status = ""

        if item.get('is_delayed', False):
            duration_status = "Extended"
        elif item.get('duration_diff', 0) < 0:
            duration_status = "Shortened"
        else:
            duration_status = ""

        tasks.append({
            'name': f" {item.get('patient_id', '?')}\n({item.get('doctor_id', '?')})",
            'room': room_id,
            'start': start_index,
            'duration': duration,
            'y': room_to_y[room_id],
            'status': status,
            'duration_status': duration_status,
            'adjustment_type': item.get('adjustment_type', '')
        })

    if not tasks:
        print("No valid tasks for adjusted Gantt chart")
        return

    fig_width = max(15, len(time_slots_list) * 0.3)
    fig_height = max(6, len(rooms) * 0.8)
    fig, ax = plt.subplots(figsize=(25, 15))

    tableau_colors = plt.cm.get_cmap('tab10').colors

    normal_color = tableau_colors[0]
    rescheduled_color = tableau_colors[1]
    conflict_color = tableau_colors[3]
    extended_color = tableau_colors[2]
    shortened_color = tableau_colors[4]

    for i, task in enumerate(tasks):
        # Determine color based on status
        color = normal_color

        if task.get('status') == 'Rescheduled':
            color = rescheduled_color
        elif task.get('status') == 'Conflict':
            color = conflict_color
        elif task.get('duration_status') == 'Extended':
            color = extended_color
        elif task.get('duration_status') == 'Shortened':
            color = shortened_color

        ax.barh(task['y'], task['duration'], left=task['start'], height=0.6,
                align='center', color=color, edgecolor='black', alpha=0.85)

        ax.text(task['start'] + task['duration'] / 2,
                task['y'],
                task['name'],
                ha='center', va='center', color='black', fontsize=9, weight='bold', wrap=True)

    ax.set_xlabel("Time Slots", fontsize=16)
    ax.set_ylabel("Operating Room", fontsize=16)
    ax.set_title(title, fontsize=21, weight='bold')

    ax.set_yticks(list(room_to_y.values()))
    ax.set_yticklabels(list(room_to_y.keys()), fontsize=12)
    ax.invert_yaxis()

    tick_spacing = max(1, len(time_slots_list) // 25)
    x_ticks = np.arange(0, len(time_slots_list), tick_spacing)
    x_labels = [time_slots_list[int(i)] for i in x_ticks if 0 <= int(i) < len(time_slots_list)]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=12)
    ax.set_xlim(0, len(time_slots_list))

    ax.grid(True, axis='x', linestyle=':', linewidth=0.6, color='gray')
    ax.grid(True, axis='y', linestyle=':', linewidth=0.6, color='gray')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=normal_color, edgecolor='black', label='Normal'),
        Patch(facecolor=extended_color, edgecolor='black', label='Extended'),
        Patch(facecolor=shortened_color, edgecolor='black', label='Shortened'),
        Patch(facecolor=rescheduled_color, edgecolor='black', label='Rescheduled'),
        Patch(facecolor=conflict_color, edgecolor='black', label='Conflict')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.show()
    return fig