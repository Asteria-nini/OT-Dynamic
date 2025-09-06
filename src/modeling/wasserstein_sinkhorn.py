import os
import time
import math
import numpy as np
import ot
from data_processing import data_loader
from visualization import scheduling_visualisation, report
from modeling import wasserstein_uncertainty
import warnings

warnings.filterwarnings("ignore", message="Glyph .* missing from current font")


def build_cost_matrix(patients,
                      time_slot_indices,
                      doctor_availability,
                      room_availability,
                      time_slots_list,
                      duration_buffer=0,
                      normal_work_blocks=300,
                      overtime_base=1.0,
                      overtime_factor=0.5):
    """
    Build cost matrix for patient-doctor-room-time assignments
    Include arrival day constraints and overtime cost calculations
    """

    if patients.empty:
        return np.array([])

    if len(patients.columns) < 4:
        return np.array([])

    patient_id_col = patients.columns[0]
    duration_col = patients.columns[1]
    doctor_id_col = patients.columns[2]
    arrival_day_col = patients.columns[3]

    n_patients = len(patients)
    doctor_ids = list(doctor_availability.keys())
    room_ids = list(room_availability.keys())
    n_doctors = len(doctor_ids)
    n_rooms = len(room_ids)
    n_time_slots = len(time_slots_list)

    if n_doctors == 0 or n_rooms == 0 or n_time_slots == 0:
        return np.array([])

    doctor_id_to_idx = {doc_id: i for i, doc_id in enumerate(doctor_ids)}
    cost_matrix = np.full((n_patients, n_doctors, n_rooms, n_time_slots), np.inf)
    print(f"Building cost matrix, dimensions: ({n_patients}, {n_doctors}, {n_rooms}, {n_time_slots})")

    # Parse room and time slot days for arrival constraints
    parsed_room_days = {room_id: data_loader.parse_day_from_room_id(room_id) for room_id in room_ids}
    parsed_slot_days = {slot: data_loader.parse_day_from_time_slot(slot) for slot in time_slots_list}

    for p_idx, patient in patients.iterrows():
        patient_id = patient[patient_id_col]

        # Get surgery duration with buffer
        try:
            surgery_duration = int(patient[duration_col])
            if surgery_duration <= 0:
                raise ValueError("Duration must be positive")
        except (ValueError, TypeError):
            print(f"Warning: Patient {patient_id} has invalid duration '{patient[duration_col]}', skipped.")
            continue

        total_duration = surgery_duration + duration_buffer
        total_duration_int = math.ceil(total_duration)

        try:
            arrival_day = int(patient[arrival_day_col])
            if arrival_day < 0:
                raise ValueError("Arrival day cannot be negative")
        except (ValueError, TypeError):
            continue

        assigned_doctor = patient[doctor_id_col]
        if assigned_doctor not in doctor_id_to_idx:
            continue
        doctor_idx = doctor_id_to_idx[assigned_doctor]

        for t_idx, time_slot in enumerate(time_slots_list):
            for r_idx, room_id in enumerate(room_ids):
                room_day = parsed_room_days[room_id]
                time_slot_day = parsed_slot_days[time_slot]

                # Check arrival day constraints
                if room_day == -1 or time_slot_day == -1:
                    continue
                if room_day != arrival_day or time_slot_day != arrival_day:
                    continue

                # Check time bounds
                if t_idx + total_duration_int > n_time_slots:
                    continue

                # Check availability
                try:
                    doc_avail = doctor_availability[assigned_doctor][t_idx: t_idx + total_duration_int]
                    room_avail = room_availability[room_id][t_idx: t_idx + total_duration_int]
                    doctor_free = np.all(doc_avail > 0)
                    room_free = np.all(room_avail > 0)

                    if doctor_free and room_free:
                        # Base cost
                        base_cost = 1.0 + 0.001 * t_idx + 0.0001 * p_idx

                        # Calculate overtime cost
                        overtime_blocks = max(0, t_idx + total_duration_int - normal_work_blocks)
                        overtime_cost = 0
                        if overtime_blocks > 0:
                            for i in range(overtime_blocks):
                                overtime_cost += overtime_base + i * overtime_factor

                        total_cost = base_cost + overtime_cost
                        cost_matrix[p_idx, doctor_idx, r_idx, t_idx] = total_cost

                except (KeyError, IndexError):
                    continue

    feasible_mask = ~np.isinf(cost_matrix)
    total_feasible = np.sum(feasible_mask)
    print(f"Cost matrix built. Feasible assignments: {total_feasible}")
    if total_feasible > 0:
        print(f"Cost range: [{cost_matrix[feasible_mask].min():.4f}, {cost_matrix[feasible_mask].max():.4f}]")
    else:
        print("Warning: No feasible assignments found.")

    return cost_matrix


def solve_transport_problem(cost_matrix, n_patients, drt_indices, doctor_ids, room_ids,
                            time_slots_list, patients, reg=0.1, max_iter=2000, duration_buffer=0):
    """
    Solve optimal transport problem using Sinkhorn algorithm
    Return hard assignment matrix and indices
    """

    n_doctors = len(doctor_ids)
    n_rooms = len(room_ids)
    n_time_slots = len(time_slots_list)
    n_slots = len(drt_indices)

    # Input validation
    if cost_matrix is None or cost_matrix.size == 0:
        return None, None

    if n_patients == 0:
        return np.array([]), []

    # Initialize flat cost matrix
    flat_cost = np.full((n_patients, n_slots), np.inf)
    indices_map = {(d, r, t): i for i, (d, r, t) in enumerate(drt_indices)}

    doctor_id_col = patients.columns[2]
    duration_col = patients.columns[1]
    patient_id_col = patients.columns[0]
    doctor_id_to_idx = {doc_id: i for i, doc_id in enumerate(doctor_ids)}

    for p_idx in range(n_patients):
        assigned_doctor_id = patients.iloc[p_idx][doctor_id_col]
        if assigned_doctor_id not in doctor_id_to_idx:
            continue
        assigned_doctor_idx = doctor_id_to_idx[assigned_doctor_id]

        # Get patient costs for assigned doctor
        patient_costs = cost_matrix[p_idx, assigned_doctor_idx, :, :]

        # Fill flat cost matrix
        for r_idx in range(n_rooms):
            for t_idx in range(n_time_slots):
                cost = patient_costs[r_idx, t_idx]
                flat_idx = indices_map.get((assigned_doctor_idx, r_idx, t_idx))
                if flat_idx is not None:
                    flat_cost[p_idx, flat_idx] = cost

    # Check infeasible patients
    infeasible_patients = np.where(np.all(np.isinf(flat_cost), axis=1))[0]
    if len(infeasible_patients) > 0:
        infeasible_ids = patients.iloc[infeasible_patients][patient_id_col].tolist()
        print(f"Warning: Infeasible patients (indices): {infeasible_patients} (IDs: {infeasible_ids})")

    # Replace inf values with large penalty
    large_penalty = np.inf
    if np.any(np.isinf(flat_cost)):
        finite_costs = flat_cost[~np.isinf(flat_cost)]
        if finite_costs.size > 0:
            max_finite = np.max(finite_costs)
            large_penalty = max_finite * 100 + 1000
        else:
            large_penalty = 1e9
        flat_cost[np.isinf(flat_cost)] = large_penalty
        print(f"Replaced inf costs with penalty: {large_penalty}")

    if n_slots == 0:
        print("Error: No available slots.")
        return None, None

    # Set uniform distributions
    a = np.ones(n_patients) / n_patients
    b = np.ones(n_slots) / n_slots

    # Size validation
    if a.shape[0] != flat_cost.shape[0] or b.shape[0] != flat_cost.shape[1]:
        print(f"Error: Distribution size mismatch. a: {a.shape}, b: {b.shape}, cost: {flat_cost.shape}")
        return None, None

    print(f"Running Sinkhorn algorithm: reg={reg}, max_iter={max_iter}")
    try:
        # Solve optimal transport
        gamma = ot.sinkhorn(a, b, flat_cost, reg, numItermax=max_iter, verbose=False)
        print("Sinkhorn algorithm completed.")
    except Exception as e:
        print(f"Sinkhorn algorithm failed: {e}")
        return None, None

    # Hard assignment
    assignment_indices = [-1] * n_patients
    occupied_slots = set()

    # Collect assignment candidates
    candidates = []
    for p_idx in range(n_patients):
        for flat_idx in range(n_slots):
            prob = gamma[p_idx, flat_idx]
            cost = flat_cost[p_idx, flat_idx]
            if cost < large_penalty * 0.99:
                candidates.append((p_idx, flat_idx, prob, cost))

    # Sort by probability (desc) and cost (asc)
    candidates.sort(key=lambda x: (-x[2], x[3]))
    assigned_patients = set()

    # Assign patients to slots
    for p_idx, flat_idx, prob, cost in candidates:
        if p_idx in assigned_patients:
            continue

        if not (0 <= flat_idx < len(drt_indices)):
            continue

        d_idx, r_idx, t_idx = drt_indices[flat_idx]

        # Validate assignment
        patient_info = patients.iloc[p_idx]
        assigned_doctor_id = patient_info[doctor_id_col]
        if assigned_doctor_id not in doctor_id_to_idx:
            continue
        assigned_doctor_idx = doctor_id_to_idx[assigned_doctor_id]

        if d_idx != assigned_doctor_idx:
            continue

        # Get duration with buffer
        try:
            duration_raw = int(patient_info[duration_col]) + duration_buffer
            duration = math.ceil(duration_raw)
            if duration <= 0:
                continue
        except:
            continue

        # Check conflicts
        is_conflict = False
        doctor_slots = set()
        room_slots = set()
        for offset in range(duration):
            cur_t = t_idx + offset
            if cur_t >= n_time_slots:
                is_conflict = True
                break
            if ('doctor', d_idx, cur_t) in occupied_slots or ('room', r_idx, cur_t) in occupied_slots:
                is_conflict = True
                break
            doctor_slots.add(('doctor', d_idx, cur_t))
            room_slots.add(('room', r_idx, cur_t))

        # Assign if no conflict
        if not is_conflict:
            assignment_indices[p_idx] = flat_idx
            occupied_slots.update(doctor_slots)
            occupied_slots.update(room_slots)
            assigned_patients.add(p_idx)

    # Build hard assignment matrix
    hard_matrix = np.zeros_like(gamma)
    for p_idx, flat_idx in enumerate(assignment_indices):
        if flat_idx != -1:
            hard_matrix[p_idx, flat_idx] = 1

    print(f"Hard assignment completed. Assigned patients: {len(assigned_patients)} / {n_patients}")
    return hard_matrix, assignment_indices


def extend_time_slots_by_day(time_slots_list, target_day, extend_n=2):
    """
    Extend time slots for a specific day by adding new slots at the end
    """
    day_slots = [ts for ts in time_slots_list if ts.startswith(f"T{target_day}-")]
    if day_slots:
        max_slot_num = max(int(ts.split('-')[1]) for ts in day_slots)
    else:
        max_slot_num = 0

    new_slots = []
    for i in range(1, extend_n + 1):
        new_slot_num = max_slot_num + i
        new_slot_name = f"T{target_day}-{new_slot_num}"
        new_slots.append(new_slot_name)

    insert_pos = len(time_slots_list)
    for i in reversed(range(len(time_slots_list))):
        if time_slots_list[i].startswith(f"T{target_day}-"):
            insert_pos = i + 1
            break

    new_time_slots = time_slots_list[:insert_pos] + new_slots + time_slots_list[insert_pos:]
    return new_time_slots, insert_pos, extend_n


def extend_availability_at_pos(availability_dict, insert_pos, extend_n=2):
    """Extend availability arrays at specified position"""
    for key in availability_dict:
        old_arr = availability_dict[key]
        extension = np.ones(extend_n, dtype=int)
        new_arr = np.concatenate([old_arr[:insert_pos], extension, old_arr[insert_pos:]])
        availability_dict[key] = new_arr


def run_scheduling_with_overtime(patients, ts_idx, doctor_availability, room_availability,
                                 time_slots_list, doctor_ids, room_ids, duration_buffer=0,
                                 max_overtime_blocks=20, overtime_step=2, reg=0.1, max_iter=2000):
    """
    Run scheduling with automatic overtime extension for unassigned patients
    """

    arrival_day_col = patients.columns[3]
    extend_count = 0
    current_time_slots = time_slots_list.copy()
    current_doctor_avail = {k: v.copy() for k, v in doctor_availability.items()}
    current_room_avail = {k: v.copy() for k, v in room_availability.items()}

    while True:
        print(f"\nScheduling attempt #{extend_count + 1}: Time slots = {len(current_time_slots)}")

        cost_matrix = build_cost_matrix(patients, ts_idx, current_doctor_avail,
                                        current_room_avail, current_time_slots, duration_buffer)
        n_patients = len(patients)
        n_time_slots = len(current_time_slots)
        n_doctors = len(current_doctor_avail)
        n_rooms = len(current_room_avail)

        drt_indices = [(d, r, t) for d in range(n_doctors)
                       for r in range(n_rooms)
                       for t in range(n_time_slots)]

        assignment_matrix, assignment_indices = solve_transport_problem(
            cost_matrix, n_patients, drt_indices,
            list(current_doctor_avail.keys()),
            list(current_room_avail.keys()),
            current_time_slots,
            patients,
            reg=reg,
            max_iter=max_iter,
            duration_buffer=duration_buffer)

        if assignment_indices is None:
            return None, None, current_time_slots

        unassigned = [i for i, idx in enumerate(assignment_indices) if idx == -1]

        if len(unassigned) == 0:
            return assignment_matrix, assignment_indices, current_time_slots
        else:
            print(f"{len(unassigned)} patients unassigned.")

            if extend_count * overtime_step > max_overtime_blocks:
                print(f"Maximum overtime limit ({max_overtime_blocks} blocks) reached.")
                return assignment_matrix, assignment_indices, current_time_slots

            unassigned_patients = patients.iloc[unassigned]
            unassigned_days = sorted(unassigned_patients[arrival_day_col].unique())

            print(f"Unassigned patient arrival days: {unassigned_days}")

            # Extend overtime for each unassigned day
            for day in unassigned_days:
                print(f"Extending day {day} by {overtime_step} time blocks.")
                current_time_slots, insert_pos, num_new = extend_time_slots_by_day(
                    current_time_slots, day, overtime_step)
                extend_availability_at_pos(current_doctor_avail, insert_pos, num_new)
                extend_availability_at_pos(current_room_avail, insert_pos, num_new)

            extend_count += 1


def main():
    """Main scheduling workflow"""
    base_path = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'
    batch_count = 20

    output_dir = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\output"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    time_slots, patients_batches, doctors_batches, operating_rooms = data_loader.load_all_data(base_path, batch_count)

    if time_slots.empty or operating_rooms.empty:
        return

    time_slot_col = time_slots.columns[0]
    time_slots_list = time_slots[time_slot_col].unique().tolist()

    # Split batches
    all_batch_keys = sorted(patients_batches.keys(), key=data_loader.batch_key_sort_key)
    training_keys = [k for k in all_batch_keys if 1 <= data_loader.batch_key_sort_key(k) <= 16]
    validation_keys = [k for k in all_batch_keys if data_loader.batch_key_sort_key(k) >= 17]

    print(f"Training batches: {len(training_keys)}, Validation batches: {len(validation_keys)}")

    # Extract training duration samples
    train_durations = []
    for key in training_keys:
        df = patients_batches.get(key, None)
        if df is not None and not df.empty and 'Duration' in df.columns:
            durations = df['Duration'].dropna().values
            train_durations.append(durations)
        else:
            print(f"Warning: Training batch {key} missing data or 'Duration' column, skipped.")

    if train_durations:
        train_durations = np.concatenate(train_durations)
        print(f"Total training duration samples: {len(train_durations)}")
        print(f"Training samples - Mean: {train_durations.mean():.2f}, Std: {train_durations.std():.2f}")
    else:
        train_durations = np.array([])
        print("Warning: No training samples available.")

    # Set parameters
    epsilon = 1
    duration_buffer = wasserstein_uncertainty.compute_dynamic_delta_duration(train_durations, epsilon, kappa=0.25)

    # Process validation batches
    for batch_key in validation_keys:
        print(f"\n===== Processing validation batch {batch_key} =====")
        patients = patients_batches.get(batch_key, None)
        doctors = doctors_batches.get(batch_key, None)

        if patients is None or patients.empty:
            print(f"Batch {batch_key} has empty patient data, skipped.")
            continue
        if doctors is None or doctors.empty:
            print(f"Batch {batch_key} has empty doctor data, skipped.")
            continue

        # Preprocess data
        ts_idx, doctor_availability, room_availability, ts_list = data_loader.preprocess_data(
            time_slots, patients, doctors, operating_rooms)
        if ts_idx is None:
            print(f"Batch {batch_key} preprocessing failed, skipped.")
            continue

        doctor_ids = list(doctor_availability.keys())
        room_ids = list(room_availability.keys())

        # Run scheduling with overtime
        assignment_matrix, assignment_indices, final_time_slots = run_scheduling_with_overtime(
            patients, ts_idx, doctor_availability, room_availability, ts_list,
            doctor_ids, room_ids,
            duration_buffer=duration_buffer,
            max_overtime_blocks=20,
            overtime_step=2,
            reg=0.1,
            max_iter=2000)

        if assignment_indices is None:
            print(f"Batch {batch_key} scheduling failed, skipped.")
            continue

        # Generate schedule and report
        schedule, assigned_indices = report.generate_schedule(
            patients, assignment_indices,
            [(d, r, t) for d in range(len(doctor_ids))
             for r in range(len(room_ids))
             for t in range(len(final_time_slots))],
            ts_idx, doctor_ids, room_ids, final_time_slots)

        report_df = report.generate_report(schedule, patients, assigned_indices)

        # Save report
        report_file = os.path.join(output_dir, f"{batch_key}_surgery_assignment_report.xlsx")
        try:
            report_df.to_excel(report_file, index=False, engine='openpyxl')
            print(f"Batch {batch_key} report saved to: {os.path.abspath(report_file)}")
        except Exception as e:
            print(f"Error saving batch {batch_key} report: {e}")

        # Generate and save Gantt chart
        fig = scheduling_visualisation.plot_gantt_chart(schedule, final_time_slots,
                                                        title=f"{batch_key}_surgical_schedule_gantt_chart")

        if fig is not None:
            gantt_file = os.path.join(output_dir, f"{batch_key}_surgical_schedule_gantt_chart.svg")
            try:
                fig.savefig(gantt_file, dpi=300, bbox_inches='tight')
                print(f"Batch {batch_key} Gantt chart saved to: {os.path.abspath(gantt_file)}")
            except Exception as e:
                print(f"Error saving batch {batch_key} Gantt chart: {e}")

        time.sleep(1)

    print("\nAll validation batches processed.")


if __name__ == '__main__':
    main()