import random
import os
import time
import numpy as np
import pandas as pd
from data_processing import data_loader
from modeling import wasserstein_uncertainty, wasserstein_sinkhorn
from visualization import scheduling_visualisation, report
from utils.Basic_costs import evaluate_basic_costs_by_day_with_visualization, print_daily_cost_report, \
    save_daily_cost_results_to_excel
from utils.comparison import compare_two_layer_schedules
from utils.evaluation import run_comprehensive_evaluation
from utils.overtime import OvertimeExtensionTracker, EnhancedOvertimeEvaluator, run_scheduling_with_overtime_enhanced


def parse_time_slot(time_slot):
    """Parse time slot like 'T3-05', return (day_number, slot_index)"""
    if isinstance(time_slot, str) and '-' in time_slot:
        parts = time_slot.split('-')
        try:
            day = int(parts[0][1:])
            slot_idx = int(parts[1])
            return day, slot_idx
        except Exception as e:
            print(f"Time slot parsing failed '{time_slot}': {e}")
    return None, None


def simulate_surgery_progression(base_schedule, time_slots_list, variation_range=(-0.15, 0.15),
                                 variation_probability=0.3, seed=None):
    """Simulate actual surgery duration variations"""
    if seed is not None:
        random.seed(seed)

    schedule_with_actual = []
    print("\n--- Simulating Surgery Duration Variations ---")

    for item in base_schedule:
        expected_duration = item.get('duration', 1)

        if random.random() < variation_probability:
            variation_pct = random.uniform(variation_range[0], variation_range[1])
            actual_duration = max(1, round(expected_duration * (1 + variation_pct)))
            status = "extended" if actual_duration > expected_duration else "shortened" if actual_duration < expected_duration else "no_change"
        else:
            actual_duration = expected_duration
            status = "no_change"

        new_item = item.copy()
        new_item['actual_duration'] = actual_duration
        new_item['is_delayed'] = actual_duration > expected_duration
        new_item['duration_diff'] = actual_duration - expected_duration
        new_item['status'] = 'waiting'

        schedule_with_actual.append(new_item)

        patient_id = item.get('patient_id', 'unknown')
        print(f"Patient {patient_id}: Expected {expected_duration} blocks, Actual {actual_duration} blocks, {status}")

    return schedule_with_actual


def create_time_progression_map(schedule, time_slots_list, current_day):
    """Create surgery start/end mapping for current day time slots"""
    day_time_slots = [ts for ts in time_slots_list if parse_time_slot(ts)[0] == current_day]
    progression_map = {slot: {'start': [], 'end': []} for slot in day_time_slots}

    max_slot_idx = max([parse_time_slot(ts)[1] for ts in day_time_slots]) if day_time_slots else -1

    for surgery in schedule:
        start_time = surgery.get('start_time')
        actual_duration = surgery.get('actual_duration', 1)
        if start_time is None:
            continue

        start_day, start_slot_idx = parse_time_slot(start_time)
        if start_day != current_day:
            continue

        end_slot_idx = min(start_slot_idx + actual_duration - 1, max_slot_idx)
        end_time = f"T{current_day}-{end_slot_idx:02d}"

        if start_time in progression_map:
            progression_map[start_time]['start'].append(surgery)

        if end_time in progression_map:
            progression_map[end_time]['end'].append(surgery)

    return progression_map


def update_surgery_status(schedule, current_time_slot, time_slots_list):
    """Update surgery status for current day"""
    time_slot_to_idx = {slot: idx for idx, slot in enumerate(time_slots_list)}

    if current_time_slot not in time_slot_to_idx:
        return schedule

    current_idx = time_slot_to_idx[current_time_slot]

    updated_schedule = []
    for surgery in schedule:
        updated_surgery = surgery.copy()
        start_time = surgery.get('start_time')
        actual_duration = surgery.get('actual_duration', 1)

        _, start_slot_idx = parse_time_slot(start_time)
        if start_slot_idx is None:
            updated_surgery['status'] = 'waiting'
            updated_schedule.append(updated_surgery)
            continue

        end_idx = start_slot_idx + actual_duration - 1

        if start_slot_idx <= current_idx + 1 <= end_idx:
            updated_surgery['status'] = 'in_progress'
        elif current_idx + 1 > end_idx:
            updated_surgery['status'] = 'completed'
        else:
            updated_surgery['status'] = 'waiting'

        updated_schedule.append(updated_surgery)

    waiting_patients = [s['patient_id'] for s in updated_schedule if s['status'] == 'waiting']
    in_progress_patients = [s['patient_id'] for s in updated_schedule if s['status'] == 'in_progress']
    completed_patients = [s['patient_id'] for s in updated_schedule if s['status'] == 'completed']

    print(f"Time slot {current_time_slot} status:")
    print(f"  Waiting: {waiting_patients}")
    print(f"  In Progress: {in_progress_patients}")
    print(f"  Completed: {completed_patients}")
    print()

    return updated_schedule


def detect_conflicts_at_time(schedule, current_time_slot, time_slots_list, current_day):
    """Detect resource conflicts at current time slot"""
    day_surgeries = [s for s in schedule if parse_time_slot(s.get('start_time', ''))[0] == current_day]
    updated_day_surgeries = update_surgery_status(day_surgeries, current_time_slot, time_slots_list)

    pid_to_status = {s['patient_id']: s['status'] for s in updated_day_surgeries}
    updated_schedule = []
    for surgery in schedule:
        s_copy = surgery.copy()
        pid = s_copy.get('patient_id')
        if pid in pid_to_status:
            s_copy['status'] = pid_to_status[pid]
        updated_schedule.append(s_copy)

    in_progress = [s for s in updated_schedule if s.get('status') == 'in_progress'
                   and parse_time_slot(s.get('start_time', ''))[0] == current_day]

    room_occupancy = {}
    doctor_occupancy = {}
    conflicts = []

    for surgery in in_progress:
        room_id = surgery.get('room_id')
        doctor_id = surgery.get('doctor_id')
        patient_id = surgery.get('patient_id')

        if room_id is not None:
            if room_id in room_occupancy:
                conflicts.append({
                    'type': 'room_conflict',
                    'room_id': room_id,
                    'time_slot': current_time_slot,
                    'patient1': patient_id,
                    'patient2': room_occupancy[room_id].get('patient_id'),
                    'surgery1': surgery,
                    'surgery2': room_occupancy[room_id]
                })
            else:
                room_occupancy[room_id] = surgery

        if doctor_id is not None:
            if doctor_id in doctor_occupancy:
                conflicts.append({
                    'type': 'doctor_conflict',
                    'doctor_id': doctor_id,
                    'time_slot': current_time_slot,
                    'patient1': patient_id,
                    'patient2': doctor_occupancy[doctor_id].get('patient_id'),
                    'surgery1': surgery,
                    'surgery2': doctor_occupancy[doctor_id]
                })
            else:
                doctor_occupancy[doctor_id] = surgery

    return conflicts, updated_schedule

def get_waiting_surgeries(schedule):
    """Get surgeries in waiting status"""
    return [s for s in schedule if s.get('status') == 'waiting']

def get_active_surgeries(schedule):
    """Get surgeries in progress"""
    return [s for s in schedule if s.get('status') == 'in_progress']

def get_completed_surgeries(schedule):
    """Get completed surgeries"""
    return [s for s in schedule if s.get('status') == 'completed']

def get_conflict_patients_ids(conflicts):
    """Extract patient IDs from conflicts"""
    patient_ids = set()
    for conflict in conflicts:
        patient_ids.add(conflict.get('patient1'))
        patient_ids.add(conflict.get('patient2'))
    return list(patient_ids)

def calculate_actual_end_time(start_time, actual_duration, time_slots_list):
    """Calculate actual end time"""
    time_slot_to_idx = {slot: idx for idx, slot in enumerate(time_slots_list)}
    start_idx = time_slot_to_idx.get(start_time, -1)

    if start_idx == -1:
        return None, []

    end_idx = min(start_idx + actual_duration - 1, len(time_slots_list) - 1)
    end_time = time_slots_list[end_idx]

    time_slots_used = []
    for i in range(start_idx, min(start_idx + actual_duration, len(time_slots_list))):
        time_slots_used.append(time_slots_list[i])

    return end_time, time_slots_used


def dynamic_time_progression_scheduling(base_schedule, time_slots_list, room_availability, doctor_availability,
                                        patients_df, time_slot_indices, d_r_t_indices, doctor_ids, room_ids,
                                        variation_range=(-0.3, 0.5), seed=42):
    """Dynamic time progression scheduling algorithm"""
    print("\n=== Starting Dynamic Scheduling ===")

    np.random.seed(seed)
    random.seed(seed)

    schedule_with_actual = simulate_surgery_progression(
        base_schedule,
        time_slots_list,
        variation_range=variation_range,
        seed=seed
    )

    working_schedule = [surgery.copy() for surgery in schedule_with_actual]
    current_surgeries = []
    completed_surgeries = []

    patient_id_column = patients_df.columns[0]
    patient_id_to_idx = {patients_df.iloc[i][patient_id_column]: i for i in range(len(patients_df))}
    patient_actual_durations = {s['patient_id']: s['actual_duration'] for s in schedule_with_actual}

    adjustment_events = []

    def parse_time_slot(ts_str):
        parts = ts_str.split('-')
        day = int(parts[0][1:])
        slot_idx = int(parts[1])
        return day, slot_idx

    unique_days = sorted(set(parse_time_slot(ts)[0] for ts in time_slots_list))

    for current_day in unique_days:
        day_time_slots = [ts for ts in time_slots_list if parse_time_slot(ts)[0] == current_day]
        time_slot_to_idx_day = {slot: idx for idx, slot in enumerate(day_time_slots)}

        print(f"\n>>> Day {current_day}, Total {len(day_time_slots)} time slots")

        for t_idx, current_time_slot in enumerate(day_time_slots):
            print(f"  Time slot: {current_time_slot} ({t_idx + 1}/{len(day_time_slots)})")

            progression_map = create_time_progression_map(working_schedule, time_slots_list, current_day)

            starting_surgeries = progression_map.get(current_time_slot, {}).get('start', [])
            if starting_surgeries:
                print(f"    {len(starting_surgeries)} surgeries starting")
                for surgery in starting_surgeries:
                    patient_id = surgery.get('patient_id')
                    for work_surgery in working_schedule:
                        if work_surgery.get('patient_id') == patient_id and work_surgery.get('status') == 'waiting':
                            work_surgery['status'] = 'in_progress'
                            current_surgeries.append(work_surgery)
                            print(f"    - Patient {patient_id} starting surgery")
                            break

            conflicts, updated_schedule = detect_conflicts_at_time(
                working_schedule, current_time_slot, day_time_slots, current_day)

            pid_to_status = {s['patient_id']: s['status'] for s in updated_schedule}
            for surgery in working_schedule:
                pid = surgery.get('patient_id')
                if pid in pid_to_status:
                    surgery['status'] = pid_to_status[pid]

            current_surgeries = [s for s in working_schedule if s.get('status') == 'in_progress']

            if conflicts:
                print(f"    Detected {len(conflicts)} conflicts")
                conflict_patient_ids = get_conflict_patients_ids(conflicts)
                print(f"    Involving patients: {conflict_patient_ids}")

                surgeries_to_reschedule = []
                surgeries_to_keep = []

                def get_start_idx(surgery):
                    st = surgery.get('start_time')
                    if st is None:
                        return -1
                    day, slot_idx = parse_time_slot(st)
                    return slot_idx

                for conflict in conflicts:
                    surgery1 = conflict['surgery1']
                    surgery2 = conflict['surgery2']

                    idx1 = get_start_idx(surgery1)
                    idx2 = get_start_idx(surgery2)

                    if idx1 > idx2:
                        surgery_to_move = surgery1
                        surgery_to_keep = surgery2
                    elif idx2 > idx1:
                        surgery_to_move = surgery2
                        surgery_to_keep = surgery1
                    else:
                        p1 = surgery1.get('priority', 1)
                        p2 = surgery2.get('priority', 1)
                        if p1 < p2:
                            surgery_to_move = surgery1
                            surgery_to_keep = surgery2
                        else:
                            surgery_to_move = surgery2
                            surgery_to_keep = surgery1

                    pid_move = surgery_to_move.get('patient_id')
                    pid_keep = surgery_to_keep.get('patient_id')

                    if pid_move not in [s.get('patient_id') for s in surgeries_to_reschedule]:
                        surgeries_to_reschedule.append(surgery_to_move)
                        for i, s in enumerate(current_surgeries):
                            if s.get('patient_id') == pid_move:
                                s['status'] = 'waiting'
                                current_surgeries.pop(i)
                                print(f"    - Patient {pid_move} set to waiting, preparing for rescheduling")
                                break

                    if pid_keep not in [s.get('patient_id') for s in surgeries_to_keep]:
                        surgeries_to_keep.append(surgery_to_keep)
                        print(f"    - Patient {pid_keep} continues")

                # Reschedule waiting patients
                waiting_patients = [s for s in working_schedule if
                                    s.get('status') == 'waiting' and parse_time_slot(s.get('start_time'))[
                                        0] == current_day]
                reschedule_patient_ids = [s.get('patient_id') for s in waiting_patients]

                if reschedule_patient_ids:
                    reschedule_patient_indices = [patient_id_to_idx.get(pid) for pid in reschedule_patient_ids if
                                                  pid in patient_id_to_idx]

                    if reschedule_patient_indices:
                        print(f"    Rescheduling {len(reschedule_patient_indices)} waiting patients")

                        reschedule_patients_df = patients_df.iloc[reschedule_patient_indices].copy().reset_index(
                            drop=True)

                        updated_room_availability = {k: v.copy() for k, v in room_availability.items()}
                        updated_doctor_availability = {k: v.copy() for k, v in doctor_availability.items()}

                        current_t_idx = time_slot_to_idx_day[current_time_slot]

                        for surgery in working_schedule:
                            room_id = surgery.get('room_id')
                            doctor_id = surgery.get('doctor_id')
                            start_time = surgery.get('start_time')
                            status = surgery.get('status')

                            if room_id is None or doctor_id is None or start_time is None:
                                continue

                            surgery_day, start_idx = parse_time_slot(start_time)
                            duration = surgery.get('actual_duration', surgery.get('duration', 1))

                            if surgery_day != current_day:
                                for i, ts in enumerate(time_slots_list):
                                    ts_day, _ = parse_time_slot(ts)
                                    if ts_day == surgery_day:
                                        if room_id in updated_room_availability and i < len(
                                                updated_room_availability[room_id]):
                                            updated_room_availability[room_id][i] = 0
                                        if doctor_id in updated_doctor_availability and i < len(
                                                updated_doctor_availability[doctor_id]):
                                            updated_doctor_availability[doctor_id][i] = 0

                        for surgery in working_schedule:
                            room_id = surgery.get('room_id')
                            doctor_id = surgery.get('doctor_id')
                            start_time = surgery.get('start_time')
                            status = surgery.get('status')

                            if room_id is None or doctor_id is None or start_time is None:
                                continue

                            surgery_day, start_idx = parse_time_slot(start_time)
                            duration = surgery.get('actual_duration', surgery.get('duration', 1))

                            if surgery_day == current_day and status in ['in_progress', 'completed']:
                                if start_idx == -1:
                                    continue
                                start_mark = max(start_idx, 0)
                                end_mark = min(start_idx + duration, len(day_time_slots))
                                for i in range(start_mark, end_mark):
                                    global_time_idx = time_slots_list.index(day_time_slots[i]) if day_time_slots[
                                                                                                      i] in time_slots_list else -1
                                    if global_time_idx >= 0:
                                        if room_id in updated_room_availability and global_time_idx < len(
                                                updated_room_availability[room_id]):
                                            updated_room_availability[room_id][global_time_idx] = 0
                                        if doctor_id in updated_doctor_availability and global_time_idx < len(
                                                updated_doctor_availability[doctor_id]):
                                            updated_doctor_availability[doctor_id][global_time_idx] = 0

                        for surgery in working_schedule:
                            pid = surgery.get('patient_id')
                            if pid in reschedule_patient_ids:
                                room_id = surgery.get('room_id')
                                doctor_id = surgery.get('doctor_id')
                                start_time = surgery.get('start_time')
                                duration = surgery.get('duration', 1)

                                surgery_day, start_idx = parse_time_slot(start_time)
                                if surgery_day != current_day or start_idx == -1:
                                    continue

                                end_idx = min(start_idx + duration - 1, len(day_time_slots) - 1)

                                for t_idx_day in range(start_idx, end_idx + 1):
                                    if t_idx_day < len(day_time_slots):
                                        time_slot_str = day_time_slots[t_idx_day]
                                        global_time_idx = time_slots_list.index(
                                            time_slot_str) if time_slot_str in time_slots_list else -1
                                        if global_time_idx >= 0:
                                            if room_id in updated_room_availability and global_time_idx < len(
                                                    updated_room_availability[room_id]):
                                                updated_room_availability[room_id][global_time_idx] = 1
                                            if doctor_id in updated_doctor_availability and global_time_idx < len(
                                                    updated_doctor_availability[doctor_id]):
                                                updated_doctor_availability[doctor_id][global_time_idx] = 1

                        consistent_doctor_ids = list(updated_doctor_availability.keys())
                        consistent_room_ids = list(updated_room_availability.keys())

                        # Reconstruct indices
                        n_doctors_sub = len(consistent_doctor_ids)
                        n_rooms_sub = len(consistent_room_ids)
                        n_time_slots_sub = len(time_slots_list)

                        sub_d_r_t_indices = [(d, r, t) for d in range(n_doctors_sub)
                                             for r in range(n_rooms_sub)
                                             for t in range(n_time_slots_sub)]

                        print(
                            f"    Subproblem: Doctors {n_doctors_sub}, Operating rooms {n_rooms_sub}, Time slots {n_time_slots_sub}")

                        consistent_time_slot_indices = {ts: idx for idx, ts in enumerate(time_slots_list)}

                        doctor_id_column = reschedule_patients_df.columns[2]
                        required_doctors = set(reschedule_patients_df[doctor_id_column].unique())
                        available_doctors = set(consistent_doctor_ids)
                        missing_doctors = required_doctors - available_doctors

                        if missing_doctors:
                            print(f"    Missing doctors {missing_doctors}")
                        else:
                            print(f"    All doctors available")

                        # Build cost matrix
                        sub_cost_matrix = wasserstein_sinkhorn.build_cost_matrix(
                            reschedule_patients_df,
                            consistent_time_slot_indices,
                            updated_doctor_availability,
                            updated_room_availability,
                            time_slots_list
                        )

                        if sub_cost_matrix is not None and not np.all(np.isinf(sub_cost_matrix)):
                            n_reschedule_patients = len(reschedule_patients_df)
                            reg = 0.2
                            numItermax = 3000

                            sub_hard_assignment_matrix, sub_hard_assignment_indices = wasserstein_sinkhorn.solve_transport_problem(
                                sub_cost_matrix,
                                n_reschedule_patients,
                                sub_d_r_t_indices,
                                consistent_doctor_ids,
                                consistent_room_ids,
                                time_slots_list,
                                reschedule_patients_df,
                                reg,
                                numItermax,
                                delta_duration=0
                            )

                            sub_schedule, sub_assigned_indices = report.generate_schedule(
                                reschedule_patients_df,
                                sub_hard_assignment_indices,
                                sub_d_r_t_indices,
                                consistent_time_slot_indices,
                                consistent_doctor_ids,
                                consistent_room_ids,
                                time_slots_list
                            )

                            if sub_schedule:
                                print(f"    Generated {len(sub_schedule)} new schedules")

                                # Validate doctor assignments
                                validation_passed = True
                                for item in sub_schedule:
                                    patient_id = item['patient_id']
                                    assigned_doctor = item['doctor_id']

                                    patient_row = reschedule_patients_df[
                                        reschedule_patients_df.iloc[:, 0] == patient_id]
                                    if not patient_row.empty:
                                        required_doctor = patient_row.iloc[0, 2]
                                        if assigned_doctor != required_doctor:
                                            print(
                                                f"    Doctor assignment error: Patient {patient_id} needs {required_doctor}, assigned {assigned_doctor}")
                                            validation_passed = False
                                        else:
                                            print(
                                                f"    Doctor assignment correct: Patient {patient_id} assigned to {assigned_doctor}")

                                if validation_passed:
                                    new_schedule_map = {item['patient_id']: item for item in sub_schedule}

                                    # Update schedule
                                    for i, surgery in enumerate(working_schedule):
                                        pid = surgery.get('patient_id')
                                        if pid in new_schedule_map and surgery.get('status') == 'waiting':
                                            new_surgery = new_schedule_map[pid]
                                            actual_duration = patient_actual_durations.get(pid,
                                                                                           new_surgery.get('duration',
                                                                                                           1))
                                            actual_end_time, actual_time_slots_used = calculate_actual_end_time(
                                                new_surgery.get('start_time'),
                                                actual_duration,
                                                time_slots_list
                                            )

                                            adj_event = {
                                                'time_slot': current_time_slot,
                                                'patient_id': pid,
                                                'original_room': surgery.get('room_id'),
                                                'original_start': surgery.get('start_time'),
                                                'new_room': new_surgery.get('room_id'),
                                                'new_start': new_surgery.get('start_time'),
                                                'reason': 'Conflict adjustment'
                                            }
                                            adjustment_events.append(adj_event)

                                            surgery['room_id'] = new_surgery.get('room_id')
                                            surgery['start_time'] = new_surgery.get('start_time')
                                            surgery['end_time'] = actual_end_time
                                            surgery['time_slots_used'] = actual_time_slots_used
                                            surgery['actual_duration'] = actual_duration
                                            surgery['adjustment_type'] = 'Dynamic adjustment'
                                            surgery['adjustment_time'] = current_time_slot

                                            print(
                                                f"    - Patient {pid} adjusted: {adj_event['original_room']}/{adj_event['original_start']} -> {adj_event['new_room']}/{adj_event['new_start']}")
                                            print(f"      Duration: {actual_duration}, End: {actual_end_time}")

                                    progression_map = create_time_progression_map(working_schedule, time_slots_list,
                                                                                  current_day)
                                else:
                                    print("    Doctor assignment validation failed, skipping adjustment")
                            else:
                                print("    Unable to generate new schedule")
                        else:
                            print("    Cost matrix has no solution")
                    else:
                        print("    No patient indices")
                else:
                    print("    No waiting patients need rescheduling")

            ending_surgeries = progression_map.get(current_time_slot, {}).get('end', [])
            if ending_surgeries:
                print(f"    {len(ending_surgeries)} surgeries scheduled to end")
                for surgery in ending_surgeries:
                    pid = surgery.get('patient_id')
                    for i, work_surgery in enumerate(current_surgeries):
                        if work_surgery.get('patient_id') == pid and work_surgery.get('status') == 'in_progress':
                            work_surgery['status'] = 'completed'
                            completed_surgeries.append(work_surgery)
                            current_surgeries.pop(i)
                            print(f"    - Patient {pid} surgery completed")
                            break

    final_schedule = []
    for surgery in working_schedule:
        surgery_copy = surgery.copy()
        if 'adjustment_type' not in surgery_copy:
            surgery_copy['adjustment_type'] = 'No adjustment'

        actual_duration = surgery_copy.get('actual_duration', surgery_copy.get('duration', 1))
        start_time = surgery_copy.get('start_time')
        if start_time:
            actual_end_time, actual_time_slots_used = calculate_actual_end_time(
                start_time,
                actual_duration,
                time_slots_list
            )
            surgery_copy['end_time'] = actual_end_time
            surgery_copy['time_slots_used'] = actual_time_slots_used

        final_schedule.append(surgery_copy)

    print(f"\n--- Scheduling Adjustment Statistics ---")
    print(f"Total adjustments made: {len(adjustment_events)}")

    final_schedule.sort(key=lambda x: x.get('patient_id'))

    return final_schedule


def run_dynamic_scheduling_layer(base_schedule, time_slots_list, room_availability, doctor_availability,
                                 patients_df, time_slot_indices, d_r_t_indices, doctor_ids, room_ids,
                                 variation_range=(-0.3, 0.5), seed=42):
    """Run dynamic scheduling layer with overtime extension support"""
    print("\n============== Running Dynamic Scheduling Layer ==============")

    updated_room_availability = {k: v.copy() for k, v in room_availability.items()}
    updated_doctor_availability = {k: v.copy() for k, v in doctor_availability.items()}

    final_schedule = dynamic_time_progression_scheduling(
        base_schedule=base_schedule,
        time_slots_list=time_slots_list,
        room_availability=updated_room_availability,
        doctor_availability=updated_doctor_availability,
        patients_df=patients_df,
        time_slot_indices=time_slot_indices,
        d_r_t_indices=d_r_t_indices,
        doctor_ids=doctor_ids,
        room_ids=room_ids,
        variation_range=variation_range,
        seed=seed
    )

    time_progression_report = report.generate_time_progression_report(
        final_schedule,
        patients_df,
        base_schedule
    )
    return final_schedule, time_progression_report

def main():
    base_path = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'
    batch_count = 20

    output_dir = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize trackers
    global_tracker = OvertimeExtensionTracker()
    overtime_evaluator = EnhancedOvertimeEvaluator()

    time_slots, patients_batches, doctors_batches, operating_rooms = data_loader.load_all_data(base_path, batch_count)

    if time_slots.empty or operating_rooms.empty:
        print("Error: Unable to load data, program exiting")
        return

    time_slot_column = time_slots.columns[0]
    time_slots_list = time_slots[time_slot_column].unique().tolist()

    # Divide training and validation batches
    all_batch_keys = sorted(patients_batches.keys(), key=data_loader.batch_key_sort_key)
    historical_keys = [k for k in all_batch_keys if 1 <= data_loader.batch_key_sort_key(k) <= 16]
    validation_keys = [k for k in all_batch_keys if data_loader.batch_key_sort_key(k) >= 17]

    print(f"Historical batches: {len(historical_keys)}, Validation batches: {len(validation_keys)}")

    # Extract historical surgery durations
    train_durations = []
    for key in historical_keys:
        df = patients_batches.get(key, None)
        if df is not None and not df.empty and 'Duration' in df.columns:
            durations = df['Duration'].dropna().values
            train_durations.append(durations)
        else:
            print(f"Batch {key} data missing, skipping")

    if train_durations:
        train_durations = np.concatenate(train_durations)
        print(f"Total historical samples: {len(train_durations)}")
        print(f"Mean: {train_durations.mean():.2f}, Std: {train_durations.std():.2f}")
    else:
        train_durations = np.array([])
        print("Warning: No historical samples")

    epsilon = 0.1
    delta_duration = wasserstein_uncertainty.compute_dynamic_delta_duration(train_durations, epsilon)

    # Collect scheduling results
    all_base_schedules = []
    all_final_schedules = []
    all_batch_names = []

    for batch_key in validation_keys:
        print(f"\n===== Processing Batch {batch_key} =====")
        patients = patients_batches.get(batch_key, None)
        doctors = doctors_batches.get(batch_key, None)

        if patients is None or patients.empty:
            continue
        if doctors is None or doctors.empty:
            continue

        # Preprocess data
        ts_idx, doctor_availability, room_availability, ts_list = data_loader.preprocess_data(
            time_slots, patients, doctors, operating_rooms)
        if ts_idx is None:
            print(f"Batch {batch_key} preprocessing failed, skipping")
            continue

        doctor_ids = list(doctor_availability.keys())
        room_ids = list(room_availability.keys())
        n_doctors = len(doctor_ids)
        n_rooms = len(room_ids)
        n_time_slots = len(ts_list)
        d_r_t_indices = [(d, r, t) for d in range(n_doctors)
                         for r in range(n_rooms)
                         for t in range(n_time_slots)]

        hard_assignment_matrix, hard_assignment_indices, final_time_slots, batch_tracker = run_scheduling_with_overtime_enhanced(
            patients=patients,
            ts_idx=ts_idx,
            doctor_availability=doctor_availability,
            room_availability=room_availability,
            time_slots_list=ts_list,
            doctor_ids=doctor_ids,
            room_ids=room_ids,
            delta_duration=delta_duration,
            max_overtime_blocks=40,
            overtime_step=2,
            reg=0.1,
            numItermax=2000,
            batch_name=batch_key,
            tracker=global_tracker
        )

        if hard_assignment_indices is None:
            print(f"Batch {batch_key} solving failed, skipping")
            continue

        schedule, assigned_patient_indices = report.generate_schedule(
            patients, hard_assignment_indices,
            [(d, r, t) for d in range(len(doctor_ids))
             for r in range(len(room_ids))
             for t in range(len(final_time_slots))],
            ts_idx, doctor_ids, room_ids, final_time_slots)

        report_df = report.generate_report(schedule, patients, assigned_patient_indices)

        report_file = os.path.join(output_dir, f"{batch_key}_Basic_schedule_report.xlsx")
        try:
            report_df.to_excel(report_file, index=False, engine='openpyxl')
            print(f"Basic scheduling report saved: {os.path.abspath(report_file)}")
        except Exception as e:
            print(f"Error saving basic report: {e}")

        fig = scheduling_visualisation.plot_gantt_chart(schedule, final_time_slots,
                                                        title=f"{batch_key}_Basic_surgical_schedule_gantt_chart")

        if fig is not None:
            gantt_file = os.path.join(output_dir, f"{batch_key}_Basic_surgical_schedule_gantt_chart.svg")
            try:
                fig.savefig(gantt_file, dpi=300, bbox_inches='tight')
                print(f"Basic Gantt chart saved: {os.path.abspath(gantt_file)}")
            except Exception as e:
                print(f"Error saving basic Gantt chart: {e}")

        # Dynamic scheduling
        final_dynamic_schedule, time_progression_report = run_dynamic_scheduling_layer(
            base_schedule=schedule,
            time_slots_list=final_time_slots,
            room_availability=room_availability,
            doctor_availability=doctor_availability,
            patients_df=patients,
            time_slot_indices=ts_idx,
            d_r_t_indices=d_r_t_indices,
            doctor_ids=doctor_ids,
            room_ids=room_ids,
            variation_range=(-0.3, 0.5),
            seed=42
        )

        # Save dynamic scheduling report
        dynamic_report_df = report.generate_time_progression_report(final_dynamic_schedule, patients, schedule)
        dynamic_report_file = os.path.join(output_dir, f"{batch_key}_Dynamic_schedule_report.xlsx")
        try:
            dynamic_report_df.to_excel(dynamic_report_file, index=False, engine='openpyxl')
            print(f"Dynamic scheduling report saved: {dynamic_report_file}")
        except Exception as e:
            print(f"Error saving dynamic report: {e}")

        # Save dynamic Gantt chart
        fig_dynamic = scheduling_visualisation.plot_adjusted_gantt_chart(final_dynamic_schedule, final_time_slots,
                                                                         title=f"{batch_key} Surgical Scheduling Gantt Chart (Adjusted)")
        if fig_dynamic:
            dynamic_gantt_file = os.path.join(output_dir, f"{batch_key}_Dynamic_schedule_gantt.svg")
            try:
                fig_dynamic.savefig(dynamic_gantt_file, dpi=300, bbox_inches='tight')
                print(f"Dynamic Gantt chart saved: {dynamic_gantt_file}")
            except Exception as e:
                print(f"Error saving dynamic Gantt chart: {e}")

        # Collect results
        all_base_schedules.append(schedule)
        all_final_schedules.append(final_dynamic_schedule)
        all_batch_names.append(batch_key)

        daily_cost_result, figures = evaluate_basic_costs_by_day_with_visualization(
            final_schedule=final_dynamic_schedule,
            patients_data=patients,
            time_slots_list=final_time_slots,
            output_dir=output_dir,
            batch_name=batch_key,
            generate_plots=False
        )

        print_daily_cost_report(daily_cost_result, batch_name=batch_key)

        daily_cost_file = os.path.join(output_dir, f"{batch_key}_Daily_Cost_Evaluation.xlsx")
        save_daily_cost_results_to_excel(daily_cost_result, daily_cost_file, batch_name=batch_key)

        try:
            print(f"\nStarting two-layer scheduling comparison analysis...")

            comparison_metrics = compare_two_layer_schedules(
                base_schedule=schedule,
                final_schedule=final_dynamic_schedule,
                patients_data=patients,
                time_slots_list=final_time_slots,
                output_dir=output_dir,
                batch_name=batch_key,
            )

            print(f"Batch {batch_key} comparison analysis completed!")
            if 'error' not in comparison_metrics:
                print(f"   - Assignment rate improvement: {comparison_metrics['assignment_improvement']:.2%}")
                print(f"   - System efficiency: {comparison_metrics['two_layer_efficiency']:.2f}")
                print(f"   - Adjusted patients count: {comparison_metrics['adjusted_patients_count']}")

        except Exception as e:
            print(f"Two-layer comparison analysis error: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(1)

    print(f"\n" + "=" * 80)
    print("              Starting Overtime Extension Evaluation")
    print("=" * 80)

    try:
        overtime_evaluation_results = overtime_evaluator.evaluate_overtime_extension_process(
            tracker=global_tracker,
            output_dir=output_dir,
            save_plots=False
        )

        print_overall_overtime_summary(overtime_evaluation_results)
        save_overtime_comprehensive_report(overtime_evaluation_results, output_dir)

        print("Overtime extension evaluation completed")

    except Exception as e:
        print(f"Overtime extension evaluation error: {e}")
        import traceback
        traceback.print_exc()

    try:
        print(f"\nStarting comprehensive evaluation...")
        comprehensive_report = run_comprehensive_evaluation(
            base_schedules=all_base_schedules,
            final_schedules=all_final_schedules,
            train_durations=train_durations,
            time_slots_list=final_time_slots,
            batch_names=all_batch_names,
            output_dir=output_dir
        )
        print("Comprehensive evaluation completed")

    except Exception as e:
        print(f"Comprehensive evaluation error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("              All Batch Scheduling Completed")
    print("=" * 80)

def print_overall_overtime_summary(evaluation_results):
    """Print overall overtime extension evaluation summary"""
    print(f"\nOvertime Extension Overall Evaluation:")
    print(f"=" * 60)

    cross_batch = evaluation_results.get('cross_batch_statistics', {})
    summary = cross_batch.get('summary_statistics', {})

    if summary:
        print(f"Average final assignment rate: {summary.get('mean_final_assignment_rate', 0):.2%}")
        print(f"Average assignment rate improvement: {summary.get('mean_improvement', 0):.2%}")
        print(f"Average extension rounds: {summary.get('mean_extensions_needed', 0):.1f}")
        print(f"Average cost effectiveness: {summary.get('mean_cost_effectiveness', 0):.4f}")

    performance = cross_batch.get('performance_consistency', {})
    if performance:
        print(f"Assignment rate consistency: {performance.get('assignment_rate_consistency', 0):.2%}")
        print(f"Extension rounds consistency: {performance.get('extension_consistency', 0):.2%}")

    if cross_batch.get('best_performing_batch'):
        print(f"Best performing batch: {cross_batch['best_performing_batch']}")
    if cross_batch.get('most_efficient_batch'):
        print(f"Most efficient batch: {cross_batch['most_efficient_batch']}")

def save_overtime_comprehensive_report(evaluation_results, output_dir):
    """Save overtime extension comprehensive report"""
    try:
        report_path = os.path.join(output_dir, 'Enhanced_Overtime_Extension_Comprehensive_Report.xlsx')

        with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
            executive_summary = []
            cross_batch = evaluation_results.get('cross_batch_statistics', {})
            summary_stats = cross_batch.get('summary_statistics', {})

            executive_summary.append({
                'Metric Category': 'Assignment Effect',
                'Metric Name': 'Average Final Assignment Rate',
                'Value': f"{summary_stats.get('mean_final_assignment_rate', 0):.2%}",
                'Description': 'Average patient assignment rate achieved across all batches'
            })

            executive_summary.append({
                'Metric Category': 'Improvement Effect',
                'Metric Name': 'Average Assignment Rate Improvement',
                'Value': f"{summary_stats.get('mean_improvement', 0):.2%}",
                'Description': 'Average assignment rate improvement achieved through overtime extension'
            })

            executive_summary.append({
                'Metric Category': 'Resource Consumption',
                'Metric Name': 'Average Extension Rounds',
                'Value': f"{summary_stats.get('mean_extensions_needed', 0):.1f}",
                'Description': 'Average extension rounds needed to achieve final assignment rate'
            })

            executive_summary.append({
                'Metric Category': 'Cost Effectiveness',
                'Metric Name': 'Average Cost Effectiveness Ratio',
                'Value': f"{summary_stats.get('mean_cost_effectiveness', 0):.4f}",
                'Description': 'Average ratio of assignment rate improvement to overtime cost'
            })

            pd.DataFrame(executive_summary).to_excel(writer, sheet_name='Executive Summary', index=False)

        print(f"Overtime extension comprehensive report saved: {report_path}")

    except Exception as e:
        print(f"Error saving comprehensive report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()