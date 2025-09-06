import pandas as pd

def generate_schedule(patients, hard_assignment_indices, d_r_t_indices, time_slot_indices, doctor_ids, room_ids, time_slots_list):
    # Generate final surgical schedule based on hard assignment results
    # Returns: schedule (list), assigned_patient_indices (set)
    schedule = []
    assigned_patient_indices = set()

    if patients.empty or hard_assignment_indices is None:
        print("Warning: Empty patient data or invalid assignment indices")
        return [], set()

    patient_id_column = patients.columns[0]
    surgery_duration_column = patients.columns[1]
    doctor_id_column = patients.columns[2]

    n_time_slots = len(time_slots_list)

    if not d_r_t_indices:
         print("Error: Empty d_r_t_indices")
         return [], set()

    print("Generating final schedule from hard assignment results...")
    for p_idx, flat_idx in enumerate(hard_assignment_indices):
        if flat_idx != -1:  # -1 means unassigned
            try:
                # Get (d_idx, r_idx, t_idx) from flat_idx
                if not (0 <= flat_idx < len(d_r_t_indices)):
                    print(f"Warning: Patient {p_idx} assignment index {flat_idx} out of range, skipping")
                    continue
                d_idx, r_idx, t_idx = d_r_t_indices[flat_idx]

                # Get patient info
                patient = patients.iloc[p_idx]
                patient_id = patient[patient_id_column]
                requested_doctor_id = patient[doctor_id_column]

                # Index to ID mapping and validation
                if not (0 <= d_idx < len(doctor_ids)):
                     print(f"Warning: Invalid doctor index {d_idx} for patient {patient_id}, skipping")
                     continue
                scheduled_doctor_id = doctor_ids[d_idx]

                # Verify assigned doctor matches requested doctor
                if scheduled_doctor_id != requested_doctor_id:
                     print(f"Error: Patient {patient_id} (needs doctor {requested_doctor_id}) assigned to wrong doctor {scheduled_doctor_id}, skipping")
                     continue

                if not (0 <= r_idx < len(room_ids)):
                     print(f"Warning: Invalid room index {r_idx} for patient {patient_id}, skipping")
                     continue
                room_id = room_ids[r_idx]

                if not (0 <= t_idx < n_time_slots):
                     print(f"Warning: Invalid time index {t_idx} for patient {patient_id}, skipping")
                     continue
                start_time_slot = time_slots_list[t_idx]

                # Get surgery duration
                try:
                    duration = int(patient[surgery_duration_column])
                    if duration <= 0: raise ValueError("Duration must be positive")
                except (ValueError, TypeError):
                    print(f"Warning: Invalid duration for patient {patient_id}, skipping")
                    continue

                # Calculate end time
                end_t_idx = t_idx + duration - 1
                if end_t_idx >= n_time_slots:
                    print(f"Warning: Patient {patient_id} end time exceeds range, using last slot")
                    end_time_slot = time_slots_list[-1]
                    actual_end_idx = n_time_slots - 1
                else:
                    end_time_slot = time_slots_list[end_t_idx]
                    actual_end_idx = end_t_idx

                # Get all occupied time slots
                surgery_time_slots_indices = list(range(t_idx, actual_end_idx + 1))
                surgery_time_slots = [time_slots_list[i] for i in surgery_time_slots_indices if 0 <= i < n_time_slots]

                # Add to schedule
                schedule.append({
                    'patient_id': patient_id,
                    'doctor_id': scheduled_doctor_id,
                    'room_id': room_id,
                    'start_time': start_time_slot,
                    'end_time': end_time_slot,
                    'duration': duration,
                    'time_slots_used': ', '.join(surgery_time_slots),
                })
                assigned_patient_indices.add(p_idx)

            except Exception as e:
                import traceback
                print(f"Error generating schedule for patient {p_idx}: {e}")
                traceback.print_exc()
                continue

    print(f"Schedule generated: {len(schedule)} surgery records")
    return schedule, assigned_patient_indices


def generate_report(schedule, patients, assigned_patient_indices):
    # Generate surgical scheduling report
    if patients.empty:
        print("Warning: Empty patient data")
        return pd.DataFrame()

    if len(patients.columns) < 4:
        print("Error: Insufficient patient data columns")
        return pd.DataFrame()

    patient_id_column = patients.columns[0]
    arrival_day_column = patients.columns[3]
    doctor_id_column = patients.columns[2]

    schedule_map = {}
    for item in schedule:
        pid = item['patient_id']
        if pid not in schedule_map:
            schedule_map[pid] = item
        else:
            print(f"Warning: Duplicate patient ID '{pid}' in schedule")

    report_data = {
        "Patient ID": [],
        "Arrival Day": [],
        "Surgery Duration": [],
        "Requested Doctor": [],
        "Assigned Room": [],
        "Start Time": [],
        "End Time": [],
        "Status": []
    }

    print("Generating final report...")
    for p_idx, patient in patients.iterrows():
        patient_id = patient[patient_id_column]
        arrival_day = patient[arrival_day_column]
        requested_doctor_id = patient[doctor_id_column]

        report_data["Patient ID"].append(patient_id)
        report_data["Arrival Day"].append(arrival_day)
        report_data["Requested Doctor"].append(requested_doctor_id)

        if p_idx in assigned_patient_indices:
            schedule_entry = schedule_map.get(patient_id)
            if schedule_entry:
                if schedule_entry['doctor_id'] != requested_doctor_id:
                    print(f"Warning: Patient {patient_id} doctor mismatch in report")
                report_data["Assigned Room"].append(schedule_entry["room_id"])
                report_data["Start Time"].append(schedule_entry["start_time"])
                report_data["End Time"].append(schedule_entry["end_time"])
                report_data["Surgery Duration"].append(schedule_entry.get("duration", "---"))
                report_data["Status"].append("Scheduled")
            else:
                print(f"Error: Patient {p_idx} ({patient_id}) missing schedule record")
                report_data["Assigned Room"].append("Error")
                report_data["Start Time"].append("Error")
                report_data["End Time"].append("Error")
                report_data["Surgery Duration"].append("Error")
                report_data["Status"].append("Assignment Error")
        else:
            report_data["Assigned Room"].append("---")
            report_data["Start Time"].append("---")
            report_data["End Time"].append("---")
            report_data["Surgery Duration"].append("---")
            report_data["Status"].append("Unscheduled")

    report_df = pd.DataFrame(report_data)
    print("Report generation complete")
    return report_df


def generate_time_progression_report(final_schedule, patients_df, original_schedule, adjustment_events=None):
    # Generate time series progression report for dynamic scheduling
    # Returns DataFrame without file writing
    print("Generating time progression report...")

    # Create lookup maps for final and original schedules
    final_map = {item['patient_id']: item for item in final_schedule}
    original_map = {item['patient_id']: item for item in original_schedule}

    report_data = {
        "Patient ID": [],
        "Original Room": [],
        "Original Start Time": [],
        "Expected Duration": [],
        "Actual Duration": [],
        "Duration Difference": [],
        "Final Room": [],
        "Final Start Time": [],
        "Final End Time": [],
        "Adjustment Type": [],
        "Adjustment Time": [],
        "Surgery Status": []
    }

    patient_id_column = patients_df.columns[0]

    for patient in patients_df.itertuples():
        patient_id = getattr(patient, patient_id_column)

        # Default values
        original_room = "---"
        original_start = "---"
        expected_duration = "---"
        actual_duration = "---"
        duration_diff = "---"
        final_room = "---"
        final_start = "---"
        final_end = "---"
        adjustment_type = "---"
        adjustment_time = "---"
        status = "Unscheduled"

        # Get original schedule info
        if patient_id in original_map:
            orig_item = original_map[patient_id]
            original_room = orig_item.get('room_id', "---")
            original_start = orig_item.get('start_time', "---")
            expected_duration = orig_item.get('duration', "---")
            status = "Scheduled"

        # Get final schedule info
        if patient_id in final_map:
            final_item = final_map[patient_id]
            final_room = final_item.get('room_id', "---")
            final_start = final_item.get('start_time', "---")
            final_end = final_item.get('end_time', "---")
            actual_duration = final_item.get('actual_duration', "---")

            diff_val = final_item.get('duration_diff', None)
            if diff_val is not None and diff_val != "---":
                duration_diff = f"{'+' if diff_val > 0 else ''}{diff_val}"
            else:
                duration_diff = "---"

            adjustment_type = final_item.get('adjustment_type', "---")
            adjustment_time = final_item.get('adjustment_time', "---")
            status = final_item.get('status', status)

        report_data["Patient ID"].append(patient_id)
        report_data["Original Room"].append(original_room)
        report_data["Original Start Time"].append(original_start)
        report_data["Expected Duration"].append(expected_duration)
        report_data["Actual Duration"].append(actual_duration)
        report_data["Duration Difference"].append(duration_diff)
        report_data["Final Room"].append(final_room)
        report_data["Final Start Time"].append(final_start)
        report_data["Final End Time"].append(final_end)
        report_data["Adjustment Type"].append(adjustment_type)
        report_data["Adjustment Time"].append(adjustment_time)
        report_data["Surgery Status"].append(status)

    return pd.DataFrame(report_data)