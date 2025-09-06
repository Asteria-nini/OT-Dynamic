import pandas as pd
import numpy as np
import re
import os
import hashlib
import itertools
from datetime import datetime, timedelta


def process_csv_to_excel(csv_file_path, output_excel_path):
    # Parse CSV and generate initial doctor and patient data Excel
    with open(csv_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Parse surgeon-patient list data
    surgeon_patient_data = []
    for line in lines:
        if "List for Surgeon" in line:
            surgeon_id, patient_ids = line.split(":")
            surgeon_id = surgeon_id.strip().split()[-1]
            patient_ids = patient_ids.strip().split()
            patient_ids = ' '.join(patient_ids)
            surgeon_patient_data.append([surgeon_id, patient_ids])
    surgeon_patient_df = pd.DataFrame(surgeon_patient_data, columns=["SurgeonID", "PatientsID"])

    # Parse surgeon requested time data
    surgeon_time_data = []
    for line in lines:
        if "Requested time Surgeon" in line:
            match = re.match(
                r"Requested time Surgeon (\d+) \(elective, addon\): Time\((\d+), (\d+)\) #\((\d+), (\d+)\)", line)
            if match:
                surgeon_id = match.group(1)
                elective_time = match.group(2)
                emergency_time = match.group(3)
                elective_count = match.group(4)
                emergency_count = match.group(5)
                surgeon_time_data.append([surgeon_id, elective_time, emergency_time, elective_count, emergency_count])
    surgeon_time_df = pd.DataFrame(surgeon_time_data,
                                   columns=["SurgeonID", "ElectiveTime", "EmergencyTime", "ElectiveCount",
                                            "EmergencyCount"])

    # Parse patient duration data
    patient_duration_data = []
    for line in lines:
        if "Duration pat" in line:
            parts = line.split(":")
            patient_id = parts[0].strip().split()[-1]
            duration = parts[1].strip() if len(parts) > 1 else '0'
            patient_duration_data.append([patient_id, duration])
    patient_duration_df = pd.DataFrame(patient_duration_data, columns=["PatientID", "Duration"])

    # Merge surgeon and time data
    merged_df = pd.merge(surgeon_patient_df, surgeon_time_df, on="SurgeonID", how="outer")

    # Determine emergency level based on emergency count and time
    emergency_level_data = []
    for _, row in merged_df.iterrows():
        surgeon_id = row["SurgeonID"]
        emergency_count = int(row["EmergencyCount"]) if pd.notna(row["EmergencyCount"]) else 0
        emergency_time = int(row["EmergencyTime"]) if pd.notna(row["EmergencyTime"]) else 0
        patient_ids = row["PatientsID"].split() if isinstance(row["PatientsID"], str) else []

        if emergency_count == 0:
            for patient_id in patient_ids:
                emergency_level_data.append([patient_id, 0])
        else:
            patient_durations = []
            for patient_id in patient_ids:
                duration_values = patient_duration_df[patient_duration_df["PatientID"] == patient_id]["Duration"].values
                if duration_values.size > 0:
                    patient_durations.append((patient_id, int(duration_values[0])))
            found_combination = None
            for combination in itertools.combinations(patient_durations, emergency_count):
                total_duration = sum([patient[1] for patient in combination])
                if total_duration == emergency_time:
                    found_combination = [patient[0] for patient in combination]
                    break
            if found_combination:
                for patient_id in patient_ids:
                    if patient_id in found_combination:
                        emergency_level_data.append([patient_id, 1])
                    else:
                        emergency_level_data.append([patient_id, 0])
            else:
                for patient_id in patient_ids:
                    emergency_level_data.append([patient_id, 0])
    emergency_level_df = pd.DataFrame(emergency_level_data, columns=["PatientID", "EmergencyLevel"])

    # Merge patient duration and emergency level
    patient_duration_with_emergency = pd.merge(patient_duration_df, emergency_level_df, on="PatientID", how="left")
    patient_duration_with_emergency = patient_duration_with_emergency.dropna(subset=['Duration', 'EmergencyLevel'])
    patient_duration_with_emergency['Duration'] = pd.to_numeric(patient_duration_with_emergency['Duration'],
                                                                errors='coerce')
    patient_duration_with_emergency = patient_duration_with_emergency[
        (patient_duration_with_emergency['Duration'] >= 0) & (patient_duration_with_emergency['Duration'] <= 48)
        ]

    # Add SurgeonID to patient data
    patient_to_surgeon = {}
    for _, row in surgeon_patient_df.iterrows():
        surgeon_id = row['SurgeonID']
        patient_ids = row['PatientsID'].split() if isinstance(row['PatientsID'], str) else []
        for patient in patient_ids:
            patient_to_surgeon[patient] = surgeon_id
    patient_duration_with_emergency['SurgeonID'] = patient_duration_with_emergency['PatientID'].map(patient_to_surgeon)

    # Save to Excel
    with pd.ExcelWriter(output_excel_path) as writer:
        merged_df.to_excel(writer, sheet_name="Surgeon Info", index=False)
        patient_duration_with_emergency.to_excel(writer, sheet_name="Patient Duration Info", index=False)

    print(f"Data saved to '{output_excel_path}'")
    return output_excel_path


def generate_seed_from_string(s):
    # Generate random seed from string for reproducibility
    return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:8], 16)


def process_patient_time_and_priority(input_excel, output_excel):
    # Add arrival time and priority to patient data
    seed = generate_seed_from_string(input_excel)
    np.random.seed(seed)

    xls = pd.ExcelFile(input_excel)
    df = pd.read_excel(xls, sheet_name='Patient Duration Info')
    df = df.sort_values(by='PatientID')

    num_patients = len(df)
    days_of_week = 5
    patients_per_day = num_patients // days_of_week
    remainder = num_patients % days_of_week

    # Distribute patients across 5 days
    arrival_days = []
    for i in range(days_of_week):
        start_idx = i * patients_per_day
        end_idx = start_idx + patients_per_day
        if i < remainder:
            end_idx += 1
        arrival_days.extend([i + 1] * (end_idx - start_idx))

    df['ArrivalDay'] = arrival_days

    def generate_arrival_times_by_patientid(df_day, elective_low=0.0, elective_high=1.0, expo_scale=2.0):
        df_day = df_day.sort_values(by="PatientID").copy()
        start_time = datetime.strptime("09:00", "%H:%M")
        end_time = datetime.strptime("19:00", "%H:%M")
        total_seconds = (end_time - start_time).seconds

        increments = []
        for _, row in df_day.iterrows():
            if row['EmergencyLevel'] == 0:
                inc = np.random.uniform(elective_low, elective_high)
            else:
                inc = np.random.exponential(scale=expo_scale)
            increments.append(inc)
        increments = np.array(increments)
        cumulative = np.cumsum(increments)

        scaling_factor = total_seconds / (cumulative[-1] - np.random.uniform(0.1, 1))
        offsets = cumulative * scaling_factor

        arrival_times = [start_time + timedelta(seconds=offset) for offset in offsets]
        df_day['ArrivalTime'] = arrival_times

        time_blocks = [((arrival_time - start_time).seconds // (12 * 60)) + 1 for arrival_time in arrival_times]
        df_day['TimeBlock'] = time_blocks

        return df_day

    result_list = []
    days = sorted(df['ArrivalDay'].unique())
    for day in days:
        df_day = df[df['ArrivalDay'] == day].copy()
        if df_day.empty:
            continue
        df_day_updated = generate_arrival_times_by_patientid(df_day)
        result_list.append(df_day_updated)

    df_final = pd.concat(result_list, axis=0, ignore_index=True)
    df_final['ArrivalTime'] = df_final['ArrivalTime'].dt.strftime('%H:%M')

    # Calculate priority score
    df_final['PriorityScore'] = (60 - df_final['TimeBlock']) + (df_final['EmergencyLevel'] == 1) * 50

    df_final.to_excel(output_excel, index=False)
    print(f"Patient arrival time and priority saved to: {output_excel}")


def process_doctor_disturbance(input_excel, output_excel_disturbance, output_excel_unavailable):
    # Generate doctor disturbance times and unavailable periods
    seed_value = generate_seed_from_string(input_excel)
    np.random.seed(seed_value)
    xls = pd.ExcelFile(input_excel)
    surgeon_info_df = pd.read_excel(xls, sheet_name='Surgeon Info')
    num_doctors = len(surgeon_info_df)

    # Select 10%-15% doctors for disturbance
    num_disturbed_doctors = int(np.random.uniform(0.10, 0.15) * num_doctors)
    disturbed_doctors = np.random.choice(surgeon_info_df['SurgeonID'], size=num_disturbed_doctors, replace=False)

    def generate_unique_time_interval(used_intervals):
        while True:
            start_time_candidate = np.random.randint(1, 49)
            time_diff = np.random.randint(15, 28)
            end_time_candidate = start_time_candidate + time_diff
            if end_time_candidate > 60:
                end_time_candidate = 60
                if end_time_candidate - start_time_candidate < 16:
                    continue
            if (start_time_candidate, end_time_candidate) not in used_intervals:
                used_intervals.add((start_time_candidate, end_time_candidate))
                return start_time_candidate, end_time_candidate

    used_intervals = set()
    all_surgeons_with_disturbance_updated = []

    # Assign disturbance times to selected doctors
    for doctor_id in surgeon_info_df['SurgeonID']:
        if doctor_id in disturbed_doctors:
            day = np.random.randint(1, 6)
            start_time, end_time = generate_unique_time_interval(used_intervals)
            all_surgeons_with_disturbance_updated.append({
                'SurgeonID': doctor_id,
                'AffairDay': day,
                'AffairStartTime': start_time,
                'AffairEndTime': end_time
            })
        else:
            all_surgeons_with_disturbance_updated.append({
                'SurgeonID': doctor_id,
                'AffairDay': None,
                'AffairStartTime': None,
                'AffairEndTime': None
            })

    full_disturbance_updated_df = pd.DataFrame(all_surgeons_with_disturbance_updated)

    # Assign 40%-70% doctors to hospital meetings
    num_meeting_doctors = int(np.random.uniform(0.40, 0.70) * num_doctors)
    meeting_doctors = np.random.choice(surgeon_info_df['SurgeonID'], size=num_meeting_doctors, replace=False)
    meeting_day = np.random.randint(1, 6)

    while True:
        meeting_start = np.random.randint(1, 61)
        duration = np.random.randint(8, 17)
        meeting_end = meeting_start + duration
        if meeting_end <= 60:
            break

    full_disturbance_updated_df['MeetingDay'] = None
    full_disturbance_updated_df['MeetingStartTime'] = None
    full_disturbance_updated_df['MeetingEndTime'] = None

    mask = full_disturbance_updated_df['SurgeonID'].isin(meeting_doctors)
    full_disturbance_updated_df.loc[mask, 'MeetingDay'] = meeting_day
    full_disturbance_updated_df.loc[mask, 'MeetingStartTime'] = meeting_start
    full_disturbance_updated_df.loc[mask, 'MeetingEndTime'] = meeting_end

    full_disturbance_updated_df.to_excel(output_excel_disturbance, index=False)

    # Generate doctor unavailable time periods
    doctor_unavailable_times = []

    for idx, row in full_disturbance_updated_df.iterrows():
        doctor_id = row['SurgeonID']
        doctor_label = f'D{doctor_id}'

        # Process personal affair periods
        if pd.notna(row['AffairStartTime']) and pd.notna(row['AffairEndTime']):
            start_time = int(row['AffairStartTime'])
            end_time = int(row['AffairEndTime'])
            affair_day = int(row['AffairDay'])

            start_time_slot = f"T{affair_day}-{start_time}"
            end_time_slot = f"T{affair_day}-{end_time}"

            doctor_unavailable_times.append({
                'doctor_id': doctor_label,
                'Unavailable_Start_Time': start_time_slot,
                'Unavailable_End_Time': end_time_slot
            })

        # Process meeting periods
        if pd.notna(row['MeetingStartTime']) and pd.notna(row['MeetingEndTime']):
            start_time = int(row['MeetingStartTime'])
            end_time = int(row['MeetingEndTime'])
            meeting_day_int = int(row['MeetingDay'])

            start_time_slot = f"T{meeting_day_int}-{start_time}"
            end_time_slot = f"T{meeting_day_int}-{end_time}"

            doctor_unavailable_times.append({
                'doctor_id': doctor_label,
                'Unavailable_Start_Time': start_time_slot,
                'Unavailable_End_Time': end_time_slot
            })

    doctor_unavailable_times_df = pd.DataFrame(doctor_unavailable_times)
    doctor_unavailable_times_df.to_excel(output_excel_unavailable, index=False)

    print(
        f"Doctor disturbance and unavailable times saved to '{output_excel_disturbance}' and '{output_excel_unavailable}'")


def organize_and_save_patients(input_arrival_time_excel, output_patients_excel):
    # Organize patients by priority and save final patient file
    df = pd.read_excel(input_arrival_time_excel)

    patient_data = []
    for day in range(1, 6):
        day_df = df[df['ArrivalDay'] == day]
        sorted_day_df = day_df.sort_values(by='PriorityScore', ascending=False)
        for _, row in sorted_day_df.iterrows():
            surgeon_label = f'D{row["SurgeonID"]}'
            patient_data.append([f'P{row["PatientID"]}', row['Duration'], surgeon_label, row['ArrivalDay']])

    patients_df = pd.DataFrame(patient_data, columns=['PatientID', 'Duration', 'SurgeonID', 'ArrivalDay'])
    patients_df.to_excel(output_patients_excel, index=False)
    print(f"{output_patients_excel} generated")


if __name__ == "__main__":
    base_csv_dir = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\interim\G_1'
    base_output_dir_1 = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\excel\G_1'
    base_output_dir_2 = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'

    os.makedirs(base_output_dir_1, exist_ok=True)
    os.makedirs(base_output_dir_2, exist_ok=True)

    # Process files 1_1DataInfo.csv to 1_20DataInfo.csv
    for i in range(1, 21):
        csv_filename = f"1_{i}DataInfo.csv"
        csv_path = os.path.join(base_csv_dir, csv_filename)

        if not os.path.exists(csv_path):
            print(f"Skip missing file: {csv_path}")
            continue

        cleaned_excel = os.path.join(base_output_dir_1, f"1_{i}DataInfo_Cleaned.xlsx")
        patient_time_output = os.path.join(base_output_dir_1, f"1_{i}DataInfo_Arrival_Time.xlsx")
        doctor_disturbance_output = os.path.join(base_output_dir_1, f"1_{i}Doctor_Availability_with_Disturbance.xlsx")
        doctor_unavailable_output = os.path.join(base_output_dir_2, f"1_{i}Doctor.xlsx")
        final_patients_output = os.path.join(base_output_dir_2, f"1_{i}Patients.xlsx")

        print(f"\nProcessing: {csv_filename}")

        try:
            process_csv_to_excel(csv_path, cleaned_excel)
            process_patient_time_and_priority(cleaned_excel, patient_time_output)
            process_doctor_disturbance(cleaned_excel, doctor_disturbance_output, doctor_unavailable_output)
            organize_and_save_patients(patient_time_output, final_patients_output)
        except Exception as e:
            print(f"Error processing {csv_filename}: {e}")

        print(f"Completed: {csv_filename}")