import pandas as pd
import numpy as np
import os
import re

def parse_day_from_room_id(room_id):
    # Extract day from room ID format R{day}{room}, e.g., R11 = day 1, room 1
    if isinstance(room_id, str) and room_id.startswith('R') and len(room_id) == 3:
        try:
            return int(room_id[1])
        except ValueError:
            return -1
    return -1

def parse_day_from_time_slot(time_slot):
    # Parse day from time slot ID like T3-1, T10-5
    if isinstance(time_slot, str) and time_slot.startswith('T') and '-' in time_slot:
        parts = time_slot.split('-')
        if len(parts) == 2 and parts[0].startswith('T') and len(parts[0]) > 1:
            try:
                return int(parts[0][1:])
            except ValueError:
                return -1
        return -1
    return -1

def batch_key_sort_key(key):
    # Sort by numeric part in "1_number" format
    m = re.match(r'1_(\d+)', key)
    return int(m.group(1)) if m else 0

def load_all_data(base_path=r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1',
                  batch_count=20):
    # Load patient and doctor data by batch, plus common time_slots and operating_room files
    patients_batches = {}
    doctors_batches = {}

    for i in range(1, batch_count + 1):
        batch_key = f'1_{i}'

        # Load patient data
        patients_file = os.path.join(base_path, f'{batch_key}Patients.xlsx')
        try:
            patients_batches[batch_key] = pd.read_excel(patients_file)
        except (FileNotFoundError, Exception) as e:
            print(f"Skip patients file {batch_key}: {e}")

        # Load doctor data
        doctors_file = os.path.join(base_path, f'{batch_key}Doctor.xlsx')
        try:
            doctors_batches[batch_key] = pd.read_excel(doctors_file)
        except (FileNotFoundError, Exception) as e:
            print(f"Skip doctors file {batch_key}: {e}")

    # Load common data
    try:
        time_slots = pd.read_excel(os.path.join(base_path, 'time_slots.xlsx'))
        operating_rooms = pd.read_excel(os.path.join(base_path, 'operating_room.xlsx'))
    except Exception as e:
        print(f"Error loading common data: {e}")
        return pd.DataFrame(), {}, {}, pd.DataFrame()

    return time_slots, patients_batches, doctors_batches, operating_rooms

def preprocess_data(time_slots, patients, doctors, operating_rooms):
    # Create time slot mappings and resource availability arrays
    # Doctors not in doctor table are considered always available
    if time_slots.empty:
        print("Error: Empty time slots data")
        return None, None, None, None

    time_slot_column = time_slots.columns[0]
    time_slot_list = time_slots[time_slot_column].unique().tolist()
    time_slot_indices = {slot: idx for idx, slot in enumerate(time_slot_list)}
    num_time_slots = len(time_slot_indices)

    if num_time_slots == 0:
        print("Error: No valid time slots")
        return None, None, None, None

    print(f"Found {num_time_slots} time slots")

    # Initialize doctor availability
    doctor_availability = {}
    if not patients.empty and len(patients.columns) >= 3:
        doctor_col = patients.columns[2]
        all_doctors = patients[doctor_col].dropna().unique().tolist()
        print(f"Found {len(all_doctors)} doctors")

        for doctor_id in all_doctors:
            doctor_availability[doctor_id] = np.ones(num_time_slots)

        # Apply doctor unavailability constraints
        if not doctors.empty and len(doctors.columns) >= 3:
            doctor_id_col, start_col, end_col = doctors.columns[:3]

            for _, row in doctors.iterrows():
                doctor_id = row[doctor_id_col]
                u_start, u_end = row[start_col], row[end_col]

                if (doctor_id in doctor_availability and
                        pd.notna(u_start) and pd.notna(u_end) and
                        u_start in time_slot_indices and u_end in time_slot_indices):

                    s_idx = time_slot_indices[u_start]
                    e_idx = time_slot_indices[u_end]
                    if s_idx <= e_idx < num_time_slots:
                        doctor_availability[doctor_id][s_idx:e_idx + 1] = 0

    # Initialize room availability
    room_availability = {}
    if not operating_rooms.empty and len(operating_rooms.columns) >= 3:
        room_col, time_col, avail_col = operating_rooms.columns[:3]
        all_rooms = operating_rooms[room_col].unique().tolist()
        print(f"Found {len(all_rooms)} operating rooms")

        for room in all_rooms:
            room_availability[room] = np.zeros(num_time_slots)

        for _, row in operating_rooms.iterrows():
            room_id, time_slot, available = row[room_col], row[time_col], row[avail_col]
            if time_slot in time_slot_indices and available == 1:
                t_idx = time_slot_indices[time_slot]
                room_availability[room_id][t_idx] = 1

    return time_slot_indices, doctor_availability, room_availability, time_slot_list


if __name__ == '__main__':
    base_path = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'
    batch_count = 20

    time_slots, patients_batches, doctors_batches, operating_rooms = load_all_data(base_path, batch_count)

    for batch_key in sorted(patients_batches.keys(), key=batch_key_sort_key):
        print(f"\n=== Processing batch {batch_key} ===")
        patients = patients_batches.get(batch_key, pd.DataFrame())
        doctors = doctors_batches.get(batch_key, pd.DataFrame())

        if patients.empty or doctors.empty:
            print(f"Skip batch {batch_key}: missing data")
            continue

        result = preprocess_data(time_slots, patients, doctors, operating_rooms)
        if result[0] is None:
            print(f"Skip batch {batch_key}: preprocessing failed")
            continue

    print("\nAll batches processed")