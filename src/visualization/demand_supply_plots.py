import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demand_distribution_visualization():
    # Demand distribution visualization and Patients.xlsx generation
    file_path = r'D:\python_project\pythonProject\OR\DATA\G1-1DataInfo_Arrival_Time.xlsx'
    df = pd.read_excel(file_path)

    # First plot: Original patient data 3D bar chart
    x = df['PatientID']
    y = df['SurgeonID']
    z = df['Duration']
    emergency_level = df['EmergencyLevel']

    x_unique = np.unique(x)
    y_unique = np.unique(y)

    X, Y = np.meshgrid(x_unique, y_unique)
    Z = np.zeros(X.shape)

    # Store Duration in Z array
    for i in range(len(x)):
        xi = np.where(x_unique == x.iloc[i])[0][0]
        yi = np.where(y_unique == y.iloc[i])[0][0]
        Z[yi, xi] = z.iloc[i]

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Emergency cases in red
    for i in range(len(x)):
        if emergency_level.iloc[i] == 1:
            xi = np.where(x_unique == x.iloc[i])[0][0]
            yi = np.where(y_unique == y.iloc[i])[0][0]
            ax.bar3d(xi, yi, 0, 1, 1, z.iloc[i], color='red', shade=True)

    # Elective cases in blue
    for i in range(len(x)):
        if emergency_level.iloc[i] == 0:
            xi = np.where(x_unique == x.iloc[i])[0][0]
            yi = np.where(y_unique == y.iloc[i])[0][0]
            ax.bar3d(xi, yi, 0, 1, 1, z.iloc[i], color='blue', shade=True)

    ax.set_xlabel('Patient ID')
    ax.set_ylabel('Surgeon ID')
    ax.set_zlabel('Duration')
    ax.set_title('3D Bar Plot for Patient ID, Surgeon ID, and Duration (Emergency vs Elective)')

    ax.set_xticks(np.arange(min(x_unique), max(x_unique)+1, 5))
    ax.set_yticks(np.arange(min(y_unique), max(y_unique)+1, 3))
    ax.set_zticks(np.arange(min(z), max(z)+1, 5))

    plt.tick_params(axis='both', length=5)
    ax.set_box_aspect((max(x_unique), max(y_unique), max(z)))

    plt.savefig(r'D:\python_project\pythonProject\OR\可视化\原始需求分布.svg', format='svg')
    plt.show()

    # Second plot: Sorted patient demand distribution
    sorted_patient_ids = []
    for day in range(1, 6):
        day_df = df[df['ArrivalDay'] == day]
        sorted_day_df = day_df.sort_values(by='PriorityScore', ascending=False)
        sorted_patient_ids.extend(sorted_day_df['PatientID'].tolist())
    sorted_patient_ids = list(dict.fromkeys(sorted_patient_ids))

    x_sorted = []
    y_sorted = []
    z_sorted = []
    color_sorted = []

    for pid in sorted_patient_ids:
        record = df[df['PatientID'] == pid].iloc[0]
        x_sorted.append(pid)
        y_sorted.append(record['SurgeonID'])
        z_sorted.append(record['Duration'])
        color_sorted.append('red' if record['EmergencyLevel'] == 1 else 'blue')

    y_unique_sorted = np.unique(y_sorted)

    fig2 = plt.figure(figsize=(35, 20))
    ax2 = fig2.add_subplot(111, projection='3d')

    for i in range(len(x_sorted)):
        xi = i  # Patient position in sorted order
        yi = np.where(y_unique_sorted == y_sorted[i])[0][0]
        ax2.bar3d(xi, yi, 0, 1, 1, z_sorted[i], color=color_sorted[i], shade=True)

    ax2.set_xlabel('Patient ID In Order')
    ax2.set_ylabel('Surgeon ID')
    ax2.set_zlabel('Duration')
    ax2.set_title('3D Bar Plot of Duration by Patient Order, SurgeonID, and Emergency Level')

    ax2.set_xticks(np.arange(0, len(sorted_patient_ids), 1))
    ax2.set_xticklabels(sorted_patient_ids, rotation=60)

    ax2.set_yticks(np.arange(0, max(y_unique)+1, 2))
    ax2.set_zticks(np.arange(0, max(z_sorted)+1, 2))

    plt.tick_params(axis='x', labelsize=8)
    plt.tick_params(axis='y', labelsize=8)
    plt.tick_params(axis='z', labelsize=8)

    ax2.set_box_aspect([len(sorted_patient_ids), max(y_unique), max(z_sorted)])

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()

    plt.savefig(r'D:\python_project\pythonProject\OR\可视化\调整后需求分布.svg', format='svg')
    plt.show()

    # Generate Patients.xlsx
    patient_data = []
    for day in range(1, 6):
        day_df = df[df['ArrivalDay'] == day]
        sorted_day_df = day_df.sort_values(by='PriorityScore', ascending=False)
        for _, row in sorted_day_df.iterrows():
            surgeon_label = f'D{row["SurgeonID"]}'
            patient_data.append([f'P{row["PatientID"]}', row['Duration'], surgeon_label, row['ArrivalDay']])

    patients_df = pd.DataFrame(patient_data, columns=['PatientID', 'Duration', 'SurgeonID', 'ArrivalDay'])
    patients_df.to_excel(r'D:\python_project\pythonProject\OR\DATA\Patients.xlsx', index=False)
    print("Patients.xlsx generated")


def supply_distribution_visualization():
    # Supply distribution visualization
    num_rooms = 4
    num_doctors = 27
    time_blocks = 48
    num_days = 5
    y_tick_interval = 2

    file_path = r'D:\python_project\pythonProject\OR\DATA\Doctor_Availability_with_Disturbance.xlsx'
    info_path = r'D:\python_project\pythonProject\OR\DATA\G1-1DataInfo_Cleaned.xlsx'

    # Read unavailable times and meeting times
    df = pd.read_excel(file_path, sheet_name='Sheet1')

    availability_data = df[['SurgeonID', 'AffairDay', 'AffairStartTime', 'AffairEndTime']].dropna()
    invalid_times = []
    for _, row in availability_data.iterrows():
        surgeon_id = int(row['SurgeonID'])
        affair_day = int(row['AffairDay'])
        start_time = int(row['AffairStartTime'])
        end_time = int(row['AffairEndTime'])
        invalid_times.append((surgeon_id, affair_day, start_time, end_time))

    meeting_data = df[['SurgeonID', 'MeetingDay', 'MeetingStartTime', 'MeetingEndTime']].dropna()
    meeting_times = []
    for _, row in meeting_data.iterrows():
        surgeon_id = int(row['SurgeonID'])
        meeting_day = int(row['MeetingDay'])
        meeting_start_time = int(row['MeetingStartTime'])
        meeting_end_time = int(row['MeetingEndTime'])
        meeting_times.append((surgeon_id, meeting_day, meeting_start_time, meeting_end_time))

    info_df = pd.read_excel(info_path)

    # Filter out doctors with empty PatientsID
    to_remove = set(info_df[info_df['PatientsID'].isna()]['SurgeonID'].unique())

    # Generate operating room data points
    room_ids_2 = []
    surgeon_ids_2 = []
    usage_capacity_2 = []

    for day in range(1, num_days + 1):
        for room in range(1, num_rooms + 1):
            for surgeon in range(num_doctors):
                room_ids_2.append((day - 1) * num_rooms + room)
                surgeon_ids_2.append(surgeon)
                usage_capacity_2.append(time_blocks)

    total_rooms = num_days * num_rooms
    room_labels = [f"Room{(i - 1) // num_rooms + 1}{(i -1) % num_rooms + 1}" for i in range(1, total_rooms + 1)]

    # Filter out data for doctors with empty PatientsID
    filtered_room_ids = []
    filtered_surgeon_ids = []
    filtered_usage_capacity = []

    for r, s, u in zip(room_ids_2, surgeon_ids_2, usage_capacity_2):
        if s not in to_remove:
            filtered_room_ids.append(r)
            filtered_surgeon_ids.append(s)
            filtered_usage_capacity.append(u)

    # Create plot
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(111, projection='3d')

    dx = 0.5
    dy = 0.5

    for x_point, y_point, height in zip(filtered_room_ids, filtered_surgeon_ids, filtered_usage_capacity):

        # Unavailable time periods
        for surgeon_id, affair_day, start_time, end_time in invalid_times:
            if surgeon_id == y_point:
                room_day = (x_point - 1) // num_rooms + 1
                if room_day == affair_day:
                    if start_time < time_blocks and end_time > 0 and end_time <= time_blocks:
                        ax2.bar3d(x_point - dx / 2, y_point - dy / 2, 0, dx, dy, start_time, color='lightgrey', alpha=0.1)
                        ax2.bar3d(x_point - dx / 2, y_point - dy / 2, end_time, dx, dy, time_blocks - end_time, color='lightgrey', alpha=0.1)
                        ax2.bar3d(x_point - dx / 2, y_point - dy / 2, start_time, dx, dy, end_time - start_time, color='red', alpha=1)

        # Meeting time periods
        for surgeon_id, meeting_day, meeting_start_time, meeting_end_time in meeting_times:
            if surgeon_id == y_point:
                room_day = (x_point - 1) // num_rooms + 1
                if room_day == meeting_day:
                    if meeting_start_time < time_blocks and meeting_end_time > 0 and meeting_end_time <= time_blocks:
                        ax2.bar3d(x_point - dx / 2, y_point - dy / 2, meeting_start_time, dx, dy, meeting_end_time - meeting_start_time, color='red', alpha=1)

        # Available time periods
        if height > 0:
            ax2.bar3d(x_point - dx / 2, y_point - dy / 2, 0, dx, dy, height, color='lightgreen', alpha=0.1)

    ax2.set_xlabel('Room', labelpad=30)
    ax2.set_ylabel('Surgeon')
    ax2.set_zlabel('Usage Capacity')
    ax2.set_title('Usage Capacity for Operating Theatres (Excluding Unavailable Times, Meetings, and Filtered Surgeons)')

    ax2.set_xticks(np.arange(1, total_rooms + 1))
    ax2.set_xticklabels(room_labels, rotation=45, ha='right')
    ax2.set_yticks(np.arange(0, num_doctors, y_tick_interval))

    ax2.set_xlim(0.5, total_rooms + 0.5)
    ax2.set_ylim(-0.5, num_doctors - 0.5)
    ax2.set_zlim(0, time_blocks + 10)

    plt.savefig(r'D:\python_project\pythonProject\OR\可视化\Usage Capacity for Operating Theatres (Excluding Unavailable Times, Meetings, and Filtered Surgeons).svg', format='svg')
    plt.show()


if __name__ == '__main__':
    demand_distribution_visualization()
    supply_distribution_visualization()
