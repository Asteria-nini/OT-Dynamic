import pandas as pd
from datetime import datetime, timedelta

# Generate time slots
start_time = datetime.strptime("09:00:00", "%H:%M:%S")
end_time = datetime.strptime("21:00:00", "%H:%M:%S")
time_delta = timedelta(minutes=12)

time_slots = []
week_days = [1, 2, 3, 4, 5]  # Mon-Fri

for day_index, day in enumerate(week_days, 1):
    current_time = start_time
    for i in range(60):
        time_slot_id = f"T{day_index}-{i+1}"
        end_of_slot = current_time + time_delta
        time_slots.append([time_slot_id, f"{day} {current_time.strftime('%H:%M')}", f"{day} {end_of_slot.strftime('%H:%M')}"])
        current_time = end_of_slot

df = pd.DataFrame(time_slots, columns=["time_slot_id", "StartTime", "EndTime"])
df.to_excel(r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1\time_slots.xlsx", index=False)
print("time_slots.xlsx generated")


# Generate operating room data
days = [1, 2, 3, 4, 5]
rooms_per_day = 4
room_data = []

for day in days:
    for room_index in range(1, rooms_per_day + 1):
        room_id = f'R{day}{room_index}'
        for i in range(1, 61):
            time_slot_id = f'T{day}-{i}'
            room_data.append([room_id, time_slot_id, 1])  # 1 = available

df = pd.DataFrame(room_data, columns=['room_id', 'time_slot_id', 'available'])
df.to_excel(r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1\operating_room.xlsx', index=False)
print("operating_room.xlsx generated")