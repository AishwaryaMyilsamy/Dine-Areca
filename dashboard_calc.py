from datetime import datetime, timedelta

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}hours:{int(minutes):02}minutes:{int(seconds):02}seconds"

def calculate_times(timestamps):
    
    cycle_times = []
    idle_times = []
    
    for i in range(1, len(timestamps) - 1, 2):
        prev_T2 = timestamps[i - 1]  # Previous T2 (end time of previous cycle)
        curr_T1 = timestamps[i]      # Current T1 (start time of current cycle)
        curr_T2 = timestamps[i + 1]  # Current T2 (end time of current cycle)
        
        cycle_time = curr_T2 - prev_T2  # Cycle time is the difference between current T2 and previous T2
        idle_time = curr_T1 - prev_T2   # Idle time is the difference between current T1 and previous T2

        cycle_times.append(cycle_time.total_seconds())
        idle_times.append(idle_time.total_seconds())
    
    total_cycle_time = sum(cycle_times)  # Sum of all cycle times
    total_idle_time = sum(idle_times)    # Sum of all idle times

    return format_time(total_cycle_time), format_time(total_idle_time)

timestamps = [
    datetime(2024, 7, 9, 9, 15, 0),  # End of first cycle
    datetime(2024, 7, 9, 9, 20, 0),  # Start of second cycle
    datetime(2024, 7, 9, 9, 35, 0),  # End of second cycle
    datetime(2024, 7, 9, 9, 40, 0),  # Start of third cycle
    datetime(2024, 7, 9, 9, 55, 0),  # End of third cycle
    datetime(2024, 7, 9, 10, 0, 0),  # Start of fourth cycle
    datetime(2024, 7, 9, 10, 15, 0)  # End of fourth cycle

]

total_cycle_time, total_idle_time = calculate_times(timestamps)

print(f"Total Cycle Time: {total_cycle_time}")
print(f"Total Idle Time: {total_idle_time}")
