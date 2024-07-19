import random
import pandas as pd
import matplotlib.pyplot as plt

# Sample data generation for different products over time
def generate_sample_data():
    start_time = pd.Timestamp('2023-06-26 08:00:00')
    end_time = pd.Timestamp('2023-06-26 18:00:00')
    times = pd.date_range(start=start_time, end=end_time, freq='30T')[:-1]  # Exclude the end time for correct interval labeling
    products = ['Product_A', 'Product_B', 'Product_C']

    data = {'Time': times}
    for product in products:
        data[product] = [random.randint(10, 50) for _ in range(len(times))]

    df = pd.DataFrame(data)
    df['Time'] = df['Time'].dt.strftime('%H:%M')
    return df

# Generate sample data
df = generate_sample_data()

# Create the time intervals for x-axis labels
time_intervals = [f'{df["Time"][i]} - {df["Time"][i+1]}' for i in range(len(df)-1)]

# Adjust DataFrame to match time intervals
df = df.iloc[:-1]  # Exclude the last row for correct interval labeling

# Plotting
plt.figure(figsize=(14, 7))

# Define bar width and positions
bar_width = 0.35
bar_positions = range(len(df))

# Create a stacked bar plot with spaced bars
bottoms = [0] * len(df)
for product in df.columns[1:]:
    plt.bar(bar_positions, df[product], bottom=bottoms, width=bar_width, label=product)
    bottoms = [bottoms[i] + df[product][i] for i in range(len(bottoms))]

# Set x-axis labels and positions
plt.xticks(bar_positions, time_intervals, rotation=45)

plt.xlabel('Time Intervals')
plt.ylabel('Units Produced')
plt.title('Throughput of Different Products Over Time')
plt.legend(title='Products')
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

