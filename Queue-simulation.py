import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RANDOM_SEED = 42
SIM_TIME = 480
WORK_DAYS = 5
HEAVY_RATIO = 0.20


NUM_STANDARD_PORTS = 3
NUM_HEAVY_PORTS = 2


SERVICE_TIME_STANDARD = 12.0
SERVICE_TIME_HEAVY = 28.0


class InspectionStation:
    def __init__(self, env, day_id):
        self.env = env
        self.day_id = day_id
        self.standard_ports = simpy.Resource(env, capacity=NUM_STANDARD_PORTS)
        self.heavy_ports = simpy.Resource(env, capacity=NUM_HEAVY_PORTS)
        self.records = []
        self.queue_monitor = []

    def log_queue(self):

        self.queue_monitor.append({
            'day': self.day_id,
            'time': self.env.now,
            'standard_queue': len(self.standard_ports.queue),
            'heavy_queue': len(self.heavy_ports.queue)
        })

    def inspect_car(self, car_type):
        mean_time = SERVICE_TIME_HEAVY if car_type == 'Heavy' else SERVICE_TIME_STANDARD
        duration = random.expovariate(1.0 / mean_time)
        yield self.env.timeout(duration)


def vehicle(env, name, car_type, station):
    arrival_time = env.now
    resource = station.heavy_ports if car_type == 'Heavy' else station.standard_ports

    station.log_queue()

    with resource.request() as req:
        yield req
        service_start = env.now
        waiting_time = service_start - arrival_time

        station.log_queue()

        yield env.process(station.inspect_car(car_type))
        departure_time = env.now
        service_duration = departure_time - service_start
        total_time = departure_time - arrival_time

        station.records.append({
            'Day': station.day_id,
            'Vehicle_ID': name,
            'Type': car_type,
            'Arrival_Time': arrival_time,
            'Service_Start': service_start,
            'Departure_Time': departure_time,
            'Waiting_Time': waiting_time,
            'Service_Duration': service_duration,
            'Total_Time': total_time
        })

        station.log_queue()


def traffic_generator(env, station, arrival_rate):
    i = 0
    while True:
        inter_arrival = random.expovariate(1.0 / arrival_rate)
        yield env.timeout(inter_arrival)
        i += 1
        car_type = 'Heavy' if random.random() < HEAVY_RATIO else 'Standard'
        env.process(vehicle(env, f"D{station.day_id}_{car_type}_{i}", car_type, station))


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

all_records = []
all_queue_logs = []
daily_rates = [4.5, 3.8, 5.0, 3.5, 4.0]

for day in range(1, WORK_DAYS + 1):
    env = simpy.Environment()
    station = InspectionStation(env, day_id=day)
    rate = daily_rates[day - 1]
    env.process(traffic_generator(env, station, rate))
    env.run(until=SIM_TIME)

    all_records.extend(station.records)
    all_queue_logs.extend(station.queue_monitor)

df = pd.DataFrame(all_records)
q_df = pd.DataFrame(all_queue_logs)


print("\n" + "="*88)
print("                    5-DAY BESIKTNING SIMULATION REPORT (DTA400)")
print("="*88)


daily_summary = df.groupby(['Day', 'Type']).agg(
    Vehicles=('Vehicle_ID', 'count'),
    Avg_Wait=('Waiting_Time', 'mean'),
    Std_Wait=('Waiting_Time', 'std'),
    Max_Wait=('Waiting_Time', 'max'),
    Avg_Service=('Service_Duration', 'mean')
).round(2)

print("\n--- [ TABLE 1: DAILY VEHICLE WAIT & SERVICE TIMES ] ---")
print(f"{'Day':<6} | {'Class':<10} | {'Vehicles':<10} | {'Avg Wait (m)':<14} | {'Std Wait (m)':<14} | {'Max Wait (m)':<12}")
print("-" * 88)
for (day_num, v_type), row in daily_summary.iterrows():
    print(f"Day {day_num:<2} | {v_type:<10} | {row['Vehicles']:<10.0f} | {row['Avg_Wait']:<14.2f} | {row['Std_Wait']:<14.2f} | {row['Max_Wait']:<12.2f}")
print("-" * 88)


queue_stats = []
for d in range(1, WORK_DAYS + 1):
    sub = q_df[q_df['day'] == d]
    std_avg = sub['standard_queue'].mean()
    std_max = sub['standard_queue'].max()
    std_empty = (sub['standard_queue'] == 0).mean() * 100

    hvy_avg = sub['heavy_queue'].mean()
    hvy_max = sub['heavy_queue'].max()
    hvy_empty = (sub['heavy_queue'] == 0).mean() * 100

    queue_stats.append({
        'Day': f"Day {d}",
        'Std_Avg_Q': std_avg, 'Std_Max_Q': std_max, 'Std_Empty_%': std_empty,
        'Hvy_Avg_Q': hvy_avg, 'Hvy_Max_Q': hvy_max, 'Hvy_Empty_%': hvy_empty
    })


queue_stats.append({
    'Day': 'ALL (Mean)',
    'Std_Avg_Q': q_df['standard_queue'].mean(),
    'Std_Max_Q': q_df['standard_queue'].max(),
    'Std_Empty_%': (q_df['standard_queue'] == 0).mean() * 100,
    'Hvy_Avg_Q': q_df['heavy_queue'].mean(),
    'Hvy_Max_Q': q_df['heavy_queue'].max(),
    'Hvy_Empty_%': (q_df['heavy_queue'] == 0).mean() * 100
})

q_table = pd.DataFrame(queue_stats).round(2)

print("\n--- [ TABLE 2: QUEUE BEHAVIOR SUMMARY ACROSS ALL DAYS (FIGURE 2 DATA) ] ---")
print(f"{'Work Period':<12} | {'Std Avg Q':<10} | {'Std Max Q':<10} | {'Std Empty %':<12} | {'Hvy Avg Q':<10} | {'Hvy Max Q':<10} | {'Hvy Empty %':<12}")
print("-" * 88)
for _, r in q_table.iterrows():
    prefix = "*** " if r['Day'] == 'ALL (Mean)' else "    "
    print(f"{prefix + r['Day']:<12} | {r['Std_Avg_Q']:<10.2f} | {r['Std_Max_Q']:<10.0f} | {r['Std_Empty_%']:<11.1f}% | {r['Hvy_Avg_Q']:<10.2f} | {r['Hvy_Max_Q']:<10.0f} | {r['Hvy_Empty_%']:<11.1f}%")
print("-" * 88)


total_std_srv = df[df['Type'] == 'Standard']['Service_Duration'].sum()
total_hvy_srv = df[df['Type'] == 'Heavy']['Service_Duration'].sum()
util_std = (total_std_srv / (NUM_STANDARD_PORTS * SIM_TIME * WORK_DAYS)) * 100
util_hvy = (total_hvy_srv / (NUM_HEAVY_PORTS * SIM_TIME * WORK_DAYS)) * 100

print("\n--- [ TABLE 3: WEEKLY RESOURCE UTILIZATION ] ---")
print(f"Standard Inspection Ports Utilization : {util_std:.2f}%")
print(f"Heavy Vehicle Inspection Ports Util   : {util_hvy:.2f}%")
print("="*88 + "\n")


fig1, ax = plt.subplots(figsize=(10, 5))
days = np.arange(1, WORK_DAYS + 1)
width = 0.35

std_means = [daily_summary.loc[(d, 'Standard'), 'Avg_Wait'] for d in days]
std_errors = [daily_summary.loc[(d, 'Standard'), 'Std_Wait'] for d in days]
hvy_means = [daily_summary.loc[(d, 'Heavy'), 'Avg_Wait'] for d in days]
hvy_errors = [daily_summary.loc[(d, 'Heavy'), 'Std_Wait'] for d in days]

ax.bar(days - width/2, std_means, width, yerr=std_errors, capsize=5, label='Standard Cars', color='#2b5c8f', alpha=0.85)
ax.bar(days + width/2, hvy_means, width, yerr=hvy_errors, capsize=5, label='Heavy Trucks', color='#d9534f', alpha=0.85)

ax.set_title('Daily Average Waiting Time Comparison Across 5 Working Days', fontsize=12, pad=12)
ax.set_xlabel('Work Day', fontsize=11)
ax.set_ylabel('Average Wait Time (Minutes)', fontsize=11)
ax.set_xticks(days)
ax.set_xticklabels([f'Day {d}' for d in days])
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()


fig2, axes = plt.subplots(WORK_DAYS, 1, figsize=(12, 10), sharex=True)

for d in days:
    day_data = q_df[q_df['day'] == d]
    day_ax = axes[d - 1]

    day_ax.plot(day_data['time'], day_data['standard_queue'],
                label='Standard Queue', color='#2b5c8f', linewidth=1.5)
    day_ax.plot(day_data['time'], day_data['heavy_queue'],
                label='Heavy Queue', color='#d9534f', linewidth=1.5)
    day_ax.set_ylabel(f'Day {d}\nQueue', fontsize=9)
    day_ax.grid(True, linestyle='--', alpha=0.4)

    if d == 1:
        day_ax.legend(loc='upper right')

axes[-1].set_xlim(0, SIM_TIME)
axes[-1].set_xlabel('Operating Time (Minutes from 0 to 480)', fontsize=11)
fig2.suptitle('Queue Length Fluctuations Day-by-Day', fontsize=13)
fig2.tight_layout()


time_grid = np.linspace(0, SIM_TIME, 100)
mean_std_curve = []
mean_hvy_curve = []

for t in time_grid:
    t_data = q_df[(q_df['time'] >= t - 5) & (q_df['time'] <= t + 5)]
    mean_std_curve.append(t_data['standard_queue'].mean() if not t_data.empty else 0)
    mean_hvy_curve.append(t_data['heavy_queue'].mean() if not t_data.empty else 0)

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(time_grid, mean_std_curve, label='Avg Standard Queue',
         color='#2b5c8f', linestyle='--', linewidth=2)
ax3.plot(time_grid, mean_hvy_curve, label='Avg Heavy Queue',
         color='#d9534f', linestyle='--', linewidth=2)
ax3.set_title('Average Queue Length Across 5 Working Days', fontsize=13, pad=12)
ax3.set_xlabel('Operating Time (Minutes from 0 to 480)', fontsize=11)
ax3.set_ylabel('5-Day Avg Queue', fontsize=11)
ax3.set_xlim(0, SIM_TIME)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='upper right')
fig3.tight_layout()

plt.show()
