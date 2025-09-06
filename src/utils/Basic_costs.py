import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os

class ScheduleEvaluatorWithVisualization:
    # Medical scheduling cost evaluator with visualization capabilities

    def __init__(self, slots_per_day=50, overtime_base=1.0, overtime_factor=0.5):
        self.slots_per_day = slots_per_day
        self.overtime_base = overtime_base
        self.overtime_factor = overtime_factor
        self._patient_cache = {}

        # Set font and style
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", 10)

    @staticmethod
    def parse_time_slot(time_slot):
        # Parse time slot, return (day, slot)
        try:
            if time_slot.startswith('T') and '-' in time_slot:
                parts = time_slot[1:].split('-', 1)
                return int(parts[0]), int(parts[1])
        except:
            pass
        return -1, -1

    def get_patient_info_cached(self, patient_id, patients_data):
        # Get patient info with caching
        if patient_id in self._patient_cache:
            return self._patient_cache[patient_id]

        try:
            patient_id_column = patients_data.columns[0]
            patient_row = patients_data[patients_data[patient_id_column] == patient_id]

            if patient_row.empty:
                return None

            row_data = patient_row.iloc[0]

            def safe_int(value, default=0):
                try:
                    return int(value) if pd.notna(value) else default
                except:
                    return default

            def safe_float(value, default=1.0):
                try:
                    return float(value) if pd.notna(value) else default
                except:
                    return default

            info = {
                'patient_idx': patients_data.index[patients_data[patient_id_column] == patient_id][0],
                'duration': safe_int(row_data.iloc[1], 1),
                'doctor_id': row_data.iloc[2] if pd.notna(row_data.iloc[2]) else 'unknown',
                'arrival_day': safe_int(row_data.iloc[3], 0),
                'urgency': safe_int(row_data.iloc[4] if len(row_data) > 4 else None, 1),
                'priority_weight': safe_float(row_data.iloc[5] if len(row_data) > 5 else None, 1.0)
            }

            self._patient_cache[patient_id] = info
            return info

        except Exception as e:
            print(f"Error getting patient {patient_id} info: {e}")
            return None

    def group_schedule_by_day(self, final_schedule, patients_data):
        # Group schedule and patient data by day
        daily_schedules = defaultdict(list)
        daily_patients = defaultdict(list)

        for surgery in final_schedule:
            start_time = surgery.get('start_time')
            if start_time:
                day, _ = self.parse_time_slot(start_time)
                if day >= 0:
                    daily_schedules[day].append(surgery)

        for _, patient in patients_data.iterrows():
            patient_id = patient.iloc[0]
            patient_info = self.get_patient_info_cached(patient_id, patients_data)
            if patient_info:
                daily_patients[patient_info['arrival_day']].append(patient_info)

        return daily_schedules, daily_patients

    def get_daily_time_slots(self, time_slots_list, target_day):
        # Get time slots for specified day
        daily_slots = []
        for slot in time_slots_list:
            day, slot_num = self.parse_time_slot(slot)
            if day == target_day:
                daily_slots.append(slot)
        return sorted(daily_slots, key=lambda x: self.parse_time_slot(x)[1])

    def calculate_statistics(self, costs_array):
        # Calculate unified statistical metrics
        if len(costs_array) == 0:
            return {key: 0 for key in ['Aver', 'Std', 'Wor', 'Dec', 'Qua', 'min', 'max', 'median',
                                       'percentile_25', 'percentile_95', 'coefficient_of_variation']}

        costs = np.array(costs_array)
        mean_val = np.mean(costs)

        return {
            'Aver': mean_val,
            'Std': np.std(costs),
            'Wor': np.max(costs),
            'Dec': np.percentile(costs, 90),
            'Qua': np.percentile(costs, 75),
            'min': np.min(costs),
            'max': np.max(costs),
            'median': np.median(costs),
            'percentile_25': np.percentile(costs, 25),
            'percentile_95': np.percentile(costs, 95),
            'coefficient_of_variation': np.std(costs) / mean_val if mean_val > 0 else 0
        }

    def calculate_single_scenario_cost(self, schedule, actual_durations, daily_patients_info,
                                       daily_time_slots, time_slots_list):
        # Calculate single scenario cost
        total_base_cost = 0
        total_overtime_cost = 0
        used_rooms = set()

        expected_patients = set(p['patient_idx'] for p in daily_patients_info)
        scheduled_patients = set()

        normal_work_blocks = min(self.slots_per_day, len(daily_time_slots))

        for surgery in schedule:
            patient_id = surgery.get('patient_id')
            room_id = surgery.get('room_id')
            start_time = surgery.get('start_time')

            if not all([patient_id, start_time]):
                continue

            if room_id:
                used_rooms.add(room_id)

            patient_info = next((p for p in daily_patients_info
                                 if str(p.get('patient_idx', '')) == str(patient_id)), None)

            if patient_info:
                scheduled_patients.add(patient_info['patient_idx'])
                patient_idx = patient_info['patient_idx']
            else:
                try:
                    scheduled_patients.add(int(patient_id))
                    patient_idx = 0
                except:
                    patient_idx = 0

            try:
                daily_start_idx = daily_time_slots.index(start_time)
                global_start_idx = time_slots_list.index(start_time)
            except ValueError:
                continue

            base_cost = 1.0 + 0.001 * global_start_idx + 0.0001 * patient_idx
            total_base_cost += base_cost

            duration = actual_durations.get(patient_id, surgery.get('duration', 1))
            daily_end_idx = daily_start_idx + math.ceil(duration)
            overtime_blocks = max(0, daily_end_idx - normal_work_blocks)

            if overtime_blocks > 0:
                overtime_cost = sum(self.overtime_base + i * self.overtime_factor
                                    for i in range(overtime_blocks))
                total_overtime_cost += overtime_cost

        room_opening_cost = len(used_rooms) * 1.0
        unassigned_count = len(expected_patients - scheduled_patients)
        unassigned_penalty = unassigned_count * 1000

        return {
            'total': room_opening_cost + total_base_cost + total_overtime_cost + unassigned_penalty,
            'room_opening': room_opening_cost,
            'base': total_base_cost,
            'overtime': total_overtime_cost,
            'unassigned_penalty': unassigned_penalty,
            'unassigned_count': unassigned_count
        }

    def evaluate_daily_costs(self, daily_schedule, daily_patients, daily_time_slots,
                             time_slots_list, day_number, n_scenarios=100):
        # Evaluate daily costs
        scenario_costs = []
        cost_components = defaultdict(list)

        for _ in range(n_scenarios):
            actual_durations = {}
            for surgery in daily_schedule:
                patient_id = surgery.get('patient_id')
                expected = surgery.get('duration', 1)
                variation = np.random.normal(0, 0.1)
                actual_durations[patient_id] = max(1, expected * (1 + variation))

            scenario_result = self.calculate_single_scenario_cost(
                daily_schedule, actual_durations, daily_patients,
                daily_time_slots, time_slots_list
            )

            scenario_costs.append(scenario_result['total'])
            for component, value in scenario_result.items():
                if component != 'total':
                    cost_components[component].append(value)

        stats = self.calculate_statistics(scenario_costs)

        scheduled_count = len(daily_schedule)
        total_patients_today = len(daily_patients)
        assignment_rate = scheduled_count / max(1, total_patients_today)

        constraint_violations = sum(1 for surgery in daily_schedule
                                    if self.parse_time_slot(surgery.get('start_time', ''))[0] != day_number)

        urgent_total = sum(1 for p in daily_patients if p.get('urgency', 1) > 1)
        scheduled_patient_ids = set(str(surgery.get('patient_id')) for surgery in daily_schedule)
        urgent_scheduled = sum(1 for p in daily_patients
                               if p.get('urgency', 1) > 1 and str(p.get('patient_idx', '')) in scheduled_patient_ids)
        urgent_assignment_rate = urgent_scheduled / max(1, urgent_total)

        used_time_slots = set()
        for surgery in daily_schedule:
            start_time = surgery.get('start_time')
            if start_time in daily_time_slots:
                start_idx = daily_time_slots.index(start_time)
                duration = surgery.get('duration', 1)
                for i in range(duration):
                    if start_idx + i < len(daily_time_slots):
                        used_time_slots.add(start_idx + i)

        time_utilization = len(used_time_slots) / max(1, len(daily_time_slots))

        result = {
            'day': day_number,
            'normal_work_blocks': min(self.slots_per_day, len(daily_time_slots)),
            'total_time_slots': len(daily_time_slots),
            **{k: v for k, v in stats.items() if k in ['Aver', 'Std', 'Wor', 'Dec', 'Qua']},
            'assignment_rate': assignment_rate,
            'scheduled_count': scheduled_count,
            'total_patients': total_patients_today,
            'constraint_violations': constraint_violations,
            'urgent_assignment_rate': urgent_assignment_rate,
            'urgent_patients_total': urgent_total,
            'urgent_patients_scheduled': urgent_scheduled,
            'time_utilization': time_utilization,
            'components': {f'{k}_avg': np.mean(v) for k, v in cost_components.items()},
            'detailed_stats': {k: v for k, v in stats.items() if k not in ['Aver', 'Std', 'Wor', 'Dec', 'Qua']},
            'scenario_costs': scenario_costs,
            'cost_components': cost_components
        }

        return result, scenario_costs

    def evaluate_basic_costs_by_day(self, final_schedule, patients_data, time_slots_list):
        # Evaluate basic costs by day
        daily_schedules, daily_patients = self.group_schedule_by_day(final_schedule, patients_data)
        all_days = sorted([day for day in set(daily_schedules.keys()) | set(daily_patients.keys()) if day >= 0])

        print(f"Found {len(all_days)} days of schedule data: {all_days}")

        daily_results = {}
        all_global_costs = []

        for day in all_days:
            print(f"Calculating evaluation metrics for day {day}...")
            day_schedule = daily_schedules.get(day, [])
            day_patients = daily_patients.get(day, [])
            day_time_slots = self.get_daily_time_slots(time_slots_list, day)

            print(f"  Day {day}: {len(day_schedule)} surgeries, {len(day_patients)} patients, {len(day_time_slots)} time slots")

            if day_time_slots:
                daily_result, daily_scenario_costs = self.evaluate_daily_costs(
                    day_schedule, day_patients, day_time_slots, time_slots_list, day
                )
                daily_results[day] = daily_result
                all_global_costs.extend(daily_scenario_costs)

        if not daily_results:
            return {'daily_results': {}, 'summary_stats': {'total_days': 0, 'error': 'No valid daily data found'},
                    'all_days': []}

        daily_metrics = {
            'averages': [r['Aver'] for r in daily_results.values()],
            'stds': [r['Std'] for r in daily_results.values()],
            'assignment_rates': [r['assignment_rate'] for r in daily_results.values()],
            'time_utilizations': [r['time_utilization'] for r in daily_results.values()]
        }

        global_stats = self.calculate_statistics(all_global_costs)

        summary_stats = {
            'total_days': len(daily_results),
            'average_daily_cost': np.mean(daily_metrics['averages']),
            'average_daily_std': np.mean(daily_metrics['stds']),
            'average_assignment_rate': np.mean(daily_metrics['assignment_rates']),
            'average_time_utilization': np.mean(daily_metrics['time_utilizations']),
            'total_scheduled': sum(r['scheduled_count'] for r in daily_results.values()),
            'total_patients': sum(r['total_patients'] for r in daily_results.values()),
            'overall_assignment_rate': (sum(r['scheduled_count'] for r in daily_results.values()) /
                                        max(1, sum(r['total_patients'] for r in daily_results.values()))),
            'total_constraint_violations': sum(r['constraint_violations'] for r in daily_results.values()),
            'cost_stats': {
                'min_daily_cost': min(daily_metrics['averages']),
                'max_daily_cost': max(daily_metrics['averages']),
                'std_of_daily_costs': np.std(daily_metrics['averages'])
            },
            'global_metrics': {f'Global_{k}': v for k, v in global_stats.items()
                               if k in ['Aver', 'Std', 'Wor', 'Dec', 'Qua']},
            'global_detailed_stats': {f'global_{k}': v for k, v in global_stats.items()
                                      if k not in ['Aver', 'Std', 'Wor', 'Dec', 'Qua']} |
                                     {'total_scenarios': len(all_global_costs)}
        }

        return {
            'daily_results': daily_results,
            'summary_stats': summary_stats,
            'all_days': all_days
        }

    def plot_daily_cost_trends(self, daily_cost_result, output_dir, batch_name="", save_plot=True):
        # Plot daily cost trends
        daily_results = daily_cost_result['daily_results']
        if not daily_results:
            print("No data for plotting")
            return None

        days = sorted(daily_results.keys())
        metrics = ['Aver', 'Std', 'Wor', 'Dec', 'Qua']

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{batch_name} - Daily Cost Trend Analysis', fontsize=16, fontweight='bold')

        # Cost metric trends
        for i, metric in enumerate(metrics):
            row, col = i // 3, i % 3
            values = [daily_results[day][metric] for day in days]

            axes[row, col].plot(days, values, 'o-', linewidth=2, markersize=6,
                                color=self.colors[i], label=metric)
            axes[row, col].set_title(f'{metric} Trend', fontweight='bold')
            axes[row, col].set_xlabel('Day')
            axes[row, col].set_ylabel('Cost Value')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()

            # Add value labels
            for day, value in zip(days, values):
                axes[row, col].annotate(f'{value:.1f}', (day, value),
                                        textcoords="offset points", xytext=(0, 10), ha='center')

        # Assignment rate and time utilization
        assignment_rates = [daily_results[day]['assignment_rate'] for day in days]
        time_utilizations = [daily_results[day]['time_utilization'] for day in days]

        axes[1, 2].plot(days, assignment_rates, 'o-', linewidth=2, markersize=6,
                        color=self.colors[5], label='Assignment Rate')
        axes[1, 2].plot(days, time_utilizations, 's-', linewidth=2, markersize=6,
                        color=self.colors[6], label='Time Utilization')
        axes[1, 2].set_title('Assignment Rate & Time Utilization', fontweight='bold')
        axes[1, 2].set_xlabel('Day')
        axes[1, 2].set_ylabel('Rate')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(output_dir, f'{batch_name}_daily_cost_trends.svg')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Daily cost trend chart saved: {plot_path}")

        return fig

    def plot_cost_distribution_analysis(self, daily_cost_result, output_dir, batch_name="", save_plot=True):
        # Plot cost distribution analysis
        daily_results = daily_cost_result['daily_results']
        if not daily_results:
            return None

        # Collect scenario cost data
        all_scenario_costs = []
        daily_averages = []
        days = sorted(daily_results.keys())

        for day in days:
            if 'scenario_costs' in daily_results[day]:
                scenario_costs = daily_results[day]['scenario_costs']
                all_scenario_costs.extend(scenario_costs)
                daily_averages.append(daily_results[day]['Aver'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{batch_name} - Cost Distribution Analysis', fontsize=16, fontweight='bold')

        # Overall cost distribution histogram
        axes[0, 0].hist(all_scenario_costs, bins=30, alpha=0.7, color=self.colors[0], edgecolor='black')
        axes[0, 0].axvline(np.mean(all_scenario_costs), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(all_scenario_costs):.2f}')
        axes[0, 0].axvline(np.median(all_scenario_costs), color='orange', linestyle='--', linewidth=2,
                           label=f'Median: {np.median(all_scenario_costs):.2f}')
        axes[0, 0].set_title('Overall Cost Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Cost Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Daily cost box plot
        daily_cost_data = []
        day_labels = []
        for day in days:
            if 'scenario_costs' in daily_results[day]:
                daily_cost_data.append(daily_results[day]['scenario_costs'])
                day_labels.append(f'Day{day}')

        if daily_cost_data:
            box_plot = axes[0, 1].boxplot(daily_cost_data, labels=day_labels, patch_artist=True)
            for i, box in enumerate(box_plot['boxes']):
                box.set_facecolor(self.colors[i % len(self.colors)])
            axes[0, 1].set_title('Daily Cost Distribution Box Plot', fontweight='bold')
            axes[0, 1].set_xlabel('Day')
            axes[0, 1].set_ylabel('Cost Value')
            axes[0, 1].grid(True, alpha=0.3)

        # Cost coefficient of variation
        cv_values = []
        for day in days:
            result = daily_results[day]
            cv = result['Std'] / result['Aver'] if result['Aver'] > 0 else 0
            cv_values.append(cv)

        bars = axes[1, 0].bar(days, cv_values, color=self.colors[2], alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Cost Coefficient of Variation (Std/Mean)', fontweight='bold')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Coefficient of Variation')
        axes[1, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, cv in zip(bars, cv_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{cv:.3f}', ha='center', va='bottom')

        # Cost metrics radar chart
        if len(days) > 0:
            metrics = ['Aver', 'Std', 'Wor', 'Dec', 'Qua']

            # Normalize data
            all_values = []
            for metric in metrics:
                values = [daily_results[day][metric] for day in days]
                all_values.append(values)

            avg_values = [np.mean(vals) for vals in all_values]
            max_values = [np.max(vals) for vals in all_values]

            # Normalize to 0-1 range
            max_val = max(max_values)
            normalized_avg = [val / max_val for val in avg_values]

            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            normalized_avg = normalized_avg + [normalized_avg[0]]

            ax_radar = plt.subplot(2, 2, 4, projection='polar')
            ax_radar.plot(angles, normalized_avg, 'o-', linewidth=2, color=self.colors[3])
            ax_radar.fill(angles, normalized_avg, alpha=0.25, color=self.colors[3])
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(metrics)
            ax_radar.set_ylim(0, 1)
            ax_radar.set_title('Cost Metrics Radar Chart\n(Normalized)', fontweight='bold', pad=20)

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(output_dir, f'{batch_name}_cost_distribution_analysis.svg')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Cost distribution analysis chart saved: {plot_path}")

        return fig

    def plot_cost_components_breakdown(self, daily_cost_result, output_dir, batch_name="", save_plot=True):
        # Plot cost components breakdown
        daily_results = daily_cost_result['daily_results']
        if not daily_results:
            return None

        days = sorted(daily_results.keys())

        # Extract cost component data
        components = ['room_opening_avg', 'base_avg', 'overtime_avg', 'unassigned_penalty_avg']
        component_labels = ['Room Opening', 'Base Cost', 'Overtime Cost', 'Unassigned Penalty']

        component_data = {comp: [] for comp in components}

        for day in days:
            comp_dict = daily_results[day].get('components', {})
            for comp in components:
                component_data[comp].append(comp_dict.get(comp, 0))

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{batch_name} - Cost Components Breakdown Analysis', fontsize=16, fontweight='bold')

        # Stacked bar chart
        bottom = np.zeros(len(days))
        for i, (comp, label) in enumerate(zip(components, component_labels)):
            values = component_data[comp]
            axes[0, 0].bar(days, values, bottom=bottom, label=label,
                           color=self.colors[i], alpha=0.8)
            bottom += values

        axes[0, 0].set_title('Cost Components Stacked Chart', fontweight='bold')
        axes[0, 0].set_xlabel('Day')
        axes[0, 0].set_ylabel('Cost Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Cost component proportion pie chart
        avg_components = [np.mean(component_data[comp]) for comp in components]
        total_avg = sum(avg_components)

        if total_avg > 0:
            percentages = [comp / total_avg * 100 for comp in avg_components]
            wedges, texts, autotexts = axes[0, 1].pie(avg_components, labels=component_labels,
                                                      autopct='%1.1f%%', startangle=90,
                                                      colors=self.colors[:len(components)])
            axes[0, 1].set_title('Average Cost Components Proportion', fontweight='bold')

        # Component trend lines
        for i, (comp, label) in enumerate(zip(components, component_labels)):
            values = component_data[comp]
            axes[1, 0].plot(days, values, 'o-', linewidth=2, markersize=5,
                            color=self.colors[i], label=label)

        axes[1, 0].set_title('Cost Components Trends', fontweight='bold')
        axes[1, 0].set_xlabel('Day')
        axes[1, 0].set_ylabel('Cost Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Overtime cost analysis
        overtime_costs = component_data['overtime_avg']
        total_costs = [sum(component_data[comp][i] for comp in components) for i in range(len(days))]
        overtime_ratios = [overtime / total if total > 0 else 0 for overtime, total in zip(overtime_costs, total_costs)]

        bars = axes[1, 1].bar(days, overtime_ratios, color=self.colors[7], alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Overtime Cost Proportion', fontweight='bold')
        axes[1, 1].set_xlabel('Day')
        axes[1, 1].set_ylabel('Overtime Cost Ratio')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels
        for bar, ratio in zip(bars, overtime_ratios):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                            f'{ratio:.2%}', ha='center', va='bottom')

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(output_dir, f'{batch_name}_cost_components_breakdown.svg')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Cost components breakdown chart saved: {plot_path}")

        return fig

    def plot_performance_quality_metrics(self, daily_cost_result, output_dir, batch_name="", save_plot=True):
        # Plot performance quality metrics
        daily_results = daily_cost_result['daily_results']
        if not daily_results:
            return None

        days = sorted(daily_results.keys())

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{batch_name} - Performance Quality Metrics Analysis', fontsize=16, fontweight='bold')

        # Assignment rate vs cost efficiency
        assignment_rates = [daily_results[day]['assignment_rate'] for day in days]
        avg_costs = [daily_results[day]['Aver'] for day in days]

        scatter = axes[0, 0].scatter(assignment_rates, avg_costs, c=days,
                                     cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        axes[0, 0].set_title('Assignment Rate vs Average Cost', fontweight='bold')
        axes[0, 0].set_xlabel('Assignment Rate')
        axes[0, 0].set_ylabel('Average Cost')
        axes[0, 0].grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=axes[0, 0])
        cbar.set_label('Day')

        for i, day in enumerate(days):
            axes[0, 0].annotate(f'Day{day}', (assignment_rates[i], avg_costs[i]),
                                xytext=(5, 5), textcoords='offset points', fontsize=8)

        # Time utilization comparison
        time_utilizations = [daily_results[day]['time_utilization'] for day in days]
        scheduled_counts = [daily_results[day]['scheduled_count'] for day in days]
        total_patients = [daily_results[day]['total_patients'] for day in days]

        x = np.arange(len(days))
        width = 0.35

        bars1 = axes[0, 1].bar(x - width / 2, time_utilizations, width, label='Time Utilization',
                               color=self.colors[0], alpha=0.7)
        bars2 = axes[0, 1].bar(x + width / 2, assignment_rates, width, label='Assignment Rate',
                               color=self.colors[1], alpha=0.7)

        axes[0, 1].set_title('Time Utilization vs Assignment Rate', fontweight='bold')
        axes[0, 1].set_xlabel('Day')
        axes[0, 1].set_ylabel('Rate')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels([f'Day{day}' for day in days])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Urgent patient handling
        urgent_assignment_rates = [daily_results[day]['urgent_assignment_rate'] for day in days]
        urgent_totals = [daily_results[day]['urgent_patients_total'] for day in days]

        bubble_sizes = [day * 50 for day in days]

        scatter2 = axes[1, 0].scatter(urgent_totals, urgent_assignment_rates,
                                      s=bubble_sizes, c=days, cmap='plasma',
                                      alpha=0.6, edgecolors='black')
        axes[1, 0].set_title('Urgent Patient Handling Performance', fontweight='bold')
        axes[1, 0].set_xlabel('Total Urgent Patients')
        axes[1, 0].set_ylabel('Urgent Patient Assignment Rate')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        cbar2 = plt.colorbar(scatter2, ax=axes[1, 0])
        cbar2.set_label('Day')

        # Comprehensive performance radar chart
        performance_metrics = {
            'Assignment Rate': assignment_rates,
            'Time Utilization': time_utilizations,
            'Urgent Assignment Rate': urgent_assignment_rates,
            'Cost Stability': [1 - (daily_results[day]['Std'] / daily_results[day]['Aver'])
                           if daily_results[day]['Aver'] > 0 else 0 for day in days],
            'Constraint Satisfaction': [1 - daily_results[day]['constraint_violations'] /
                           max(1, daily_results[day]['scheduled_count']) for day in days]
        }

        avg_performance = {metric: np.mean(values) for metric, values in performance_metrics.items()}

        metrics = list(avg_performance.keys())
        values = list(avg_performance.values())

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        values = values + [values[0]]

        ax_radar = plt.subplot(2, 2, 4, projection='polar')
        ax_radar.plot(angles, values, 'o-', linewidth=2, color=self.colors[4])
        ax_radar.fill(angles, values, alpha=0.25, color=self.colors[4])
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Comprehensive Performance Radar Chart', fontweight='bold', pad=20)

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(output_dir, f'{batch_name}_performance_quality_metrics.svg')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Performance quality metrics chart saved: {plot_path}")

        return fig

    def generate_comprehensive_cost_visualization(self, daily_cost_result, output_dir, batch_name=""):
        # Generate comprehensive cost visualization report
        print(f"Generate {batch_name} Comprehensive Cost Visualization Report")

        os.makedirs(output_dir, exist_ok=True)

        figures = []

        try:
            fig1 = self.plot_daily_cost_trends(daily_cost_result, output_dir, batch_name)
            if fig1:
                figures.append(('daily_cost_trends', fig1))

            fig2 = self.plot_cost_distribution_analysis(daily_cost_result, output_dir, batch_name)
            if fig2:
                figures.append(('cost_distribution_analysis', fig2))

            fig3 = self.plot_cost_components_breakdown(daily_cost_result, output_dir, batch_name)
            if fig3:
                figures.append(('cost_components_breakdown', fig3))

            fig4 = self.plot_performance_quality_metrics(daily_cost_result, output_dir, batch_name)
            if fig4:
                figures.append(('performance_quality_metrics', fig4))

            print(f"Successfully generated {len(figures)} visualization charts")

            try:
                self.create_summary_report(daily_cost_result, output_dir, batch_name)
            except Exception as e:
                print(f"Error creating summary report: {e}")

            return figures

        except Exception as e:
            print(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return []

    def create_summary_report(self, daily_cost_result, output_dir, batch_name=""):
        # Create text summary report
        summary_stats = daily_cost_result['summary_stats']
        daily_results = daily_cost_result['daily_results']

        report_path = os.path.join(output_dir, f'{batch_name}_cost_visualization_summary.txt')

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"        {batch_name} Cost Evaluation Visualization Summary Report\n")
            f.write(f"{'=' * 60}\n\n")

            if 'global_metrics' in summary_stats:
                f.write("Global Key Metrics:\n")
                for metric, value in summary_stats['global_metrics'].items():
                    f.write(f"   {metric}: {value:.4f}\n")
                f.write("\n")

            f.write("Summary Statistics:\n")
            key_metrics = ['total_days', 'average_daily_cost', 'overall_assignment_rate',
                           'average_time_utilization']
            for metric in key_metrics:
                if metric in summary_stats:
                    value = summary_stats[metric]
                    if 'rate' in metric:
                        f.write(f"   {metric}: {value:.2%}\n")
                    else:
                        f.write(f"   {metric}: {value:.4f}\n")
            f.write("\n")

            if daily_results:
                f.write("Daily Key Metrics:\n")
                f.write("Day  Average Cost  Assignment Rate  Time Utilization\n")
                f.write("-" * 50 + "\n")
                for day in sorted(daily_results.keys()):
                    result = daily_results[day]
                    f.write(
                        f"{day:>3}  {result['Aver']:>12.2f}  {result['assignment_rate']:>14.2%}  {result['time_utilization']:>15.2%}\n")

            f.write(f"\nGeneration Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Summary report saved: {report_path}")

    def print_daily_cost_report(self, daily_cost_result, batch_name=""):
        # Print detailed cost evaluation report
        print(f"\n{'=' * 80}")
        print(f"                  Daily Basic Scheduling Cost Evaluation Report - {batch_name}")
        print(f"{'=' * 80}")

        daily_results = daily_cost_result['daily_results']
        summary_stats = daily_cost_result['summary_stats']

        if 'error' in summary_stats:
            print(f"Error: {summary_stats['error']}")
            return

        if 'global_metrics' in summary_stats:
            global_metrics = summary_stats['global_metrics']
            print(f"\nGlobal Key Metrics:")
            for metric, value in global_metrics.items():
                print(f"   {metric}: {value:.4f}")

        if 'global_detailed_stats' in summary_stats:
            global_stats = summary_stats['global_detailed_stats']
            print(f"\nGlobal Detailed Statistics:")
            for metric, value in global_stats.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        print(f"\nDaily Summary Statistics:")
        metrics_to_show = ['total_days', 'average_daily_cost', 'average_daily_std',
                           'average_assignment_rate', 'average_time_utilization',
                           'overall_assignment_rate', 'total_constraint_violations']

        for metric in metrics_to_show:
            if metric in summary_stats:
                value = summary_stats[metric]
                if isinstance(value, float) and 'rate' in metric:
                    print(f"   {metric}: {value:.2%}")
                elif isinstance(value, float):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        print(f"\nDaily Detailed Results:")
        print(
            f"{'Day':<4} {'Avg Cost':<10} {'Std Dev':<8} {'Worst':<10} {'90%ile':<10} {'75%ile':<10} {'Assign Rate':<12} {'Time Util':<10}")
        print("-" * 88)

        for day in sorted(daily_results.keys()):
            result = daily_results[day]
            print(f"{day:<4} {result['Aver']:<10.4f} {result['Std']:<8.4f} "
                  f"{result['Wor']:<10.4f} {result['Dec']:<10.4f} {result['Qua']:<10.4f} "
                  f"{result['assignment_rate']:<12.2%} {result['time_utilization']:<10.2%}")

        if len(daily_results) > 1:
            best_day = min(daily_results.keys(), key=lambda d: daily_results[d]['Aver'])
            worst_day = max(daily_results.keys(), key=lambda d: daily_results[d]['Aver'])
            print(f"\nBest performing day: Day {best_day} (Average cost: {daily_results[best_day]['Aver']:.4f})")
            print(f"Worst performing day: Day {worst_day} (Average cost: {daily_results[worst_day]['Aver']:.4f})")

    def save_daily_cost_results_to_excel(self, daily_cost_result, file_path, batch_name=""):
        # Save results to Excel
        daily_results = daily_cost_result['daily_results']
        summary_stats = daily_cost_result['summary_stats']

        if 'error' in summary_stats:
            print(f"Cannot save results: {summary_stats['error']}")
            return

        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                if 'global_metrics' in summary_stats:
                    global_df = pd.DataFrame([
                        {'Metric': k, 'Value': v, 'Description': f'Global {k.split("_")[1]}'}
                        for k, v in summary_stats['global_metrics'].items()
                    ])
                    global_df.to_excel(writer, sheet_name='Global_Main_Metrics', index=False)

                summary_df = pd.DataFrame([
                    {'Metric': k, 'Value': v} for k, v in summary_stats.items()
                    if k not in ['global_metrics', 'global_detailed_stats', 'cost_stats']
                       and isinstance(v, (int, float))
                ])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

                daily_main_data = []
                for day in sorted(daily_results.keys()):
                    result = daily_results[day]
                    daily_main_data.append({
                        'Day': day,
                        **{k: v for k, v in result.items()
                           if k in ['Aver', 'Std', 'Wor', 'Dec', 'Qua', 'assignment_rate',
                                    'time_utilization', 'scheduled_count', 'total_patients',
                                    'normal_work_blocks', 'total_time_slots']}
                    })

                pd.DataFrame(daily_main_data).to_excel(writer, sheet_name='Daily_Main_Metrics', index=False)

            print(f"Daily cost evaluation results saved to: {file_path}")

        except Exception as e:
            print(f"Error saving Excel file: {e}")


# Maintain compatibility with original function interfaces
def evaluate_basic_costs_by_day_with_visualization(final_schedule, patients_data, time_slots_list,
                                                   slots_per_day=50, overtime_base=1.0, overtime_factor=0.5,
                                                   output_dir=None, batch_name="", generate_plots=True):
    # Cost evaluation function with visualization
    evaluator = ScheduleEvaluatorWithVisualization(slots_per_day, overtime_base, overtime_factor)

    daily_cost_result = evaluator.evaluate_basic_costs_by_day(final_schedule, patients_data, time_slots_list)

    if generate_plots and output_dir:
        figures = evaluator.generate_comprehensive_cost_visualization(daily_cost_result, output_dir, batch_name)
        return daily_cost_result, figures

    return daily_cost_result, None


def evaluate_basic_costs_by_day(final_schedule, patients_data, time_slots_list,
                                slots_per_day=50, overtime_base=1.0, overtime_factor=0.5):
    # Compatible with original interface
    evaluator = ScheduleEvaluatorWithVisualization(slots_per_day, overtime_base, overtime_factor)
    return evaluator.evaluate_basic_costs_by_day(final_schedule, patients_data, time_slots_list)


def print_daily_cost_report(daily_cost_result, batch_name=""):
    # Compatible with original interface
    evaluator = ScheduleEvaluatorWithVisualization()
    evaluator.print_daily_cost_report(daily_cost_result, batch_name)


def save_daily_cost_results_to_excel(daily_cost_result, file_path, batch_name=""):
    # Compatible with original interface
    evaluator = ScheduleEvaluatorWithVisualization()
    evaluator.save_daily_cost_results_to_excel(daily_cost_result, file_path, batch_name)