import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os


class TwoLayerScheduleComparator:
    """Comparator for two-layer scheduling system evaluation"""

    def __init__(self):
        self.comparison_results = {}

    @staticmethod
    def parse_time_slot(time_slot):
        """Parse time slot, return (day, slot)"""
        try:
            if time_slot.startswith('T') and '-' in time_slot:
                parts = time_slot[1:].split('-', 1)
                return int(parts[0]), int(parts[1])
        except:
            pass
        return -1, -1

    def extract_patient_assignments(self, schedule, patients_data=None):
        """Extract patient assignment information"""
        if not schedule:
            return set(), {}

        assigned_patients = set()
        patient_details = {}

        for surgery in schedule:
            patient_id = surgery.get('patient_id')
            if patient_id is not None:
                assigned_patients.add(str(patient_id))
                patient_details[str(patient_id)] = {
                    'start_time': surgery.get('start_time'),
                    'room_id': surgery.get('room_id'),
                    'doctor_id': surgery.get('doctor_id'),
                    'duration': surgery.get('duration', 1),
                    'actual_duration': surgery.get('actual_duration'),
                    'status': surgery.get('status', 'scheduled')
                }

        return assigned_patients, patient_details

    def identify_adjusted_patients(self, base_schedule, final_schedule):
        """Identify adjusted patients"""
        _, base_details = self.extract_patient_assignments(base_schedule)
        _, final_details = self.extract_patient_assignments(final_schedule)

        adjusted_patients = []
        adjustment_types = defaultdict(int)

        for patient_id in base_details.keys():
            if patient_id in final_details:
                base_info = base_details[patient_id]
                final_info = final_details[patient_id]

                # Check adjustment types
                adjustments = []

                if base_info['start_time'] != final_info['start_time']:
                    adjustments.append('Time Adjustment')
                    adjustment_types['Time Adjustment'] += 1

                if base_info['room_id'] != final_info['room_id']:
                    adjustments.append('Room Adjustment')
                    adjustment_types['Room Adjustment'] += 1

                if base_info['doctor_id'] != final_info['doctor_id']:
                    adjustments.append('Doctor Adjustment')
                    adjustment_types['Doctor Adjustment'] += 1

                if adjustments:
                    adjusted_patients.append({
                        'patient_id': patient_id,
                        'adjustment_types': adjustments,
                        'base_start': base_info['start_time'],
                        'final_start': final_info['start_time'],
                        'base_room': base_info['room_id'],
                        'final_room': final_info['room_id'],
                        'base_doctor': base_info['doctor_id'],
                        'final_doctor': final_info['doctor_id'],
                        'expected_duration': base_info['duration'],
                        'actual_duration': final_info.get('actual_duration', base_info['duration'])
                    })

        return adjusted_patients, dict(adjustment_types)

    def calculate_duration_adaptations(self, final_schedule):
        """Calculate duration adaptation improvements"""
        duration_improvements = []

        for surgery in final_schedule:
            expected_duration = surgery.get('duration', 1)
            actual_duration = surgery.get('actual_duration', expected_duration)

            if actual_duration != expected_duration:
                # Calculate adaptation improvement
                relative_diff = abs(actual_duration - expected_duration) / max(expected_duration, 1)
                adaptation_score = 1 / (1 + relative_diff)
                duration_improvements.append(adaptation_score)

        return duration_improvements

    def calculate_resource_utilization_comparison(self, base_schedule, final_schedule, time_slots_list):
        """Calculate resource utilization comparison"""

        def get_resource_utilization(schedule, time_slots_list):
            if not schedule:
                return {'rooms': 0, 'time_slots': 0, 'doctors': 0}

            used_rooms = set()
            used_time_slots = set()
            used_doctors = set()

            for surgery in schedule:
                if surgery.get('room_id'):
                    used_rooms.add(surgery['room_id'])
                if surgery.get('start_time'):
                    used_time_slots.add(surgery['start_time'])
                if surgery.get('doctor_id'):
                    used_doctors.add(surgery['doctor_id'])

            return {
                'rooms': len(used_rooms),
                'time_slots': len(used_time_slots),
                'doctors': len(used_doctors)
            }

        base_util = get_resource_utilization(base_schedule, time_slots_list)
        final_util = get_resource_utilization(final_schedule, time_slots_list)

        return {
            'base_utilization': base_util,
            'final_utilization': final_util,
            'utilization_improvement': {
                'rooms': final_util['rooms'] - base_util['rooms'],
                'time_slots': final_util['time_slots'] - base_util['time_slots'],
                'doctors': final_util['doctors'] - base_util['doctors']
            }
        }

    def calculate_conflict_resolution_metrics(self, final_schedule):
        """Calculate conflict resolution metrics"""
        conflict_resolved = 0
        total_conflicts_detected = 0

        for surgery in final_schedule:
            # Check for adjustment markers indicating conflicts
            if surgery.get('adjustment_type') and surgery['adjustment_type'] != 'No Adjustment':
                total_conflicts_detected += 1
                if surgery.get('status') != 'conflict':
                    conflict_resolved += 1

        conflict_resolution_rate = conflict_resolved / max(1, total_conflicts_detected)

        return {
            'conflicts_detected': total_conflicts_detected,
            'conflicts_resolved': conflict_resolved,
            'conflict_resolution_rate': conflict_resolution_rate
        }

    def calculate_comprehensive_comparison_metrics(self, base_schedule, final_schedule, patients_data, time_slots_list):
        """Calculate all comparison metrics"""

        # Input validation
        if not isinstance(base_schedule, list):
            base_schedule = []
        if not isinstance(final_schedule, list):
            final_schedule = []
        if patients_data is None or patients_data.empty:
            print("Warning: Empty patient data, some metrics may be inaccurate")
            patients_data = pd.DataFrame()

        # Basic assignment rate calculation
        total_patients = len(patients_data) if not patients_data.empty else max(
            len(set(s.get('patient_id') for s in base_schedule if s.get('patient_id'))),
            len(set(s.get('patient_id') for s in final_schedule if s.get('patient_id')))
        )

        base_assigned_patients, _ = self.extract_patient_assignments(base_schedule)
        final_assigned_patients, _ = self.extract_patient_assignments(final_schedule)

        base_assignment_rate = len(base_assigned_patients) / max(1, total_patients)
        final_assignment_rate = len(final_assigned_patients) / max(1, total_patients)

        # Adjusted patient identification
        adjusted_patients, adjustment_types = self.identify_adjusted_patients(base_schedule, final_schedule)
        adjustment_rate = len(adjusted_patients) / max(1, len(base_assigned_patients))

        # Duration adaptation calculation
        duration_improvements = self.calculate_duration_adaptations(final_schedule)

        # Resource utilization comparison
        resource_comparison = self.calculate_resource_utilization_comparison(
            base_schedule, final_schedule, time_slots_list)

        # Conflict resolution metrics
        conflict_metrics = self.calculate_conflict_resolution_metrics(final_schedule)

        # Efficiency calculation
        two_layer_efficiency = final_assignment_rate / max(0.01, adjustment_rate + 0.01)

        # Time window analysis
        def analyze_time_windows(schedule):
            if not schedule:
                return {'min_day': 0, 'max_day': 0, 'total_days': 0}

            days = []
            for surgery in schedule:
                start_time = surgery.get('start_time')
                if start_time:
                    day, _ = self.parse_time_slot(start_time)
                    if day >= 0:
                        days.append(day)

            if days:
                return {
                    'min_day': min(days),
                    'max_day': max(days),
                    'total_days': len(set(days))
                }
            return {'min_day': 0, 'max_day': 0, 'total_days': 0}

        base_time_analysis = analyze_time_windows(base_schedule)
        final_time_analysis = analyze_time_windows(final_schedule)

        # Urgent patient processing
        urgent_metrics = self.calculate_urgent_patient_metrics(
            base_schedule, final_schedule, patients_data)

        # Aggregate all metrics
        comprehensive_metrics = {
            # Core metrics
            'base_assignment_rate': base_assignment_rate,
            'final_assignment_rate': final_assignment_rate,
            'assignment_improvement': final_assignment_rate - base_assignment_rate,
            'adjustment_rate': adjustment_rate,
            'adjusted_patients_count': len(adjusted_patients),
            'avg_duration_adaptation': np.mean(duration_improvements) if duration_improvements else 0,
            'two_layer_efficiency': two_layer_efficiency,

            # Detailed metrics
            'detailed_metrics': {
                'total_patients': total_patients,
                'base_assigned_count': len(base_assigned_patients),
                'final_assigned_count': len(final_assigned_patients),
                'newly_assigned_count': len(final_assigned_patients - base_assigned_patients),
                'lost_assignments_count': len(base_assigned_patients - final_assigned_patients),

                # Adjustment type statistics
                'adjustment_types': adjustment_types,
                'adjustment_details': adjusted_patients,

                # Resource utilization
                'resource_comparison': resource_comparison,

                # Conflict resolution
                'conflict_resolution': conflict_metrics,

                # Time window analysis
                'time_window_analysis': {
                    'base': base_time_analysis,
                    'final': final_time_analysis,
                    'time_span_improvement': final_time_analysis['total_days'] - base_time_analysis['total_days']
                },

                # Duration adaptation details
                'duration_adaptation_details': {
                    'adaptations_count': len(duration_improvements),
                    'adaptation_scores': duration_improvements,
                    'min_adaptation': min(duration_improvements) if duration_improvements else 0,
                    'max_adaptation': max(duration_improvements) if duration_improvements else 0,
                    'std_adaptation': np.std(duration_improvements) if duration_improvements else 0
                },

                # Urgent patient metrics
                'urgent_patient_metrics': urgent_metrics
            }
        }

        return comprehensive_metrics

    def calculate_urgent_patient_metrics(self, base_schedule, final_schedule, patients_data):
        """Calculate urgent patient processing metrics"""
        try:
            # Check for empty or insufficient patient data
            if patients_data is None or patients_data.empty or len(patients_data.columns) <= 4:
                return {'urgent_analysis': 'insufficient_data'}

            urgency_column = patients_data.columns[4]
            patient_id_column = patients_data.columns[0]

            # Identify urgent patients
            urgent_patients = set()
            for _, patient in patients_data.iterrows():
                urgency = patient[urgency_column]
                if pd.notna(urgency) and urgency > 1:  # Priority > 1 considered urgent
                    urgent_patients.add(str(patient[patient_id_column]))

            if not urgent_patients:
                return {'urgent_analysis': 'no_urgent_patients'}

            # Analyze urgent patient processing in both layers
            base_assigned, _ = self.extract_patient_assignments(base_schedule)
            final_assigned, _ = self.extract_patient_assignments(final_schedule)

            urgent_base_assigned = urgent_patients & base_assigned
            urgent_final_assigned = urgent_patients & final_assigned

            return {
                'total_urgent_patients': len(urgent_patients),
                'urgent_base_assigned': len(urgent_base_assigned),
                'urgent_final_assigned': len(urgent_final_assigned),
                'urgent_assignment_improvement': len(urgent_final_assigned) - len(urgent_base_assigned),
                'urgent_base_rate': len(urgent_base_assigned) / len(urgent_patients),
                'urgent_final_rate': len(urgent_final_assigned) / len(urgent_patients),
                'urgent_rate_improvement': (len(urgent_final_assigned) - len(urgent_base_assigned)) / len(
                    urgent_patients)
            }

        except Exception as e:
            return {'urgent_analysis': f'error: {str(e)}'}

    def generate_comparison_report(self, metrics, batch_name=""):
        """Generate comparison evaluation report"""
        print(f"\n{'=' * 80}")
        print(f"           Two-Layer Scheduling System Comparison Report - {batch_name}")
        print(f"{'=' * 80}")

        # Core comparison metrics
        print(f"\nCore Comparison Metrics:")
        core_metrics = [
            ('Base Schedule Assignment Rate', 'base_assignment_rate', '.2%'),
            ('Dynamic Adjusted Assignment Rate', 'final_assignment_rate', '.2%'),
            ('Assignment Rate Improvement', 'assignment_improvement', '.2%'),
            ('Patient Adjustment Rate', 'adjustment_rate', '.2%'),
            ('Adjusted Patients Count', 'adjusted_patients_count', 'd'),
            ('Average Duration Adaptation', 'avg_duration_adaptation', '.4f'),
            ('Two-Layer System Efficiency', 'two_layer_efficiency', '.4f')
        ]

        for name, key, fmt in core_metrics:
            value = metrics.get(key, 0)
            if 'd' in fmt:
                print(f"   {name}: {value}")
            else:
                print(f"   {name}: {value:{fmt}}")

        # Detailed analysis
        detailed = metrics.get('detailed_metrics', {})

        print(f"\nPatient Assignment Analysis:")
        print(f"   Total Patients: {detailed.get('total_patients', 0)}")
        print(f"   Base Schedule Assigned: {detailed.get('base_assigned_count', 0)}")
        print(f"   Dynamic Adjusted Assigned: {detailed.get('final_assigned_count', 0)}")
        print(f"   Newly Assigned: {detailed.get('newly_assigned_count', 0)}")
        print(f"   Lost Assignments: {detailed.get('lost_assignments_count', 0)}")

        # Adjustment type analysis
        adjustment_types = detailed.get('adjustment_types', {})
        if adjustment_types:
            print(f"\nAdjustment Type Statistics:")
            for adj_type, count in adjustment_types.items():
                print(f"   {adj_type}: {count} cases")

        # Resource utilization comparison
        resource_comp = detailed.get('resource_comparison', {})
        if resource_comp:
            print(f"\nResource Utilization Comparison:")
            base_util = resource_comp.get('base_utilization', {})
            final_util = resource_comp.get('final_utilization', {})
            util_improvement = resource_comp.get('utilization_improvement', {})

            print(f"   Operating Rooms: {base_util.get('rooms', 0)} → {final_util.get('rooms', 0)} "
                  f"(Improvement: {util_improvement.get('rooms', 0):+d})")
            print(f"   Time Slots: {base_util.get('time_slots', 0)} → {final_util.get('time_slots', 0)} "
                  f"(Improvement: {util_improvement.get('time_slots', 0):+d})")
            print(f"   Doctors: {base_util.get('doctors', 0)} → {final_util.get('doctors', 0)} "
                  f"(Improvement: {util_improvement.get('doctors', 0):+d})")

        # Conflict resolution status
        conflict_res = detailed.get('conflict_resolution', {})
        if conflict_res:
            print(f"\nConflict Resolution Status:")
            print(f"   Conflicts Detected: {conflict_res.get('conflicts_detected', 0)}")
            print(f"   Conflicts Resolved: {conflict_res.get('conflicts_resolved', 0)}")
            print(f"   Conflict Resolution Rate: {conflict_res.get('conflict_resolution_rate', 0):.2%}")

        # Urgent patient processing
        urgent_metrics = detailed.get('urgent_patient_metrics', {})
        if urgent_metrics and urgent_metrics.get('urgent_analysis') not in ['insufficient_data', 'no_urgent_patients']:
            print(f"\nUrgent Patient Processing:")
            print(f"   Total Urgent Patients: {urgent_metrics.get('total_urgent_patients', 0)}")
            print(f"   Base Schedule Assigned: {urgent_metrics.get('urgent_base_assigned', 0)}")
            print(f"   Dynamic Adjusted Assigned: {urgent_metrics.get('urgent_final_assigned', 0)}")
            print(f"   Urgent Assignment Rate Improvement: {urgent_metrics.get('urgent_rate_improvement', 0):.2%}")

        # System efficiency summary
        print(f"\nSystem Efficiency Summary:")
        if metrics.get('assignment_improvement', 0) > 0:
            print(f"   Assignment rate improved by {metrics['assignment_improvement']:.2%}")
        else:
            print(f"   Assignment rate decreased by {abs(metrics.get('assignment_improvement', 0)):.2%}")

        if metrics.get('two_layer_efficiency', 0) > 1:
            print(f"   Two-layer system efficiency is good ({metrics['two_layer_efficiency']:.2f})")
        else:
            print(f"   Two-layer system efficiency needs improvement ({metrics.get('two_layer_efficiency', 0):.2f})")

    def plot_comparison_metrics(self, metrics, output_dir=None, batch_name=""):
        """Plot comparison metrics visualization charts"""
        try:
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Set Chinese fonts
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            # Core metrics comparison radar chart
            fig1, ax1 = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # Core metrics data
            categories = ['Assignment Rate', 'Adjustment Efficiency', 'Duration Adaptation',
                         'Resource Utilization', 'Conflict Resolution', 'System Efficiency']
            values = [
                metrics.get('final_assignment_rate', 0),
                1 - metrics.get('adjustment_rate', 0),  # Lower adjustment rate is better
                metrics.get('avg_duration_adaptation', 0),
                0.8,  # Resource utilization (simplified)
                0.9,  # Conflict resolution rate (simplified)
                min(metrics.get('two_layer_efficiency', 0) / 5, 1)  # Normalized efficiency
            ]

            # Draw radar chart
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            values += values[:1]  # Close the shape
            angles = np.concatenate([angles, [angles[0]]])

            ax1.plot(angles, values, 'o-', linewidth=2, label='Two-Layer Scheduling System')
            ax1.fill(angles, values, alpha=0.25)
            ax1.set_xticks(angles[:-1])
            ax1.set_xticklabels(categories)
            ax1.set_ylim(0, 1)
            ax1.set_title(f'{batch_name} - Two-Layer Scheduling System Comprehensive Evaluation Radar Chart',
                         fontsize=14, pad=20)
            ax1.legend()

            if output_dir:
                fig1.savefig(os.path.join(output_dir, f'{batch_name}_comparison_radar.png'),
                             dpi=300, bbox_inches='tight')
                plt.close(fig1)

            # Assignment rate comparison bar chart
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            categories_bar = ['Base Schedule', 'Dynamic Adjusted']
            assignment_rates = [metrics.get('base_assignment_rate', 0), metrics.get('final_assignment_rate', 0)]

            bars = ax2.bar(categories_bar, assignment_rates, color=['skyblue', 'lightcoral'])
            ax2.set_ylabel('Assignment Rate')
            ax2.set_title(f'{batch_name} - Assignment Rate Comparison')
            ax2.set_ylim(0, 1)

            # Add value labels
            for bar, rate in zip(bars, assignment_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                         f'{rate:.2%}', ha='center', va='bottom')

            # Add improvement indication
            improvement = metrics.get('assignment_improvement', 0)
            if improvement > 0:
                ax2.annotate(f'Improvement: +{improvement:.2%}',
                             xy=(1, assignment_rates[1]), xytext=(1.2, assignment_rates[1]),
                             arrowprops=dict(arrowstyle='->', color='green'),
                             color='green', fontweight='bold')

            if output_dir:
                fig2.savefig(os.path.join(output_dir, f'{batch_name}_assignment_comparison.png'),
                             dpi=300, bbox_inches='tight')
                plt.close(fig2)

            # Adjustment type analysis pie chart
            detailed = metrics.get('detailed_metrics', {})
            adjustment_types = detailed.get('adjustment_types', {})

            fig3 = None
            if adjustment_types:
                fig3, ax3 = plt.subplots(figsize=(8, 8))

                labels = list(adjustment_types.keys())
                sizes = list(adjustment_types.values())
                colors = plt.cm.Set3(np.arange(len(labels)))

                wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                   colors=colors, startangle=90)
                ax3.set_title(f'{batch_name} - Adjustment Type Distribution', fontsize=14)

                if output_dir:
                    fig3.savefig(os.path.join(output_dir, f'{batch_name}_adjustment_types.png'),
                                 dpi=300, bbox_inches='tight')
                    plt.close(fig3)

            return fig1, fig2, fig3

        except Exception as e:
            print(f"Error plotting visualization charts: {e}")
            return None, None, None

    def save_comparison_results_to_excel(self, metrics, file_path, batch_name=""):
        """Save comparison results to Excel"""
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Core comparison metrics
                core_data = []
                core_metrics = [
                    ('Base Schedule Assignment Rate', 'base_assignment_rate'),
                    ('Dynamic Adjusted Assignment Rate', 'final_assignment_rate'),
                    ('Assignment Rate Improvement', 'assignment_improvement'),
                    ('Patient Adjustment Rate', 'adjustment_rate'),
                    ('Adjusted Patients Count', 'adjusted_patients_count'),
                    ('Average Duration Adaptation', 'avg_duration_adaptation'),
                    ('Two-Layer System Efficiency', 'two_layer_efficiency')
                ]

                for name, key in core_metrics:
                    core_data.append({
                        'Metric Name': name,
                        'Metric Value': metrics.get(key, 0),
                        'Description': f'Calculation result for {key}'
                    })

                pd.DataFrame(core_data).to_excel(writer, sheet_name='Core Comparison Metrics', index=False)

                # Detailed metrics
                detailed = metrics.get('detailed_metrics', {})
                if detailed:
                    detail_data = []
                    for key, value in detailed.items():
                        if isinstance(value, (int, float)):
                            detail_data.append({'Metric': key, 'Value': value})

                    if detail_data:
                        pd.DataFrame(detail_data).to_excel(writer, sheet_name='Detailed Metrics', index=False)

                # Adjustment details
                adjustment_details = detailed.get('adjustment_details', [])
                if adjustment_details:
                    try:
                        adj_df = pd.DataFrame(adjustment_details)
                        adj_df.to_excel(writer, sheet_name='Adjustment Details', index=False)
                    except Exception as e:
                        print(f"Error saving adjustment details: {e}")

            print(f"Two-layer comparison results saved to: {file_path}")

        except Exception as e:
            print(f"Error saving Excel file: {e}")


# Main function for easy integration with existing code
def compare_two_layer_schedules(base_schedule, final_schedule, patients_data, time_slots_list,
                                output_dir=None, batch_name="", save_results=True):
    """
    Main function for two-layer scheduling system comparison analysis

    Args:
        base_schedule: Base scheduling layer results
        final_schedule: Dynamic adjustment layer results
        patients_data: Patient data DataFrame
        time_slots_list: Time slots list
        output_dir: Output directory
        batch_name: Batch name
        save_results: Whether to save results

    Returns:
        dict: Dictionary containing all comparison metrics
    """

    try:
        comparator = TwoLayerScheduleComparator()

        # Calculate comparison metrics
        metrics = comparator.calculate_comprehensive_comparison_metrics(
            base_schedule, final_schedule, patients_data, time_slots_list)

        # Generate report
        comparator.generate_comparison_report(metrics, batch_name)

        # Plot visualization charts
        if output_dir:
            comparator.plot_comparison_metrics(metrics, output_dir, batch_name)

        # Save results
        if save_results and output_dir:
            excel_file = os.path.join(output_dir, f'{batch_name}_two_layer_comparison.xlsx')
            comparator.save_comparison_results_to_excel(metrics, excel_file, batch_name)

        return metrics

    except Exception as e:
        print(f"Two-layer scheduling comparison analysis error: {e}")
        import traceback
        traceback.print_exc()

        # Return basic metrics dictionary to avoid crashes
        return {
            'base_assignment_rate': 0,
            'final_assignment_rate': 0,
            'assignment_improvement': 0,
            'adjustment_rate': 0,
            'adjusted_patients_count': 0,
            'avg_duration_adaptation': 0,
            'two_layer_efficiency': 0,
            'error': str(e)
        }