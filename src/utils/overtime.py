import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import time
import math
import os


class OvertimeTracker:
    """Track overtime extension details"""

    def __init__(self):
        self.records = []
        self.current_batch = None

    def start_tracking(self, batch_name, initial_patients, initial_slots):
        """Start tracking new batch"""
        self.current_batch = {
            'batch_name': batch_name,
            'initial_patients': initial_patients,
            'initial_slots': initial_slots,
            'extensions': [],
            'start_time': time.time()
        }

    def record_extension(self, round_num, assigned, unassigned, total_slots,
                        extended_days, step_size, cost_data=None):
        """Record single extension information"""
        if self.current_batch is None:
            return

        total_patients = assigned + unassigned
        assignment_rate = assigned / total_patients if total_patients > 0 else 0

        extension_data = {
            'round': round_num,
            'assigned': assigned,
            'unassigned': unassigned,
            'total_patients': total_patients,
            'assignment_rate': assignment_rate,
            'total_slots': total_slots,
            'extended_days': extended_days,
            'step_size': step_size,
            'overtime_blocks': total_slots - self.current_batch['initial_slots'],
            'timestamp': time.time(),
            'cost_data': cost_data or {}
        }

        self.current_batch['extensions'].append(extension_data)

    def finish_tracking(self, final_schedule=None):
        """Complete batch tracking"""
        if self.current_batch is None:
            return

        self.current_batch['end_time'] = time.time()
        self.current_batch['duration'] = self.current_batch['end_time'] - self.current_batch['start_time']

        if final_schedule:
            self.current_batch['final_summary'] = self._analyze_schedule(final_schedule)

        self.records.append(self.current_batch.copy())
        self.current_batch = None

    def _analyze_schedule(self, schedule):
        """Analyze final schedule"""
        total_surgeries = len(schedule)
        overtime_surgeries = 0

        for surgery in schedule:
            start_time = surgery.get('start_time', '')
            if start_time and '-' in start_time:
                try:
                    slot = int(start_time.split('-')[1])
                    if slot >= 60:  # overtime threshold
                        overtime_surgeries += 1
                except:
                    pass

        return {
            'total_surgeries': total_surgeries,
            'overtime_surgeries': overtime_surgeries,
            'overtime_rate': overtime_surgeries / max(1, total_surgeries)
        }

    def get_batch_data(self, batch_name):
        """Get specific batch data"""
        for record in self.records:
            if record['batch_name'] == batch_name:
                return record
        return None

    def get_all_records(self):
        """Get all records"""
        return self.records


def run_overtime_scheduling(patients, ts_idx, doctor_availability, room_availability,
                          time_slots_list, doctor_ids, room_ids, delta_duration=0,
                          max_overtime=20, overtime_step=2, reg=0.1, max_iter=2000,
                          batch_name="", tracker=None):
    """
    Run scheduling with overtime extension
    """
    from modeling import wasserstein_sinkhorn
    from visualization import report

    if tracker is None:
        tracker = OvertimeTracker()

    tracker.start_tracking(batch_name, len(patients), len(time_slots_list))

    arrival_day_col = patients.columns[3]
    extend_count = 0
    current_slots = time_slots_list.copy()
    current_doctor_avail = {k: v.copy() for k, v in doctor_availability.items()}
    current_room_avail = {k: v.copy() for k, v in room_availability.items()}

    print(f"\nStarting overtime scheduling - Batch: {batch_name}")

    while True:
        print(f"\nAttempt #{extend_count + 1}: Time slots = {len(current_slots)}")

        # Build cost matrix
        cost_matrix = wasserstein_sinkhorn.build_cost_matrix(
            patients, ts_idx, current_doctor_avail, current_room_avail,
            current_slots, delta_duration
        )

        n_patients = len(patients)
        n_slots = len(current_slots)
        n_doctors = len(current_doctor_avail)
        n_rooms = len(current_room_avail)

        drt_indices = [(d, r, t) for d in range(n_doctors)
                       for r in range(n_rooms)
                       for t in range(n_slots)]

        # Solve transport problem
        assignment_matrix, assignment_indices = wasserstein_sinkhorn.solve_transport_problem(
            cost_matrix, n_patients, drt_indices,
            list(current_doctor_avail.keys()),
            list(current_room_avail.keys()),
            current_slots,
            patients,
            reg=reg,
            numItermax=max_iter,
            delta_duration=delta_duration
        )

        if assignment_indices is None:
            print("Solving failed, terminating.")
            tracker.finish_tracking()
            return None, None, current_slots, tracker

        # Count assignments
        unassigned = [i for i, idx in enumerate(assignment_indices) if idx == -1]
        assigned_count = n_patients - len(unassigned)
        unassigned_count = len(unassigned)

        # Basic cost statistics
        cost_stats = {}
        if cost_matrix is not None and cost_matrix.size > 0:
            finite_costs = cost_matrix[~np.isinf(cost_matrix)]
            if len(finite_costs) > 0:
                cost_stats = {
                    'mean_cost': float(np.mean(finite_costs)),
                    'min_cost': float(np.min(finite_costs)),
                    'max_cost': float(np.max(finite_costs))
                }

        # Record extension
        unassigned_patients = patients.iloc[unassigned] if unassigned else pd.DataFrame()
        extended_days = sorted(unassigned_patients[arrival_day_col].unique()) if not unassigned_patients.empty else []

        tracker.record_extension(
            round_num=extend_count,
            assigned=assigned_count,
            unassigned=unassigned_count,
            total_slots=len(current_slots),
            extended_days=extended_days,
            step_size=overtime_step,
            cost_data=cost_stats
        )

        print(f"Round {extend_count + 1} result: {assigned_count}/{n_patients} ({assigned_count/n_patients:.2%})")

        if len(unassigned) == 0:
            print("All patients scheduled successfully.")
            schedule, _ = report.generate_schedule(
                patients, assignment_indices, drt_indices, ts_idx,
                list(current_doctor_avail.keys()), list(current_room_avail.keys()), current_slots
            )
            tracker.finish_tracking(schedule)
            return assignment_matrix, assignment_indices, current_slots, tracker
        else:
            print(f"{len(unassigned)} patients unassigned.")

            if extend_count * overtime_step > max_overtime:
                print(f"Maximum overtime limit ({max_overtime} blocks) reached.")
                schedule, _ = report.generate_schedule(
                    patients, assignment_indices, drt_indices, ts_idx,
                    list(current_doctor_avail.keys()), list(current_room_avail.keys()), current_slots
                )
                tracker.finish_tracking(schedule)
                return assignment_matrix, assignment_indices, current_slots, tracker

            print(f"Unassigned patients arrival days: {extended_days}")

            # Extend overtime for unassigned days
            for day in extended_days:
                print(f"Extending day {day} by {overtime_step} time blocks.")
                current_slots, insert_pos, num_new = wasserstein_sinkhorn.extend_time_slots_by_day(
                    current_slots, day, overtime_step
                )
                wasserstein_sinkhorn.extend_availability_at_pos(current_doctor_avail, insert_pos, num_new)
                wasserstein_sinkhorn.extend_availability_at_pos(current_room_avail, insert_pos, num_new)

            extend_count += 1


class OvertimeEvaluator:
    """Evaluate overtime extension process"""

    def __init__(self):
        self.results = {}

    def evaluate_process(self, tracker, output_dir=None, save_plots=True):
        """Evaluate complete overtime extension process"""
        print("\nStarting overtime extension evaluation...")

        all_records = tracker.get_all_records()
        if not all_records:
            print("No extension records found")
            return {}

        results = {}

        for batch_record in all_records:
            batch_name = batch_record['batch_name']
            print(f"\nEvaluating batch: {batch_name}")

            batch_eval = self._evaluate_batch(batch_record)
            results[batch_name] = batch_eval

            self._print_summary(batch_name, batch_eval)

        # Cross-batch statistics
        cross_stats = self._calculate_cross_stats(results)
        results['cross_batch_stats'] = cross_stats

        # Generate outputs
        if output_dir and save_plots:
            self._generate_plots(results, output_dir)
            self._save_report(results, output_dir)

        self.results = results
        return results

    def _evaluate_batch(self, batch_record):
        """Evaluate single batch"""
        extensions = batch_record.get('extensions', [])
        if not extensions:
            return {'error': 'No extension records'}

        total_extensions = len(extensions)
        final_ext = extensions[-1]
        initial_rate = extensions[0]['assignment_rate'] if extensions else 0
        final_rate = final_ext['assignment_rate']

        # Calculate improvements
        improvements = []
        for i in range(1, len(extensions)):
            improvement = extensions[i]['assignment_rate'] - extensions[i-1]['assignment_rate']
            improvements.append(improvement)

        return {
            'basic_stats': {
                'total_extensions': total_extensions,
                'initial_rate': initial_rate,
                'final_rate': final_rate,
                'improvement': final_rate - initial_rate,
                'overtime_blocks': final_ext['overtime_blocks'],
                'duration': batch_record.get('duration', 0)
            },
            'efficiency': {
                'avg_improvement': np.mean(improvements) if improvements else 0,
                'improvement_stability': np.std(improvements) if improvements else 0,
                'blocks_efficiency': (final_rate - initial_rate) / max(1, final_ext['overtime_blocks'])
            },
            'extension_details': extensions
        }

    def _calculate_cross_stats(self, results):
        """Calculate cross-batch statistics"""
        batch_results = {k: v for k, v in results.items() if k != 'cross_batch_stats'}

        if not batch_results:
            return {}

        final_rates = []
        improvements = []
        extensions = []

        for batch_name, result in batch_results.items():
            if 'error' not in result:
                basic = result.get('basic_stats', {})
                final_rates.append(basic.get('final_rate', 0))
                improvements.append(basic.get('improvement', 0))
                extensions.append(basic.get('total_extensions', 0))

        return {
            'mean_final_rate': np.mean(final_rates) if final_rates else 0,
            'mean_improvement': np.mean(improvements) if improvements else 0,
            'mean_extensions': np.mean(extensions) if extensions else 0,
            'rate_consistency': 1 - (np.std(final_rates) / np.mean(final_rates)
                                   if np.mean(final_rates) > 0 and final_rates else 0)
        }

    def _print_summary(self, batch_name, evaluation):
        """Print batch evaluation summary"""
        if 'error' in evaluation:
            print(f"   Error in {batch_name}: {evaluation['error']}")
            return

        basic = evaluation.get('basic_stats', {})
        efficiency = evaluation.get('efficiency', {})

        print(f"   {batch_name} Results:")
        print(f"      Extensions: {basic.get('total_extensions', 0)}")
        print(f"      Final rate: {basic.get('final_rate', 0):.2%}")
        print(f"      Improvement: {basic.get('improvement', 0):.2%}")
        print(f"      Avg improvement/round: {efficiency.get('avg_improvement', 0):.3%}")

    def _generate_plots(self, results, output_dir):
        """Generate visualization plots"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False

            os.makedirs(output_dir, exist_ok=True)

            self._plot_convergence(results, output_dir)
            self._plot_efficiency(results, output_dir)

            print(f"Plots saved to: {output_dir}")

        except Exception as e:
            print(f"Error generating plots: {e}")

    def _plot_convergence(self, results, output_dir):
        """Plot assignment rate convergence"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        batch_results = {k: v for k, v in results.items() if k != 'cross_batch_stats'}

        # Assignment rate convergence
        for batch_name, result in batch_results.items():
            if 'error' not in result:
                extensions = result.get('extension_details', [])
                rounds = [ext['round'] for ext in extensions]
                rates = [ext['assignment_rate'] for ext in extensions]
                ax1.plot(rounds, rates, marker='o', label=batch_name, linewidth=2)

        ax1.set_xlabel('Extension Round')
        ax1.set_ylabel('Assignment Rate')
        ax1.set_title('Assignment Rate Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Overtime blocks accumulation
        for batch_name, result in batch_results.items():
            if 'error' not in result:
                extensions = result.get('extension_details', [])
                rounds = [ext['round'] for ext in extensions]
                blocks = [ext['overtime_blocks'] for ext in extensions]
                ax2.plot(rounds, blocks, marker='s', label=batch_name, linewidth=2)

        ax2.set_xlabel('Extension Round')
        ax2.set_ylabel('Overtime Blocks')
        ax2.set_title('Overtime Blocks Accumulation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overtime_convergence.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_efficiency(self, results, output_dir):
        """Plot efficiency analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        batch_results = {k: v for k, v in results.items() if k != 'cross_batch_stats'}

        # Efficiency comparison
        names = []
        final_rates = []
        improvements = []

        for batch_name, result in batch_results.items():
            if 'error' not in result:
                basic = result.get('basic_stats', {})
                names.append(batch_name)
                final_rates.append(basic.get('final_rate', 0))
                improvements.append(basic.get('improvement', 0))

        if names:
            x = np.arange(len(names))
            width = 0.35

            ax1.bar(x - width/2, final_rates, width, label='Final Rate', alpha=0.8)
            ax1.bar(x + width/2, improvements, width, label='Improvement', alpha=0.8)
            ax1.set_xlabel('Batch')
            ax1.set_ylabel('Rate')
            ax1.set_title('Final Rates and Improvements')
            ax1.set_xticks(x)
            ax1.set_xticklabels(names, rotation=45)
            ax1.legend()

        # Cost-benefit scatter
        costs = []
        benefits = []
        for batch_name, result in batch_results.items():
            if 'error' not in result:
                basic = result.get('basic_stats', {})
                cost = basic.get('overtime_blocks', 0)
                benefit = basic.get('improvement', 0)
                costs.append(cost)
                benefits.append(benefit)

        if costs and benefits:
            scatter = ax2.scatter(costs, benefits, s=100, alpha=0.7)
            ax2.set_xlabel('Overtime Blocks')
            ax2.set_ylabel('Assignment Rate Improvement')
            ax2.set_title('Cost-Benefit Analysis')
            ax2.grid(True, alpha=0.3)

            for i, name in enumerate(names):
                ax2.annotate(name, (costs[i], benefits[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overtime_efficiency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_report(self, results, output_dir):
        """Save detailed report to Excel"""
        try:
            report_file = os.path.join(output_dir, 'overtime_report.xlsx')

            with pd.ExcelWriter(report_file, engine='openpyxl') as writer:
                self._create_summary_sheet(results, writer)

                batch_results = {k: v for k, v in results.items() if k != 'cross_batch_stats'}
                for batch_name, result in batch_results.items():
                    if 'error' not in result:
                        self._create_detail_sheet(batch_name, result, writer)

            print(f"Report saved to: {report_file}")

        except Exception as e:
            print(f"Error saving report: {e}")

    def _create_summary_sheet(self, results, writer):
        """Create summary sheet"""
        summary_data = []
        batch_results = {k: v for k, v in results.items() if k != 'cross_batch_stats'}

        for batch_name, result in batch_results.items():
            if 'error' not in result:
                basic = result.get('basic_stats', {})
                efficiency = result.get('efficiency', {})

                summary_data.append({
                    'Batch': batch_name,
                    'Extensions': basic.get('total_extensions', 0),
                    'Initial Rate': f"{basic.get('initial_rate', 0):.2%}",
                    'Final Rate': f"{basic.get('final_rate', 0):.2%}",
                    'Improvement': f"{basic.get('improvement', 0):.2%}",
                    'Overtime Blocks': basic.get('overtime_blocks', 0),
                    'Avg Improvement': f"{efficiency.get('avg_improvement', 0):.3%}"
                })

        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

    def _create_detail_sheet(self, batch_name, result, writer):
        """Create batch detail sheet"""
        extensions = result.get('extension_details', [])

        detail_data = []
        for ext in extensions:
            detail_data.append({
                'Round': ext['round'],
                'Assigned': ext['assigned'],
                'Unassigned': ext['unassigned'],
                'Rate': f"{ext['assignment_rate']:.2%}",
                'Total Slots': ext['total_slots'],
                'Overtime Blocks': ext['overtime_blocks'],
                'Extended Days': str(ext['extended_days']),
                'Step Size': ext['step_size']
            })

        sheet_name = batch_name[:31]  # Excel sheet name limit
        pd.DataFrame(detail_data).to_excel(writer, sheet_name=sheet_name, index=False)