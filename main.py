import os
import time
import numpy as np
import pandas as pd
from data_processing import data_loader
from visualization import scheduling_visualisation,report
from modeling import wasserstein_uncertainty,wasserstein_sinkhorn,dynamic_surgery
import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from current font")



def main():
    """主函数，协调整个手术调度过程，包括基础排程层和动态调整层"""
    print("=" * 30 + " 开始手术调度过程 " + "=" * 30)

    base_path = r'D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\processed\G_1'
    batch_count = 20

    output_dir = r"D:\python_project\pythonProject\surgical_scheduling_optimal_transport\data\output"
    os.makedirs(output_dir, exist_ok=True)

    try:
        # ==== 第一层：基础排程层 ====
        print("\n" + "=" * 20 + " 第一层：基础排程层 " + "=" * 20)

        #  加载数据
        print("\n---  加载数据 ---")
        time_slots, patients_batches, doctors_batches, operating_rooms = data_loader.load_all_data(base_path, batch_count)
        if time_slots.empty or operating_rooms.empty:
            return

        time_slot_column = time_slots.columns[0]
        time_slots_list = time_slots[time_slot_column].unique().tolist()

        #  预处理数据
        print("\n---  预处理数据 ---")
        # 按批次号排序，划分训练（历史）和验证批次
        all_batch_keys = sorted(patients_batches.keys(), key=data_loader.batch_key_sort_key)
        historical_keys = [k for k in all_batch_keys if 1 <= data_loader.batch_key_sort_key(k) <= 16]
        validation_keys = [k for k in all_batch_keys if data_loader.batch_key_sort_key(k) >= 17]

        print(f"历史数据批次数量: {len(historical_keys)}, 验证数据批次数量: {len(validation_keys)}")

        # 从历史批次中提取手术时长样本
        train_durations = []
        for key in historical_keys:
            df = patients_batches.get(key, None)
            if df is not None and not df.empty and 'Duration' in df.columns:
                durations = df['Duration'].dropna().values
                train_durations.append(durations)
            else:
                print(f"[警告] 训练批次 {key} 不存在数据或无'Duration'列，已跳过。")

        if train_durations:
            train_durations = np.concatenate(train_durations)
            print(f"历史数据所有批次合并手术时长样本总数: {len(train_durations)}")
            print(f"训练样本均值: {train_durations.mean():.2f}, 标准差: {train_durations.std():.2f}")
        else:
            train_durations = np.array([])
            print("警告: 训练样本为空，无法计算动态delta_duration。")

        # Wasserstein不确定球半径设定
        epsilon = 1

        # 计算动态 delta_duration
        delta_duration = wasserstein_uncertainty.compute_dynamic_delta_duration(train_durations, epsilon, alpha=0.1)

        for batch_key in validation_keys:
            print(f"\n===== 处理验证批次 {batch_key} =====")
            patients = patients_batches.get(batch_key, None)
            doctors = doctors_batches.get(batch_key, None)

            if patients is None or patients.empty:
                print(f"批次 {batch_key} 的患者数据为空，跳过。")
                continue
            if doctors is None or doctors.empty:
                print(f"批次 {batch_key} 的医生数据为空，跳过。")
                continue

            # 预处理数据生成索引和可用性
            ts_idx, doctor_availability, room_availability, ts_list = data_loader.preprocess_data(
                time_slots, patients, doctors, operating_rooms)
            if ts_idx is None:
                print(f"批次 {batch_key} 预处理失败，跳过该批次。")
                continue

            doctor_ids = list(doctor_availability.keys())
            room_ids = list(room_availability.keys())

            #  求解最优传输与硬分配
            print("\n---  求解最优传输与硬分配 ---")
            # 调用自动加班调度主流程，传入动态delta_duration
            hard_assignment_matrix, hard_assignment_indices, final_time_slots = wasserstein_sinkhorn.run_scheduling_with_overtime(
                patients, ts_idx, doctor_availability, room_availability, ts_list,
                doctor_ids, room_ids,
                delta_duration=delta_duration,
                max_overtime_blocks=36,
                overtime_step=2,
                reg=0.1,
                numItermax=2000)

            if hard_assignment_indices is None:
                print(f"批次 {batch_key} 求解失败，跳过调度生成。")
                continue

            base_schedule, assigned_patient_indices = report.generate_schedule(
                patients, hard_assignment_indices,
                [(d, r, t) for d in range(len(doctor_ids))
                           for r in range(len(room_ids))
                           for t in range(len(final_time_slots))],
                ts_idx, doctor_ids, room_ids, final_time_slots)

            report_df = report.generate_report(base_schedule, patients, assigned_patient_indices)

            report_file = os.path.join(output_dir, f"{batch_key}_Basic_surgery_assignment_report.xlsx")
            try:
                report_df.to_excel(report_file, index=False, engine='openpyxl')
                print(f"批次 {batch_key} 调度报告已保存至: {os.path.abspath(report_file)}")
            except Exception as e:
                print(f"保存批次 {batch_key} 报告时出错: {e}")

            # ==== 第二层：动态调整层 ====
            print("\n" + "=" * 20 + " 第二层：动态调整层 " + "=" * 20)

            #  模拟实际手术时长
            print("\n---  模拟实际手术时长 ---")
            schedule_with_actual = dynamic_surgery.simulate_actual_durations(
                base_schedule,
                variation_range=(-0.3, 0.5),  # 实际时长为预期的70%-150%
                seed=42  # 设置随机数种子以保证可重复性
            )

            fig = scheduling_visualisation.plot_gantt_chart(base_schedule, final_time_slots, title=f"{batch_key}_Basic_surgical_schedule_gantt_chart")

            if fig is not None:
                gantt_file = os.path.join(output_dir, f"{batch_key}_Basic_surgical_schedule_gantt_chart.svg")
                try:
                    fig.savefig(gantt_file, dpi=300, bbox_inches='tight')
                    print(f"批次 {batch_key} 甘特图已保存至: {os.path.abspath(gantt_file)}")
                except Exception as e:
                    print(f"保存批次 {batch_key} 甘特图时出错: {e}")

            # 10. 动态调整层调用
            final_schedule = dynamic_surgery.dynamic_time_progression_scheduling(
                base_schedule,
                final_time_slots,
                room_availability,
                doctor_availability,
                patients,
                ts_idx,
                [(d, r, t) for d in range(len(doctor_ids))
                           for r in range(len(room_ids))
                           for t in range(len(final_time_slots))],
                doctor_ids,
                room_ids,
                variation_range=(-0.3, 0.5),
                seed=42
            )

            # 11. 生成手术时长对比图（动态数据）
            fig_bar, fig_scatter = scheduling_visualisation.plot_simple_duration_comparison(
                schedule_with_actual=dynamic_surgery.schedule_with_actual,
                max_patients=40,
                batch_key=batch_key,
                output_dir=output_dir
            )

            # 统一保存图片文件
            if output_dir and batch_key:
                os.makedirs(output_dir, exist_ok=True)
                bar_file = os.path.join(output_dir, f"{batch_key}_surgery_duration_comparison.svg")
                try:
                    fig_bar.savefig(bar_file, dpi=300)
                    print(f"条形图已保存：{bar_file}")
                except Exception as e:
                    print(f"保存条形图出错：{e}")

                # 保存散点图
                scatter_file = os.path.join(output_dir, f"{batch_key}_surgery_duration_scatterplot.svg")
                try:
                    fig_scatter.savefig(scatter_file, dpi=300)
                    print(f"散点图已保存：{scatter_file}")
                except Exception as e:
                    print(f"保存散点图出错：{e}")
            else:
                print("未指定output_dir或batch_key，图表未保存。")

            # 12. 生成和保存动态调整报告
            # 动态调整生成报告
            dynamic_report_df = report.generate_time_progression_report(
                final_schedule,
                patients,
                base_schedule
            )

            # 统一保存文件，确保路径和批次号唯一
            if output_dir and batch_key:
                os.makedirs(output_dir, exist_ok=True)
                report_path = os.path.join(output_dir, f"{batch_key}_time_progression_report.xlsx")
                try:
                    dynamic_report_df.to_excel(report_path, index=False, engine='openpyxl')
                    print(f"时间序列推进报告已保存：{report_path}")
                except Exception as e:
                    print(f"保存报告出错：{e}")
            else:
                print("未指定output_dir或batch_key，动态调整报告未保存。")

            # 13. 生成和保存动态调整甘特图
            fig_adj_gantt = scheduling_visualisation.plot_adjusted_gantt_chart(
                final_schedule,
                ts_list,
                title=f"{batch_key} Adjusted Surgery Scheduling Gantt Chart")
            if fig_adj_gantt:
                adj_gantt_path = os.path.join(output_dir, f"{batch_key}_Adjusted_surgical_schedule_gantt_chart.svg")
                try:
                    fig_adj_gantt.savefig(adj_gantt_path, dpi=300, bbox_inches='tight')
                    print(f"批次 {batch_key} 动态调整甘特图已保存: {adj_gantt_path}")
                except Exception as e:
                    print(f"保存动态调整甘特图时出错: {e}")

            # 14. 等待避免系统压力（可选）
            time.sleep(1)

        print("\n所有批次处理完成。")


    except Exception as e:
        print(f"\n!!! 主程序执行过程中发生未捕获的严重错误: {e} !!!")
        import traceback
        traceback.print_exc()
    #     # 尝试返回已生成的报告（如果有）
    #     if final_report_df is not None:
    #          print("程序异常终止，但尝试返回已生成的报告。")
    #          return final_report_df
    #     else:
    #          return None # 指示失败
    #
    # return final_report_df # 返回生成的报告DataFrame

if __name__ == "__main__":
    main()

