import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


def read_matrix(filepath):
    """Читает матрицу из файла (без информации о размерах)"""
    matrix = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                row = list(map(int, line.strip().split()))
                matrix.append(row)
    return np.array(matrix)


def verify_directory(dir_path):
    """Проверяет умножение матриц в одной директории"""
    try:
        m1 = read_matrix(os.path.join(dir_path, "matrix1.txt"))
        m2 = read_matrix(os.path.join(dir_path, "matrix2.txt"))
        cpp_res = read_matrix(os.path.join(dir_path, "result.txt"))

        np_res = np.dot(m1, m2)

        if np.array_equal(cpp_res, np_res):
            return True, 0
        else:
            max_diff = np.max(np.abs(cpp_res - np_res))
            return False, max_diff

    except Exception as e:
        return False, str(e)


def verify_all_combinations(base_dir="."):
    """Проверяет все комбинации размеров и процессов"""
    process_counts = [2, 4, 6, 8, 10]  # Изменил на ваши значения из run.py
    sizes = [100, 200, 300, 400, 500, 1000, 1500, 2000]
    results = defaultdict(list)

    for procs in process_counts:
        for size in sizes:
            dir_name = f"{size}_procs_{procs}"  # Изменил threads на procs
            dir_path = os.path.join(base_dir, dir_name)

            if not os.path.exists(dir_path):
                results[procs].append((size, "Directory not found"))
                continue

            is_ok, diff_or_error = verify_directory(dir_path)

            if is_ok:
                results[procs].append((size, "OK"))
            elif isinstance(diff_or_error, (int, float)):
                results[procs].append((size, f"Max diff: {diff_or_error}"))
            else:
                results[procs].append((size, f"Error: {diff_or_error}"))

    return results


def save_verification_report(results, filename="verification_report.txt"):
    """Сохраняет отчет о верификации"""
    with open(filename, "w") as f:
        f.write("MPI Matrix Multiplication Verification Report\n")  # Изменил OpenMP на MPI
        f.write("=" * 60 + "\n\n")

        for procs in sorted(results.keys()):
            f.write(f"Processes: {procs}\n")  # Изменил Threads на Processes
            f.write("-" * 40 + "\n")

            for size, status in results[procs]:
                f.write(f"{size}x{size}: {status}\n")

            f.write("\n")

        f.write("\nVerification completed.\n")


def plot_timings(timing_file="timings_mpi.txt"):  # Изменил имя файла
    """Строит графики времени выполнения"""
    data = defaultdict(list)

    with open(timing_file) as f:
        # Пропускаем заголовок
        next(f)

        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        procs = int(parts[0])
                        size = int(parts[1])
                        time = int(parts[2])
                        data[procs].append((size, time))
                    except ValueError as e:
                        print(f"Skipping invalid line: {line.strip()} (Error: {e})")

    if not data:
        print("No valid timing data found!")
        return

    plt.figure(figsize=(12, 8))

    for procs in sorted(data.keys()):
        sizes = [x[0] for x in data[procs]]
        times = [x[1] for x in data[procs]]
        plt.plot(sizes, times, marker='o', label=f'{procs} processes')  # Изменил threads на processes

    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Time (microseconds)')
    plt.title('MPI Matrix Multiplication Performance')  # Изменил OpenMP на MPI
    plt.grid(True)
    plt.legend()
    plt.savefig("mpi_timings_plot.png")  # Изменил имя файла
    plt.show()


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("Starting verification...")
    verification_results = verify_all_combinations(base_dir)
    save_verification_report(verification_results)
    print("Verification completed. Report saved to verification_report.txt")

    print("\nGenerating timing plots...")
    plot_timings()
    print("Timing plots saved to mpi_timings_plot.png")