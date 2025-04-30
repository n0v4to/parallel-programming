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
    """Проверяет все комбинации размеров и потоков"""
    thread_counts = [2, 4, 6, 8, 10, 12, 14]
    sizes = [100, 200, 300, 400, 500, 1000, 1500, 2000]
    results = defaultdict(list)

    for threads in thread_counts:
        for size in sizes:
            dir_name = f"{size}_threads_{threads}"
            dir_path = os.path.join(base_dir, dir_name)

            if not os.path.exists(dir_path):
                results[threads].append((size, "Directory not found"))
                continue

            is_ok, diff_or_error = verify_directory(dir_path)

            if is_ok:
                results[threads].append((size, "OK"))
            elif isinstance(diff_or_error, (int, float)):
                results[threads].append((size, f"Max diff: {diff_or_error}"))
            else:
                results[threads].append((size, f"Error: {diff_or_error}"))

    return results


def save_verification_report(results, filename="verification_report.txt"):
    """Сохраняет отчет о верификации"""
    with open(filename, "w") as f:
        f.write("OpenMP Matrix Multiplication Verification Report\n")
        f.write("=" * 60 + "\n\n")

        for threads in sorted(results.keys()):
            f.write(f"Threads: {threads}\n")
            f.write("-" * 40 + "\n")

            for size, status in results[threads]:
                f.write(f"{size}x{size}: {status}\n")

            f.write("\n")

        f.write("\nVerification completed.\n")


def plot_timings(timing_file="timings.txt"):
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
                        threads = int(parts[0])
                        size = int(parts[1])
                        time = int(parts[2])
                        data[threads].append((size, time))
                    except ValueError as e:
                        print(f"Skipping invalid line: {line.strip()} (Error: {e})")

    if not data:
        print("No valid timing data found!")
        return

    plt.figure(figsize=(12, 8))

    for threads in sorted(data.keys()):
        sizes = [x[0] for x in data[threads]]
        times = [x[1] for x in data[threads]]
        plt.plot(sizes, times, marker='o', label=f'{threads} threads')

    plt.xlabel('Matrix Size (N x N)')
    plt.ylabel('Time (microseconds)')
    plt.title('OpenMP Matrix Multiplication Performance')
    plt.grid(True)
    plt.legend()
    plt.savefig("openmp_timings_plot.png")
    plt.show()


if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    print("Starting verification...")
    verification_results = verify_all_combinations(base_dir)
    save_verification_report(verification_results)
    print("Verification completed. Report saved to verification_report.txt")

    print("\nGenerating timing plots...")
    plot_timings()
    print("Timing plots saved to openmp_timings_plot.png")