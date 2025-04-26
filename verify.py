import numpy as np
import os
import sys
import matplotlib.pyplot as plt

def read_matrix(filepath):
    matrix = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                matrix.append(list(map(int, line.strip().split())))

    row_lengths = set(len(row) for row in matrix)
    if len(row_lengths) != 1:
        raise ValueError(f"Inconsistent row lengths in matrix: {row_lengths}")

    return np.array(matrix)

def verify_all_directories(base_dir="."):
    sizes = [100, 200, 300, 400, 500, 1000, 1500, 2000]
    verification_results = []

    for size in sizes:
        dir_path = os.path.join(base_dir, str(size))
        if not os.path.exists(dir_path):
            verification_results.append(f"{size}x{size} folder not found")
            continue

        try:
            m1 = read_matrix(os.path.join(dir_path, "matrix1.txt"))
            m2 = read_matrix(os.path.join(dir_path, "matrix2.txt"))
            cpp_res = read_matrix(os.path.join(dir_path, "result.txt"))

            np_res = np.dot(m1, m2)

            if np.array_equal(cpp_res, np_res):
                verification_results.append(f"Multiplication {size}x{size} - OK")
            else:
                max_diff = np.max(np.abs(cpp_res - np_res))
                verification_results.append(f"Multiplication {size}x{size} - ERROR (max diff: {max_diff})")

        except Exception as e:
            verification_results.append(f"Multiplication {size}x{size} - ERROR: {str(e)}")

    with open("verification_report.txt", "w") as f:
        f.write("Verification report:\n")
        f.write("=" * 50 + "\n")
        for result in verification_results:
            f.write(result + "\n")
        f.write("\nVerification completed.\n")

    print("Verification done. Results saved in verification_report.txt")

def plot_timings(timing_file="timings.txt"):
    sizes = []
    times = []
    with open(timing_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                size, time = map(int, parts)
                sizes.append(size)
                times.append(time)

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, times, marker="o", linestyle="-", color="blue")
    plt.xlabel("Matrix size (N x N)")
    plt.ylabel("Time (microseconds)")
    plt.title("Matrix Multiplication Time vs Size")
    plt.grid(True)
    plt.savefig("multiplication_time_plot.png")
    plt.show()

if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    verify_all_directories(base_dir)
    plot_timings()
