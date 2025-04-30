#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <chrono>
#include <omp.h>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

std::vector<std::vector<int>> generate_matrix(int size) {
    std::vector<std::vector<int>> matrix(size, std::vector<int>(size));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = std::rand() % 100;
        }
    }
    return matrix;
}

void write_matrix_to_file(const std::vector<std::vector<int>>& matrix, const std::string& filename) {
    std::ofstream out(filename);
    for (const auto& row : matrix) {
        for (const auto& val : row) {
            out << val << " ";
        }
        out << "\n";
    }
}

std::vector<std::vector<int>> multiply_matrices(
    const std::vector<std::vector<int>>& a,
    const std::vector<std::vector<int>>& b
) {
    size_t n = a.size();
    std::vector<std::vector<int>> c(n, std::vector<int>(n, 0));

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

void create_directory(const std::string& dir_name) {
    fs::create_directory(dir_name);
}

int main() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::vector<int> thread_counts = { 2, 14 };
    //std::vector<int> thread_counts = { 2, 4, 6, 8, 10, 12, 14 };
    std::vector<int> sizes = { 100, 200, 300, 400, 500, 1000, 1500, 2000 };

    std::ofstream timing_file("timings2.txt");
    timing_file << "Threads Size Time_microseconds\n"; // Заголовок файла

    for (int threads : thread_counts) {
        omp_set_num_threads(threads);
        std::cout << "Running with " << threads << " threads..." << std::endl;

        for (int size : sizes) {
            std::string dir_name = std::to_string(size) + "_threads_" + std::to_string(threads);
            create_directory(dir_name);

            auto matrix1 = generate_matrix(size);
            auto matrix2 = generate_matrix(size);

            write_matrix_to_file(matrix1, dir_name + "/matrix1.txt");
            write_matrix_to_file(matrix2, dir_name + "/matrix2.txt");

            auto start = std::chrono::high_resolution_clock::now();
            auto result = multiply_matrices(matrix1, matrix2);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            write_matrix_to_file(result, dir_name + "/result.txt");

            timing_file << threads << " " << size << " " << duration << "\n";

            std::cout << "Size " << size << " done in " << duration << " microseconds.\n";
        }
    }

    timing_file.close();
    std::cout << "All timings written to timings.txt\n";
    return 0;
}
