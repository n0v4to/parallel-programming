#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <string>
#include <chrono>
#include <mpi.h>

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

std::vector<std::vector<int>> multiply_matrices_mpi(
    const std::vector<std::vector<int>>& a,
    const std::vector<std::vector<int>>& b,
    int size,
    int rank,
    int num_procs
) {
    std::vector<std::vector<int>> c(size, std::vector<int>(size, 0));

    int rows_per_proc = size / num_procs;
    int extra_rows = size % num_procs;

    int start_row = rank * rows_per_proc + std::min(rank, extra_rows);
    int end_row = start_row + rows_per_proc + (rank < extra_rows ? 1 : 0);

    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    if (rank == 0) {
        for (int src = 1; src < num_procs; ++src) {
            int src_start = src * rows_per_proc + std::min(src, extra_rows);
            int src_end = src_start + rows_per_proc + (src < extra_rows ? 1 : 0);

            for (int i = src_start; i < src_end; ++i) {
                MPI_Recv(c[i].data(), size, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else {
        for (int i = start_row; i < end_row; ++i) {
            MPI_Send(c[i].data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    return c;
}

void create_directory(const std::string& dir_name) {
    fs::create_directory(dir_name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    std::srand(static_cast<unsigned int>(std::time(nullptr)) + rank);
    std::vector<int> sizes = { 100, 200, 300, 400, 500, 1000, 1500, 2000 };

    std::ofstream timing_file;
    if (rank == 0) {
        timing_file.open("timings_mpi.txt");
        timing_file << "Processes Size Time_microseconds\n";
    }

    for (int size : sizes) {
        if (rank == 0) {
            std::string dir_name = std::to_string(size) + "_procs_" + std::to_string(num_procs);
            fs::create_directory(dir_name);

            auto matrix1 = generate_matrix(size);
            auto matrix2 = generate_matrix(size);
            write_matrix_to_file(matrix1, dir_name + "/matrix1.txt");
            write_matrix_to_file(matrix2, dir_name + "/matrix2.txt");

            // Рассылка данных
            for (int i = 0; i < size; ++i) {
                MPI_Bcast(matrix1[i].data(), size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(matrix2[i].data(), size, MPI_INT, 0, MPI_COMM_WORLD);
            }

            auto start = std::chrono::high_resolution_clock::now();
            auto result = multiply_matrices_mpi(matrix1, matrix2, size, rank, num_procs);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            write_matrix_to_file(result, dir_name + "/result.txt");
            timing_file << num_procs << " " << size << " " << duration << "\n";
            std::cout << "Size " << size << " with " << num_procs << " processes: " << duration << " μs\n";
        }
        else {
            std::vector<std::vector<int>> matrix1(size, std::vector<int>(size));
            std::vector<std::vector<int>> matrix2(size, std::vector<int>(size));
            for (int i = 0; i < size; ++i) {
                MPI_Bcast(matrix1[i].data(), size, MPI_INT, 0, MPI_COMM_WORLD);
                MPI_Bcast(matrix2[i].data(), size, MPI_INT, 0, MPI_COMM_WORLD);
            }
            multiply_matrices_mpi(matrix1, matrix2, size, rank, num_procs);
        }
    }

    if (rank == 0) {
        timing_file.close();
        std::cout << "All timings saved to timings_mpi.txt\n";
    }

    MPI_Finalize();
    return 0;
}