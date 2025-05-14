#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>

// Конфигурация
const int MASTER_RANK = 0;
const int MATRIX_TAG = 1;
const int RESULT_TAG = 2;

// Генерация матрицы с улучшенной производительностью
void generate_matrix(int* matrix, int size, int seed_offset = 0) {
    std::srand(static_cast<unsigned int>(std::time(nullptr)) + seed_offset);
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = std::rand() % 100;
    }
}

// Блочное умножение матриц с оптимизацией для кэша
void matrix_multiply(const int* A, const int* B, int* C, int rows, int cols, int common_dim) {
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < common_dim; ++k) {
            int a_ik = A[i * common_dim + k];
            for (int j = 0; j < cols; ++j) {
                C[i * cols + j] += a_ik * B[k * cols + j];
            }
        }
    }
}

// Основная функция умножения с MPI
void distributed_matrix_multiply(int size, int rank, int num_procs, double& computation_time) {
    // 1. Распределение работы
    int block_size = size / num_procs;
    int remainder = size % num_procs;
    
    int local_rows = block_size + (rank < remainder ? 1 : 0);
    int row_offset = rank * block_size + std::min(rank, remainder);
    
    // 2. Выделение памяти
    std::vector<int> local_A(local_rows * size);
    std::vector<int> local_B(size * size);
    std::vector<int> local_C(local_rows * size, 0);
    
    // 3. Распределение данных
    if (rank == MASTER_RANK) {
        std::vector<int> full_A(size * size);
        std::vector<int> full_B(size * size);
        
        generate_matrix(full_A.data(), size, 0);
        generate_matrix(full_B.data(), size, 1);
        
        // Раздача блоков матрицы A
        std::vector<MPI_Request> send_requests(num_procs - 1);
        for (int dest = 1; dest < num_procs; ++dest) {
            int dest_rows = block_size + (dest < remainder ? 1 : 0);
            int dest_offset = dest * block_size + std::min(dest, remainder);
            
            MPI_Isend(&full_A[dest_offset * size], dest_rows * size, MPI_INT, 
                     dest, MATRIX_TAG, MPI_COMM_WORLD, &send_requests[dest-1]);
        }
        
        // Копирование своего блока
        std::copy_n(full_A.begin() + row_offset * size, local_rows * size, local_A.begin());
        
        // Рассылка матрицы B всем процессам
        MPI_Bcast(full_B.data(), size * size, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
        
        MPI_Waitall(num_procs - 1, send_requests.data(), MPI_STATUSES_IGNORE);
    } else {
        // Получение блока матрицы A
        MPI_Recv(local_A.data(), local_rows * size, MPI_INT, MASTER_RANK, 
                MATRIX_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Получение матрицы B
        MPI_Bcast(local_B.data(), size * size, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
    }
    
    // 4. Локальное умножение (замеряем только время вычислений)
    MPI_Barrier(MPI_COMM_WORLD);
    double comp_start = MPI_Wtime();
    
    matrix_multiply(local_A.data(), local_B.data(), local_C.data(), 
                   local_rows, size, size);
    
    double comp_end = MPI_Wtime();
    computation_time = comp_end - comp_start;
    
    // 5. Сбор результатов
    if (rank == MASTER_RANK) {
        std::vector<int> full_C(size * size);
        
        // Копирование своего блока
        std::copy_n(local_C.begin(), local_rows * size, full_C.begin() + row_offset * size);
        
        // Получение блоков от других процессов
        std::vector<MPI_Request> recv_requests(num_procs - 1);
        for (int src = 1; src < num_procs; ++src) {
            int src_rows = block_size + (src < remainder ? 1 : 0);
            int src_offset = src * block_size + std::min(src, remainder);
            
            MPI_Irecv(&full_C[src_offset * size], src_rows * size, MPI_INT, 
                     src, RESULT_TAG, MPI_COMM_WORLD, &recv_requests[src-1]);
}
        
        MPI_Waitall(num_procs - 1, recv_requests.data(), MPI_STATUSES_IGNORE);
    } else {
        MPI_Send(local_C.data(), local_rows * size, MPI_INT, MASTER_RANK, 
                RESULT_TAG, MPI_COMM_WORLD);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    std::vector<int> sizes;
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            sizes.push_back(std::stoi(argv[i]));
        }
    } else {
        sizes = {100, 200, 500, 1000, 2000};
    }
    
    if (rank == MASTER_RANK) {
        std::cout << "Running matrix multiplication benchmark on " 
                  << num_procs << " processes\n";
        std::cout << "Size\tTotal Time(s)\tComp Time(s)\tComm Time(s)\n";
    }
    
    for (int size : sizes) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        double computation_time = 0.0;
        
        distributed_matrix_multiply(size, rank, num_procs, computation_time);
        
        double end_time = MPI_Wtime();
        double total_time = end_time - start_time;
        double communication_time = total_time - computation_time;
        
        if (rank == MASTER_RANK) {
            std::cout << size << "\t" << total_time << "\t" 
                      << computation_time << "\t" << communication_time << "\n";
        }
    }
    
    MPI_Finalize();
    return 0;
}