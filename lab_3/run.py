import subprocess
import sys
import os


def run_mpi_program(executable, num_processes):
    try:
        print(f"\nЗапуск {num_processes} процессов...")

        # Формируем команду
        command = ["mpiexec", "-n", str(num_processes), executable]

        # Запускаем процесс
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        # Выводим результаты
        if process.stdout:
            print("Вывод программы:")
            print(process.stdout)

        if process.stderr:
            print("Ошибки программы:")
            print(process.stderr)

        if process.returncode != 0:
            print(f"Процесс завершился с ошибкой: код {process.returncode}")
        else:
            print("Выполнение завершено успешно")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    executable = "D:/Code/lessons/year_3/semak_2/paral_prog/lab_1/pythonProject1/lab_3/out/build/x64-Debug/MPI_Project.exe"
    processes_to_run = [2, 4, 6, 8, 10]

    # Проверяем существование файла
    if not os.path.exists(executable):
        print(f"Ошибка: исполняемый файл {executable} не существует!")
        sys.exit(1)

    # Запускаем для каждого количества процессов
    for num_processes in processes_to_run:
        run_mpi_program(executable, num_processes)