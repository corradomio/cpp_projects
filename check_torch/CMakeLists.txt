cmake_minimum_required(VERSION 3.26)
project(check_torch)

set(Torch_DIR "D:\\Projects\\ml\\libtorch-cpu\\share\\cmake\\Torch")

include_directories(
    "D:\\Projects\\ml\\libtorch-cpu\\include"
    "D:\\Projects\\ml\\libtorch-cpu\\include\\torch\\csrc\\api\\include"
)

link_directories(
    "D:\\Projects\\ml\\libtorch-cpu\\lib"
)

set(CMAKE_CXX_STANDARD 17)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(check_torch main.cpp)

target_link_libraries(check_torch "${TORCH_LIBRARIES}")
