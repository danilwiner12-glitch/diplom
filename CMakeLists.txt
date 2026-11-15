cmake_minimum_required(VERSION 3.16)
project(pic3d_cherenkov LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(pic3d main.cpp)
target_include_directories(pic3d PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(pic3d PRIVATE ${HDF5_CXX_LIBRARIES})
target_compile_options(pic3d PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)

enable_testing()
add_executable(pic3d_tests tests.cpp)
target_compile_options(pic3d_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
)
add_test(NAME pic3d_tests COMMAND pic3d_tests)
