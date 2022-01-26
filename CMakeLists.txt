cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(vortigaunt LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# !!! win32 here includes both 32-bit and 64-bit OS !!!
if(WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    # suppress windows crt security related warnings
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(UNIX AND NOT APPLE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
        add_compile_options(-fdiagnostics-color=always -m64)
        add_compile_options(-Wuninitialized -Wpedantic -Wextra -Wall -Wshadow)
        add_compile_options(-fvisibility=hidden -fvisibility-inlines-hidden)
    endif()
endif()


# handle cuda
find_package(CUDAToolkit)
message(INFO "--> CUDA Version: ${CUDAToolkit_VERSION}")

if(CUDAToolkit_VERSION VERSION_LESS "11.6")
    string(CONCAT ERROR_MSG "--> Must upgrade CUDA to 11.6 or newer.")
    message(FATAL_ERROR ${ERROR_MSG})
endif()

# With CMake v3.22 there is still no optimal way to specify the compile flags
# for virtual/real GPUs. The new CUDA_ARCHITECTURES property is half-baked. (:/)
# Here we use the primitive (and actually easiest) way.
#
# GPU         compute capability
# 1080 Ti     6.1
# 2080 Ti     7.5
# 3080 Ti     8.6
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
-Xptxas -dlcm=ca \
-Xptxas -v \
-Xptxas -warn-spills \
--generate-code arch=compute_61,code=sm_61 \
--generate-code arch=compute_75,code=sm_75 \
--generate-code arch=compute_86,code=sm_86"
)

# It appears this flag for JIT. On my PC it defaults to
# a Maxwell architecture. Therefore we need to overwrite it.
# This results in --generate-code arch=compute_75,code=compute_75 --> JIT
set(CMAKE_CUDA_ARCHITECTURES 75)

add_subdirectory(test)


