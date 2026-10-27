cmake_minimum_required(VERSION 3.20)

# Clang on some images does not search the GCC directory that holds the
# libstdc++.so linker script. Point the link line at that directory before
# the compiler probe runs.
if(CMAKE_CXX_COMPILER MATCHES "clang")
    execute_process(
        COMMAND g++ -print-file-name=libstdc++.so
        OUTPUT_VARIABLE KVEXP_LIBSTDCXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE KVEXP_LIBSTDCXX_RC)
    if(KVEXP_LIBSTDCXX_RC EQUAL 0 AND EXISTS "${KVEXP_LIBSTDCXX}")
        get_filename_component(KVEXP_LIBSTDCXX_DIR "${KVEXP_LIBSTDCXX}" DIRECTORY)
        foreach(flag_var CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
            string(FIND "${${flag_var}}" "${KVEXP_LIBSTDCXX_DIR}" already)
            if(already EQUAL -1)
                set(${flag_var} "${${flag_var}} -L${KVEXP_LIBSTDCXX_DIR}" CACHE STRING "" FORCE)
            endif()
        endforeach()
        string(FIND "${CMAKE_CXX_FLAGS}" "gcc-install-dir" already_gcc)
        if(already_gcc EQUAL -1)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-install-dir=${KVEXP_LIBSTDCXX_DIR}" CACHE STRING "" FORCE)
        endif()
    endif()
endif()

project(kvexp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_library(kvexp
    src/codec.cpp
    src/cpu.cpp
    src/attention_scalar.cpp
    src/attention_avx2.cpp
    src/attention_avx512.cpp
    src/dispatch.cpp)
target_include_directories(kvexp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(kvexp PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(kvexp PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)

set_source_files_properties(src/attention_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(src/attention_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512bf16;-mfma")

option(KVEXP_SANITIZER "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
if(KVEXP_SANITIZER)
    target_compile_options(kvexp PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(kvexp PUBLIC -fsanitize=address,undefined)
endif()

add_executable(kvexp_tests tests/test_main.cpp)
target_link_libraries(kvexp_tests PRIVATE kvexp)
target_compile_options(kvexp_tests PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)

add_executable(kvexp_bench
    src/bench.cpp
    src/peak_avx2.cpp
    src/peak_avx512.cpp)
target_link_libraries(kvexp_bench PRIVATE kvexp OpenMP::OpenMP_CXX)
target_compile_options(kvexp_bench PRIVATE -Wall -Wextra -Wpedantic)
set_source_files_properties(src/peak_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(src/peak_avx512.cpp PROPERTIES COMPILE_OPTIONS
    "-mavx512f;-mavx512bw;-mavx512vl;-mavx512bf16;-mfma")

enable_testing()
add_test(NAME kvexp_tests COMMAND kvexp_tests)
