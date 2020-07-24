cmake_minimum_required(VERSION 3.12)

project(gmphd)

if (NOT EXISTS "${CMAKE_BINARY_DIR}/cmake/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.15/conan.cmake"
            "${CMAKE_BINARY_DIR}/cmake/conan.cmake")
endif ()
include(${CMAKE_BINARY_DIR}/conan.cmake)
conan_cmake_run(CONANFILE
        conanfile.txt
        BASIC_SETUP CMAKE_TARGETS)

# Required modules
include(GoogleTest)
include(CTest)

## Libraries
add_library(tracker
        modules/tracker/src/update.cpp)
target_include_directories(tracker PUBLIC ${PROJECT_SOURCE_DIR}/modules/tracker/include)
target_link_libraries(tracker PUBLIC CONAN_PKG::boost)
target_compile_features(tracker PRIVATE cxx_std_17)

## Executables

## Tests
add_executable(test_gmphd
        tests/TEST_tracker.cpp)
target_link_libraries(test_gmphd CONAN_PKG::gtest pthread tracker)
target_compile_features(test_gmphd PRIVATE cxx_std_17)
add_test(NAME test_gmphd COMMAND test_gmphd)
gtest_discover_tests(test_gmphd)
