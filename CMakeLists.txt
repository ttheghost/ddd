cmake_minimum_required(VERSION 3.25)

project(dl VERSION 0.1.0 LANGUAGES CXX)

option(DL_BUILD_MODELS "Build model executables" ON)
option(DL_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(core)

if (DL_BUILD_MODELS)
    add_subdirectory(models/perceptron)
    add_subdirectory(models/ann)
    add_subdirectory(models/cnn)
    add_subdirectory(models/rnn)
    add_subdirectory(models/lstm)
    add_subdirectory(models/gru)
    add_subdirectory(models/transformer)
endif ()

if (DL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()