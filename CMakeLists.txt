cmake_minimum_required(VERSION 3.18)
project(credit_mc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(creditmc STATIC src/engine.cpp)
target_include_directories(creditmc PUBLIC include)
target_link_libraries(creditmc PUBLIC Threads::Threads)
set_target_properties(creditmc PROPERTIES POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(creditmc_core src/bindings.cpp)
target_link_libraries(creditmc_core PRIVATE creditmc)
set_target_properties(creditmc_core PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python)