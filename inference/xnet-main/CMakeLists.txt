cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(xnet VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# plain version
add_executable(xnet-plain xnet_plain.cpp)

# SEAL version
find_package(SEAL 4.1 REQUIRED)
add_executable(xnet-seal xnet_seal.cpp)
if(TARGET SEAL::seal)
    target_link_libraries(xnet-seal SEAL::seal)
elseif(TARGET SEAL::seal_shared)
    target_link_libraries(xnet-seal SEAL::seal_shared)
else()
    message(FATAL_ERROR "Cannot find SEAL target")
endif()

# CAHEL version
find_package(CAHEL REQUIRED)
add_executable(xnet-cahel xnet_cahel.cu)
target_link_libraries(xnet-cahel CAHEL::CAHEL)
