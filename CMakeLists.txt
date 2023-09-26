cmake_minimum_required(VERSION 3.10)

project(SFMLDemo)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ensure debug symbols are generated
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Find required packages
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)

# Specify the source file and its executable output
add_executable(main sfml_demo.cpp)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g")

# Link libraries to the executable
target_link_libraries(main PRIVATE 
    sfml-graphics sfml-window sfml-system 
    Eigen3::Eigen 
    fmt::fmt
    Sophus::Sophus
)
