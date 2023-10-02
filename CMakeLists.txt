cmake_minimum_required(VERSION 3.10)

project(SDF)

# Specify the required C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ensure debug symbols are generated
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Location of SPICE headers and libraries
set(SPICE_DIR "cspice")  # Replace with the path where you extracted SPICE
include_directories(${SPICE_DIR}/include)

# Find required packages
find_package(SFML 2.6 COMPONENTS graphics window system REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Sophus REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

# Include directories
include_directories(includes)
include_directories(includes/aaplus)  # Add AA+ include directory

# Include ImGui
include_directories(${CMAKE_SOURCE_DIR}/imgui)

# Add ImGui source files to the build
file(GLOB IMGUI_SOURCES imgui/*.cpp imgui/backends/imgui_impl_glfw.cpp imgui/backends/imgui_impl_opengl3.cpp)

# add_executable(main test_spice.cpp "includes/stb_image.h" ${AAPLUS_SOURCES} ${IMGUI_SOURCES})
add_executable(main solar.cpp "includes/stb_image.h" ${AAPLUS_SOURCES} ${IMGUI_SOURCES})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -g -fno-omit-frame-pointer")

# Link libraries to the executable
target_link_libraries(main PRIVATE 
    sfml-graphics sfml-window sfml-system 
    Eigen3::Eigen 
    fmt::fmt
    Sophus::Sophus
    GLEW::GLEW
    glfw
    # Linking SPICE library
    ${SPICE_DIR}/lib/cspice.a
    # Add any required ImGui linking if necessary
)
