cmake_minimum_required(VERSION 3.16)
project(SNN_GPU LANGUAGES CXX)

# For IDE IntelliSense (C/C++ extension, compile_commands.json)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# train | train-debug | live | live-debug
set(SNN_GPU_APP_MODE "train-debug" CACHE STRING "App entry: train, train-debug, live, live-debug")
option(SNN_GPU_RUN_TESTS "Run speciation file I/O tests at startup, then exit" OFF)

include(FetchContent)

# Optional overrides: -DGLFW_ROOT=...  -DOpenCL_INCLUDE_DIR=...  -DOpenCL_LIBRARY=...
set(GLFW_ROOT "" CACHE PATH "Optional GLFW root (folder with include/GLFW/glfw3.h)")
if(GLFW_ROOT)
    list(PREPEND CMAKE_PREFIX_PATH "${GLFW_ROOT}")
elseif(DEFINED ENV{GLFW_ROOT})
    set(GLFW_ROOT "$ENV{GLFW_ROOT}" CACHE PATH "Optional GLFW root (folder with include/GLFW/glfw3.h)" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${GLFW_ROOT}")
endif()
if(DEFINED ENV{MSYSTEM_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{MSYSTEM_PREFIX}")
endif()
if(CMAKE_CXX_COMPILER)
    get_filename_component(_snn_cxx_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_snn_toolchain_prefix "${_snn_cxx_bin}" DIRECTORY)
    if(EXISTS "${_snn_toolchain_prefix}/include/CL/cl.h")
        list(APPEND CMAKE_PREFIX_PATH "${_snn_toolchain_prefix}")
    endif()
endif()
if(EXISTS "$ENV{SystemDrive}/msys64/mingw64/include/CL/cl.h")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{SystemDrive}/msys64/mingw64")
endif()

find_package(OpenCL REQUIRED)

find_package(glfw3 CONFIG QUIET)
if(TARGET glfw)
    set(_snn_glfw glfw)
elseif(TARGET glfw3)
    set(_snn_glfw glfw3)
else()
    find_path(GLFW_INCLUDE_DIR
        NAMES GLFW/glfw3.h
        HINTS "${GLFW_ROOT}/include" "$ENV{GLFW_ROOT}/include"
        PATHS "$ENV{ProgramFiles}/GLFW/glfw-3.4.bin.WIN64/include" "$ENV{ProgramFiles}/GLFW"
        PATH_SUFFIXES include
    )
    find_library(GLFW_LIBRARY
        NAMES glfw3 glfw
        HINTS "${GLFW_ROOT}" "$ENV{GLFW_ROOT}"
        PATHS "$ENV{ProgramFiles}/GLFW/glfw-3.4.bin.WIN64" "$ENV{ProgramFiles}/GLFW"
        PATH_SUFFIXES lib-mingw-w64 lib-vc2022 lib-vc2019 lib-vc2017 lib lib64
    )
    if(NOT GLFW_INCLUDE_DIR OR NOT GLFW_LIBRARY)
        message(FATAL_ERROR "GLFW not found. Set GLFW_ROOT to a GLFW tree (include/GLFW/glfw3.h)")
    endif()
    add_library(snn_glfw UNKNOWN IMPORTED)
    set_target_properties(snn_glfw PROPERTIES
        IMPORTED_LOCATION "${GLFW_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GLFW_INCLUDE_DIR}"
    )
    set(_snn_glfw snn_glfw)
endif()

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

set(_snn_app_mode "${SNN_GPU_APP_MODE}")
string(TOLOWER "${_snn_app_mode}" _snn_app_mode_lower)

set(_snn_app_defs
    CL_TARGET_OPENCL_VERSION=300
    SNN_GPU_PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
)

if (_snn_app_mode_lower STREQUAL "train" OR _snn_app_mode_lower STREQUAL "train-debug")
    list(APPEND _snn_app_defs SNN_GPU_APP_TRAIN)
elseif (_snn_app_mode_lower STREQUAL "live" OR _snn_app_mode_lower STREQUAL "live-debug")
    list(APPEND _snn_app_defs SNN_GPU_APP_LIVE)
else()
    message(FATAL_ERROR "SNN_GPU_APP_MODE must be train, train-debug, live, or live-debug (got '${SNN_GPU_APP_MODE}')")
endif()

if (_snn_app_mode_lower STREQUAL "train-debug" OR _snn_app_mode_lower STREQUAL "live-debug")
    list(APPEND _snn_app_defs SNN_GPU_DEBUG_LOGS)
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND _snn_app_defs SNN_GPU_DEBUG_LOGS)
endif()

if (SNN_GPU_RUN_TESTS)
    list(APPEND _snn_app_defs SNN_GPU_RUN_TESTS)
endif()

add_executable(snn_app
    app/main.cpp
    app/opencl-context/CLBoilerplate.cpp
    app/util/Log.cpp
    app/util/RecordClips.cpp
    app/life-cycle/App.cpp
    app/life-cycle/MouseCapture.cpp
    app/life-cycle/MouseOutput.cpp
    app/life-cycle/SNNEngine.cpp
    app/util/Test.cpp
    app/evolution/DNAReader.cpp
    app/life-cycle/ErrorFeedback.cpp
    app/evolution/FitnessEvaluation.cpp
    app/evolution/NeatPipeline.cpp
    app/evolution/LiveOrganism.cpp
    app/evolution/Reproduction.cpp
    app/evolution/Speciation.cpp
)

target_compile_definitions(snn_app PRIVATE ${_snn_app_defs})

target_include_directories(snn_app PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/app"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/evolution"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/life-cycle"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/POD structs"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/opencl-context"
)

target_link_libraries(snn_app PRIVATE
    ${_snn_glfw}
    nlohmann_json::nlohmann_json
    OpenCL::OpenCL
    opengl32
    gdi32
    user32
    shell32
)

if (WIN32)
    target_link_libraries(snn_app PRIVATE dwmapi)
endif()

message(STATUS "OpenCL: ${OpenCL_LIBRARIES}")
message(STATUS "GLFW: ${_snn_glfw} ${GLFW_LIBRARY}")
message(STATUS "SNN_GPU_APP_MODE: ${SNN_GPU_APP_MODE}")

add_custom_target(run
    COMMAND $<TARGET_FILE:snn_app>
    DEPENDS snn_app
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
