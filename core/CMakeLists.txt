add_library(dl STATIC
        src/dl.cpp
)

target_include_directories(dl
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(dl PUBLIC cxx_std_20)