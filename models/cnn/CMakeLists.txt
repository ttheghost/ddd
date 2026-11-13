add_executable(cnn
        cnn.cpp
)

set_target_properties(cnn PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(cnn PRIVATE dl)
