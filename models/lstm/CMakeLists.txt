add_executable(lstm
        lstm.cpp
)

set_target_properties(lstm PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(lstm PRIVATE dl)
