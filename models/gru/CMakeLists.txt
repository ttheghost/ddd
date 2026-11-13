add_executable(gru
        gru.cpp
)

set_target_properties(gru PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(gru PRIVATE dl)
