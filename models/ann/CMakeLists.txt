add_executable(ann
        ann.cpp
)

set_target_properties(ann PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(ann PRIVATE dl)
