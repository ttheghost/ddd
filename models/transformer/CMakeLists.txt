add_executable(transformer
        transformer.cpp
)

set_target_properties(transformer PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(transformer PRIVATE dl)
