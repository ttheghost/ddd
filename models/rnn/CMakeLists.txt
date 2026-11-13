add_executable(rnn
        rnn.cpp
)

set_target_properties(rnn PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        BUILD_RPATH "${CMAKE_SOURCE_DIR}"
)

target_link_libraries(rnn PRIVATE dl)
