add_executable(dl_tensor_tests
        tensor.cpp
)

target_link_libraries(dl_tensor_tests PRIVATE dl)

add_test(NAME dl_tensor_tests COMMAND dl_tensor_tests)