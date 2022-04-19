add_executable(mcl-tests
    main.cpp
)
target_include_directories(mcl-tests PUBLIC .)
target_compile_options(mcl-tests PRIVATE ${STAMINA_CXX_FLAGS})
target_link_libraries(mcl-tests PRIVATE Catch2::Catch2 mcl)

include(CTest)
include(Catch)
catch_discover_tests(mcl-tests)
enable_testing()
