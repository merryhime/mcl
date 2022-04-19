add_library(mcl
    ../include/mcl/assert.hpp
    ../include/mcl/bit/bit_count.hpp
    ../include/mcl/bit/bit_field.hpp
    ../include/mcl/bit/rotate.hpp
    ../include/mcl/bit/swap.hpp
    ../include/mcl/bit_cast.hpp
    ../include/mcl/concepts/bit_integral.hpp
    ../include/mcl/concepts/is_any_of.hpp
    ../include/mcl/concepts/same_as.hpp
    ../include/mcl/container/intrusive_list.hpp
    ../include/mcl/hint/assume.hpp
    ../include/mcl/macro/anonymous_variable.hpp
    ../include/mcl/macro/architecture.hpp
    ../include/mcl/macro/concatenate_tokens.hpp
    ../include/mcl/scope_exit.hpp
    ../include/mcl/stdint.hpp
    ../include/mcl/stdint.hpp
    assert.cpp
)
target_include_directories(mcl PUBLIC ../include/)
target_compile_options(mcl PRIVATE ${MCL_CXX_FLAGS})
target_link_libraries(mcl PUBLIC fmt::fmt)

include(CreateTargetDirectoryGroups)
create_target_directory_groups(mcl)
