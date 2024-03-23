//
// Created by Corrado Mio on 08/03/2024.
//
/*
    C++ Exceptions
    --------------

    std::exception
        logic_error
            invalid_argument
            domain_error
            length_error
            out_of_range
            future_error
        runtime_error
            range_error
            overflow_error
            underflow_error
            regex_error
            system_error
                ios_base::failure
                filesystem::filesystem_error
            tx_exception
            nonexistent_local_time
            ambiguous_local_time
            format_error
        bad_typeid
        bad_cast
            bad_any_cast
        bad_optional_access
        bad_expected_access
        bad_weak_ptr
        bad_function_call
        bad_alloc
            bad_array_new_length
        bad_exception
        ios_base::failure
        bad_variant_access
 */

#include <stdexcept>

#ifndef STDX_EXCEPTIONS_H
#define STDX_EXCEPTIONS_H

namespace stdx {

    struct not_implemented: public std::runtime_error {
        not_implemented(const std::string what): std::runtime_error(what) {}
    };

    struct bad_dimensions : public std::runtime_error {
        bad_dimensions(): std::runtime_error("Incompatible dimensions") {}
    };

}

#endif //STDX_EXCEPTIONS_H
