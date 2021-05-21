https://www.cplusplus.com/reference/stl/

    array
    vector
    deque
    forward_list
    list

    stack
    queue
    priority_queue

    set
    multiset
    map
    multimap

    unordered_set
    unordered_multiset
    unordered_map
    unordered_multimap


https://en.cppreference.com/w/cpp/algorithm

    Non-modifying sequence operations
    Defined in header <algorithm>

        all_ofany_ofnone_of
            checks if a predicate is true for all, any or none of the elements in a range

        ranges::all_ofranges::any_ofranges::none_of
            checks if a predicate is true for all, any or none of the elements in a range

        for_each
            applies a function to a range of elements

        ranges::for_each
            applies a function to a range of elements

        for_each_n
            applies a function object to the first n elements of a sequence

        ranges::for_each_n
            applies a function object to the first n elements of a sequence

        countcount_if
            returns the number of elements satisfying specific criteria

        ranges::countranges::count_if
            returns the number of elements satisfying specific criteria

        mismatch
            finds the first position where two ranges differ

        ranges::mismatch
            finds the first position where two ranges differ

        findfind_iffind_if_not
            finds the first element satisfying specific criteria

        ranges::findranges::find_ifranges::find_if_not
            finds the first element satisfying specific criteria

        find_end
            finds the last sequence of elements in a certain range

        ranges::find_end
            finds the last sequence of elements in a certain range

        find_first_of
            searches for any one of a set of elements

        ranges::find_first_of
            searches for any one of a set of elements

        adjacent_find
            finds the first two adjacent items that are equal (or satisfy a given predicate)

        ranges::adjacent_find
            finds the first two adjacent items that are equal (or satisfy a given predicate)

        search
            searches for a range of elements

        ranges::search
            searches for a range of elements

        search_n
            searches a range for a number of consecutive copies of an element

        ranges::search_n
            searches for a number consecutive copies of an element in a range

    Modifying sequence operations
    Defined in header <algorithm>

        copycopy_if
            copies a range of elements to a new location

        ranges::copyranges::copy_if
            copies a range of elements to a new location

        copy_n
            copies a number of elements to a new location

        ranges::copy_n
            copies a number of elements to a new location

        copy_backward
            copies a range of elements in backwards order

        ranges::copy_backward
            copies a range of elements in backwards order

        move
            moves a range of elements to a new location

        ranges::move
            moves a range of elements to a new location

        move_backward
            moves a range of elements to a new location in backwards order

        ranges::move_backward
            moves a range of elements to a new location in backwards order

        fill
            copy-assigns the given value to every element in a range

        ranges::fill
            assigns a range of elements a certain value

        fill_n
            copy-assigns the given value to N elements in a range

        ranges::fill_n
            assigns a value to a number of elements

        transform
            applies a function to a range of elements, storing results in a destination range

        ranges::transform
            applies a function to a range of elements

        generate
            assigns the results of successive function calls to every element in a range

        ranges::generate
            saves the result of a function in a range

        generate_n
            assigns the results of successive function calls to N elements in a range

        ranges::generate_n
            saves the result of N applications of a function

        removeremove_if
            removes elements satisfying specific criteria

        ranges::removeranges::remove_if
            removes elements satisfying specific criteria

        remove_copyremove_copy_if
            copies a range of elements omitting those that satisfy specific criteria

        ranges::remove_copyranges::remove_copy_if
            copies a range of elements omitting those that satisfy specific criteria

        replacereplace_if
            replaces all values satisfying specific criteria with another value

        ranges::replaceranges::replace_if
            replaces all values satisfying specific criteria with another value

        replace_copyreplace_copy_if
            copies a range, replacing elements satisfying specific criteria with another value

        ranges::replace_copyranges::replace_copy_if
            copies a range, replacing elements satisfying specific criteria with another value

        swap
            swaps the values of two objects

        swap_ranges
            swaps two ranges of elements

        ranges::swap_ranges
            swaps two ranges of elements

        iter_swap
            swaps the elements pointed to by two iterators

        reverse
            reverses the order of elements in a range

        ranges::reverse
            reverses the order of elements in a range

        reverse_copy
            creates a copy of a range that is reversed

        ranges::reverse_copy
            creates a copy of a range that is reversed

        rotate
            rotates the order of elements in a range

        ranges::rotate
            rotates the order of elements in a range

        rotate_copy
            copies and rotate a range of elements

        ranges::rotate_copy
            copies and rotate a range of elements

        shift_leftshift_right
            shifts elements in a range

        random_shuffleshuffle
            randomly re-orders elements in a range

        ranges::shuffle
            randomly re-orders elements in a range

        sample
            selects n random elements from a sequence

        ranges::sample
            selects n random elements from a sequence

        unique
            removes consecutive duplicate elements in a range

        ranges::unique
            removes consecutive duplicate elements in a range

        unique_copy
            creates a copy of some range of elements that contains no consecutive duplicates

        ranges::unique_copy
            creates a copy of some range of elements that contains no consecutive duplicates

    Partitioning operations
    Defined in header <algorithm>

        is_partitioned
            determines if the range is partitioned by the given predicate

        ranges::is_partitioned
            determines if the range is partitioned by the given predicate

        partition
            divides a range of elements into two groups

        ranges::partition
            divides a range of elements into two groups

        partition_copy
            copies a range dividing the elements into two groups

        ranges::partition_copy
            copies a range dividing the elements into two groups

        stable_partition
            divides elements into two groups while preserving their relative order

        ranges::stable_partition
            divides elements into two groups while preserving their relative order

        partition_point
            locates the partition point of a partitioned range

        ranges::partition_point
            locates the partition point of a partitioned range

    Sorting operations
    Defined in header <algorithm>

        is_sorted
            checks whether a range is sorted into ascending order

        ranges::is_sorted
            checks whether a range is sorted into ascending order

        is_sorted_until
            finds the largest sorted subrange

        ranges::is_sorted_until
            finds the largest sorted subrange

        sort
            sorts a range into ascending order

        ranges::sort
            sorts a range into ascending order

        partial_sort
            sorts the first N elements of a range

        ranges::partial_sort
            sorts the first N elements of a range

        partial_sort_copy
            copies and partially sorts a range of elements

        ranges::partial_sort_copy
            copies and partially sorts a range of elements

        stable_sort
            sorts a range of elements while preserving order between equal elements

        ranges::stable_sort
            sorts a range of elements while preserving order between equal elements

        nth_element
            partially sorts the given range making sure that it is partitioned by the given element

        ranges::nth_element
            partially sorts the given range making sure that it is partitioned by the given element

    Binary search operations (on sorted ranges)
    Defined in header <algorithm>

        lower_bound
            returns an iterator to the first element not less than the given value

        ranges::lower_bound
            returns an iterator to the first element not less than the given value

        upper_bound
            returns an iterator to the first element greater than a certain value

        ranges::upper_bound
            returns an iterator to the first element greater than a certain value

        binary_search
            determines if an element exists in a certain range

        ranges::binary_search
            determines if an element exists in a certain range

        equal_range
            returns range of elements matching a specific key

        ranges::equal_range
            returns range of elements matching a specific key

    Other operations on sorted ranges
    Defined in header <algorithm>

        merge
            merges two sorted ranges

        ranges::merge
            merges two sorted ranges

        inplace_merge
            merges two ordered ranges in-place

        ranges::inplace_merge
            merges two ordered ranges in-place

    Set operations (on sorted ranges)
    Defined in header <algorithm>

        includes
            returns true if one sequence is a subsequence of another

        ranges::includes
            returns true if one sequence is a subsequence of another

        set_difference
            computes the difference between two sets

        ranges::set_difference
            computes the difference between two sets

        set_intersection
            computes the intersection of two sets

        ranges::set_intersection
            computes the intersection of two sets

        set_symmetric_difference
            computes the symmetric difference between two sets

        ranges::set_symmetric_difference
            computes the symmetric difference between two sets

        set_union
            computes the union of two sets

        ranges::set_union
            computes the union of two sets

    Heap operations
    Defined in header <algorithm>

        is_heap
            checks if the given range is a max heap

        ranges::is_heap
            checks if the given range is a max heap

        is_heap_until
            finds the largest subrange that is a max heap

        ranges::is_heap_until
            finds the largest subrange that is a max heap

        make_heap
            creates a max heap out of a range of elements

        ranges::make_heap
            creates a max heap out of a range of elements

        push_heap
            adds an element to a max heap

        ranges::push_heap
            adds an element to a max heap

        pop_heap
            removes the largest element from a max heap

        ranges::pop_heap
            removes the largest element from a max heap

        sort_heap
            turns a max heap into a range of elements sorted in ascending order

        ranges::sort_heap
            turns a max heap into a range of elements sorted in ascending order

    Minimum/maximum operations
    Defined in header <algorithm>

        max
            returns the greater of the given values

        ranges::max
            returns the greater of the given values

        max_element
            returns the largest element in a range

        ranges::max_element
            returns the largest element in a range

        min
            returns the smaller of the given values

        ranges::min
            returns the smaller of the given values

        min_element
            returns the smallest element in a range

        ranges::min_element
            returns the smallest element in a range

        minmax
            returns the smaller and larger of two elements

        ranges::minmax
            returns the smaller and larger of two elements

        minmax_element
            returns the smallest and the largest elements in a range

        ranges::minmax_element
            returns the smallest and the largest elements in a range

        clamp
            clamps a value between a pair of boundary values

        ranges::clamp
            clamps a value between a pair of boundary values

    Comparison operations
    Defined in header <algorithm>

        equal
            determines if two sets of elements are the same

        ranges::equal
            determines if two sets of elements are the same

        lexicographical_compare
            returns true if one range is lexicographically less than another

        ranges::lexicographical_compare
            returns true if one range is lexicographically less than another

        lexicographical_compare_three_way
            compares two ranges using three-way comparison

    Permutation operations
    Defined in header <algorithm>

        is_permutation
            determines if a sequence is a permutation of another sequence

        ranges::is_permutation
            determines if a sequence is a permutation of another sequence

        next_permutation
            generates the next greater lexicographic permutation of a range of elements

        ranges::next_permutation
            generates the next greater lexicographic permutation of a range of elements

        prev_permutation
            generates the next smaller lexicographic permutation of a range of elements

        ranges::prev_permutation
            generates the next smaller lexicographic permutation of a range of elements

    Numeric operations
    Defined in header <numeric>

        iota
            fills a range with successive increments of the starting value

        accumulate
            sums up a range of elements

        inner_product
            computes the inner product of two ranges of elements

        adjacent_difference
            computes the differences between adjacent elements in a range

        partial_sum
            computes the partial sum of a range of elements

        reduce
            similar to std::accumulate, except out of order

        exclusive_scan
            similar to std::partial_sum, excludes the ith input element from the ith sum

        inclusive_scan
            similar to std::partial_sum, includes the ith input element in the ith sum

        transform_reduce
            applies an invocable, then reduces out of order

        transform_exclusive_scan
            applies an invocable, then calculates exclusive scan

        transform_inclusive_scan
            applies an invocable, then calculates inclusive scan

    Operations on uninitialized memory
    Defined in header <memory>

        uninitialized_copy
            copies a range of objects to an uninitialized area of memory

        ranges::uninitialized_copy
            copies a range of objects to an uninitialized area of memory

        uninitialized_copy_n
            copies a number of objects to an uninitialized area of memory

        ranges::uninitialized_copy_n
            copies a number of objects to an uninitialized area of memory

        uninitialized_fill
            copies an object to an uninitialized area of memory, defined by a range

        ranges::uninitialized_fill
            copies an object to an uninitialized area of memory, defined by a range

        uninitialized_fill_n
            copies an object to an uninitialized area of memory, defined by a start and a count

        ranges::uninitialized_fill_n
            copies an object to an uninitialized area of memory, defined by a start and a count

        uninitialized_move
            moves a range of objects to an uninitialized area of memory

        ranges::uninitialized_move
            moves a range of objects to an uninitialized area of memory

        uninitialized_move_n
            moves a number of objects to an uninitialized area of memory

        ranges::uninitialized_move_n
            moves a number of objects to an uninitialized area of memory

        uninitialized_default_construct
            constructs objects by default-initialization in an uninitialized area of memory, defined by a range

        ranges::uninitialized_default_construct
            constructs objects by default-initialization in an uninitialized area of memory, defined by a range

        uninitialized_default_construct_n
            constructs objects by default-initialization in an uninitialized area of memory, defined by a start and a count

        ranges::uninitialized_default_construct_n
            constructs objects by default-initialization in an uninitialized area of memory, defined by a start and count

        uninitialized_value_construct
            constructs objects by value-initialization in an uninitialized area of memory, defined by a range

        ranges::uninitialized_value_construct
            constructs objects by value-initialization in an uninitialized area of memory, defined by a range

        uninitialized_value_construct_n
            constructs objects by value-initialization in an uninitialized area of memory, defined by a start and a count

        ranges::uninitialized_value_construct_n
            constructs objects by value-initialization in an uninitialized area of memory, defined by a start and a count

        destroy
            destroys a range of objects

        ranges::destroy
            destroys a range of objects

        destroy_n
            destroys a number of objects in a range

        ranges::destroy_n
            destroys a number of objects in a range

        destroy_at
            destroys an object at a given address

        ranges::destroy_at
            destroys an object at a given address

        construct_at
            creates an object at a given address

        ranges::construct_at
            creates an object at a given address

    C library
    Defined in header <cstdlib>

        qsort
            sorts a range of elements with unspecified type

        bsearch
            searches an array for an element of unspecified type
