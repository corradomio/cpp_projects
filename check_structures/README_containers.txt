
    https://cplusplus.com/reference/stl/

Containers member types

    value_type          T
    reference	        value_type&
    const_reference	    const value_type&
    pointer	            value_type*
    const_pointer	    const value_type*
    size_type           size_t
    difference_type	p   ptrdiff_t

    iterator
    const_iterator
    reverse_iterator        reverse_iterator<iterator>
    const_reverse_iterator  reverse_iterator<const_iterator>

Containers iterators

     begin(),  end(),  cbegin(),  cend(),
    rbegin(), rend(), crbegin(), crend()


Container class templates

    array           template < class T, size_t N > class array;
                    operator[](i)
                    at(i)
                    front()
                    back()
                    data()

                    size()
                    max_size()

                    fill(T v)
                    swap(array& that)
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
