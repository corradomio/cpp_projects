/*
 * Copyright (C) 2008-2010, 2012 Peter Colberg
 * Copyright (C) 2020            Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_WRAPPER_COPY_HPP
#define CUDA_WRAPPER_COPY_HPP

#include <iterator>
#include <type_traits>
#include <cstring>

#include <cuda.h>

#include "error.hpp"
#include "iterator_category.hpp"
#include "stream.hpp"
#include "thread.hpp"

namespace cuda {

/**
 * Copy from host or device to host or device.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    (std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    || std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value)
    && (std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    || std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value)
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    cuMemcpy(
        reinterpret_cast<CUdeviceptr>(&*result)
      , reinterpret_cast<CUdeviceptr>(&*first)
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
    );
    return result + size;
}

/**
 * Asynchronous copy from host or device to host or device.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    (std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    || std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value)
    && (std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    || std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value)
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy_async(InputIterator first, InputIterator last, OutputIterator result)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(&*result)
      , reinterpret_cast<CUdeviceptr>(&*first)
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , 0
    );
    return result + size;
}

/**
 * Asynchronous copy from host or device to host or device in dedicated stream.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    (std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    || std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value)
    && (std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    || std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value)
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy_async(InputIterator first, InputIterator last, OutputIterator result, stream const& stream)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    cuMemcpyAsync(
        reinterpret_cast<CUdeviceptr>(&*result)
      , reinterpret_cast<CUdeviceptr>(&*first)
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , stream.data()
    );
    return result + size;
}

/**
 * Fill device memory area with constant byte value.
 */
template <typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
  , void>::type memset(OutputIterator first, OutputIterator last, unsigned char value)
{
    typename std::iterator_traits<OutputIterator>::difference_type size = last - first;
    CU_CALL(cuMemsetD8(
        reinterpret_cast<CUdeviceptr>(&*first)
      , value
      , size * sizeof(typename std::iterator_traits<OutputIterator>::value_type)
    ));
}

} // namespace cuda

#endif /* ! CUDA_WRAPPER_COPY_HPP */
