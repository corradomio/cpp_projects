/* cuda_wrapper/cuda_wrapper.hpp
 *
 * Copyright (C) 2013 Felix Höfling
 * Copyright (C) 2007 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

/*
 * CUDA runtime API wrapper classes
 */

#ifndef CUDA_WRAPPER_HPP
#define CUDA_WRAPPER_HPP

/* Disable warning for CUDA 5.5 headers emitted by Clang ≤ 3.3:
 *   /usr/local/cuda-5.5/include/cuda_runtime.h:225:33: warning: function
 *   'cudaMallocHost' is not needed and will not be emitted [-Wunneeded-internal-declaration]
 *
 * The CUDA runtime API version cannot be checked at this stage since the macro
 * CUDART_VERSION is not yet defined.
 */
#if (defined(__clang__) && __clang_major__ == 3 && __clang_minor__ <= 3)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunneeded-internal-declaration"
# include <cuda_runtime.h>
# pragma GCC diagnostic pop
#else
# include <cuda_runtime.h>
#endif

#include <cuda.h>

/*
 * C++ wrappers requiring runtime functionality (e.g. exceptions)
 */

#ifndef __CUDACC__
# include "copy.hpp"
# include "device.hpp"
# include "error.hpp"
# include "event.hpp"
# include "memory/device/allocator.hpp"
# include "memory/device/vector.hpp"
# include "memory/host/allocator.hpp"
# include "memory/host/vector.hpp"
# include "memory/managed/allocator.hpp"
# include "memory/managed/vector.hpp"
# include "stream.hpp"
# include "texture.hpp"
# include "thread.hpp"
# include "version.hpp"
#endif /* ! __CUDACC__ */

/*
 * C++ wrappers *not* requiring runtime functionality
 */

#include "function.hpp"
#include "symbol.hpp"

#endif /* ! CUDA_WRAPPER_HPP */
