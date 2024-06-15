//
// Created by Corrado Mio on 02/06/2024.
//

#include "cudacpp/cudacpp.h"
#include <map>

#define info  (*this->_info)

namespace cudacpp {

    std::map<std::string, int> ATTRIBUTES = {
        // extra attributes
        {"total_memory_mb", -1},
        {"device_count", -2},
        {"driver_version", -3},

        // standard attributes
        {"max_threads_per_block", CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK},

        {"max_block_dim_x", CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X},
        {"max_block_dim_y", CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y},
        {"max_block_dim_z", CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z},
        {"max_grid_dim_x", CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X},
        {"max_grid_dim_y", CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y},
        {"max_grid_dim_z", CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z},

        {"max_shared_memory_per_block", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK},
        {    "shared_memory_per_block", CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK},
        {"total_constant_memory", CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY},
        {"warp_size", CU_DEVICE_ATTRIBUTE_WARP_SIZE},
        {"max_pitch", CU_DEVICE_ATTRIBUTE_MAX_PITCH},
        {"max_registers_per_block", CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK},
        {"registers_per_block", CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK},
        {"clock_rate", CU_DEVICE_ATTRIBUTE_CLOCK_RATE},
        {"texture_alignment", CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT},
        {"gpu_overlap", CU_DEVICE_ATTRIBUTE_GPU_OVERLAP},
        {"multiprocessor_count", CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT},
        {"kernel_exec_timeout", CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT},
        {"integrated", CU_DEVICE_ATTRIBUTE_INTEGRATED},
        {"can_map_host_memory", CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY},
        {"compute_mode", CU_DEVICE_ATTRIBUTE_COMPUTE_MODE},

        {"maximum_texture1d_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH},
        {"maximum_texture2d_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH},
        {"maximum_texture2d_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT},
        {"maximum_texture3d_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH},
        {"maximum_texture3d_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT},
        {"maximum_texture3d_depth", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH},
        {"maximum_texture1d_layered_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH},
        {"maximum_texture1d_layered_layers", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS},
        {"maximum_texture2d_layered_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH},
        {"maximum_texture2d_layered_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT},
        {"maximum_texture2d_layered_layers", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS},
        {"maximum_texture2d_array_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH},
        {"maximum_texture2d_array_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT},
        {"maximum_texture2d_array_numslices", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES},
        {"maximum_texture2d_gather_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH},
        {"maximum_texture2d_gather_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT},
        {"maximum_texture3d_width_alternate", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE},
        {"maximum_texture3d_height_alternate", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE},
        {"maximum_texture3d_depth_alternate", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE},

        {"maximum_texturecubemap_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH},
        {"maximum_texturecubemap_layered_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH},
        {"maximum_texturecubemap_layered_layers", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS},

        {"maximum_texture1d_linear_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH},
        {"maximum_texture2d_linear_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH},
        {"maximum_texture2d_linear_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT},
        {"maximum_texture2d_linear_pitch", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH},

        {"maximum_texture1d_mipmapped_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH},
        {"maximum_texture2d_mipmapped_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH},
        {"maximum_texture2d_mipmapped_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT},

        {"maximum_surface1d_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH},
        {"maximum_surface2d_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH},
        {"maximum_surface2d_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT},
        {"maximum_surface3d_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH},
        {"maximum_surface3d_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT},
        {"maximum_surface3d_depth", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH},

        {"maximum_surface1d_layered_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH},
        {"maximum_surface1d_layered_layers", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS},
        {"maximum_surface2d_layered_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH},
        {"maximum_surface2d_layered_height", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT},
        {"maximum_surface2d_layered_layers", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS},

        {"maximum_surfacecubemap_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH},
        {"maximum_surfacecubemap_layered_width", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH},
        {"maximum_surfacecubemap_layered_layers", CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS},

        {"surface_alignment", CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT},
        {"concurrent_kernels", CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS},
        {"ecc_enabled", CU_DEVICE_ATTRIBUTE_ECC_ENABLED},
        {"pci_bus_id", CU_DEVICE_ATTRIBUTE_PCI_BUS_ID},
        {"pci_device_id", CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID},
        {"tcc_driver", CU_DEVICE_ATTRIBUTE_TCC_DRIVER},
        {"memory_clock_rate", CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE},
        {"global_memory_bus_width", CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH},
        {"l2_cache_size", CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE},
        {"max_threads_per_multiprocessor", CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR},
        {"async_engine_count", CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT},
        {"unified_addressing", CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING},
        {"can_tex2d_gather", CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER},
        {"pci_domain_id", CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID},
        {"texture_pitch_alignment", CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT},
        {"compute_capability_major", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR},
        {"compute_capability_minor", CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR},

        {"stream_priorities_supported", CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED},
        {"global_l1_cache_supported", CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED},
        {"local_l1_cache_supported", CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED},
        {"max_shared_memory_per_multiprocessor", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR},
        {"max_registers_per_multiprocessor", CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR},
        {"managed_memory", CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY},
        {"multi_gpu_board", CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD},
        {"multi_gpu_board_group_id", CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID},
        {"host_native_atomic_supported", CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED},
        {"single_to_double_precision_perf_ratio", CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO},
        {"pageable_memory_access", CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS},
        {"concurrent_managed_access", CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS},
        {"compute_preemption_supported", CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED},
        {"can_use_host_pointer_for_registered_mem", CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM},
        {"can_use_stream_mem_ops_v1", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1},
        {"can_use_64_bit_stream_mem_ops_v1", CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1},
        {"can_use_stream_wait_value_nor_v1", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1},
        {"cooperative_launch", CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH},
        {"cooperative_multi_device_launch", CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH},
        {"max_shared_memory_per_block_optin", CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN},
        {"can_flush_remote_writes", CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES},
        {"host_register_supported", CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED},
        {"pageable_memory_access_uses_host_page_tables", CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES},
        {"direct_managed_mem_access_from_host", CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST},
        {"virtual_address_management_supported", CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED},
        {"virtual_memory_management_supported", CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED},
        {"handle_type_posix_file_descriptor_supported", CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED},
        {"handle_type_win32_handle_supported", CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED},
        {"handle_type_win32_kmt_handle_supported", CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED},
        {"max_blocks_per_multiprocessor", CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR},
        {"generic_compression_supported", CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED},
        {"max_persisting_l2_cache_size", CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE},
        {"max_access_policy_window_size", CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE},
        {"gpu_direct_rdma_with_cuda_vmm_supported", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED},
        {"reserved_shared_memory_per_block", CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK},
        {"sparse_cuda_array_supported", CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED},
        {"read_only_host_register_supported", CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED},
        {"timeline_semaphore_interop_supported", CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED},
        {"memory_pools_supported", CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED},
        {"gpu_direct_rdma_supported", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED},
        {"gpu_direct_rdma_flush_writes_options", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS},
        {"gpu_direct_rdma_writes_ordering", CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING},
        {"mempool_supported_handle_types", CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES},
        {"cluster_launch", CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH},
        {"deferred_mapping_cuda_array_supported", CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED},
        {"can_use_64_bit_stream_mem_ops", CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS},
        {"can_use_stream_wait_value_nor", CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR},
        {"dma_buf_supported", CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED},
        {"ipc_event_supported", CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED},
        {"mem_sync_domain_count", CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT},
        {"tensor_map_access_supported", CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED},
        {"handle_type_fabric_supported", CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED},     // v12.4
        {"unified_function_pointer", CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS},
        {"numa_config", CU_DEVICE_ATTRIBUTE_NUMA_CONFIG},
        {"numa_id", CU_DEVICE_ATTRIBUTE_NUMA_ID},
        {"multicast_supported", CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED},
        {"mps_enabled", CU_DEVICE_ATTRIBUTE_MPS_ENABLED},               // v12.4
        {"host_numa_id", CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID},
        {"d3d12_cig_supported", CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED}    // v12.5

        // {"max", CU_DEVICE_ATTRIBUTE_MAX},
    };

    // ----------------------------------------------------------------------
    // cuda_error
    // ----------------------------------------------------------------------

    cuda_error::cuda_error(CUresult error)
    : _error(error), std::runtime_error("CUDA error") {

    }

    const char* cuda_error::what() const noexcept {
        const int MSG_LEN = 512;
        const char *name = nullptr;
        const char *message = nullptr;
        char stream[MSG_LEN + 2];
        ::cuGetErrorName(self._error, &name);
        ::cuGetErrorString(self._error, &message);
        ::snprintf(stream, MSG_LEN, "%s: %s", name, message);
        self._what = std::string(stream);
        return self._what.c_str();
    }

    void check(CUresult res) {
        if (res != CUDA_SUCCESS) {
            throw cuda_error(res);
        }
    }

    // void check(cudaError_t res) {
    //     if (res != cudaError_t::cudaSuccess) {
    //         throw cuda_error((CUresult)res);
    //     }
    // }

    // ----------------------------------------------------------------------
    // cuda_device_t
    //     cuda_info_t
    // ----------------------------------------------------------------------

    cuda_t* this_device = nullptr;

    cuda_t::cuda_t() {
        self.ordinal = 0;
        check(::cuInit(0));
        check(::cuDeviceGet(&self.dev, self.ordinal));
        check(::cuCtxCreate(&self.ctx, 0, self.dev));
        check(::cuCtxSetCurrent(self.ctx));
        check(::cuDevicePrimaryCtxRetain(&self.pctx, self.dev));
    }

    cuda_t::~cuda_t() {
        check(::cuDevicePrimaryCtxRelease(self.dev));
        check(::cuCtxDestroy(self.ctx));
        this_device = nullptr;
    }

    std::string cuda_t::name() const {
        char name[32+1];
        check(::cuDeviceGetName(name, 32, self.dev));
        name[32] = 0;
        return {name};
    }

    const cuda_attributes_t& cuda_t::attributes() const {
        cuda_attributes_t& cap = self.attrs;
        if (cap.initialized)
        return cap;

        {
            size_t bytes = 0;
            check(::cuDeviceTotalMem(&bytes, self.dev));
            cap.total_memory_mb = int(bytes/(1024*0124)+1);
        }

        {
            ::cuDeviceGetAttribute(&cap.multiprocessors, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, self.dev);
            ::cuDeviceGetAttribute(&cap.compute_capability.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev);
            ::cuDeviceGetAttribute(&cap.compute_capability.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev);
            ::cuDeviceGetAttribute(&cap.max.threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, self.dev);
            ::cuDeviceGetAttribute(&cap.max.threads_per_multiprocessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, self.dev);
            ::cuDeviceGetAttribute(&cap.max.shared_memory_per_block, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, self.dev);
            ::cuDeviceGetAttribute(&cap.max.blocks_per_multiprocessor, CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, self.dev);
            ::cuDeviceGetAttribute(&cap.warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, self.dev);
            ::cuDeviceGetAttribute(&cap.concurrent_kernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, self.dev);
        }

        {
            ::cuDeviceGetAttribute(&cap.max.grid_dim.x, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, self.dev);
            ::cuDeviceGetAttribute(&cap.max.grid_dim.y, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, self.dev);
            ::cuDeviceGetAttribute(&cap.max.grid_dim.z, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, self.dev);

            ::cuDeviceGetAttribute(&cap.max.block_dim.x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, self.dev);
            ::cuDeviceGetAttribute(&cap.max.block_dim.x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, self.dev);
            ::cuDeviceGetAttribute(&cap.max.block_dim.x, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, self.dev);
        }

        cap.initialized = true;
        return cap;
    }

    int cuda_t::attribute(const std::string& name) const {
        int attrib = ATTRIBUTES[name];
        return self.attribute(attrib);
    }

    int cuda_t::attribute(int attrib) const {
        int value = -1;
        if (attrib >= 0)
            // check(::cuDeviceGetAttribute(&value, CUdevice_attribute(attrib), self.dev));
            ::cuDeviceGetAttribute(&value, CUdevice_attribute(attrib), self.dev);
        elif (attrib == -1) {
            size_t bytes;
            ::cuDeviceTotalMem(&bytes, self.dev);
            value = int(bytes/(1024*1024));
        }
        elif (attrib == -2) {
            ::cuDeviceGetCount(&value);
        }
        elif (attrib == -3) {
            ::cuDriverGetVersion(&value);
        }
        return value;
    }


    // ----------------------------------------------------------------------
    // modules
    // ----------------------------------------------------------------------
    // load_modules([file1,...]) -> module
    // module.destroy();

    module_t::module_t(const char* module_path) {
        self.hmod = nullptr;
        self.load(module_path);
    }

    module_t::~module_t() {
        self.unload();
    }

    void module_t::load(const char* module_path) {
        self.unload();
        check(::cuModuleLoad(&self.hmod, module_path));
    }

    void module_t::unload() {
        if (self.hmod != nullptr) {
            check(::cuModuleUnload(self.hmod));
            self.hmod = nullptr;
        }
    }

    // ----------------------------------------------------------------------
    // end
    // ----------------------------------------------------------------------
}