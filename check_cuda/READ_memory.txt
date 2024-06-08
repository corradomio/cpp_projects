Unified memory: same memory address for host & device
    cuMemAllocManaged/cuMemFree

Device memory
    cuMemAlloc cuMemAllocAsync/cuMemFree

Pitch memory:   memory aligned to 2/4/8/16 bytes
    cuMemAllocPitch/cuMemFree

Memory pool
    cuMemAllocFromPoolAsync/cuMemFree

-----------------------------------------------------------------------------

cuMemFree           cuMemAlloc cuMemAllocPitch cuMemAllocManaged
                    cuMemAllocAsync cuMemAllocFromPoolAsync
cuMemFreeHost       cuMemAllocHost

-----------------------------------------------------------------------------

cuArrayDestroy      cuArrayCreate cuArray3DCreate
cuMipmappedArrayDestroy cuMipmappedArrayCreate

cuMemRelease        cuMemCreate
cuMemUnmap          cuMemMap

-----------------------------------------------------------------------------

cuMemAlloc                  Allocates device memory.
cuMemHostAlloc              Allocates page-locked host memory.
cuMemAllocHost              Allocates page-locked host memory.
cuMemAllocManaged           Allocates memory that will be automatically managed by the Unified Memory system.
cuMemAllocPitch             Allocates pitched device memory
cuMemFree
cuMemFreeHost

cuMemHostRegister
cuMemHostUnregister

cuMemGetAddressRange
cuMemHostGetDevicePointer

H: host
D: device
A: array

cuMemcpy            unified memory
cuMemcpyPeer        Copies device memory between two contexts.
cuMemcpyHtoD
cuMemcpyDtoH
cuMemcpyDtoD
cuMemcpyDtoA
cuMemcpyAtoD
cuMemcpyHtoA
cuMemcpyAtoH
cuMemcpyAtoA

cuMemcpyAsync       unified memory
cuMemcpyPeerAsync
cuMemcpyHtoAAsync
cuMemcpyAtoHAsync
cuMemcpyHtoDAsync
cuMemcpyDtoHAsync
cuMemcpyDtoDAsync

cuMemcpy2D
cuMemcpy2DUnaligned
cuMemcpy2DAsync
cuMemcpy3D
cuMemcpy3DAsync
cuMemcpy3Peer
cuMemcpy3PeerAsync

cuMemsetD8
cuMemsetD16
cuMemsetD32
cuMemsetD2D8
cuMemsetD2D16
cuMemsetD2D32

cuMemsetD8Async
cuMemsetD16Async
cuMemsetD2D16Async
cuMemsetD2D32Async

cuArrayCreate
cuArrayGetDescriptor
cuArrayDestroy
cuArrayGetMemoryRequirement
cuArray3DCreate
cuArray3DGetDescriptor

cuMipmappedArrayCreate
cuMipmappedArrayDestroy
cuMipmappedArrayGetMemoryRequirement