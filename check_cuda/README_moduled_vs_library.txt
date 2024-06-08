Note
----

    module
        function
            kernel



Modules
-------

    cuModuleLoad             /cuModuleUnload
    cuModuleLoadData
    cuModuleLoadDataEx
    cuModuleLoadFatBinary

    cuModuleEnumerateFunctions
    cuModuleGetFunction
    cuModuleGetFunctionCount


Libraries
---------

    cuLibraryLoadFromFile   /cuLibraryUnload
    cuLibraryLoadData

    cuLibraryGetModule      <<<
    cuLibraryGetKernel
    cuLibraryGetKernelCount

    cuKernelGetFunction
    cuKernelGetLibrary
    cuKernelGetAttribute/cuKernelSetAttributels -la
    cuKernelGetName
    cuKernelGetParamInfo

Functions
---------

    cuLaunchKernel/cuLaunchKernelEx



    CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                    unsigned int gridDimX,
                                    unsigned int gridDimY,
                                    unsigned int gridDimZ,
                                    unsigned int blockDimX,
                                    unsigned int blockDimY,
                                    unsigned int blockDimZ,
                                    unsigned int sharedMemBytes,
                                    CUstream hStream,
                                    void **kernelParams,
                                    void **extra);
    CUresult CUDAAPI cuLaunchKernelEx(const CUlaunchConfig *config,
                                      CUfunction f,
                                      void **kernelParams,
                                      void **extra);