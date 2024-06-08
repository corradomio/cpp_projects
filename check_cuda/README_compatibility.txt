nvcuda ha version 12.2

Non funziona usando CUDA 12.4
Ma  funziona usando CUDA 12.2

Comunque, nvcc dice una cosa strana:

    D:\Programming\CUDA\v12.2\include\crt/host_config.h(164): fatal error C1189: #error:
    -- unsupported Microsoft Visual Studio version! Only the versions between 2017 and 2022 (inclusive) are supported!
     The nvcc flag '-allow-unsupported-compiler' can be used to override this version check; however, using an
     unsupported host compiler may cause compilation failure or incorrect run time execution. Use at your own risk.


il limite e':

    #if _MSC_VER < 1910 || _MSC_VER >= 1940     UNSUPPORTED
    #if _MSC_VER >= 1910 && _MSC_VER < 1910     DEPRECATED


bisogna usare:

    nvcc vecadd.cu -ptx -allow-unsupported-compiler

trick:
    modificato "host_config.h" in modo da avere

        _MSC_VER > 1940

Microsoft cl macros
-------------------

    echo // > foo.cpp
    cl /Zc:preprocessor /PD foo.cpp

    _MSC_VER 1940:      TROPPO recente




Visual Studio Compatibility
---------------------------

    https://quasar.ugent.be/files/doc/cuda-msvc-compatibility.html

    VS 2022     v12.*   v11.6
    VS 2019     v12.*   v11.6
    VS 2017     v12.*   v11.6   v9.*    v8.0
    VS 2015             v11.6   v9.*    v8.0
    VS 2013             v11.6   v9.*    v8.0    v7.*    v6.5
    VS 2012             v11.6   v9.*    v8.0    v7.*    v6.5    v6.0    v5.5



CUDA architecture
-----------------

    https://en.wikipedia.org/wiki/CUDA
    https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

    Fermi       sm_20
    Kepler      sm_30   sm_35   sm_37
    Maxwell     sm_50   sm_52   sm_53
    Pascal      sm_60   sm_61   sm_61
    Volta       sm_70   sm_72 (Xavier)
    Turing      sm_75
    Ampere      sm_80   sm_86   sm_87 (Orin)
    Ada         sm_89
    Hopper      sm_90   sm_90a (Thor)
    Blackwell   ???

    Fermi   deprecated after CUDA 9
    Kepler  deprecated after CUDA 11
    Maxwell deprecated after CUDA 11.6

