#include <stddef.h>
#include <string.h>
#include "../../../include/sys/sysctl.h"
#include <windows.h>


static DWORD NumberOfThreads()
{
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

extern int sysctlnametomib(const char *name, int *mibp, size_t *sizep)
{
    if (0 == strcmp("hw.logicalcpu", name))
    {
        mibp[0] = CTL_HW;
        mibp[1] = HW_NCPU;
        *sizep = 2;
        return 0;
    }

    return -1;
}

extern int sysctl(int *name, unsigned int namelen, void *oldp, size_t *oldlenp, void *newp, size_t newlen)
{
    if (namelen == 2 && name[0] == CTL_HW && name[1] == HW_NCPU)
    {
        DWORD nOfThreads = NumberOfThreads();
        (*(DWORD*)oldp) = nOfThreads;
        (*oldlenp) = sizeof(DWORD);
        return 0;
    }

    return -1;
}
