#include <stdio.h>
#include <mkl.h>

int main() {
    printf("Hello cruel world\n");
    MKLVersion mkl_version;
    mkl_get_version(&mkl_version);
    printf("You are using oneMKL %d.%d\n", mkl_version.MajorVersion, mkl_version.UpdateVersion);
    return 0;
}