#include <stdio.h>
#include <stdint.h>

#include <functional>
#include <ostream>
#include "main.h"

int main() {
    printf("sizeof(enum) = %d\n", sizeof(c10::DeviceType));
    return 0;
}