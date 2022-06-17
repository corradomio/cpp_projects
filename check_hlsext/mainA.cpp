//
// Created by Corrado Mio on 10/10/2020.
//

#include <iostream>
#include <stdx/properties.h>
#include <stdio.h>
#include <stdx/cmathx.h>
#include <winsock2.h>
#include <iphlpapi.h>

//  (ULONG Family, ULONG Flags, PVOID Reserved, PIP_ADAPTER_ADDRESSES AdapterAddresses, PULONG SizePointer);

int main() {
    printf("%g\n", stdx::math::epsof(0));
    printf("%g\n", stdx::math::epsof(1));
    GetAdaptersAddresses(0, 0, NULL, NULL, NULL);
}