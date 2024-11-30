#include <iostream>
#include <intrin.h>
#include <inttypes.h>

int main()
{
    std::cout << "Hello, World!" << std::endl;

    try
    {
        // POPCNT
        std::cout << _popcnt64(0x0F0F0F0F) << std::endl;
        // SSE 4.2
        std::cout << _mm_crc32_u16((unsigned int)uint32_t(0x0F0F0F0F), (unsigned short)uint16_t(0xFF)) << std::endl;
    }
    catch(...)
    {
        std::cout << "opps" << std::endl;
    }

    return 0;
}
