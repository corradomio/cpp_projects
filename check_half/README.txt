https://en.wikipedia.org/wiki/IEEE_754

https://github.com/intheswim/float24/blob/master/float24.h
https://github.com/biovault/biovault_bfloat16
https://github.com/oprecomp/FloatX
https://github.com/oprecomp/flexfloat
https://github.com/acgessler/half_float     (umHalf.h, umHalf.inl)
https://github.com/x448/float16             (go)
https://github.com/fengwang/float16_t
https://iis-git.ee.ethz.ch/pulp-sw/softFloat

---------------------------------------------------------------------------
{s, e, m}   con |e| e |m|

bias esponente  q = 2^(|e|-1) - 1       |e|=8   -> q=127

        s   e-q        m           (-1)^s   e          m
f = (-1)  2     (1 + ------)   =  -------- 2   (1 + -------)
                      2^|m|          2^q             2^|m|


ATTENZIONE: se e = 0, allora f va calcolato come

                                       m
           f = (-1)^s s^(-q+1) (0 + -------)
                                     2^|m|


---------------------------------------------------------------------------
FP = ExMy

q = 2^(|e|-1)-1

+-+-----+----------+
|s|e    |m         |       E5M10
+-+-----+----------+


f = (-1)^s 1/(2^q) 2^e 1.m      if e > 0
f = (-1)^s 1/(2^q) 2^e 0.m      if e = 0

https://en.wikipedia.org/wiki/Half-precision_floating-point_format


|e| = 5     q = 15

0 00000 0000000000      0
0 00000 0000000001      2^(-15) x (0 +    1/1024)
0 00000 1111111111      2^(-15) x (0 + 1023/1024)
0 00001 0000000000      2^(-14) x (1 +    0/1024)
0 01101 0101010101      2^( -2) x (1 +  341/1024)
0 01110 1111111111      2^( -1) x (1 + 1024/1024)
0 01111 0000000000      2^(  0) x (1 +    0/1024)
0 01111 0000000001      2^(  0) x (1 +    1/1024)
0 11110 1111111111      2^( 15) x (1 + 1024/1024)
0 11111 0000000000      +infinity
1 00000 0000000000      -0
1 10000 0000000000      -2
1 11111 0000000000      -infinity
-----------------------------------------------------------------------------

stdint.h
stdint-gcc.h


IEEE 754
--------

int8_t
int16_t
int32_t
int64_t

uint8_t
uint16_t
uint32_t
uint64_t

---------------

int_least8_t
int_least16_t
int_least32_t
int_least64_t

uint_least8_t
uint_least16_t
uint_least32_t
uint_least64_t

int_fast8_t
int_fast16_t
int_fast32_t
int_fast64_t

uint_fast8_t
uint_fast16_t
uint_fast32_t
uint_fast64_t

intptr_t
uintptr_t

intmax_t
uintmax_t

----------

float16_t
float32_t
float64_t
float128_t
bfloat16_t


name        format      bits
----------------------------
FP64        E10F53      64

FP32        E8M23       32
BF16        E8M7        16

FP16        E5M10       16
FP8         E4M3         8
            E5M2         8

FP24        E8M15       24
TF32        E8M10       19
