//
// Created by Corrado Mio on 15/06/2024.
//
#include <cassert>
#include <map>
#include "stdx/number/i32/numbers.h"

// n!/k!
// 9! / 3!*6!
//      1 2 3  4 5 6 7 8 9
//      1 2 3
//             1 2 3 4 5 6

//
// binomial(n, k) = (n!) / (k! (n-k)!)
//                = binomial(n-1, k-1) + binomial(n-1, k)
//
#include "stdx/number/i32/numbers.h"

namespace stdx::number::i32::cns {

    static std::map<std::pair<int, int>, iset_t> _binomial_cache;

    /// (n, k) = (n!)/(k!(n-k)!)
    iset_t binomial(int n, int k) {
        if (k < 0 or n < k)
            return 0;
        if (k == 0 or n == k)
            return 1;

        std::pair<int, int> nk = std::make_pair(n, k);
        auto it = _binomial_cache.find(nk);
        if (it != _binomial_cache.end())
            return it->second;

        iset_t b = 1;

        // if (n < k || k <= 0 || n <= 0) return 0;

        for (int i = 1; i<=(n - k); ++i) {
            b *= k + i;
            b /= i;
        }

        _binomial_cache[nk] = b;

        return b;
    }

    int hibit(iset_t S) {
        assert (S >= 0);

        int bit = -1;
        while (S != 0) {
            S >>= 1;
            bit += 1;
        }

        return bit;
    }


    int ihighbit(iset_t S) {
        int h = -1;
        while (S != 0) {
            h += 1;
            S >>= 1;
        }
        return h;
    }

    static int _BIT_COUNTS[] = {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3,
        3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
        3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4,
        4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
        3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6,
        6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5,
        4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8};

    int icard(iset_t S) {
        int c = 0;
        while (S != 0) {
            c += _BIT_COUNTS[S & 0xFF];
            S >>= 8;
        }
        return c;
    }

    iset_t iadd(iset_t S, int i) {
        return S | (ONE << i);
    }

    iset_t ilexset(iset_t L, int n) {
        iset_t S = 0;
        int k = -1;
        iset_t nk = binomial(n, k);

        while (nk <= L) {
            L -= nk;
            k += 1;
            nk = binomial(n, k);
        }

        while (k > 0) {
            iset_t ck = 0;
            iset_t ckk = binomial(ck, k);
            while (ckk <= L) {
                ck += 1;
                ckk = binomial(ck, k);
            }
            ck -= 1;
            L -= binomial(ck, k);
            S = iadd(S, ck);
            k -= 1;
        }
        return S;
    }

    iset_t ilexidx(iset_t S, int n) {
        int m = icard(S);

        iset_t L = 0;
        for (int k=0; k<m; ++k)
            L += binomial(n, k);

        int i = 0;
        iset_t ci;
        for (int e=0; e<n && S; ++e) {
            if (S & ONE) {
                i += 1;
                ci = binomial(e, i);
                L += ci;
            }

            S >>= 1;
        }
        return L;
    }

}

