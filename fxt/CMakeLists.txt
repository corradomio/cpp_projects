cmake_minimum_required(VERSION 3.15)
project(fxt)

set(CMAKE_CXX_STANDARD 14)

include_directories(demo/bits)
include_directories(src)
include_directories(src/aux0)
include_directories(src/aux0/attic)
include_directories(src/aux1)
include_directories(src/aux1/attic)
include_directories(src/aux2)
include_directories(src/aux2/attic)
include_directories(src/bits)
include_directories(src/bits/attic)
include_directories(src/bmat)
include_directories(src/bpol)
include_directories(src/bpol/attic)
include_directories(src/chirpzt)
include_directories(src/comb)
include_directories(src/comb/attic)
include_directories(src/convolution)
include_directories(src/correlation)
include_directories(src/dctdst)
include_directories(src/ds)
include_directories(src/ds/attic)
include_directories(src/fft)
include_directories(src/fht)
include_directories(src/graph)
include_directories(src/graph/attic)
include_directories(src/haar)
include_directories(src/haar/attic)
include_directories(src/matrix)
include_directories(src/matrix/attic)
include_directories(src/mod)
include_directories(src/mod/attic)
include_directories(src/ntt)
include_directories(src/perm)
include_directories(src/perm/attic)
include_directories(src/realfft)
include_directories(src/sort)
include_directories(src/sort/attic)
include_directories(src/walsh)
include_directories(src/walsh/attic)
include_directories(src/wavelet)
include_directories(test)

add_executable(fxt
#    demo/arith/briggs-log-demo.cc
#    demo/arith/cayley-dickson-demo.cc
#    demo/arith/cordic-circ-demo.cc
#    demo/arith/cordic-hyp-demo.cc
#    demo/arith/power-costs-demo.cc
#    demo/arith/power-l2r-demo.cc
#    demo/arith/power-r2l-demo.cc
#    demo/arith/shiftadd-exp-demo.cc
#    demo/arith/shiftadd-log-demo.cc
#    demo/arith/zero-divisors-demo.cc
#    demo/arith/zero-divisors-func-demo.cc
#    demo/arith/zero-divisors-pairs-demo.cc
#    demo/bits/all-dbs-demo.cc
#    demo/bits/bin-to-sl-gray-demo.cc
#    demo/bits/bin2naf-demo.cc
#    demo/bits/bin2sbin-demo.cc
#    demo/bits/bit-necklace-demo.cc
#    demo/bits/bit-nextgray-demo.cc
#    demo/bits/bit-paper-fold-alt-demo.cc
#    demo/bits/bit-paper-fold-demo.cc
#    demo/bits/bit-paper-fold-general-demo.cc
#    demo/bits/bit-rll2-demo.cc
#    demo/bits/bit-sl-gray-demo.cc
#    demo/bits/bit2adic-demo.cc
#    demo/bits/bitcombcolex-demo.cc
#    demo/bits/bitcomblex-demo.cc
#    demo/bits/bitcombminchange-demo.cc
#    demo/bits/bitcombshifts-demo.cc
#    demo/bits/bitcount-v-demo.cc
#    demo/bits/bitdemos.h
#    demo/bits/bitfibgray-demo.cc
#    demo/bits/bithilo-demo.cc
#    demo/bits/bitlex-demo.cc
#    demo/bits/bitlex-rec-demo.cc
#    demo/bits/bitmisc-demo.cc
#    demo/bits/bitmisc2-demo.cc
#    demo/bits/bitperm1-demo.cc
#    demo/bits/bitsequency-demo.cc
#    demo/bits/bitsubset-demo.cc
#    demo/bits/bitsubset-gray-demo.cc
#    demo/bits/bitsubset-shift-demo.cc
#    demo/bits/bittransforms-blue-demo.cc
#    demo/bits/bittransforms-blue-fp-demo.cc
#    demo/bits/bittransforms-red-demo.cc
#    demo/bits/bitxtransforms-demo.cc
#    demo/bits/crc64-demo.cc
#    demo/bits/debruijn-lookup-demo.cc
#    demo/bits/dragon-curve-moves-demo.cc
#    demo/bits/dragon-curve-texpic-demo.cc
#    demo/bits/dragon-hex-texpic-demo.cc
#    demo/bits/dragon-r13-texpic-demo.cc
#    demo/bits/dragon-r4-texpic-demo.cc
#    demo/bits/dragon-r5-texpic-demo.cc
#    demo/bits/dragon-r7-2-texpic-demo.cc
#    demo/bits/dragon-r7-texpic-demo.cc
#    demo/bits/dragon-r9-texpic-demo.cc
#    demo/bits/dragon3-texpic-demo.cc
#    demo/bits/fibrep-subset-lexrev-demo.cc
#    demo/bits/fibrep2-demo.cc
#    demo/bits/gotcha-demo.cc
#    demo/bits/gray-demo.cc
#    demo/bits/gray-rec-demo.cc
#    demo/bits/graycode-demo.cc
#    demo/bits/grs-demo.cc
#    demo/bits/grs-next-demo.cc
#    demo/bits/hanoi-demo.cc
#    demo/bits/hilbert-demo.cc
#    demo/bits/hilbert-gray-demo.cc
#    demo/bits/hilbert-moves-demo.cc
#    demo/bits/hilbert-texpic-demo.cc
#    demo/bits/kolakoski-seq-demo.cc
#    demo/bits/modular-lookup-demo.cc
#    demo/bits/negbin-demo.cc
#    demo/bits/negbin2-demo.cc
#    demo/bits/parenword-colex-demo.cc
#    demo/bits/parenword-demo.cc
#    demo/bits/pcrc64-demo.cc
#    demo/bits/radix-2i-demo.cc
#    demo/bits/radix-2i-to-z-demo.cc
#    demo/bits/radix-m1pi-demo.cc
#    demo/bits/radix-m1pi-to-z-demo.cc
#    demo/bits/radix-m4-demo.cc
#    demo/bits/revbin-rec-demo.cc
#    demo/bits/revbin-steps-demo.cc
#    demo/bits/revbin-tab-demo.cc
#    demo/bits/revbin-update-demo.cc
#    demo/bits/thue-morse-demo.cc
#    demo/bits/zorder-texpic-demo.cc
#    demo/comb/attic/dyck-pref2-demo.cc
#    demo/comb/attic/mixedradix-gray2-demo.cc
#    demo/comb/attic/mixedradix-gslex-alt-demo.cc
#    demo/comb/attic/mixedradix-gslex-demo.cc
#    demo/comb/attic/mixedradix-modular-gray-demo.cc
#    demo/comb/attic/mixedradix-rfact-demo.cc
#    demo/comb/attic/perm-gray-ffact2-demo.cc
#    demo/comb/attic/perm-lex-demo.cc
#    demo/comb/attic/perm-lex2-demo.cc
#    demo/comb/attic/perm-rev2-demo.cc
#    demo/comb/acgray-demo.cc
#    demo/comb/acyclic-map-demo.cc
#    demo/comb/arrangement-lex-demo.cc
#    demo/comb/arrangement-rgs-demo.cc
#    demo/comb/ascent-alt-rgs-demo.cc
#    demo/comb/ascent-nonflat-rgs-demo.cc
#    demo/comb/ascent-rgs-demo.cc
#    demo/comb/ascent-rgs-stats-demo.cc
#    demo/comb/ascent-rgs-subset-lex-demo.cc
#    demo/comb/balanced-ordered-tree-lev-seq-demo.cc
#    demo/comb/balanced-ordered-tree-lev-seq-stats-demo.cc
#    demo/comb/ballot-seq-stats-demo.cc
#    demo/comb/bell-number-demo.cc
#    demo/comb/big-fact2perm-demo.cc
#    demo/comb/binary-debruijn-demo.cc
#    demo/comb/binary-huffman-demo.cc
#    demo/comb/binary-necklace-demo.cc
#    demo/comb/binary-sl-gray-demo.cc
#    demo/comb/binomial-demo.cc
#    demo/comb/catalan-demo.cc
#    demo/comb/catalan-flat-path-lex-demo.cc
#    demo/comb/catalan-number-demo.cc
#    demo/comb/catalan-path-lex-demo.cc
#    demo/comb/catalan-rgs-demo.cc
#    demo/comb/catalan-rgs-gray-demo.cc
#    demo/comb/catalan-rgs-gslex-demo.cc
#    demo/comb/catalan-rgs-subset-lex-demo.cc
#    demo/comb/catalan-rgs-to-noncrossing-setpart-rgs-demo.cc
#    demo/comb/catalan-step-rgs-colex-demo.cc
#    demo/comb/catalan-step-rgs-lex-demo.cc
#    demo/comb/catalan-step-rgs-subset-lexrev-demo.cc
#    demo/comb/cayley-perm-demo.cc
#    demo/comb/cayley-perm-stats-demo.cc
#    demo/comb/change-rgs-demo.cc
#    demo/comb/comb2comp-demo.cc
#    demo/comb/combination-chase-demo.cc
#    demo/comb/combination-colex-demo.cc
#    demo/comb/combination-emk-demo.cc
#    demo/comb/combination-emk-rec-demo.cc
#    demo/comb/combination-endo-demo.cc
#    demo/comb/combination-enup-demo.cc
#    demo/comb/combination-enup-rec-demo.cc
#    demo/comb/combination-gray-rec-demo.cc
#    demo/comb/combination-lam-demo.cc
#    demo/comb/combination-lex-demo.cc
#    demo/comb/combination-mod-demo.cc
#    demo/comb/combination-pref-demo.cc
#    demo/comb/combination-rank-demo.cc
#    demo/comb/combination-rec-demo.cc
#    demo/comb/combination-revdoor-demo.cc
#    demo/comb/composition-colex-demo.cc
#    demo/comb/composition-colex2-demo.cc
#    demo/comb/composition-dist-unimodal-demo.cc
#    demo/comb/composition-ex-colex-demo.cc
#    demo/comb/composition-ex-lex-demo.cc
#    demo/comb/composition-gray-rec-demo.cc
#    demo/comb/composition-nz-binary-demo.cc
#    demo/comb/composition-nz-carlitz-demo.cc
#    demo/comb/composition-nz-demo.cc
#    demo/comb/composition-nz-first-max-demo.cc
#    demo/comb/composition-nz-gray-demo.cc
#    demo/comb/composition-nz-gray-rec-demo.cc
#    demo/comb/composition-nz-gray2-demo.cc
#    demo/comb/composition-nz-i-smooth-demo.cc
#    demo/comb/composition-nz-left-2smooth-demo.cc
#    demo/comb/composition-nz-left-smooth-demo.cc
#    demo/comb/composition-nz-max-demo.cc
#    demo/comb/composition-nz-min-demo.cc
#    demo/comb/composition-nz-minc-demo.cc
#    demo/comb/composition-nz-numparts-demo.cc
#    demo/comb/composition-nz-odd-demo.cc
#    demo/comb/composition-nz-odd-subset-lex-demo.cc
#    demo/comb/composition-nz-restrpref-demo.cc
#    demo/comb/composition-nz-rl-demo.cc
#    demo/comb/composition-nz-smooth-demo.cc
#    demo/comb/composition-nz-sorts-demo.cc
#    demo/comb/composition-nz-sorts2-demo.cc
#    demo/comb/composition-nz-sorts2-pp-demo.cc
#    demo/comb/composition-nz-stats-demo.cc
#    demo/comb/composition-nz-subset-lex-demo.cc
#    demo/comb/composition-nz-subset-lex-rec-demo.cc
#    demo/comb/composition-nz-superdiagonal-demo.cc
#    demo/comb/composition-nz-upstep-demo.cc
#    demo/comb/composition-nz-weakly-unimodal-demo.cc
#    demo/comb/composition-rank-demo.cc
#    demo/comb/composition-unimodal-demo.cc
#    demo/comb/conference-quadres-demo.cc
#    demo/comb/cyclic-perm-demo.cc
#    demo/comb/debruijn-demo.cc
#    demo/comb/descent-rgs-demo.cc
#    demo/comb/descent-rgs-stats-demo.cc
#    demo/comb/dyck-gray-demo.cc
#    demo/comb/dyck-gray2-demo.cc
#    demo/comb/dyck-pref-demo.cc
#    demo/comb/dyck-rgs-demo.cc
#    demo/comb/dyck-rgs-subset-lex-demo.cc
#    demo/comb/fact2cyclic-demo.cc
#    demo/comb/fact2perm-demo.cc
#    demo/comb/fact2perm-rev-demo.cc
#    demo/comb/fact2perm-rot-demo.cc
#    demo/comb/fact2perm-swp-demo.cc
#    demo/comb/ffact2kperm-demo.cc
#    demo/comb/fib-alt-gray-demo.cc
#    demo/comb/fibgray-rec-demo.cc
#    demo/comb/gexz-gray-demo.cc
#    demo/comb/hadamard-srs-demo.cc
#    demo/comb/hanoi-rec-demo.cc
#    demo/comb/hilbert-ndim-demo.cc
#    demo/comb/hilbert-ndim-rec-demo.cc
#    demo/comb/id-tree-lev-seq-demo.cc
#    demo/comb/id-tree-lev-seq-stats-demo.cc
#    demo/comb/involution-stats-demo.cc
#    demo/comb/involution-zero-map-rgs-demo.cc
#    demo/comb/isoscent-rgs-demo.cc
#    demo/comb/isoscent-rgs-stats-demo.cc
#    demo/comb/kperm-gray-demo.cc
#    demo/comb/kperm-lex-demo.cc
#    demo/comb/kproducts-colex-demo.cc
#    demo/comb/ksubset-gray-demo.cc
#    demo/comb/ksubset-lex-demo.cc
#    demo/comb/ksubset-rec-demo.cc
#    demo/comb/ksubset-twoclose-demo.cc
#    demo/comb/lyndon-factorization-demo.cc
#    demo/comb/lyndon-words-demo.cc
#    demo/comb/lyndon-words-restrpref-demo.cc
#    demo/comb/map23-rgs-demo.cc
#    demo/comb/maxrep-gray-demo.cc
#    demo/comb/mixedradix-demo.cc
#    demo/comb/mixedradix-endo-demo.cc
#    demo/comb/mixedradix-endo-gray-demo.cc
#    demo/comb/mixedradix-gray-demo.cc
#    demo/comb/mixedradix-gslex-alt-demo.cc
#    demo/comb/mixedradix-gslex-demo.cc
#    demo/comb/mixedradix-modular-gray-demo.cc
#    demo/comb/mixedradix-naf-demo.cc
#    demo/comb/mixedradix-naf-gray-demo.cc
#    demo/comb/mixedradix-naf-subset-lex-demo.cc
#    demo/comb/mixedradix-restrpref-demo.cc
#    demo/comb/mixedradix-rev-demo.cc
#    demo/comb/mixedradix-sl-gray-demo.cc
#    demo/comb/mixedradix-sl-gray-rec-demo.cc
#    demo/comb/mixedradix-sod-lex-demo.cc
#    demo/comb/mixedradix-subset-lex-demo.cc
#    demo/comb/mixedradix-subset-lexrev-demo.cc
#    demo/comb/monotonicgray-demo.cc
#    demo/comb/motzkin-nonflat-rgs-lex-demo.cc
#    demo/comb/motzkin-path-lex-demo.cc
#    demo/comb/motzkin-rgs-lex-demo.cc
#    demo/comb/motzkin-step-rgs-lex-demo.cc
#    demo/comb/mpartition-demo.cc
#    demo/comb/mpartition2-demo.cc
#    demo/comb/mset-kperm-lex-demo.cc
#    demo/comb/mset-ksubset-demo.cc
#    demo/comb/mset-perm-gray-demo.cc
#    demo/comb/mset-perm-lex-demo.cc
#    demo/comb/mset-perm-lex-rec-demo.cc
#    demo/comb/mset-perm-lex-rec2-demo.cc
#    demo/comb/mset-perm-pref-demo.cc
#    demo/comb/mset-subset-lex-demo.cc
#    demo/comb/naf-gray-rec-demo.cc
#    demo/comb/naf-pos-rec-demo.cc
#    demo/comb/necklace-cat-demo.cc
#    demo/comb/necklace-demo.cc
#    demo/comb/necklace-fkm-demo.cc
#    demo/comb/necklace-gray-demo.cc
#    demo/comb/necklace-gray3-demo.cc
#    demo/comb/necklace-sigma-tau-demo.cc
#    demo/comb/necklaces-via-gray-leaders-demo.cc
#    demo/comb/no111-gray-demo.cc
#    demo/comb/no1111-gray-demo.cc
#    demo/comb/no1x1-gray-demo.cc
#    demo/comb/no1xy1-gray-demo.cc
#    demo/comb/ntnz-gray-demo.cc
#    demo/comb/ntz-gray-demo.cc
#    demo/comb/num-partitions-demo.cc
#    demo/comb/ordered-tree-branches-demo.cc
#    demo/comb/ordered-tree-branching-seq-demo.cc
#    demo/comb/ordered-tree-lev-seq-demo.cc
#    demo/comb/ordered-tree-lev-seq-stats-demo.cc
#    demo/comb/paren-demo.cc
#    demo/comb/paren-gray-demo.cc
#    demo/comb/paren-gray-rec-demo.cc
#    demo/comb/paren-lex-demo.cc
#    demo/comb/paren-pref-demo.cc
#    demo/comb/partition-2fall-asc-demo.cc
#    demo/comb/partition-2fall-asc-subset-lex-demo.cc
#    demo/comb/partition-2fall-desc-demo.cc
#    demo/comb/partition-asc-2rep-demo.cc
#    demo/comb/partition-asc-2rep-subset-lex-demo.cc
#    demo/comb/partition-asc-demo.cc
#    demo/comb/partition-asc-perim-demo.cc
#    demo/comb/partition-asc-sorts-demo.cc
#    demo/comb/partition-asc-sorts2-demo.cc
#    demo/comb/partition-asc-sorts2-pp-demo.cc
#    demo/comb/partition-asc-stats-demo.cc
#    demo/comb/partition-asc-subset-lex-csh-demo.cc
#    demo/comb/partition-asc-subset-lex-demo.cc
#    demo/comb/partition-binary-asc-demo.cc
#    demo/comb/partition-binary-desc-demo.cc
#    demo/comb/partition-demo.cc
#    demo/comb/partition-desc-bb-demo.cc
#    demo/comb/partition-desc-demo.cc
#    demo/comb/partition-dist-asc-demo.cc
#    demo/comb/partition-dist-asc-len-demo.cc
#    demo/comb/partition-dist-asc-stats-demo.cc
#    demo/comb/partition-dist-asc-subset-lex-demo.cc
#    demo/comb/partition-dist-d-asc-demo.cc
#    demo/comb/partition-dist-desc-demo.cc
#    demo/comb/partition-gen-demo.cc
#    demo/comb/partition-nonsquashing-desc-demo.cc
#    demo/comb/partition-odd-asc-demo.cc
#    demo/comb/partition-odd-asc-stats-demo.cc
#    demo/comb/partition-odd-asc-subset-lex-csh-demo.cc
#    demo/comb/partition-odd-asc-subset-lex-demo.cc
#    demo/comb/partition-odd-desc-demo.cc
#    demo/comb/partition-odd-nonsquashing-desc-demo.cc
#    demo/comb/partition-rgs-lex-demo.cc
#    demo/comb/partition-s-desc-demo.cc
#    demo/comb/partition-strongly-decr-desc-demo.cc
#    demo/comb/pascal-demo.cc
#    demo/comb/peano-ndim-demo.cc
#    demo/comb/pellgen-gray-demo.cc
#    demo/comb/pellgray-rec-demo.cc
#    demo/comb/perm-colex-demo.cc
#    demo/comb/perm-derange-demo.cc
#    demo/comb/perm-dist1-gray-demo.cc
#    demo/comb/perm-genus-demo.cc
#    demo/comb/perm-gray-ffact-demo.cc
#    demo/comb/perm-gray-lipski-demo.cc
#    demo/comb/perm-gray-rfact-demo.cc
#    demo/comb/perm-gray-rot1-demo.cc
#    demo/comb/perm-gray-wells-demo.cc
#    demo/comb/perm-heap-demo.cc
#    demo/comb/perm-heap2-demo.cc
#    demo/comb/perm-heap2-swaps-demo.cc
#    demo/comb/perm-involution-demo.cc
#    demo/comb/perm-involution-naf-demo.cc
#    demo/comb/perm-ives-demo.cc
#    demo/comb/perm-l1r2-gray-demo.cc
#    demo/comb/perm-lex-cycles-demo.cc
#    demo/comb/perm-lex-demo.cc
#    demo/comb/perm-mv0-demo.cc
#    demo/comb/perm-pref-demo.cc
#    demo/comb/perm-rec-demo.cc
#    demo/comb/perm-restrpref-demo.cc
#    demo/comb/perm-rev-demo.cc
#    demo/comb/perm-right1-gray-demo.cc
#    demo/comb/perm-rot-demo.cc
#    demo/comb/perm-rot-unrank-demo.cc
#    demo/comb/perm-st-demo.cc
#    demo/comb/perm-st-gray-demo.cc
#    demo/comb/perm-st-pref-demo.cc
#    demo/comb/perm-star-demo.cc
#    demo/comb/perm-star-inv-demo.cc
#    demo/comb/perm-star-swaps-demo.cc
#    demo/comb/perm-trotter-demo.cc
#    demo/comb/perm-trotter-lg-demo.cc
#    demo/comb/perm2fact-demo.cc
#    demo/comb/rgs-fincr-demo.cc
#    demo/comb/rgs-kincr-demo.cc
#    demo/comb/rgs-maxincr-demo.cc
#    demo/comb/rll-rec-demo.cc
#    demo/comb/root-sums-demo.cc
#    demo/comb/ruler-func-demo.cc
#    demo/comb/ruler-func-s-demo.cc
#    demo/comb/ruler-func1-demo.cc
#    demo/comb/schroeder-path-lex-demo.cc
#    demo/comb/schroeder-rgs-lex-demo.cc
#    demo/comb/schroeder-tree-demo.cc
#    demo/comb/score-sequence-demo.cc
#    demo/comb/setpart-ccf-rgs-lex-demo.cc
#    demo/comb/setpart-ck-rgs-demo.cc
#    demo/comb/setpart-demo.cc
#    demo/comb/setpart-p-rgs-lex-demo.cc
#    demo/comb/setpart-rgs-gray-demo.cc
#    demo/comb/setpart-rgs-lex-demo.cc
#    demo/comb/setpart-rgs-subset-lex-demo.cc
#    demo/comb/setpart-s-zero-map-rgs-demo.cc
#    demo/comb/setpart-zero-map-rgs-demo.cc
#    demo/comb/shift-subsets-demo.cc
#    demo/comb/skew-binary-demo.cc
#    demo/comb/smooth-rfact-rgs-demo.cc
#    demo/comb/stirling1-demo.cc
#    demo/comb/stirling2-demo.cc
#    demo/comb/string-subst-demo.cc
#    demo/comb/string-subst-hilbert3d-demo.cc
#    demo/comb/subset-debruijn-demo.cc
#    demo/comb/subset-deltalex-demo.cc
#    demo/comb/subset-gray-delta-demo.cc
#    demo/comb/subset-gray-demo.cc
#    demo/comb/subset-lex-demo.cc
#    demo/comb/tree-lev-seq-demo.cc
#    demo/comb/tree-lev-seq-stats-demo.cc
#    demo/comb/weakly-unimodal-rgs-lex-demo.cc
#    demo/comb/wfl-hilbert-demo.cc
#    demo/comb/young-tab-rgs-demo.cc
#    demo/comb/young-tab-rgs-subset-lex-demo.cc
#    demo/ds/bitarray-demo.cc
#    demo/ds/bitarray2d-demo.cc
#    demo/ds/deque-demo.cc
#    demo/ds/left-right-array-demo.cc
#    demo/ds/priorityqueue-demo.cc
#    demo/ds/queue-demo.cc
#    demo/ds/ringbuffer-demo.cc
#    demo/ds/stack-demo.cc
#    demo/fft/butterfly-texpic-demo.cc
#    demo/fft/fft-arblen-demo.cc
#    demo/fft/fht-bench.cc
#    demo/fft/fib-haar-demo.cc
#    demo/fft/fib-walsh-demo.cc
#    demo/fft/gen-walsh-demo.cc
#    demo/fft/locrec-demo.cc
#    demo/fft/mers-haar-demo.cc
#    demo/fft/mers-walsh-demo.cc
#    demo/fft/walsh-basis-demo.cc
#    demo/fft/walsh-bench.cc
#    demo/fft/walsh-eigenvec-demo.cc
#    demo/gf2n/all-irredpoly-demo.cc
#    demo/gf2n/all-normalpoly-demo.cc
#    demo/gf2n/all-primpoly-srs-demo.cc
#    demo/gf2n/bitmat-circulant-demo.cc
#    demo/gf2n/bitmat-kronecker-demo.cc
#    demo/gf2n/bitpol-inverse-demo.cc
#    demo/gf2n/bitpol-search-irred-demo.cc
#    demo/gf2n/bitpol-srp-demo.cc
#    demo/gf2n/bitpolfactor-demo.cc
#    demo/gf2n/bitpolmodmult-demo.cc
#    demo/gf2n/bitpolmult-demo.cc
#    demo/gf2n/clhca-demo.cc
#    demo/gf2n/fcsr-demo.cc
#    demo/gf2n/gf2n-demo.cc
#    demo/gf2n/gf2n-matrix-demo.cc
#    demo/gf2n/gf2n-minpoly-demo.cc
#    demo/gf2n/gf2n-normal-demo.cc
#    demo/gf2n/gf2n-solvequadratic-demo.cc
#    demo/gf2n/gf2n-trace0-generators-demo.cc
#    demo/gf2n/lfsr-demo.cc
#    demo/gf2n/lfsr-fibonacci-demo.cc
#    demo/gf2n/lfsr-galois-demo.cc
#    demo/gf2n/lfsr-revbin-demo.cc
#    demo/gf2n/lfsr-trace0-generators-demo.cc
#    demo/gf2n/lhca-demo.cc
#    demo/gf2n/lhca2poly-demo.cc
#    demo/gf2n/lowbit-lhca-demo.cc
#    demo/gf2n/minweight-lowbit-lhca-demo.cc
#    demo/gf2n/necklace2irred-demo.cc
#    demo/gf2n/normalbasis-demo.cc
#    demo/gf2n/normalpoly-dual-demo.cc
#    demo/gf2n/poly2lhca-demo.cc
#    demo/gf2n/primefact-demo.cc
#    demo/gf2n/qmatrix-demo.cc
#    demo/graph/graph-acgray-demo.cc
#    demo/graph/graph-complementshift-demo.cc
#    demo/graph/graph-debruijn-demo.cc
#    demo/graph/graph-debruijn-m-demo.cc
#    demo/graph/graph-fibrepgray-demo.cc
#    demo/graph/graph-gray-demo.cc
#    demo/graph/graph-lyndon-gray-demo.cc
#    demo/graph/graph-macgray-demo.cc
#    demo/graph/graph-monotonicgray-demo.cc
#    demo/graph/graph-mtl-demo.cc
#    demo/graph/graph-parengray-demo.cc
#    demo/graph/graph-perm-demo.cc
#    demo/graph/graph-perm-doubly-adjacent-gray-demo.cc
#    demo/graph/graph-perm-pref-rev-demo.cc
#    demo/graph/graph-perm-pref-rot-demo.cc
#    demo/graph/graph-perm-star-transpositions-demo.cc
#    demo/graph/lyndon-gray-demo.cc
#    demo/graph/sta-graph-acgray-demo.cc
#    demo/graph/sta-graph-macgray-demo.cc
#    demo/mod/divisors-demo.cc
#    demo/mod/eratosthenes-demo.cc
#    demo/mod/fftprimes-demo.cc
#    demo/mod/kronecker-demo.cc
#    demo/mod/mod-residues-demo.cc
#    demo/mod/modarithtables-demo.cc
#    demo/mod/modinfo-demo.cc
#    demo/mod/modsincos-demo.cc
#    demo/mod/rabinmiller-demo.cc
#    demo/perm/ccf2perm-demo.cc
#    demo/perm/cycles-demo.cc
#    demo/perm/perm-decomp-demo.cc
#    demo/perm/perm-invert-notag-demo.cc
#    demo/perm/permfunc-demo.cc
#    demo/perm/permgray-leaders-demo.cc
#    demo/perm/permrand-connected-demo.cc
#    demo/perm/permrand-cyclic-demo.cc
#    demo/perm/permrand-derange-demo.cc
#    demo/perm/permrand-sdc-demo.cc
#    demo/perm/permrand-self-inverse-demo.cc
#    demo/perm/permutation-matrix-demo.cc
#    demo/perm/revbin-perm-rec-demo.cc
#    demo/perm/swap-blocks-demo.cc
#    demo/seq/A000793-demo.cc
#    demo/seq/A003040-demo.cc
#    demo/seq/A003043-demo.cc
#    demo/seq/A005351-demo.cc
#    demo/seq/A005576-demo.cc
#    demo/seq/A005943-demo.cc
#    demo/seq/A006951-demo.cc
#    demo/seq/A007608-demo.cc
#    demo/seq/A039724-demo.cc
#    demo/seq/A042942-demo.cc
#    demo/seq/A062200-demo.cc
#    demo/seq/A066411-demo.cc
#    demo/seq/A069999-2-demo.cc
#    demo/seq/A069999-demo.cc
#    demo/seq/A080936-demo.cc
#    demo/seq/A100749-demo.cc
#    demo/seq/A175498-demo.cc
#    demo/seq/A177505-demo.cc
#    demo/seq/A178514-demo.cc
#    demo/seq/A178715-demo.cc
#    demo/seq/A179009-demo.cc
#    demo/seq/A179080-demo.cc
#    demo/seq/A182039-demo.cc
#    demo/seq/A182372-demo.cc
#    demo/seq/A186053-demo.cc
#    demo/seq/A187081-demo.cc
#    demo/seq/A191755-demo.cc
#    demo/seq/A206464-demo.cc
#    demo/seq/A206702-demo.cc
#    demo/seq/A207018-demo.cc
#    demo/seq/A209411-demo.cc
#    demo/seq/A212494-demo.cc
#    demo/seq/A213243-demo.cc
#    demo/seq/A217262-demo.cc
#    demo/seq/A217605-demo.cc
#    demo/seq/A218293-demo.cc
#    demo/seq/A218396-demo.cc
#    demo/seq/A218694-demo.cc
#    demo/seq/A218757-demo.cc
#    demo/seq/A225084-demo.cc
#    demo/seq/A225095-demo.cc
#    demo/seq/A225616-demo.cc
#    demo/seq/A226893-demo.cc
#    demo/seq/A227344-demo.cc
#    demo/seq/A240949-demo.cc
#    demo/seq/A247386-demo.cc
#    demo/seq/A249872-demo.cc
#    demo/seq/A259095-demo.cc
#    demo/seq/A276691-demo.cc
#    demo/seq/A278479-demo.cc
#    demo/sort/equivclass-bitstring-demo.cc
#    demo/sort/equivclass-bracelets-demo.cc
#    demo/sort/equivclass-necklaces-demo.cc
#    demo/sort/heapsort-demo.cc
#    demo/sort/merge-sort-demo.cc
#    demo/sort/merge-sort4-demo.cc
#    demo/sort/radixsort-demo.cc
#    demo/sort/selection-sort-demo.cc
#    demo/sort/sort-complex-demo.cc
#    demo/sort/sort-demo.cc
#    demo/sort/sort-string-demo.cc
#    demo/sort/sortidx-demo.cc
#    demo/sort/sortptr-demo.cc
#    demo/sort/unique-demo.cc

#    src/aux0/attic/aux0double.cc
#    src/aux0/attic/aux0double.h
#    src/aux0/attic/binomial.cc
#    src/aux0/attic/constants.h
#    src/aux0/attic/funcemu.h
#    src/aux0/attic/gcd.cc
#    src/aux0/attic/range.h
#    src/aux0/attic/sign.h
#    src/aux0/attic/sincos.cc
#    src/aux0/attic/timer.h
#    src/aux0/attic/trigrec.h

    src/aux0/binomial.h
    src/aux0/cayley-dickson-mult.h
    src/aux0/cmult.h
    src/aux0/csincos.h
    src/aux0/factorial.h
    src/aux0/fibonacci.h
    src/aux0/gcd.h
    src/aux0/ipow.h
    src/aux0/jjassert.cc
    src/aux0/ldn2rc.h
    src/aux0/print-fixed.cc
    src/aux0/print-fixed.h
    src/aux0/rand-idx.h
    src/aux0/randf.cc
    src/aux0/randf.h
    src/aux0/sincos.h
    src/aux0/sumdiff.h
    src/aux0/swap.h
    src/aux0/tex-line.cc
    src/aux0/tex-line.h
    src/aux0/version.cc
    src/aux0/version.h

#    src/aux1/attic/applyfunc.h
#    src/aux1/attic/array.h
#    src/aux1/attic/aux1double.cc
#    src/aux1/attic/aux1double.h
#    src/aux1/attic/auxprint.cc
#    src/aux1/attic/auxprint.h
#    src/aux1/attic/delta.h
#    src/aux1/attic/diff.h
#    src/aux1/attic/graph-print.cc
#    src/aux1/attic/graph-print.h
#    src/aux1/attic/misc.h
#    src/aux1/attic/norm.h
#    src/aux1/attic/resample.cc
#    src/aux1/attic/resample.h
#    src/aux1/attic/scale.h
#    src/aux1/attic/scanbox.h
#    src/aux1/attic/shift.h
#    src/aux1/attic/str2ulong.cc
#    src/aux1/attic/symmetry.cc
#    src/aux1/attic/symmetry.h

    src/aux1/arith1.h
    src/aux1/bytescan.cc
    src/aux1/bytescan.h
    src/aux1/copy.h
    src/aux1/num2str.cc
    src/aux1/num2str.h
    src/aux1/wordgray.h

#    src/aux2/attic/applyfunc2d.h
#    src/aux2/attic/arith2d.h
#    src/aux2/attic/copy2d.h
#    src/aux2/attic/minmax2d.h
#    src/aux2/attic/reverse2d.h
#    src/aux2/attic/rotate2d.h
#    src/aux2/attic/scale2d.h
#    src/aux2/attic/shift2d.h
#    src/aux2/attic/transpose2.h

    src/aux2/transpose.h
    src/aux2/transpose2.h

#    src/bits/attic/bit-necklace-demo.cc
#    src/bits/attic/bit2composition.h
#    src/bits/attic/bitmrotate.h
#    src/bits/attic/bitsasm.cc
#    src/bits/attic/fibrep-demo.cc

    src/bits/average.h
    src/bits/bin-to-sl-gray.h
    src/bits/bin2naf.h
    src/bits/bit-dragon-r13.h
    src/bits/bit-dragon-r4.h
    src/bits/bit-dragon-r5.h
    src/bits/bit-dragon-r7.h
    src/bits/bit-dragon-r9.h
    src/bits/bit-dragon3.h
    src/bits/bit-isolate.h
    src/bits/bit-necklace.h
    src/bits/bit-paper-fold.h
    src/bits/bit-rll2.h
    src/bits/bit-sl-gray.h
    src/bits/bit2adic.h
    src/bits/bit2pow.h
    src/bits/bitasm-amd64.h
    src/bits/bitasm-i386.h
    src/bits/bitasm-sse.h
    src/bits/bitasm.h
    src/bits/bitblock.h
    src/bits/bitbutterfly.h
    src/bits/bitcombcolex.h
    src/bits/bitcombminchange.h
    src/bits/bitcombshifts.h
    src/bits/bitcopy.h
    src/bits/bitcount-v.cc
    src/bits/bitcount.h
    src/bits/bitcyclic-dist.h
    src/bits/bitcyclic-match.h
    src/bits/bitcyclic-minmax.h
    src/bits/bitcyclic-period.h
    src/bits/bitcyclic-xor.h
    src/bits/bitfibgray.h
    src/bits/bitgather.h
    src/bits/bitgraypermute.h
    src/bits/bithigh-edge.h
    src/bits/bithigh.h
    src/bits/bitldeq.h
    src/bits/bitlex.h
    src/bits/bitlow-edge.h
    src/bits/bitlow.h
    src/bits/bitperiodic.h
    src/bits/bitrotate.h
    src/bits/bitseparate.h
    src/bits/bitsequency.h
    src/bits/bitset2set.h
    src/bits/bitsperlong.h
    src/bits/bitsubset-gray.h
    src/bits/bitsubset.h
    src/bits/bitsubsetq.h
    src/bits/bitswap.h
    src/bits/bittest.h
    src/bits/bittransforms.h
    src/bits/bitxtransforms.h
    src/bits/bitzip-pairs.h
    src/bits/bitzip.h
    src/bits/blue-fixed-points.h
    src/bits/branchless.h
    src/bits/colormix-fl.h
    src/bits/colormix.h
    src/bits/colormixp.h
    src/bits/crc32.cc
    src/bits/crc32.h
    src/bits/crc64.cc
    src/bits/crc64.h
    src/bits/cswap.h
    src/bits/evenodd.h
    src/bits/fibrep-subset-lexrev.h
    src/bits/fibrep.h
    src/bits/graycode.h
    src/bits/graypower.h
    src/bits/grsnegative.h
    src/bits/hilbert.h
    src/bits/ith-one-idx.h
    src/bits/kolakoski-seq.h
    src/bits/lin2hilbert.cc
    src/bits/negbin.h
    src/bits/nextgray.h
    src/bits/parenwords.h
    src/bits/parity.h
    src/bits/pcrc64.h
    src/bits/print-bin.cc
    src/bits/print-bin.h
    src/bits/print-bindiff.cc
    src/bits/print-bitset.cc
    src/bits/radix-2i.h
    src/bits/radix-m1pi.h
    src/bits/radix-m4.h
    src/bits/revbin-tab.cc
    src/bits/revbin-upd.cc
    src/bits/revbin-upd.h
    src/bits/revbin.h
    src/bits/revgraycode.h
    src/bits/tcrc64.h
    src/bits/thue-morse.h
    src/bits/tinyfactors.cc
    src/bits/tinyfactors.h
    src/bits/zerobyte.h
    src/bits/zorder.h
    src/bmat/bitmat-charpoly.cc
    src/bmat/bitmat-funcs.h
    src/bmat/bitmat-hessenberg.cc
    src/bmat/bitmat-inline.h
    src/bmat/bitmat-inverse.cc
    src/bmat/bitmat-kronecker.cc
    src/bmat/bitmat-mult-mm.cc
    src/bmat/bitmat-nullspace.cc
    src/bmat/bitmat-print.cc
    src/bmat/bitmat-transpose.cc
    src/bpol/attic/bitpolmod-multrev.h
    src/bpol/all-irredpoly.h
    src/bpol/berlekamp.cc
    src/bpol/bitpol-arith.h
    src/bpol/bitpol-degree.h
    src/bpol/bitpol-deriv.h
    src/bpol/bitpol-factor.cc
    src/bpol/bitpol-factor.h
    src/bpol/bitpol-gcd.h
    src/bpol/bitpol-irred-ben-or.cc
    src/bpol/bitpol-irred-rabin.cc
    src/bpol/bitpol-irred.h
    src/bpol/bitpol-normal.cc
    src/bpol/bitpol-order.cc
    src/bpol/bitpol-order.h
    src/bpol/bitpol-primitive.h
    src/bpol/bitpol-print.cc
    src/bpol/bitpol-print.h
    src/bpol/bitpol-spi.cc
    src/bpol/bitpol-squarefree.cc
    src/bpol/bitpol-squarefree.h
    src/bpol/bitpol-srp.h
    src/bpol/bitpol2lhca.cc
    src/bpol/bitpolmod-arith.h
    src/bpol/bitpolmod-minpoly.cc
    src/bpol/bitpolmod-minpoly.h
    src/bpol/bitpolmod-solvequadratic.cc
    src/bpol/bitpolmod-solvequadratic.h
    src/bpol/clhca.h
    src/bpol/fcsr.h
    src/bpol/gf2n-minpoly.cc
    src/bpol/gf2n-order.cc
    src/bpol/gf2n-solvequadratic.cc
    src/bpol/gf2n-trace.cc
    src/bpol/gf2n-trace.h
    src/bpol/gf2n.cc
    src/bpol/gf2n.h
    src/bpol/lfsr.h
    src/bpol/lfsr64.h
    src/bpol/lhca.h
    src/bpol/lhcarule-minweight.cc
    src/bpol/mersenne-coprime.h
    src/bpol/necklace2bitpol.h
    src/bpol/normal-irredpoly.cc
    src/bpol/normal-mult.cc
    src/bpol/normal-primpoly.cc
    src/bpol/normal-solvequadratic.h
    src/bpol/normalbasis.h
    src/bpol/normalpoly-dual.cc
    src/bpol/normalpoly-dual.h
    src/bpol/num-bitpol.cc
    src/bpol/num-bitpol.h
    src/bpol/poly-tab.h
    src/bpol/primpoly-lowbit.cc
    src/bpol/primpoly-minweight.cc
    src/bpol/primpoly-rand.cc
    src/chirpzt/chirpzt.h
    src/chirpzt/fftarblen.cc
    src/chirpzt/fftfract.cc
    src/chirpzt/slowfracft.cc
    src/chirpzt/slowzt.cc
    src/comb/attic/all-compositions-minchange.cc
    src/comb/attic/all-compositions-minchange.h
    src/comb/attic/allsetpartitions.cc
    src/comb/attic/allsetpartitions.h
    src/comb/attic/ascent-rgs.h
    src/comb/attic/binarydebruijn.cc
    src/comb/attic/binarydebruijn.h
    src/comb/attic/combcolex.h
    src/comb/attic/combination-altminchange.h
    src/comb/attic/combination-minchange.cc
    src/comb/attic/combination-minchange.h
    src/comb/attic/combination-rec-demo.cc
    src/comb/attic/comblex.h
    src/comb/attic/combminchange.cc
    src/comb/attic/combminchange.h
    src/comb/attic/composition-chase.h
    src/comb/attic/compositionalt.h
    src/comb/attic/dyck-pref2.h
    src/comb/attic/mixedradix-gray2.h
    src/comb/attic/mixedradix-gslex-alt.h
    src/comb/attic/mixedradix-gslex.h
    src/comb/attic/mixedradix-modular-gray.h
    src/comb/attic/mixedradix-rfact.h
    src/comb/attic/mset-perm-invtab.h
    src/comb/attic/num2mixedradix.h
    src/comb/attic/partition-dist-lex-rec-demo.cc
    src/comb/attic/partition-lex-rec-demo.cc
    src/comb/attic/perm-gray-ffact2.h
    src/comb/attic/perm-lex.h
    src/comb/attic/perm-lex2.h
    src/comb/attic/perm-minchange.h
    src/comb/attic/perm-rev2.h
    src/comb/attic/perm-trotter.h
    src/comb/attic/permvisit.cc
    src/comb/attic/permvisit.h
    src/comb/attic/stringsubst-demo.cc
    src/comb/attic/stringsubst.cc
    src/comb/attic/stringsubst.h
    src/comb/attic/subset-monotone-demo.cc
    src/comb/attic/subset-monotone.h
    src/comb/attic/subsetdebruijn.h
    src/comb/attic/tree-lev-seq-branching-num.h
    src/comb/acgray.cc
    src/comb/acgray.h
    src/comb/acyclic-map.h
    src/comb/arith-3-progression.h
    src/comb/arrangement-lex.h
    src/comb/arrangement-rgs.h
    src/comb/ascent-alt-rgs.h
    src/comb/ascent-nonflat-rgs.h
    src/comb/ascent-rgs-subset-lex.h
    src/comb/ascent-rgs.h
    src/comb/balanced-ordered-tree-lev-seq.h
    src/comb/big-fact2perm.cc
    src/comb/big-fact2perm.h
    src/comb/binary-debruijn.h
    src/comb/binary-huffman.cc
    src/comb/binary-huffman.h
    src/comb/binary-necklace.h
    src/comb/binary-sl-gray.h
    src/comb/catalan-flat-path-lex.h
    src/comb/catalan-path-lex.h
    src/comb/catalan-rgs-gray.h
    src/comb/catalan-rgs-gslex.h
    src/comb/catalan-rgs-subset-lex.h
    src/comb/catalan-rgs-to-noncrossing-setpart-rgs.h
    src/comb/catalan-rgs.h
    src/comb/catalan-step-rgs-colex.h
    src/comb/catalan-step-rgs-lex.h
    src/comb/catalan-step-rgs-subset-lexrev.h
    src/comb/catalan-step-rgs-to-paren-string.h
    src/comb/catalan.cc
    src/comb/catalan.h
    src/comb/cayley-perm.cc
    src/comb/cayley-perm.h
    src/comb/change-rgs.h
    src/comb/check-kpermgen.h
    src/comb/check-mixedradix.h
    src/comb/check-permgen.h
    src/comb/comb-print.h
    src/comb/combination-chase.h
    src/comb/combination-colex.h
    src/comb/combination-emk.h
    src/comb/combination-endo.h
    src/comb/combination-enup.h
    src/comb/combination-lex.h
    src/comb/combination-mod.h
    src/comb/combination-pref.h
    src/comb/combination-rec.cc
    src/comb/combination-rec.h
    src/comb/combination-revdoor.h
    src/comb/comp2comb.h
    src/comb/composition-colex.h
    src/comb/composition-colex2.h
    src/comb/composition-dist-unimodal.h
    src/comb/composition-ex-colex.h
    src/comb/composition-ex-lex.h
    src/comb/composition-nz-binary.h
    src/comb/composition-nz-carlitz.h
    src/comb/composition-nz-conj.h
    src/comb/composition-nz-first-max.h
    src/comb/composition-nz-gray.h
    src/comb/composition-nz-gray2.h
    src/comb/composition-nz-i-smooth.h
    src/comb/composition-nz-left-2smooth.h
    src/comb/composition-nz-left-smooth.h
    src/comb/composition-nz-max.h
    src/comb/composition-nz-min.h
    src/comb/composition-nz-minc.cc
    src/comb/composition-nz-minc.h
    src/comb/composition-nz-numparts.h
    src/comb/composition-nz-odd-subset-lex.h
    src/comb/composition-nz-odd.h
    src/comb/composition-nz-rank.cc
    src/comb/composition-nz-rank.h
    src/comb/composition-nz-restrpref.h
    src/comb/composition-nz-rl.h
    src/comb/composition-nz-smooth.h
    src/comb/composition-nz-sorts.h
    src/comb/composition-nz-sorts2-pp.h
    src/comb/composition-nz-sorts2.h
    src/comb/composition-nz-subset-lex.h
    src/comb/composition-nz-superdiagonal.h
    src/comb/composition-nz-upstep.h
    src/comb/composition-nz-weakly-unimodal.h
    src/comb/composition-nz.h
    src/comb/composition-rank.cc
    src/comb/composition-rank.h
    src/comb/composition-unimodal.h
    src/comb/cyclic-perm.h
    src/comb/cyclic-words.h
    src/comb/debruijn.h
    src/comb/delta2gray.cc
    src/comb/delta2gray.h
    src/comb/descent-rgs.h
    src/comb/dyck-gray.h
    src/comb/dyck-gray2.h
    src/comb/dyck-pref.h
    src/comb/dyck-rgs-subset-lex.h
    src/comb/dyck-rgs.h
    src/comb/endo-enup.h
    src/comb/fact2cyclic.cc
    src/comb/fact2num.cc
    src/comb/fact2num.h
    src/comb/fact2num2perm.h
    src/comb/fact2perm-rev.cc
    src/comb/fact2perm-rot.cc
    src/comb/fact2perm-swp.cc
    src/comb/fact2perm.cc
    src/comb/fact2perm.h
    src/comb/gray-compare.h
    src/comb/gray-cycle-leaders.h
    src/comb/hilbert-ndim-rec.h
    src/comb/hilbert-ndim.h
    src/comb/id-tree-lev-seq.h
    src/comb/involution-zero-map-rgs.h
    src/comb/is-arrangement-rgs.h
    src/comb/is-ascent-rgs.h
    src/comb/is-catalan-path.h
    src/comb/is-catalan-rgs.h
    src/comb/is-catalan-step-rgs.h
    src/comb/is-cayley-perm.h
    src/comb/is-change-rgs.h
    src/comb/is-composition-nz.h
    src/comb/is-descent-rgs.h
    src/comb/is-dyck-rgs.h
    src/comb/is-isoscent-rgs.h
    src/comb/is-mixedradix-num.h
    src/comb/is-motzkin-path.h
    src/comb/is-motzkin-rgs.h
    src/comb/is-motzkin-step-rgs.h
    src/comb/is-noncrossing-setpart-rgs.h
    src/comb/is-nonsquashing.h
    src/comb/is-paren-position-word.h
    src/comb/is-paren-string.h
    src/comb/is-partition-asc.h
    src/comb/is-partition-desc.h
    src/comb/is-partition-rgs.h
    src/comb/is-schroeder-path.h
    src/comb/is-schroeder-rgs.h
    src/comb/is-setpart-ccf-perm.h
    src/comb/is-setpart-rgs.h
    src/comb/is-shifted-young-tab-rgs.h
    src/comb/is-smooth.h
    src/comb/is-sorts-in-runs-sorted.h
    src/comb/is-stack-sortable.h
    src/comb/is-symmetric.h
    src/comb/is-unimodal.h
    src/comb/is-young-tab-rgs.h
    src/comb/is-zero-map-rgs.h
    src/comb/isoscent-rgs.h
    src/comb/kperm-gray.h
    src/comb/kperm-lex.h
    src/comb/ksubset-gray.h
    src/comb/ksubset-lex.h
    src/comb/ksubset-rec.cc
    src/comb/ksubset-rec.h
    src/comb/ksubset-twoclose.h
    src/comb/lex-compare.h
    src/comb/lindenmayer-system.cc
    src/comb/lindenmayer-system.h
    src/comb/lyndon-factorization.h
    src/comb/lyndon-words-restrpref.h
    src/comb/lyndon-words.h
    src/comb/map23-rgs.h
    src/comb/mixedradix-aux.h
    src/comb/mixedradix-endo-gray.h
    src/comb/mixedradix-endo.h
    src/comb/mixedradix-gray.h
    src/comb/mixedradix-gslex-alt.h
    src/comb/mixedradix-gslex.h
    src/comb/mixedradix-init.cc
    src/comb/mixedradix-modular-gray.h
    src/comb/mixedradix-naf-gray.h
    src/comb/mixedradix-naf-subset-lex.h
    src/comb/mixedradix-naf.h
    src/comb/mixedradix-restrpref.h
    src/comb/mixedradix-rev.h
    src/comb/mixedradix-sl-gray-compare.h
    src/comb/mixedradix-sl-gray-rank.h
    src/comb/mixedradix-sl-gray.h
    src/comb/mixedradix-sod-lex.h
    src/comb/mixedradix-subset-lex-rank.h
    src/comb/mixedradix-subset-lex.h
    src/comb/mixedradix-subset-lexrev.h
    src/comb/mixedradix.h
    src/comb/mixedradix2num.cc
    src/comb/monotonic-gray.cc
    src/comb/monotonic-gray.h
    src/comb/motzkin-nonflat-rgs-lex.h
    src/comb/motzkin-path-lex.h
    src/comb/motzkin-rgs-lex.h
    src/comb/motzkin-step-rgs-lex.h
    src/comb/mpartition.h
    src/comb/mpartition2.h
    src/comb/mset-kperm-lex.h
    src/comb/mset-perm-gray.h
    src/comb/mset-perm-lex-rec.cc
    src/comb/mset-perm-lex-rec.h
    src/comb/mset-perm-lex.h
    src/comb/mset-perm-pref.h
    src/comb/necklace.h
    src/comb/num-compositions.h
    src/comb/num-necklaces.cc
    src/comb/num-necklaces.h
    src/comb/num2perm.cc
    src/comb/num2perm.h
    src/comb/ordered-tree-branches.cc
    src/comb/ordered-tree-branches.h
    src/comb/ordered-tree-branching-seq.cc
    src/comb/ordered-tree-branching-seq.h
    src/comb/ordered-tree-lev-seq.h
    src/comb/paren-gray.h
    src/comb/paren-lex.h
    src/comb/paren-pref.h
    src/comb/paren-string-to-rgs.cc
    src/comb/paren-string-to-rgs.h
    src/comb/paren.h
    src/comb/partition-2fall-asc-subset-lex.h
    src/comb/partition-2fall-asc.h
    src/comb/partition-2fall-desc.h
    src/comb/partition-asc-2rep-subset-lex.h
    src/comb/partition-asc-2rep.h
    src/comb/partition-asc-perim.h
    src/comb/partition-asc-sorts.h
    src/comb/partition-asc-sorts2-pp.h
    src/comb/partition-asc-sorts2.h
    src/comb/partition-asc-subset-lex-csh.h
    src/comb/partition-asc-subset-lex.h
    src/comb/partition-asc.h
    src/comb/partition-binary-asc.h
    src/comb/partition-binary-desc.h
    src/comb/partition-boundary.h
    src/comb/partition-conj.cc
    src/comb/partition-conj.h
    src/comb/partition-desc-bb.h
    src/comb/partition-desc.h
    src/comb/partition-dist-asc-len.h
    src/comb/partition-dist-asc-subset-lex.h
    src/comb/partition-dist-asc.h
    src/comb/partition-dist-d-asc.h
    src/comb/partition-dist-desc.h
    src/comb/partition-gen.cc
    src/comb/partition-gen.h
    src/comb/partition-hook-prod.h
    src/comb/partition-nonsquashing-desc.h
    src/comb/partition-odd-asc-subset-lex-csh.h
    src/comb/partition-odd-asc-subset-lex.h
    src/comb/partition-odd-asc.h
    src/comb/partition-odd-desc.h
    src/comb/partition-odd-nonsquashing-desc.h
    src/comb/partition-odd-to-dist.h
    src/comb/partition-rgs-lex.cc
    src/comb/partition-rgs-lex.h
    src/comb/partition-s-desc.h
    src/comb/partition-strongly-decr-desc.h
    src/comb/partition.cc
    src/comb/partition.h
    src/comb/peano-ndim.h
    src/comb/perm-colex.h
    src/comb/perm-derange.h
    src/comb/perm-gray-ffact.h
    src/comb/perm-gray-lipski.h
    src/comb/perm-gray-rfact.h
    src/comb/perm-gray-rot1.h
    src/comb/perm-gray-wells.h
    src/comb/perm-heap.h
    src/comb/perm-heap2-swaps.h
    src/comb/perm-heap2.h
    src/comb/perm-involution.h
    src/comb/perm-ives.h
    src/comb/perm-lex.h
    src/comb/perm-mv0.h
    src/comb/perm-pref.h
    src/comb/perm-rec.h
    src/comb/perm-restrpref.h
    src/comb/perm-rev.h
    src/comb/perm-rot.h
    src/comb/perm-st-gray.h
    src/comb/perm-st-pref.h
    src/comb/perm-st.h
    src/comb/perm-star-swaps.h
    src/comb/perm-star.h
    src/comb/perm-trotter-lg.h
    src/comb/perm-trotter.h
    src/comb/print-arrangement-rgs-perm.cc
    src/comb/print-arrangement-rgs-perm.h
    src/comb/print-catalan-path-aa.cc
    src/comb/print-catalan-path-aa.h
    src/comb/print-catalan-step-rgs-aa.cc
    src/comb/print-catalan-step-rgs-aa.h
    src/comb/print-composition-aa.cc
    src/comb/print-composition-aa.h
    src/comb/print-composition-by-sorts.cc
    src/comb/print-composition-by-sorts.h
    src/comb/print-composition-unimodal.cc
    src/comb/print-composition-unimodal.h
    src/comb/print-gray.cc
    src/comb/print-mixedradix.cc
    src/comb/print-mset.cc
    src/comb/print-partition-aa.cc
    src/comb/print-partition-aa.h
    src/comb/print-partition-conj.cc
    src/comb/print-partition-conj.h
    src/comb/print-perm.cc
    src/comb/print-set.cc
    src/comb/print-setpart.cc
    src/comb/print-vec.cc
    src/comb/print-young-tab-rgs-aa.cc
    src/comb/print-young-tab-rgs-aa.h
    src/comb/print-zero-map-rgs.cc
    src/comb/print-zero-map-rgs.h
    src/comb/reverse-paren-string.h
    src/comb/rgs-fincr.h
    src/comb/rgs-kincr.h
    src/comb/rgs-maxincr.h
    src/comb/ruler-func-s.h
    src/comb/ruler-func.h
    src/comb/ruler-func1.h
    src/comb/schroeder-path-lex.h
    src/comb/schroeder-rgs-lex.h
    src/comb/score-sequence.h
    src/comb/setpart-ccf-rgs-lex.h
    src/comb/setpart-ck-rgs.h
    src/comb/setpart-p-rgs-lex.h
    src/comb/setpart-rgs-gray.cc
    src/comb/setpart-rgs-gray.h
    src/comb/setpart-rgs-lex.h
    src/comb/setpart-rgs-subset-lex.h
    src/comb/setpart-s-zero-map-rgs.h
    src/comb/setpart-zero-map-rgs.h
    src/comb/setpart.cc
    src/comb/setpart.h
    src/comb/skew-binary.h
    src/comb/smooth-rfact-rgs.h
    src/comb/string-subst.cc
    src/comb/string-subst.h
    src/comb/subset-debruijn.h
    src/comb/subset-deltalex.h
    src/comb/subset-gray-delta.h
    src/comb/subset-gray.h
    src/comb/subset-lex-compare.h
    src/comb/subset-lex.h
    src/comb/test-gray.cc
    src/comb/test-gray.h
    src/comb/tree-lev-seq-aux.cc
    src/comb/tree-lev-seq-aux.h
    src/comb/tree-lev-seq.h
    src/comb/weakly-unimodal-rgs-lex.h
    src/comb/wfl-hilbert.h
    src/comb/word-stats.h
    src/comb/young-tab-rgs-descents.h
    src/comb/young-tab-rgs-subset-lex.h
    src/comb/young-tab-rgs.h
    src/convolution/cfhtcnvl.cc
    src/convolution/cfhtcnvla.cc
    src/convolution/cfhtcnvlacore.cc
    src/convolution/cfhtcnvlcore.cc
    src/convolution/complexconvolution.h
    src/convolution/fftcnvl.cc
    src/convolution/fftcnvla.cc
    src/convolution/fftcocnvl.cc
    src/convolution/fhtcnvl.cc
    src/convolution/fhtcnvla.cc
    src/convolution/fhtcnvlacore.cc
    src/convolution/fhtcnvlcore.cc
    src/convolution/fhtloccnvl.cc
    src/convolution/fhtloccnvla.cc
    src/convolution/fhtmulsqr.h
    src/convolution/fhtnegacnvl.cc
    src/convolution/matrixfftcnvl.cc
    src/convolution/matrixfftcnvla.cc
    src/convolution/matrixfftcocnvla.cc
    src/convolution/realconvolution.h
    src/convolution/slowcnvl-lin.h
    src/convolution/slowcnvl.h
    src/convolution/slowcnvla.h
    src/convolution/slowcnvlhalf.h
    src/convolution/slowcocnvl.cc
    src/convolution/slowconvolution.h
    src/convolution/slowtwodimcnvl.h
    src/convolution/slowweightedcnvl.h
    src/convolution/twodimfhtcnvl.cc
    src/convolution/weightedconv.cc
    src/convolution/weightedconvolution.h
    src/correlation/correlation.h
    src/correlation/fftcocorr.cc
    src/correlation/fftcorr.cc
    src/correlation/fhtcorr.cc
    src/correlation/slowcocorr.cc
    src/correlation/slowcorr.h
    src/dctdst/cosrot.cc
    src/dctdst/dct.cc
    src/dctdst/dct4.cc
    src/dctdst/dctdst.h
    src/dctdst/dcth.cc
    src/dctdst/dctzapata.cc
    src/dctdst/dst.cc
    src/dctdst/dsth.cc
    src/ds/attic/coroutine-paren-demo.cc
    src/ds/attic/coroutine.h
    src/ds/attic/orderedrarray-demo.cc
    src/ds/attic/orderedrarray.h
    src/ds/attic/priorityqueue1.h
    src/ds/attic/rarray-demo.cc
    src/ds/attic/rarray.h
    src/ds/attic/rset-demo.cc
    src/ds/attic/rset.h
    src/ds/array2d.h
    src/ds/array3d.h
    src/ds/bitarray.cc
    src/ds/bitarray.h
    src/ds/bitarray2d.cc
    src/ds/bitarray2d.h
    src/ds/bounding-box.h
    src/ds/centered-array2d.h
    src/ds/centered-array3d.h
    src/ds/deque.h
    src/ds/heap.h
    src/ds/left-right-array.h
    src/ds/point2d.h
    src/ds/point3d.h
    src/ds/priorityqueue.h
    src/ds/queue.h
    src/ds/ringbuffer.h
    src/ds/stack.h
    src/ds/vector2d.h
    src/ds/vector3d.h
    src/fft/attic/gfft.cc
    src/fft/attic/ndimfft.cc
    src/fft/cfftdif4.cc
    src/fft/cfftdit4.cc
    src/fft/cfftsplitradix.cc
    src/fft/cfftwrap.cc
    src/fft/columnffts.cc
    src/fft/fft-default.h
    src/fft/fft.h
    src/fft/fft8difcore.cc
    src/fft/fft8ditcore.cc
    src/fft/fft9.cc
    src/fft/fftdif2.cc
    src/fft/fftdif4.cc
    src/fft/fftdif4l.cc
    src/fft/fftdit2.cc
    src/fft/fftdit4.cc
    src/fft/fftdit4l.cc
    src/fft/fftspect.cc
    src/fft/fftsplitradix.cc
    src/fft/fhtcfft.cc
    src/fft/fhtfft.cc
    src/fft/fouriershift.cc
    src/fft/matrixfft.cc
    src/fft/matrixfft.h
    src/fft/recfft2.cc
    src/fft/rowcnvls.cc
    src/fft/rowffts.cc
    src/fft/shortfft.h
    src/fft/skipfft.cc
    src/fft/slowft.cc
    src/fft/slowft.h
    src/fft/twodimfft.cc
    src/fft/weightedfft.cc
    src/fht/attic/fhtdit2loc.cc
    src/fht/attic/fhttable.cc
    src/fht/attic/trigtable.cc
    src/fht/cfht0.cc
    src/fht/cfhtdif.cc
    src/fht/cfhtdit.cc
    src/fht/fht-default.h
    src/fht/fht.h
    src/fht/fht0.cc
    src/fht/fht2d.h
    src/fht/fhtdif.cc
    src/fht/fhtdif2.cc
    src/fht/fhtdit.cc
    src/fht/fhtdit2.cc
    src/fht/fhtloc2.h
    src/fht/fhtspect.cc
    src/fht/hartleyshift.h
    src/fht/recfht2.cc
    src/fht/shortfhtdifcore.h
    src/fht/shortfhtditcore.h
    src/fht/skipfht.cc
    src/fht/slowht.cc
    src/fht/slowht.h
    src/fht/twodimfht.cc
    src/graph/attic/debruijngraph.cc
    src/graph/attic/debruijngraph.h
    src/graph/digraph-paths.cc
    src/graph/digraph-paths.h
    src/graph/digraph.cc
    src/graph/digraph.h
    src/graph/lyndon-cmp.cc
    src/graph/lyndon-gray.h
    src/graph/mk-complete-digraph.cc
    src/graph/mk-debruijn-digraph.cc
    src/graph/mk-fibrep-gray-digraph.cc
    src/graph/mk-gray-digraph.cc
    src/graph/mk-lyndon-gray-digraph.cc
    src/graph/mk-mtl-digraph.cc
    src/graph/mk-paren-gray-digraph.cc
    src/graph/mk-perm-gray-digraph.cc
    src/graph/mk-perm-pref-rev-digraph.cc
    src/graph/mk-perm-pref-rot-digraph.cc
    src/graph/mk-special-digraphs.h
    src/graph/print-path.cc
    src/graph/print-path.h
    src/graph/search-digraph-cond.cc
    src/graph/search-digraph-trylucky.cc
    src/graph/search-digraph.cc
    src/haar/attic/haari2i.cc
    src/haar/attic/haari2i.h
    src/haar/fib-haar.h
    src/haar/haar.h
    src/haar/haarnn.h
    src/haar/haarrevnn.h
    src/haar/mers-haar.h
    src/haar/prefix-transform.h
    src/haar/transposedhaarnn.h
    src/haar/transposedhaarrevnn.h
    src/matrix/attic/cholesky.h
    src/matrix/attic/matrixsymmetryq.h
    src/matrix/attic/mmult.h
    src/matrix/attic/reorth.h
    src/matrix/attic/symmetricpart.h
    src/matrix/attic/testmatrix.h
    src/matrix/attic/toeplitz.h
    src/matrix/matrix.h
    src/matrix/vector.h
    src/mod/attic/auxfuncs.cc
    src/mod/attic/gcd.cc
    src/mod/attic/mersenne-generators.cc
    src/mod/attic/modaux.cc
    src/mod/attic/primroot.cc
    src/mod/attic/testnum.cc
    src/mod/attic/testnum.h
    src/mod/chebyshev.h
    src/mod/chebyshev1.cc
    src/mod/chebyshev2.cc
    src/mod/chinese.cc
    src/mod/cyclic.cc
    src/mod/divisors.h
    src/mod/eratosthenes.cc
    src/mod/euler-phi.cc
    src/mod/factor.cc
    src/mod/factor.h
    src/mod/factormake.cc
    src/mod/isqrt.h
    src/mod/kronecker.cc
    src/mod/maxorder.cc
    src/mod/mersenne-exponents.cc
    src/mod/mersenne-factors.cc
    src/mod/mersenne.h
    src/mod/mod.cc
    src/mod/mod.h
    src/mod/modarith.h
    src/mod/modinfo.cc
    src/mod/modinit.cc
    src/mod/mtypes.h
    src/mod/numtheory.h
    src/mod/order.cc
    src/mod/perfpow.cc
    src/mod/primes.cc
    src/mod/primes.h
    src/mod/quadresidue.cc
    src/mod/rabinmiller.cc
    src/mod/sqrtmod.cc
    src/ntt/attic/nttdif2nc.cc
    src/ntt/attic/nttndim.cc
    src/ntt/attic/nttskip.cc
    src/ntt/ntt.h
    src/ntt/nttcnvl.cc
    src/ntt/nttdif2.cc
    src/ntt/nttdif4.cc
    src/ntt/nttdit2.cc
    src/ntt/nttdit4.cc
    src/ntt/nttlearn.cc
    src/ntt/slowntt.cc
    src/ntt/slownttcnvl.cc
    src/perm/attic/cycles.cc
    src/perm/attic/cycles.h
    src/perm/attic/graypermute.cc
    src/perm/attic/graypermute.h
    src/perm/attic/greenpermute.h
    src/perm/attic/greenrevpermute.h
    src/perm/attic/mrotate.h
    src/perm/attic/partition.cc
    src/perm/attic/partition.h
    src/perm/attic/revbinpermute.h
    src/perm/even2lower.h
    src/perm/fact2perm-swp-apply.h
    src/perm/graypermute.h
    src/perm/grayrevpermute.h
    src/perm/haarpermute.h
    src/perm/perm-genus.h
    src/perm/perm2ccf.cc
    src/perm/perm2ccf.h
    src/perm/permapply.h
    src/perm/permapplyfunc.h
    src/perm/permcomplement.h
    src/perm/permcompose.cc
    src/perm/permcompose.h
    src/perm/perminvert.cc
    src/perm/perminvert.h
    src/perm/permq.cc
    src/perm/permq.h
    src/perm/permrand-2cycles.h
    src/perm/permrand-connected.h
    src/perm/permrand-cycle-type.h
    src/perm/permrand-cyclic.h
    src/perm/permrand-derange.h
    src/perm/permrand-derange3.h
    src/perm/permrand-inv-mod-m.h
    src/perm/permrand-ncm2.h
    src/perm/permrand-ord.h
    src/perm/permrand-parity.h
    src/perm/permrand-pref.h
    src/perm/permrand-sdc.h
    src/perm/permrand-self-inverse.h
    src/perm/permrand.h
    src/perm/printcycles.cc
    src/perm/printcycles.h
    src/perm/radixpermute.h
    src/perm/revbinpermute.h
    src/perm/revbinpermute0.h
    src/perm/reverse.h
    src/perm/rotate.h
    src/perm/shortgraypermute.h
    src/perm/shortrevbinpermute.h
    src/perm/shortrevbinpermute0.h
    src/perm/swapblocks.h
    src/perm/xorpermute.h
    src/perm/zip.cc
    src/perm/zip.h
    src/perm/ziprev.h
    src/realfft/realfft.h
    src/realfft/realfftbyfht.cc
    src/realfft/realffteasyord.cc
    src/realfft/realfftsplitradix.cc
    src/realfft/realfftwrap.cc
    src/realfft/skiprealfft.cc
    src/sort/attic/grep.h
    src/sort/attic/monotone.h
    src/sort/attic/searchdown.h
    src/sort/attic/searchunsorted.h
    src/sort/attic/sort.h
    src/sort/attic/sortcomplex.cc
    src/sort/attic/sortcomplex.h
    src/sort/attic/symbolify-demo.cc
    src/sort/attic/symbolify.h
    src/sort/attic/ucount.h
    src/sort/attic/uniqueapprox.h
    src/sort/attic/usearchfunc.h
    src/sort/bsearch.h
    src/sort/bsearchapprox.h
    src/sort/bsearchfunc.h
    src/sort/bsearchidx.h
    src/sort/bsearchidxfunc.h
    src/sort/bsearchptr.h
    src/sort/bsearchptrfunc.h
    src/sort/convex.h
    src/sort/equivclasses.h
    src/sort/heapsort.h
    src/sort/merge-sort.h
    src/sort/minmax.h
    src/sort/minmaxfunc.h
    src/sort/minmaxidx.h
    src/sort/minmaxidxfunc.h
    src/sort/minmaxmed23.h
    src/sort/minmaxmed23func.h
    src/sort/minmaxmed23idx.h
    src/sort/minmaxptr.h
    src/sort/minmaxptrfunc.h
    src/sort/quantize.h
    src/sort/radixsort.cc
    src/sort/radixsort.h
    src/sort/sort.h
    src/sort/sort23.h
    src/sort/sort23func.h
    src/sort/sortbykey.h
    src/sort/sortfunc.h
    src/sort/sortidx.h
    src/sort/sortidxfunc.h
    src/sort/sortptr.h
    src/sort/sortptrfunc.h
    src/sort/unique.h
    src/sort/uniquefunc.h
    src/sort/usearch.h
    src/walsh/attic/parity-transform.h
    src/walsh/attic/slowwalsh.cc
    src/walsh/attic/walsh0.cc
    src/walsh/attic/walshwakmatrix.h
    src/walsh/and-convolution.h
    src/walsh/arithtransform.h
    src/walsh/bitcount-weight.h
    src/walsh/dyadiccnvl.h
    src/walsh/fib-walsh.h
    src/walsh/grsnegate.h
    src/walsh/max-convolution.h
    src/walsh/mers-walsh.h
    src/walsh/or-convolution.h
    src/walsh/reedmuller.h
    src/walsh/shortwalshwakdif.h
    src/walsh/shortwalshwakdit.h
    src/walsh/slant.cc
    src/walsh/slant.h
    src/walsh/square-wave-transform.h
    src/walsh/subset-convolution.h
    src/walsh/walsh-basis.h
    src/walsh/walsheigen.cc
    src/walsh/walsheigen.h
    src/walsh/walshgray.h
    src/walsh/walshpal.h
    src/walsh/walshq.h
    src/walsh/walshseq.h
    src/walsh/walshwak.h
    src/walsh/walshwak2.h
    src/walsh/walshwak4.h
    src/walsh/walshwak8.h
    src/walsh/walshwakloc2.h
    src/walsh/walshwal.h
    src/walsh/walshwalrev.h
    src/walsh/weighted-arithtransform.h
    src/walsh/weighted-or-convolution.h
    src/wavelet/attic/waveletfilter.cc
    src/wavelet/daubechies.cc
    src/wavelet/daubechies.h
    src/wavelet/harmonic-wavelet.cc
    src/wavelet/harmonic-wavelet.h
    src/wavelet/invwavelet.cc
    src/wavelet/wavelet.cc
    src/wavelet/wavelet.h
    src/wavelet/waveletfilter.h
    src/array-len.h
    src/aux0-all.h
    src/aux1-all.h
    src/aux2-all.h
    src/bits-all.h
    src/bmat-all.h
    src/bpol-all.h
    src/chirpzt-all.h
    src/comb-all.h
    src/complextype.h
    src/convolution-all.h
    src/correlation-all.h
    src/dctdst-all.h
    src/ds-all.h
    src/fft-all.h
    src/fht-all.h
    src/fxtalloca.h
    src/fxtio.h
    src/fxttypes.h
    src/graph-all.h
    src/haar-all.h
    src/jjassert.h
    src/matrix-all.h
    src/mod-all.h
    src/nextarg.cc
    src/nextarg.h
    src/ntt-all.h
    src/perm-all.h
    src/realfft-all.h
    src/realloc.h
    src/restrict.h
    src/sort-all.h
    src/walsh-all.h
    src/wavelet-all.h
#    test/aux-test.cc
#    test/dummy.cc
#    test/test-bits.cc
#    test/test-convolution.cc
#    test/test-fact.cc
#    test/test-fft.cc
#    test/test-fht.cc
#    test/test-include-all.cc
#    test/test-misc.cc
#    test/test-permute.cc
#    test/test-transforms.cc
#    test/test-transpose.cc
#    test/test.h
)
