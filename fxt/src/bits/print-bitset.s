	.file	"print-bitset.cc"
	.text
	.p2align 4
	.def	__tcf_0;	.scl	3;	.type	32;	.endef
	.seh_proc	__tcf_0
__tcf_0:
.LFB2425:
	.seh_endprologue
	leaq	_ZStL8__ioinit(%rip), %rcx
	jmp	_ZNSt8ios_base4InitD1Ev
	.seh_endproc
	.section .rdata,"dr"
.LC0:
	.ascii "{ \0"
.LC1:
	.ascii "}\0"
.LC2:
	.ascii ",\0"
.LC3:
	.ascii " \0"
	.text
	.p2align 4
	.globl	_Z13print_bit_setPKcmm
	.def	_Z13print_bit_setPKcmm;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z13print_bit_setPKcmm
_Z13print_bit_setPKcmm:
.LFB2089:
	pushq	%r12
	.seh_pushreg	%r12
	pushq	%rbp
	.seh_pushreg	%rbp
	pushq	%rdi
	.seh_pushreg	%rdi
	pushq	%rsi
	.seh_pushreg	%rsi
	pushq	%rbx
	.seh_pushreg	%rbx
	subq	$32, %rsp
	.seh_stackalloc	32
	.seh_endprologue
	movq	%rcx, %r12
	movl	%edx, %ebx
	movl	%r8d, %esi
	testq	%rcx, %rcx
	je	.L4
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L4:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$2, %r8d
	leaq	.LC0(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testl	%esi, %esi
	je	.L5
	leal	(%rbx,%rbx), %eax
	shrl	%ebx
	movl	$32, %ecx
	andl	$-1431655766, %eax
	andl	$1431655765, %ebx
	subl	%esi, %ecx
	orl	%eax, %ebx
	leal	0(,%rbx,4), %eax
	shrl	$2, %ebx
	andl	$858993459, %ebx
	andl	$-858993460, %eax
	orl	%ebx, %eax
	movl	%eax, %ebx
	shrl	$4, %eax
	sall	$4, %ebx
	andl	$252645135, %eax
	andl	$-252645136, %ebx
	orl	%eax, %ebx
/APP
 # 62 "src/bits/bitasm-amd64.h" 1
	bswap %ebx
 # 0 "" 2
/NO_APP
	shrl	%cl, %ebx
.L5:
	xorl	%esi, %esi
	testl	%ebx, %ebx
	je	.L9
	movq	.refptr._ZSt4cout(%rip), %rdi
	movq	%rdi, %r12
	movq	%rdi, %rbp
	jmp	.L6
	.p2align 4,,10
	.p2align 3
.L7:
	addl	$1, %esi
	testl	%ebx, %ebx
	je	.L9
.L6:
	movl	%ebx, %eax
	shrl	%ebx
	andl	$1, %eax
	testl	%eax, %eax
	je	.L7
	movl	%esi, %edx
	movq	%rdi, %rcx
	call	_ZNSo9_M_insertImEERSoT_
	testl	%ebx, %ebx
	jne	.L27
.L8:
	movl	$1, %r8d
	leaq	.LC3(%rip), %rdx
	movq	%r12, %rcx
	addl	$1, %esi
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testl	%ebx, %ebx
	jne	.L6
.L9:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC1(%rip), %rdx
	addq	$32, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	jmp	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	.p2align 4,,10
	.p2align 3
.L27:
	movl	$1, %r8d
	leaq	.LC2(%rip), %rdx
	movq	%rbp, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	jmp	.L8
	.seh_endproc
	.section	.text.startup,"x"
	.p2align 4
	.def	_GLOBAL__sub_I__Z13print_bit_setPKcmm;	.scl	3;	.type	32;	.endef
	.seh_proc	_GLOBAL__sub_I__Z13print_bit_setPKcmm
_GLOBAL__sub_I__Z13print_bit_setPKcmm:
.LFB2426:
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	leaq	_ZStL8__ioinit(%rip), %rcx
	call	_ZNSt8ios_base4InitC1Ev
	leaq	__tcf_0(%rip), %rcx
	addq	$40, %rsp
	jmp	atexit
	.seh_endproc
	.section	.ctors,"w"
	.align 8
	.quad	_GLOBAL__sub_I__Z13print_bit_setPKcmm
.lcomm _ZStL8__ioinit,1,1
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
	.def	_ZNSt8ios_base4InitD1Ev;	.scl	2;	.type	32;	.endef
	.def	strlen;	.scl	2;	.type	32;	.endef
	.def	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x;	.scl	2;	.type	32;	.endef
	.def	_ZNSo9_M_insertImEERSoT_;	.scl	2;	.type	32;	.endef
	.def	_ZNSt8ios_base4InitC1Ev;	.scl	2;	.type	32;	.endef
	.def	atexit;	.scl	2;	.type	32;	.endef
	.section	.rdata$.refptr._ZSt4cout, "dr"
	.globl	.refptr._ZSt4cout
	.linkonce	discard
.refptr._ZSt4cout:
	.quad	_ZSt4cout
