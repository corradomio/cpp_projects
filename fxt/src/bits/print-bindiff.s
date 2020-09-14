	.file	"print-bindiff.cc"
	.text
	.p2align 4
	.def	__tcf_0;	.scl	3;	.type	32;	.endef
	.seh_proc	__tcf_0
__tcf_0:
.LFB2426:
	.seh_endprologue
	leaq	_ZStL8__ioinit(%rip), %rcx
	jmp	_ZNSt8ios_base4InitD1Ev
	.seh_endproc
	.p2align 4
	.globl	_Z14print_bin_diffPKcyymS0_
	.def	_Z14print_bin_diffPKcyymS0_;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z14print_bin_diffPKcyymS0_
_Z14print_bin_diffPKcyymS0_:
.LFB2089:
	pushq	%r15
	.seh_pushreg	%r15
	pushq	%r14
	.seh_pushreg	%r14
	pushq	%r13
	.seh_pushreg	%r13
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
	subq	$56, %rsp
	.seh_stackalloc	56
	.seh_endprologue
	movq	160(%rsp), %rbx
	movq	%rdx, 136(%rsp)
	movq	%rcx, %r13
	movq	%r8, %rdi
	movl	%r9d, %ebp
	testq	%rcx, %rcx
	je	.L19
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r13, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L5:
	movq	136(%rsp), %rsi
	xorq	%rdi, %rsi
	notq	%rsi
	testl	%ebp, %ebp
	je	.L6
	movl	$64, %ecx
	leaq	_ZL5n01pm(%rip), %rax
	movabsq	$-9223372036854775808, %r9
	subl	%ebp, %ecx
	shrq	%cl, %r9
	testq	%rbx, %rbx
	cmove	%rax, %rbx
	movq	%r9, %r12
	testq	%r9, %r9
	je	.L3
.L8:
	movq	.refptr._ZSt4cout(%rip), %r13
	leaq	47(%rsp), %rbp
	movq	%r13, %r14
	movq	%r13, %r15
	jmp	.L13
	.p2align 4,,10
	.p2align 3
.L21:
	movzbl	2(%rbx), %eax
	movl	$1, %r8d
	movq	%rbp, %rdx
	movq	%r14, %rcx
	movb	%al, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L10:
	shrq	%r12
	je	.L3
.L13:
	testq	%rsi, %r12
	jne	.L20
	testq	%r12, %rdi
	jne	.L21
	movzbl	3(%rbx), %eax
	movl	$1, %r8d
	movq	%rbp, %rdx
	movq	%r13, %rcx
	movb	%al, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	shrq	%r12
	jne	.L13
.L3:
	addq	$56, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
	.p2align 4,,10
	.p2align 3
.L20:
	xorl	%eax, %eax
	testq	%r12, 136(%rsp)
	movq	%rbp, %rdx
	movq	%r15, %rcx
	setne	%al
	movl	$1, %r8d
	movzbl	(%rbx,%rax), %eax
	movb	%al, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	jmp	.L10
	.p2align 4,,10
	.p2align 3
.L6:
	testq	%rbx, %rbx
	je	.L14
	movl	$2147483648, %r12d
	jmp	.L8
	.p2align 4,,10
	.p2align 3
.L19:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L5
	.p2align 4,,10
	.p2align 3
.L14:
	leaq	_ZL5n01pm(%rip), %rbx
	movl	$2147483648, %r12d
	jmp	.L8
	.seh_endproc
	.p2align 4
	.globl	_Z18print_bin_vec_diffPKcyymS0_
	.def	_Z18print_bin_vec_diffPKcyymS0_;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z18print_bin_vec_diffPKcyymS0_
_Z18print_bin_vec_diffPKcyymS0_:
.LFB2090:
	.seh_endprologue
	movq	%rcx, %r10
	movq	%rdx, %rcx
	leal	(%rdx,%rdx), %edx
	shrl	%ecx
	andl	$-1431655766, %edx
	leal	(%r8,%r8), %eax
	shrl	%r8d
	andl	$1431655765, %ecx
	andl	$-1431655766, %eax
	andl	$1431655765, %r8d
	orl	%ecx, %edx
	orl	%eax, %r8d
	leal	0(,%rdx,4), %ecx
	shrl	$2, %edx
	leal	0(,%r8,4), %eax
	shrl	$2, %r8d
	andl	$858993459, %edx
	andl	$-858993460, %ecx
	andl	$-858993460, %eax
	andl	$858993459, %r8d
	orl	%edx, %ecx
	orl	%r8d, %eax
	movl	%ecx, %edx
	movl	%eax, %r8d
	shrl	$4, %ecx
	sall	$4, %edx
	andl	$252645135, %ecx
	sall	$4, %r8d
	andl	$-252645136, %edx
	shrl	$4, %eax
	andl	$-252645136, %r8d
	orl	%ecx, %edx
	andl	$252645135, %eax
	movl	$32, %ecx
	subl	%r9d, %ecx
	orl	%r8d, %eax
	movl	%eax, %r8d
/APP
 # 62 "src/bits/bitasm-amd64.h" 1
	bswap %edx
 # 0 "" 2
 # 62 "src/bits/bitasm-amd64.h" 1
	bswap %r8d
 # 0 "" 2
/NO_APP
	shrl	%cl, %edx
	shrl	%cl, %r8d
	movq	%r10, %rcx
	jmp	_Z14print_bin_diffPKcyymS0_
	.seh_endproc
	.section	.text.startup,"x"
	.p2align 4
	.def	_GLOBAL__sub_I__Z14print_bin_diffPKcyymS0_;	.scl	3;	.type	32;	.endef
	.seh_proc	_GLOBAL__sub_I__Z14print_bin_diffPKcyymS0_
_GLOBAL__sub_I__Z14print_bin_diffPKcyymS0_:
.LFB2427:
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
	.quad	_GLOBAL__sub_I__Z14print_bin_diffPKcyymS0_
	.section .rdata,"dr"
_ZL5n01pm:
	.ascii ".1+-"
.lcomm _ZStL8__ioinit,1,1
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
	.def	_ZNSt8ios_base4InitD1Ev;	.scl	2;	.type	32;	.endef
	.def	strlen;	.scl	2;	.type	32;	.endef
	.def	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x;	.scl	2;	.type	32;	.endef
	.def	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate;	.scl	2;	.type	32;	.endef
	.def	_ZNSt8ios_base4InitC1Ev;	.scl	2;	.type	32;	.endef
	.def	atexit;	.scl	2;	.type	32;	.endef
	.section	.rdata$.refptr._ZSt4cout, "dr"
	.globl	.refptr._ZSt4cout
	.linkonce	discard
.refptr._ZSt4cout:
	.quad	_ZSt4cout
