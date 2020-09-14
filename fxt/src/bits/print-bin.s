	.file	"print-bin.cc"
	.text
	.p2align 4
	.def	__tcf_0;	.scl	3;	.type	32;	.endef
	.seh_proc	__tcf_0
__tcf_0:
.LFB2401:
	.seh_endprologue
	leaq	_ZStL8__ioinit(%rip), %rcx
	jmp	_ZNSt8ios_base4InitD1Ev
	.seh_endproc
	.p2align 4
	.globl	_Z9print_binPKcymS0_
	.def	_Z9print_binPKcymS0_;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z9print_binPKcymS0_
_Z9print_binPKcymS0_:
.LFB2062:
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
	subq	$48, %rsp
	.seh_stackalloc	48
	.seh_endprologue
	movq	%rcx, %r12
	movq	%rdx, %rdi
	movl	%r8d, %ebx
	movq	%r9, %rsi
	testq	%rcx, %rcx
	je	.L17
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L5:
	testl	%ebx, %ebx
	je	.L6
	movl	$64, %ecx
	leaq	_ZL3n01(%rip), %rax
	subl	%ebx, %ecx
	movabsq	$-9223372036854775808, %rbx
	shrq	%cl, %rbx
	testq	%rsi, %rsi
	cmove	%rax, %rsi
	testq	%rbx, %rbx
	je	.L3
.L10:
	movq	.refptr._ZSt4cout(%rip), %rbp
	leaq	47(%rsp), %r12
	.p2align 4,,10
	.p2align 3
.L9:
	xorl	%eax, %eax
	testq	%rbx, %rdi
	movl	$1, %r8d
	movq	%r12, %rdx
	setne	%al
	movq	%rbp, %rcx
	movzbl	(%rsi,%rax), %eax
	movb	%al, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	shrq	%rbx
	jne	.L9
.L3:
	addq	$48, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	ret
	.p2align 4,,10
	.p2align 3
.L6:
	testq	%rsi, %rsi
	je	.L11
	movl	$2147483648, %ebx
	jmp	.L10
	.p2align 4,,10
	.p2align 3
.L17:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L5
	.p2align 4,,10
	.p2align 3
.L11:
	leaq	_ZL3n01(%rip), %rsi
	movl	$2147483648, %ebx
	jmp	.L10
	.seh_endproc
	.p2align 4
	.globl	_Z13print_bin_vecPKcymS0_
	.def	_Z13print_bin_vecPKcymS0_;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z13print_bin_vecPKcymS0_
_Z13print_bin_vecPKcymS0_:
.LFB2063:
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
	movq	%rcx, %r12
	movq	%rdx, %rbx
	movl	%r8d, %ebp
	movq	%r9, %rdi
	testq	%rcx, %rcx
	je	.L25
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L20:
	testl	%ebp, %ebp
	movl	$32, %eax
	leaq	47(%rsp), %r13
	movq	.refptr._ZSt4cout(%rip), %r12
	cmove	%eax, %ebp
	testq	%rdi, %rdi
	leaq	_ZL3n01(%rip), %rax
	cmove	%rax, %rdi
	xorl	%esi, %esi
	.p2align 4,,10
	.p2align 3
.L23:
	movq	%rbx, %rax
	movl	$1, %r8d
	movq	%r13, %rdx
	andl	$1, %eax
	movq	%r12, %rcx
	addl	$1, %esi
	shrq	%rbx
	movzbl	(%rdi,%rax), %eax
	movb	%al, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	cmpl	%esi, %ebp
	jne	.L23
	addq	$56, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	popq	%r13
	ret
	.p2align 4,,10
	.p2align 3
.L25:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L20
	.seh_endproc
	.section .rdata,"dr"
.LC0:
	.ascii "[\0"
.LC1:
	.ascii ", \0"
.LC2:
	.ascii "]\0"
	.text
	.p2align 4
	.globl	_Z13print_idx_seqPKcym
	.def	_Z13print_idx_seqPKcym;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z13print_idx_seqPKcym
_Z13print_idx_seqPKcym:
.LFB2064:
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
	movq	%rdx, %rbx
	movl	%r8d, %edi
	testq	%rcx, %rcx
	je	.L39
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L28:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	movl	$1, %esi
	subl	$1, %edi
	leaq	.LC0(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %r12
	movq	%r12, %rbp
	jmp	.L32
	.p2align 4,,10
	.p2align 3
.L29:
	shrq	%rbx
	je	.L30
.L31:
	addl	$1, %esi
.L32:
	testb	$1, %bl
	je	.L29
	leal	(%rdi,%rsi), %edx
	movq	%r12, %rcx
	call	_ZNSo9_M_insertImEERSoT_
	cmpq	$1, %rbx
	je	.L30
	movl	$2, %r8d
	leaq	.LC1(%rip), %rdx
	movq	%rbp, %rcx
	shrq	%rbx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	jmp	.L31
	.p2align 4,,10
	.p2align 3
.L30:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC2(%rip), %rdx
	addq	$32, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	jmp	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	.p2align 4,,10
	.p2align 3
.L39:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L28
	.seh_endproc
	.section	.text.startup,"x"
	.p2align 4
	.def	_GLOBAL__sub_I__Z9print_binPKcymS0_;	.scl	3;	.type	32;	.endef
	.seh_proc	_GLOBAL__sub_I__Z9print_binPKcymS0_
_GLOBAL__sub_I__Z9print_binPKcymS0_:
.LFB2402:
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
	.quad	_GLOBAL__sub_I__Z9print_binPKcymS0_
	.section .rdata,"dr"
_ZL3n01:
	.ascii ".1"
.lcomm _ZStL8__ioinit,1,1
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
	.def	_ZNSt8ios_base4InitD1Ev;	.scl	2;	.type	32;	.endef
	.def	strlen;	.scl	2;	.type	32;	.endef
	.def	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x;	.scl	2;	.type	32;	.endef
	.def	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate;	.scl	2;	.type	32;	.endef
	.def	_ZNSo9_M_insertImEERSoT_;	.scl	2;	.type	32;	.endef
	.def	_ZNSt8ios_base4InitC1Ev;	.scl	2;	.type	32;	.endef
	.def	atexit;	.scl	2;	.type	32;	.endef
	.section	.rdata$.refptr._ZSt4cout, "dr"
	.globl	.refptr._ZSt4cout
	.linkonce	discard
.refptr._ZSt4cout:
	.quad	_ZSt4cout
