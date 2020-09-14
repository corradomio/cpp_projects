	.file	"bitpol-print.cc"
	.text
	.p2align 4
	.def	__tcf_0;	.scl	3;	.type	32;	.endef
	.seh_proc	__tcf_0
__tcf_0:
.LFB2430:
	.seh_endprologue
	leaq	_ZStL8__ioinit(%rip), %rcx
	jmp	_ZNSt8ios_base4InitD1Ev
	.seh_endproc
	.section .rdata,"dr"
.LC0:
	.ascii "x\0"
.LC1:
	.ascii "1\0"
.LC2:
	.ascii "x^\0"
.LC3:
	.ascii " + \0"
.LC4:
	.ascii "+\0"
	.text
	.p2align 4
	.globl	_Z12bitpol_printPKcmb
	.def	_Z12bitpol_printPKcmb;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z12bitpol_printPKcmb
_Z12bitpol_printPKcmb:
.LFB2086:
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
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	movq	%rcx, %r13
	movl	%edx, %ebx
	movl	%r8d, %r12d
	testq	%rcx, %rcx
	je	.L20
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r13, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L5:
	testl	%ebx, %ebx
	je	.L3
	movq	.refptr._ZSt4cout(%rip), %r15
	movl	$-2147483648, %esi
	movl	$31, %edi
	movq	%r15, %r14
	.p2align 4,,10
	.p2align 3
.L13:
	movl	%ebx, %r13d
	andl	%esi, %r13d
	je	.L7
.L21:
	movl	%r13d, %ebp
	xorl	%ebx, %ebp
	cmpl	$1, %edi
	ja	.L8
	testl	%edi, %edi
	leaq	.LC0(%rip), %rdx
	leaq	.LC1(%rip), %rax
	movq	.refptr._ZSt4cout(%rip), %rcx
	cmove	%rax, %rdx
	movl	$1, %r8d
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L14:
	cmpl	%ebx, %r13d
	je	.L11
	testb	%r12b, %r12b
	je	.L12
	movl	$3, %r8d
	leaq	.LC3(%rip), %rdx
	movq	%r15, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L11:
	shrl	%esi
	subl	$1, %edi
	cmpl	%ebx, %r13d
	je	.L3
.L22:
	movl	%ebp, %ebx
	movl	%ebx, %r13d
	andl	%esi, %r13d
	jne	.L21
.L7:
	shrl	%esi
	subl	$1, %edi
	jmp	.L13
	.p2align 4,,10
	.p2align 3
.L8:
	movq	%r14, %rcx
	movl	$2, %r8d
	leaq	.LC2(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	%edi, %edx
	call	_ZNSo9_M_insertImEERSoT_
	jmp	.L14
	.p2align 4,,10
	.p2align 3
.L12:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	shrl	%esi
	subl	$1, %edi
	leaq	.LC4(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	cmpl	%ebx, %r13d
	jne	.L22
.L3:
	addq	$40, %rsp
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
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L5
	.seh_endproc
	.p2align 4
	.globl	_Z16bitpol_print_binPKcm
	.def	_Z16bitpol_print_binPKcm;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z16bitpol_print_binPKcm
_Z16bitpol_print_binPKcm:
.LFB2087:
	.seh_endprologue
	xorl	%r9d, %r9d
	movl	%edx, %eax
	movl	%edx, %edx
/APP
 # 53 "src/bits/bitasm-amd64.h" 1
	bsrq %eax, %eax
 # 0 "" 2
/NO_APP
	leal	1(%rax), %r8d
	jmp	_Z9print_binPKcymS0_
	.seh_endproc
	.section .rdata,"dr"
.LC5:
	.ascii "[\0"
.LC6:
	.ascii ",\0"
.LC7:
	.ascii "]\0"
	.text
	.p2align 4
	.globl	_Z18bitpol_print_coeffPKcm
	.def	_Z18bitpol_print_coeffPKcm;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z18bitpol_print_coeffPKcm
_Z18bitpol_print_coeffPKcm:
.LFB2088:
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
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	movq	%rcx, %r12
	movl	%edx, %esi
	testq	%rcx, %rcx
	je	.L35
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L26:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC5(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testl	%esi, %esi
	je	.L27
	movq	.refptr._ZSt4cout(%rip), %r13
	movl	$-2147483648, %ebx
	movl	$31, %edi
	leaq	.LC6(%rip), %r15
	movq	%r13, %r14
.L30:
	movl	%esi, %ebp
	andl	%ebx, %ebp
	jne	.L36
.L28:
	shrl	%ebx
	movl	%esi, %ebp
	subl	$1, %edi
	andl	%ebx, %ebp
	je	.L28
.L36:
	movl	%ebp, %r12d
	movl	%edi, %edx
	movq	%r13, %rcx
	xorl	%esi, %r12d
	call	_ZNSo9_M_insertImEERSoT_
	cmpl	%esi, %ebp
	je	.L29
	movl	$1, %r8d
	movq	%r15, %rdx
	movq	%r14, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L29:
	shrl	%ebx
	subl	$1, %edi
	cmpl	%esi, %ebp
	je	.L27
	movl	%r12d, %esi
	jmp	.L30
	.p2align 4,,10
	.p2align 3
.L27:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC7(%rip), %rdx
	addq	$40, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	jmp	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	.p2align 4,,10
	.p2align 3
.L35:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L26
	.seh_endproc
	.section .rdata,"dr"
.LC8:
	.ascii "{\0"
.LC9:
	.ascii "}\0"
	.text
	.p2align 4
	.globl	_Z16bitpol_print_texPKcm
	.def	_Z16bitpol_print_texPKcm;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z16bitpol_print_texPKcm
_Z16bitpol_print_texPKcm:
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
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	movq	%rcx, %r12
	movl	%edx, %ebx
	testq	%rcx, %rcx
	je	.L53
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L39:
	testl	%ebx, %ebx
	je	.L37
	movq	.refptr._ZSt4cout(%rip), %r13
	movl	$-2147483648, %esi
	movl	$31, %edi
	movq	%r13, %r15
	movq	%r13, %r14
	.p2align 4,,10
	.p2align 3
.L46:
	movl	%ebx, %ebp
	andl	%esi, %ebp
	je	.L41
.L54:
	movl	%ebp, %r12d
	xorl	%ebx, %r12d
	cmpl	$1, %edi
	ja	.L42
	testl	%edi, %edi
	leaq	.LC0(%rip), %rdx
	leaq	.LC1(%rip), %rax
	movq	.refptr._ZSt4cout(%rip), %rcx
	cmove	%rax, %rdx
	movl	$1, %r8d
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L47:
	cmpl	%ebx, %ebp
	je	.L45
	movl	$3, %r8d
	leaq	.LC3(%rip), %rdx
	movq	%r13, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L45:
	shrl	%esi
	subl	$1, %edi
	cmpl	%ebx, %ebp
	je	.L37
	movl	%r12d, %ebx
	movl	%ebx, %ebp
	andl	%esi, %ebp
	jne	.L54
.L41:
	shrl	%esi
	subl	$1, %edi
	jmp	.L46
	.p2align 4,,10
	.p2align 3
.L42:
	movq	%r15, %rcx
	movl	$2, %r8d
	leaq	.LC2(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	$1, %r8d
	movq	%r14, %rcx
	leaq	.LC8(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	%edi, %edx
	call	_ZNSo9_M_insertImEERSoT_
	movl	$1, %r8d
	leaq	.LC9(%rip), %rdx
	movq	%rax, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	jmp	.L47
	.p2align 4,,10
	.p2align 3
.L37:
	addq	$40, %rsp
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
.L53:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L39
	.seh_endproc
	.section .rdata,"dr"
.LC10:
	.ascii " (\0"
.LC11:
	.ascii "\0"
.LC12:
	.ascii ")\0"
.LC13:
	.ascii "^\0"
.LC14:
	.ascii " * \0"
	.text
	.p2align 4
	.globl	_Z26bitpol_print_factorizationPKcPKmS2_m
	.def	_Z26bitpol_print_factorizationPKcPKmS2_m;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z26bitpol_print_factorizationPKcPKmS2_m
_Z26bitpol_print_factorizationPKcPKmS2_m:
.LFB2090:
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
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	movq	%rcx, %r12
	movq	%rdx, %rdi
	movq	%r8, %rsi
	movl	%r9d, %ebp
	testq	%rcx, %rcx
	je	.L64
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L57:
	leal	-1(%rbp), %r13d
	xorl	%ebx, %ebx
	movq	%r13, %r12
	testl	%ebp, %ebp
	je	.L55
	movq	.refptr._ZSt4cout(%rip), %rbp
	movq	%rbp, %r14
	movq	%rbp, %r15
	jmp	.L62
	.p2align 4,,10
	.p2align 3
.L60:
	leaq	1(%rbx), %rax
	cmpq	%r13, %rbx
	je	.L55
.L65:
	movq	%rax, %rbx
.L62:
	movl	$2, %r8d
	leaq	.LC10(%rip), %rdx
	movq	%rbp, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rdi,%rbx,4), %edx
	xorl	%r8d, %r8d
	leaq	.LC11(%rip), %rcx
	call	_Z12bitpol_printPKcmb
	movl	$1, %r8d
	leaq	.LC12(%rip), %rdx
	movq	%r14, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	cmpl	$1, (%rsi,%rbx,4)
	jbe	.L59
	leaq	.LC13(%rip), %rdx
	movq	%r15, %rcx
	movl	$1, %r8d
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rsi,%rbx,4), %edx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo9_M_insertImEERSoT_
.L59:
	cmpl	%ebx, %r12d
	jbe	.L60
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$3, %r8d
	leaq	.LC14(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	leaq	1(%rbx), %rax
	cmpq	%r13, %rbx
	jne	.L65
.L55:
	addq	$40, %rsp
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
.L64:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L57
	.seh_endproc
	.p2align 4
	.globl	_Z30bitpol_print_bin_factorizationPKcPKmS2_m
	.def	_Z30bitpol_print_bin_factorizationPKcPKmS2_m;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z30bitpol_print_bin_factorizationPKcPKmS2_m
_Z30bitpol_print_bin_factorizationPKcPKmS2_m:
.LFB2091:
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
	subq	$32, %rsp
	.seh_stackalloc	32
	.seh_endprologue
	movq	%rcx, %r12
	movq	%rdx, %rdi
	movq	%r8, %rsi
	movl	%r9d, %ebx
	testq	%rcx, %rcx
	je	.L76
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L68:
	testl	%ebx, %ebx
	je	.L66
	movq	.refptr._ZSt4cout(%rip), %rbp
	leal	-1(%rbx), %r13d
	xorl	%ebx, %ebx
	movq	%rbp, %r12
	movq	%rbp, %r14
	jmp	.L71
	.p2align 4,,10
	.p2align 3
.L72:
	movq	%rax, %rbx
.L71:
	movl	$2, %r8d
	leaq	.LC10(%rip), %rdx
	movq	%rbp, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rdi,%rbx,4), %edx
	xorl	%r9d, %r9d
	leaq	.LC11(%rip), %rcx
	movl	%edx, %eax
/APP
 # 53 "src/bits/bitasm-amd64.h" 1
	bsrq %eax, %eax
 # 0 "" 2
/NO_APP
	leal	1(%rax), %r8d
	call	_Z9print_binPKcymS0_
	movl	$1, %r8d
	leaq	.LC12(%rip), %rdx
	movq	%r12, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	cmpl	$1, (%rsi,%rbx,4)
	jbe	.L70
	leaq	.LC13(%rip), %rdx
	movq	%r14, %rcx
	movl	$1, %r8d
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rsi,%rbx,4), %edx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo9_M_insertImEERSoT_
.L70:
	leaq	1(%rbx), %rax
	cmpq	%rbx, %r13
	jne	.L72
.L66:
	addq	$32, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
	.p2align 4,,10
	.p2align 3
.L76:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L68
	.seh_endproc
	.p2align 4
	.globl	_Z32bitpol_print_coeff_factorizationPKcPKmS2_m
	.def	_Z32bitpol_print_coeff_factorizationPKcPKmS2_m;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z32bitpol_print_coeff_factorizationPKcPKmS2_m
_Z32bitpol_print_coeff_factorizationPKcPKmS2_m:
.LFB2092:
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
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	movq	%rcx, %r12
	movq	%rdx, %rdi
	movq	%r8, %rsi
	movl	%r9d, %r13d
	testq	%rcx, %rcx
	je	.L86
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L79:
	leal	-1(%r13), %r12d
	xorl	%ebx, %ebx
	movq	%r12, %rbp
	testl	%r13d, %r13d
	je	.L77
	movq	.refptr._ZSt4cout(%rip), %r13
	movq	%r13, %r14
	movq	%r13, %r15
	jmp	.L84
	.p2align 4,,10
	.p2align 3
.L82:
	leaq	1(%rbx), %rax
	cmpq	%rbx, %r12
	je	.L77
.L87:
	movq	%rax, %rbx
.L84:
	movl	(%rdi,%rbx,4), %edx
	leaq	.LC11(%rip), %rcx
	call	_Z18bitpol_print_coeffPKcm
	cmpl	$1, (%rsi,%rbx,4)
	jbe	.L81
	leaq	.LC13(%rip), %rdx
	movq	%r13, %rcx
	movl	$1, %r8d
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rsi,%rbx,4), %edx
	movq	%r14, %rcx
	call	_ZNSo9_M_insertImEERSoT_
.L81:
	cmpl	%ebx, %ebp
	jbe	.L82
	movl	$3, %r8d
	leaq	.LC14(%rip), %rdx
	movq	%r15, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	leaq	1(%rbx), %rax
	cmpq	%rbx, %r12
	jne	.L87
.L77:
	addq	$40, %rsp
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
.L86:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L79
	.seh_endproc
	.section .rdata,"dr"
.LC15:
	.ascii " \\left(\0"
.LC16:
	.ascii "\\right)\0"
.LC17:
	.ascii "^{\0"
.LC18:
	.ascii " \\cdot \0"
	.text
	.p2align 4
	.globl	_Z30bitpol_print_tex_factorizationPKcPKmS2_m
	.def	_Z30bitpol_print_tex_factorizationPKcPKmS2_m;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z30bitpol_print_tex_factorizationPKcPKmS2_m
_Z30bitpol_print_tex_factorizationPKcPKmS2_m:
.LFB2093:
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
	subq	$40, %rsp
	.seh_stackalloc	40
	.seh_endprologue
	movq	%rcx, %r12
	movq	%rdx, %rdi
	movq	%r8, %rsi
	movl	%r9d, %ebp
	testq	%rcx, %rcx
	je	.L97
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L90:
	leal	-1(%rbp), %r13d
	xorl	%ebx, %ebx
	movq	%r13, %r12
	testl	%ebp, %ebp
	je	.L88
	movq	.refptr._ZSt4cout(%rip), %rbp
	movq	%rbp, %r14
	movq	%rbp, %r15
	jmp	.L95
	.p2align 4,,10
	.p2align 3
.L93:
	leaq	1(%rbx), %rax
	cmpq	%r13, %rbx
	je	.L88
.L98:
	movq	%rax, %rbx
.L95:
	movl	$7, %r8d
	leaq	.LC15(%rip), %rdx
	movq	%rbp, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rdi,%rbx,4), %edx
	leaq	.LC11(%rip), %rcx
	call	_Z16bitpol_print_texPKcm
	movl	$7, %r8d
	leaq	.LC16(%rip), %rdx
	movq	%r14, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	cmpl	$1, (%rsi,%rbx,4)
	jbe	.L92
	movl	$2, %r8d
	leaq	.LC17(%rip), %rdx
	movq	%r15, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rsi,%rbx,4), %edx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo9_M_insertImEERSoT_
	movl	$1, %r8d
	leaq	.LC9(%rip), %rdx
	movq	%rax, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L92:
	cmpl	%ebx, %r12d
	jbe	.L93
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$7, %r8d
	leaq	.LC18(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	leaq	1(%rbx), %rax
	cmpq	%r13, %rbx
	jne	.L98
.L88:
	addq	$40, %rsp
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
.L97:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L90
	.seh_endproc
	.section .rdata,"dr"
.LC19:
	.ascii " [\0"
.LC20:
	.ascii " \0"
	.text
	.p2align 4
	.globl	_Z32bitpol_print_short_factorizationPKcPKmS2_m
	.def	_Z32bitpol_print_short_factorizationPKcPKmS2_m;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z32bitpol_print_short_factorizationPKcPKmS2_m
_Z32bitpol_print_short_factorizationPKcPKmS2_m:
.LFB2094:
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
	subq	$32, %rsp
	.seh_stackalloc	32
	.seh_endprologue
	movq	%rcx, %r12
	movq	%rdx, %rsi
	movq	%r8, %rdi
	movl	%r9d, %ebx
	testq	%rcx, %rcx
	je	.L108
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r12, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L101:
	testl	%ebx, %ebx
	je	.L99
	movq	.refptr._ZSt4cout(%rip), %rbp
	leal	-1(%rbx), %r14d
	xorl	%ebx, %ebx
	movq	%rbp, %r13
	movq	%rbp, %r12
	.p2align 4,,10
	.p2align 3
.L103:
	movl	$2, %r8d
	leaq	.LC19(%rip), %rdx
	movq	%rbp, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	%r13, %rcx
	movl	(%rsi,%rbx,4), %edx
/APP
 # 53 "src/bits/bitasm-amd64.h" 1
	bsrq %edx, %edx
 # 0 "" 2
/NO_APP
	call	_ZNSo9_M_insertImEERSoT_
	movl	$1, %r8d
	leaq	.LC20(%rip), %rdx
	movq	%r12, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	(%rdi,%rbx,4), %edx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo9_M_insertImEERSoT_
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC7(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	%rbx, %rax
	addq	$1, %rbx
	cmpq	%rax, %r14
	jne	.L103
.L99:
	addq	$32, %rsp
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	popq	%r13
	popq	%r14
	ret
	.p2align 4,,10
	.p2align 3
.L108:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	(%rcx), %rax
	addq	-24(%rax), %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L101
	.seh_endproc
	.section	.text.startup,"x"
	.p2align 4
	.def	_GLOBAL__sub_I__Z12bitpol_printPKcmb;	.scl	3;	.type	32;	.endef
	.seh_proc	_GLOBAL__sub_I__Z12bitpol_printPKcmb
_GLOBAL__sub_I__Z12bitpol_printPKcmb:
.LFB2431:
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
	.quad	_GLOBAL__sub_I__Z12bitpol_printPKcmb
.lcomm _ZStL8__ioinit,1,1
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
	.def	_ZNSt8ios_base4InitD1Ev;	.scl	2;	.type	32;	.endef
	.def	strlen;	.scl	2;	.type	32;	.endef
	.def	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x;	.scl	2;	.type	32;	.endef
	.def	_ZNSo9_M_insertImEERSoT_;	.scl	2;	.type	32;	.endef
	.def	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate;	.scl	2;	.type	32;	.endef
	.def	_Z9print_binPKcymS0_;	.scl	2;	.type	32;	.endef
	.def	_ZNSt8ios_base4InitC1Ev;	.scl	2;	.type	32;	.endef
	.def	atexit;	.scl	2;	.type	32;	.endef
	.section	.rdata$.refptr._ZSt4cout, "dr"
	.globl	.refptr._ZSt4cout
	.linkonce	discard
.refptr._ZSt4cout:
	.quad	_ZSt4cout
