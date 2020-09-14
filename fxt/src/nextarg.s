	.file	"nextarg.cc"
	.text
	.section	.text$_ZNKSt5ctypeIcE8do_widenEc,"x"
	.linkonce discard
	.align 2
	.p2align 4
	.globl	_ZNKSt5ctypeIcE8do_widenEc
	.def	_ZNKSt5ctypeIcE8do_widenEc;	.scl	2;	.type	32;	.endef
	.seh_proc	_ZNKSt5ctypeIcE8do_widenEc
_ZNKSt5ctypeIcE8do_widenEc:
.LFB1294:
	.seh_endprologue
	movl	%edx, %eax
	ret
	.seh_endproc
	.text
	.p2align 4
	.def	__tcf_0;	.scl	3;	.type	32;	.endef
	.seh_proc	__tcf_0
__tcf_0:
.LFB2407:
	.seh_endprologue
	leaq	_ZStL8__ioinit(%rip), %rcx
	jmp	_ZNSt8ios_base4InitD1Ev
	.seh_endproc
	.section .rdata,"dr"
.LC0:
	.ascii "arg \0"
.LC1:
	.ascii ": \0"
.LC2:
	.ascii " == \0"
.LC3:
	.ascii "  [\0"
.LC4:
	.ascii "]\0"
.LC5:
	.ascii "  default=\0"
	.text
	.p2align 4
	.globl	_Z14next_float_argRdPKcS1_iPPc
	.def	_Z14next_float_argRdPKcS1_iPPc;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z14next_float_argRdPKcS1_iPPc
_Z14next_float_argRdPKcS1_iPPc:
.LFB2064:
	pushq	%r14
	.seh_pushreg	%r14
	pushq	%r13
	.seh_pushreg	%r13
	pushq	%r12
	.seh_pushreg	%r12
	subq	$48, %rsp
	.seh_stackalloc	48
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm	%xmm6, 32
	.seh_endprologue
	movl	nextarg_act(%rip), %eax
	addl	$1, %eax
	movsd	(%rcx), %xmm6
	movl	%eax, nextarg_act(%rip)
	movq	%rdx, %r14
	movq	%rcx, %r12
	movq	112(%rsp), %rdx
	movq	%r8, %r13
	cmpl	%r9d, %eax
	jge	.L5
	cltq
	movq	(%rdx,%rax,8), %rcx
	cmpb	$46, (%rcx)
	je	.L5
	cmpb	$0, 1(%rcx)
	jne	.L14
.L5:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$4, %r8d
	leaq	.LC0(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	nextarg_act(%rip), %edx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSolsEi
	movl	$2, %r8d
	leaq	.LC1(%rip), %rdx
	movq	%rax, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movsd	(%r12), %xmm1
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo9_M_insertIdEERSoT_
	movl	$4, %r8d
	leaq	.LC2(%rip), %rdx
	movq	%rax, %rcx
	movq	%rax, %r12
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testq	%r14, %r14
	je	.L15
	movq	%r14, %rcx
	call	strlen
	movq	%r14, %rdx
	movq	%r12, %rcx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L7:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$3, %r8d
	leaq	.LC3(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testq	%r13, %r13
	je	.L16
	movq	%r13, %rcx
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r13, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %r12
.L9:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC4(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$10, %r8d
	leaq	.LC5(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movapd	%xmm6, %xmm1
	call	_ZNSo9_M_insertIdEERSoT_
	movq	(%r12), %rax
	movq	-24(%rax), %rax
	movq	240(%r12,%rax), %r12
	testq	%r12, %r12
	je	.L17
	cmpb	$0, 56(%r12)
	je	.L11
	movsbl	67(%r12), %edx
.L12:
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo3putEc
	nop
	movaps	32(%rsp), %xmm6
	movq	%rax, %rcx
	addq	$48, %rsp
	popq	%r12
	popq	%r13
	popq	%r14
	jmp	_ZNSo5flushEv
	.p2align 4,,10
	.p2align 3
.L11:
	movq	%r12, %rcx
	call	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movl	$10, %edx
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rcx
	movq	48(%rax), %rax
	cmpq	%rcx, %rax
	je	.L12
	movq	%r12, %rcx
	call	*%rax
	movsbl	%al, %edx
	jmp	.L12
	.p2align 4,,10
	.p2align 3
.L16:
	movq	.refptr._ZSt4cout(%rip), %r12
	movq	(%r12), %rax
	movq	-24(%rax), %rcx
	addq	%r12, %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L9
	.p2align 4,,10
	.p2align 3
.L15:
	movq	(%r12), %rax
	movq	-24(%rax), %rcx
	addq	%r12, %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L7
	.p2align 4,,10
	.p2align 3
.L14:
	xorl	%edx, %edx
	call	__mingw_strtod
	movsd	%xmm0, (%r12)
	jmp	.L5
.L17:
	call	_ZSt16__throw_bad_castv
	nop
	.seh_endproc
	.p2align 4
	.globl	_Z15next_string_argRPcPKcS2_iPS_S_
	.def	_Z15next_string_argRPcPKcS2_iPS_S_;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z15next_string_argRPcPKcS2_iPS_S_
_Z15next_string_argRPcPKcS2_iPS_S_:
.LFB2065:
	pushq	%r15
	.seh_pushreg	%r15
	pushq	%r14
	.seh_pushreg	%r14
	pushq	%r13
	.seh_pushreg	%r13
	pushq	%r12
	.seh_pushreg	%r12
	pushq	%rsi
	.seh_pushreg	%rsi
	pushq	%rbx
	.seh_pushreg	%rbx
	subq	$56, %rsp
	.seh_stackalloc	56
	.seh_endprologue
	movl	nextarg_act(%rip), %eax
	movq	152(%rsp), %r15
	addl	$1, %eax
	movl	%eax, nextarg_act(%rip)
	movq	%rdx, %r14
	movq	%rcx, %rsi
	movq	%r8, %r13
	movq	%r15, %rdx
	cmpl	%r9d, %eax
	jge	.L19
	movq	144(%rsp), %rdx
	cltq
	movq	(%rdx,%rax,8), %rdx
.L19:
	movq	%rdx, (%rsi)
	movl	$4, %r8d
	leaq	47(%rsp), %rbx
	movq	.refptr._ZSt4cout(%rip), %rcx
	leaq	.LC0(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	nextarg_act(%rip), %edx
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSolsEi
	movl	$2, %r8d
	leaq	.LC1(%rip), %rdx
	movq	%rax, %rcx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	movq	%rbx, %rdx
	movb	$34, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	(%rsi), %rsi
	movq	%rax, %r12
	testq	%rsi, %rsi
	je	.L33
	movq	%rsi, %rcx
	call	strlen
	movq	%rsi, %rdx
	movq	%r12, %rcx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L21:
	movq	%r12, %rcx
	movl	$1, %r8d
	movq	%rbx, %rdx
	movb	$34, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movl	$4, %r8d
	leaq	.LC2(%rip), %rdx
	movq	%rax, %rcx
	movq	%rax, %r12
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testq	%r14, %r14
	je	.L34
	movq	%r14, %rcx
	call	strlen
	movq	%r14, %rdx
	movq	%r12, %rcx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L23:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$3, %r8d
	leaq	.LC3(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	testq	%r13, %r13
	je	.L35
	movq	%r13, %rcx
	call	strlen
	movq	.refptr._ZSt4cout(%rip), %rcx
	movq	%r13, %rdx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rsi
.L25:
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	leaq	.LC4(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$10, %r8d
	leaq	.LC5(%rip), %rdx
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	.refptr._ZSt4cout(%rip), %rcx
	movl	$1, %r8d
	movq	%rbx, %rdx
	movb	$34, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	%rax, %r12
	testq	%r15, %r15
	je	.L36
	movq	%r15, %rcx
	call	strlen
	movq	%r15, %rdx
	movq	%r12, %rcx
	movq	%rax, %r8
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
.L27:
	movq	%r12, %rcx
	movl	$1, %r8d
	movq	%rbx, %rdx
	movb	$34, 47(%rsp)
	call	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x
	movq	(%rsi), %rax
	movq	-24(%rax), %rax
	movq	240(%rsi,%rax), %r12
	testq	%r12, %r12
	je	.L37
	cmpb	$0, 56(%r12)
	je	.L29
	movsbl	67(%r12), %edx
.L30:
	movq	.refptr._ZSt4cout(%rip), %rcx
	call	_ZNSo3putEc
	movq	%rax, %rcx
	call	_ZNSo5flushEv
	nop
	addq	$56, %rsp
	popq	%rbx
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
	.p2align 4,,10
	.p2align 3
.L29:
	movq	%r12, %rcx
	call	_ZNKSt5ctypeIcE13_M_widen_initEv
	movq	(%r12), %rax
	movl	$10, %edx
	leaq	_ZNKSt5ctypeIcE8do_widenEc(%rip), %rcx
	movq	48(%rax), %rax
	cmpq	%rcx, %rax
	je	.L30
	movq	%r12, %rcx
	call	*%rax
	movsbl	%al, %edx
	jmp	.L30
	.p2align 4,,10
	.p2align 3
.L36:
	movq	(%rax), %rax
	movq	-24(%rax), %rcx
	addq	%r12, %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L27
	.p2align 4,,10
	.p2align 3
.L35:
	movq	.refptr._ZSt4cout(%rip), %rsi
	movq	(%rsi), %rax
	movq	-24(%rax), %rcx
	addq	%rsi, %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L25
	.p2align 4,,10
	.p2align 3
.L34:
	movq	(%r12), %rax
	movq	-24(%rax), %rcx
	addq	%r12, %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L23
	.p2align 4,,10
	.p2align 3
.L33:
	movq	(%rax), %rax
	movq	-24(%rax), %rcx
	addq	%r12, %rcx
	movl	32(%rcx), %edx
	orl	$1, %edx
	call	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate
	jmp	.L21
.L37:
	call	_ZSt16__throw_bad_castv
	nop
	.seh_endproc
	.section	.text.startup,"x"
	.p2align 4
	.def	_GLOBAL__sub_I_nextarg_act;	.scl	3;	.type	32;	.endef
	.seh_proc	_GLOBAL__sub_I_nextarg_act
_GLOBAL__sub_I_nextarg_act:
.LFB2408:
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
	.quad	_GLOBAL__sub_I_nextarg_act
	.globl	nextarg_act
	.bss
	.align 4
nextarg_act:
	.space 4
.lcomm _ZStL8__ioinit,1,1
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
	.def	_ZNSt8ios_base4InitD1Ev;	.scl	2;	.type	32;	.endef
	.def	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_x;	.scl	2;	.type	32;	.endef
	.def	_ZNSolsEi;	.scl	2;	.type	32;	.endef
	.def	_ZNSo9_M_insertIdEERSoT_;	.scl	2;	.type	32;	.endef
	.def	strlen;	.scl	2;	.type	32;	.endef
	.def	_ZNSo3putEc;	.scl	2;	.type	32;	.endef
	.def	_ZNSo5flushEv;	.scl	2;	.type	32;	.endef
	.def	_ZNKSt5ctypeIcE13_M_widen_initEv;	.scl	2;	.type	32;	.endef
	.def	_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate;	.scl	2;	.type	32;	.endef
	.def	__mingw_strtod;	.scl	2;	.type	32;	.endef
	.def	_ZSt16__throw_bad_castv;	.scl	2;	.type	32;	.endef
	.def	_ZNSt8ios_base4InitC1Ev;	.scl	2;	.type	32;	.endef
	.def	atexit;	.scl	2;	.type	32;	.endef
	.section	.rdata$.refptr._ZSt4cout, "dr"
	.globl	.refptr._ZSt4cout
	.linkonce	discard
.refptr._ZSt4cout:
	.quad	_ZSt4cout
