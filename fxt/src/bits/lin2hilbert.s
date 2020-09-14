	.file	"lin2hilbert.cc"
	.text
	.p2align 4
	.globl	_Z11lin2hilbertmRmS_
	.def	_Z11lin2hilbertmRmS_;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z11lin2hilbertmRmS_
_Z11lin2hilbertmRmS_:
.LFB17:
	pushq	%rdi
	.seh_pushreg	%rdi
	pushq	%rsi
	.seh_pushreg	%rsi
	pushq	%rbx
	.seh_pushreg	%rbx
	.seh_endprologue
	movl	$16, %ebx
	xorl	%eax, %eax
	xorl	%r10d, %r10d
	xorl	%r11d, %r11d
	leaq	_ZL4htab(%rip), %rsi
	movq	%rdx, %rdi
	.p2align 4,,10
	.p2align 3
.L2:
	movl	%ecx, %edx
	sall	$2, %eax
	addl	%r10d, %r10d
	addl	%r11d, %r11d
	shrl	$30, %edx
	sall	$2, %ecx
	orl	%edx, %eax
	movl	(%rsi,%rax,4), %r9d
	movl	%r9d, %edx
	movl	%r9d, %eax
	shrl	$3, %r9d
	shrl	$2, %edx
	andl	$3, %eax
	orl	%r9d, %r11d
	andl	$1, %edx
	orl	%edx, %r10d
	subl	$1, %ebx
	jne	.L2
	movl	%r11d, (%rdi)
	movl	%r10d, (%r8)
	popq	%rbx
	popq	%rsi
	popq	%rdi
	ret
	.seh_endproc
	.p2align 4
	.globl	_Z11hilbert2linmm
	.def	_Z11hilbert2linmm;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z11hilbert2linmm
_Z11hilbert2linmm:
.LFB18:
	pushq	%rbx
	.seh_pushreg	%rbx
	.seh_endprologue
	movl	$16, %r10d
	xorl	%r8d, %r8d
	xorl	%eax, %eax
	leaq	_ZL5ihtab(%rip), %r11
	.p2align 4,,10
	.p2align 3
.L6:
	sall	$2, %r8d
	movl	%edx, %ebx
	movl	%ecx, %r9d
	sall	$2, %eax
	shrl	$15, %ebx
	shrl	$14, %r9d
	addl	%ecx, %ecx
	addl	%edx, %edx
	andl	$1, %ebx
	andl	$2, %r9d
	orl	%ebx, %r8d
	orl	%r9d, %r8d
	movl	(%r11,%r8,4), %r9d
	movl	%r9d, %r8d
	shrl	$2, %r9d
	andl	$3, %r8d
	orl	%r9d, %eax
	subl	$1, %r10d
	jne	.L6
	popq	%rbx
	ret
	.seh_endproc
	.p2align 4
	.globl	_Z17hilbert_gray_codem
	.def	_Z17hilbert_gray_codem;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z17hilbert_gray_codem
_Z17hilbert_gray_codem:
.LFB19:
	.seh_endprologue
	movl	$16, %r9d
	movl	$2, %edx
	xorl	%eax, %eax
	leaq	_ZL4htab(%rip), %r10
	.p2align 4,,10
	.p2align 3
.L9:
	sall	$2, %edx
	movl	%ecx, %r8d
	sall	$2, %eax
	shrl	$30, %r8d
	sall	$2, %ecx
	orl	%r8d, %edx
	movl	(%r10,%rdx,4), %r8d
	movl	%r8d, %edx
	shrl	$2, %r8d
	andl	$3, %edx
	orl	%r8d, %eax
	subl	$1, %r9d
	jne	.L9
	movl	%eax, %r8d
	shrl	$2, %r8d
	andl	$357913941, %r8d
	xorl	%eax, %r8d
	movl	%r8d, %eax
	shrl	$2, %eax
	andl	$715827882, %eax
	xorl	%r8d, %eax
	ret
	.seh_endproc
	.p2align 4
	.globl	_Z25inverse_hilbert_gray_codem
	.def	_Z25inverse_hilbert_gray_codem;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z25inverse_hilbert_gray_codem
_Z25inverse_hilbert_gray_codem:
.LFB20:
	pushq	%rbx
	.seh_pushreg	%rbx
	.seh_endprologue
	movl	$16, %r11d
	leaq	_ZL5ihtab(%rip), %rbx
	movl	%ecx, %edx
	leal	(%rcx,%rcx), %eax
	andl	$-1717986919, %ecx
	shrl	%edx
	andl	$1145324612, %eax
	andl	$572662306, %edx
	orl	%eax, %edx
	orl	%ecx, %edx
	movl	%edx, %eax
	leal	0(,%rdx,4), %ecx
	andl	$-1010580541, %edx
	shrl	$2, %eax
	andl	$808464432, %ecx
	andl	$202116108, %eax
	orl	%ecx, %eax
	orl	%edx, %eax
	movl	%eax, %edx
	movl	%eax, %ecx
	andl	$-267390961, %eax
	shrl	$4, %edx
	sall	$4, %ecx
	andl	$251662080, %ecx
	andl	$15728880, %edx
	orl	%ecx, %edx
	orl	%eax, %edx
	movl	%edx, %eax
	movl	%edx, %ecx
	andl	$-16776961, %edx
	shrl	$8, %eax
	sall	$8, %ecx
	andl	$65280, %eax
	andl	$16711680, %ecx
	orl	%ecx, %eax
	orl	%edx, %eax
	movzwl	%ax, %edx
	movl	%eax, %r9d
	movl	%edx, %r10d
	shrl	%r10d
	xorl	%edx, %r10d
	movl	%r10d, %edx
	shrl	$2, %edx
	xorl	%edx, %r10d
	movl	%r10d, %edx
	shrl	$4, %edx
	xorl	%edx, %r10d
	movl	%r10d, %edx
	shrl	$8, %edx
	shrl	$17, %r9d
	xorl	%r8d, %r8d
	shrl	$16, %eax
	xorl	%edx, %r10d
	xorl	%r9d, %eax
	movl	%eax, %r9d
	shrl	$2, %r9d
	xorl	%eax, %r9d
	movl	%r9d, %eax
	shrl	$4, %eax
	xorl	%eax, %r9d
	movl	%r9d, %eax
	shrl	$8, %eax
	xorl	%eax, %r9d
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L12:
	sall	$2, %r8d
	movl	%r9d, %edx
	movl	%r10d, %ecx
	sall	$2, %eax
	shrl	$15, %edx
	shrl	$14, %ecx
	addl	%r10d, %r10d
	addl	%r9d, %r9d
	andl	$1, %edx
	andl	$2, %ecx
	orl	%r8d, %edx
	orl	%ecx, %edx
	movl	(%rbx,%rdx,4), %ecx
	movl	%ecx, %r8d
	shrl	$2, %ecx
	andl	$3, %r8d
	orl	%ecx, %eax
	subl	$1, %r11d
	jne	.L12
	popq	%rbx
	ret
	.seh_endproc
	.section .rdata,"dr"
	.align 32
_ZL5ihtab:
	.long	2
	.long	4
	.long	13
	.long	8
	.long	9
	.long	5
	.long	12
	.long	3
	.long	0
	.long	15
	.long	6
	.long	10
	.long	11
	.long	14
	.long	7
	.long	1
	.align 32
_ZL4htab:
	.long	2
	.long	4
	.long	12
	.long	9
	.long	15
	.long	5
	.long	1
	.long	8
	.long	0
	.long	10
	.long	14
	.long	7
	.long	13
	.long	11
	.long	3
	.long	6
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
