	.file	"bitcount-v.cc"
	.text
	.p2align 4
	.globl	_Z11bit_count_vPKmm
	.def	_Z11bit_count_vPKmm;	.scl	2;	.type	32;	.endef
	.seh_proc	_Z11bit_count_vPKmm
_Z11bit_count_vPKmm:
.LFB24:
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
	.seh_endprologue
	xorl	%r8d, %r8d
	movl	%edx, %edx
	leaq	4(%rcx,%rdx,4), %r9
	leaq	60(%rcx), %rdx
	cmpq	%rdx, %r9
	jbe	.L2
	.p2align 4,,10
	.p2align 3
.L3:
	movl	-60(%rdx), %r11d
	movl	-56(%rdx), %esi
	movq	%rdx, %rcx
	movl	-52(%rdx), %r10d
	movl	%r11d, %ebx
	andl	%esi, %r11d
	xorl	%esi, %ebx
	movl	%ebx, %eax
	xorl	%r10d, %eax
	andl	%ebx, %r10d
	movl	-44(%rdx), %ebx
	xorl	%r11d, %r10d
	movl	-48(%rdx), %r11d
	movl	%eax, %edi
	movl	%r10d, %esi
	andl	%r11d, %edi
	xorl	%r11d, %eax
	xorl	%edi, %esi
	movl	%eax, %r11d
	xorl	%ebx, %eax
	andl	%edi, %r10d
	andl	%ebx, %r11d
	movl	%esi, %ebx
	xorl	%r11d, %ebx
	andl	%r11d, %esi
	movl	-40(%rdx), %r11d
	xorl	%esi, %r10d
	movl	%eax, %esi
	andl	%r11d, %esi
	xorl	%r11d, %eax
	movl	%ebx, %r11d
	xorl	%esi, %r11d
	andl	%esi, %ebx
	movl	-36(%rdx), %esi
	xorl	%ebx, %r10d
	movl	%eax, %ebx
	movl	%r11d, %edi
	andl	%esi, %ebx
	xorl	%esi, %eax
	movl	-28(%rdx), %esi
	xorl	%ebx, %edi
	andl	%ebx, %r11d
	movl	-32(%rdx), %ebx
	xorl	%r10d, %r11d
	movl	%eax, %r10d
	andl	%ebx, %r10d
	xorl	%ebx, %eax
	movl	%edi, %ebx
	movl	%r11d, %r12d
	andl	%r10d, %ebx
	xorl	%edi, %r10d
	movl	%eax, %edi
	xorl	%esi, %eax
	xorl	%ebx, %r12d
	andl	%esi, %edi
	movl	%r10d, %esi
	andl	%edi, %esi
	movl	%r12d, %ebp
	xorl	%edi, %r10d
	andl	%ebx, %r11d
	movl	-24(%rdx), %ebx
	xorl	%esi, %ebp
	andl	%esi, %r12d
	movl	%eax, %esi
	xorl	%r12d, %r11d
	movl	-16(%rdx), %edi
	andl	%ebx, %esi
	xorl	%ebx, %eax
	movl	%r10d, %ebx
	andl	%esi, %ebx
	xorl	%esi, %r10d
	movl	%ebp, %esi
	xorl	%ebx, %esi
	andl	%ebx, %ebp
	movl	-20(%rdx), %ebx
	xorl	%r11d, %ebp
	movl	%eax, %r11d
	andl	%ebx, %r11d
	xorl	%ebx, %eax
	movl	%r10d, %ebx
	andl	%r11d, %ebx
	xorl	%r11d, %r10d
	movl	%esi, %r11d
	xorl	%ebx, %r11d
	andl	%ebx, %esi
	movl	%eax, %ebx
	xorl	%edi, %eax
	andl	%edi, %ebx
	movl	%r10d, %edi
	xorl	%ebp, %esi
	andl	%ebx, %edi
	xorl	%ebx, %r10d
	movl	%r11d, %ebx
	andl	%edi, %r11d
	xorl	%edi, %ebx
	movl	%eax, %edi
	xorl	%esi, %r11d
	movl	-12(%rdx), %esi
	andl	%esi, %edi
	xorl	%esi, %eax
	movl	%r10d, %esi
	andl	%edi, %esi
	xorl	%edi, %r10d
	movl	%ebx, %edi
	andl	%esi, %ebx
	xorl	%esi, %edi
	movl	%eax, %esi
	xorl	%r11d, %ebx
	movl	-8(%rdx), %r11d
	movl	%edi, %ebp
	andl	%r11d, %esi
	xorl	%r11d, %eax
	movl	%r10d, %r11d
	andl	%esi, %r11d
	xorl	%esi, %r10d
	movl	-4(%rdx), %esi
	xorl	%r11d, %ebp
	andl	%r11d, %edi
	movl	%eax, %r11d
	andl	%esi, %r11d
	xorl	%ebx, %edi
	movl	%r10d, %ebx
	xorl	%eax, %esi
	andl	%r11d, %ebx
	xorl	%r10d, %r11d
	movl	%ebp, %r10d
	movl	%ebp, %eax
	andl	%ebx, %r10d
	xorl	%ebx, %eax
	movl	%r10d, %ebx
	movl	%esi, %r10d
	shrl	%r10d
	xorl	%edi, %ebx
	andl	$1431655765, %r10d
	subl	%r10d, %esi
	movl	%esi, %r10d
	andl	$858993459, %esi
	shrl	$2, %r10d
	andl	$858993459, %r10d
	addl	%r10d, %esi
	movl	%esi, %r10d
	shrl	$4, %r10d
	addl	%r10d, %esi
	movl	%r11d, %r10d
	shrl	%r10d
	andl	$252645135, %esi
	andl	$1431655765, %r10d
	subl	%r10d, %r11d
	movl	%r11d, %r10d
	shrl	$2, %r11d
	andl	$858993459, %r11d
	andl	$858993459, %r10d
	addl	%r11d, %r10d
	movl	%r10d, %r11d
	shrl	$4, %r11d
	imull	$16843009, %esi, %esi
	addl	%r10d, %r11d
	movl	%eax, %r10d
	shrl	%r10d
	andl	$252645135, %r11d
	andl	$1431655765, %r10d
	imull	$16843009, %r11d, %r11d
	shrl	$24, %esi
	subl	%r10d, %eax
	movl	%eax, %r10d
	andl	$858993459, %eax
	shrl	$2, %r10d
	shrl	$24, %r11d
	andl	$858993459, %r10d
	leal	(%rsi,%r11,2), %r11d
	addl	%r10d, %eax
	movl	%eax, %r10d
	shrl	$4, %r10d
	addl	%eax, %r10d
	movl	%ebx, %eax
	andl	$252645135, %r10d
	shrl	%eax
	imull	$16843009, %r10d, %r10d
	andl	$1431655765, %eax
	subl	%eax, %ebx
	movl	%ebx, %eax
	shrl	$24, %r10d
	shrl	$2, %eax
	leal	(%r11,%r10,4), %r11d
	movl	%ebx, %r10d
	andl	$858993459, %eax
	andl	$858993459, %r10d
	addl	%eax, %r10d
	movl	%r10d, %eax
	shrl	$4, %eax
	addl	%r10d, %eax
	andl	$252645135, %eax
	imull	$16843009, %eax, %eax
	shrl	$24, %eax
	leal	(%r11,%rax,8), %eax
	addl	%eax, %r8d
	addq	$60, %rdx
	cmpq	%rdx, %r9
	ja	.L3
.L2:
	subq	%rcx, %r9
	sarq	$2, %r9
	cmpl	$1, %r9d
	je	.L1
	leal	-2(%r9), %eax
	leaq	4(%rcx,%rax,4), %r9
	.p2align 4,,10
	.p2align 3
.L5:
	movl	(%rcx), %edx
	addq	$4, %rcx
	movl	%edx, %eax
	shrl	%eax
	andl	$1431655765, %eax
	subl	%eax, %edx
	movl	%edx, %eax
	andl	$858993459, %edx
	shrl	$2, %eax
	andl	$858993459, %eax
	addl	%eax, %edx
	movl	%edx, %eax
	shrl	$4, %eax
	addl	%edx, %eax
	andl	$252645135, %eax
	imull	$16843009, %eax, %eax
	shrl	$24, %eax
	addl	%eax, %r8d
	cmpq	%rcx, %r9
	jne	.L5
.L1:
	movl	%r8d, %eax
	popq	%rbx
	popq	%rsi
	popq	%rdi
	popq	%rbp
	popq	%r12
	ret
	.seh_endproc
	.ident	"GCC: (Rev2, Built by MSYS2 project) 9.3.0"
