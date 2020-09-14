#ifndef _SYS_SYSCTL_H_
#define _SYS_SYSCTL_H_

/*
 * Top-level identifiers
 */
#define CTL_UNSPEC  0        /* unused */
#define CTL_KERN    1        /* "high kernel": proc, limits */
#define CTL_VM      2        /* virtual memory */
#define CTL_FS      3        /* file system, mount type is next */
#define CTL_NET     4        /* network, see socket.h */
#define CTL_DEBUG   5        /* debugging parameters */
#define CTL_HW      6        /* generic cpu/io */
#define CTL_MACHDEP 7        /* machine dependent */
#define CTL_USER    8        /* user-level */
#define CTL_MAXID   9        /* number of valid top-level ids */

#define CTL_NAMES { \
    { 0, 0 }, \
    { "kern", CTLTYPE_NODE }, \
    { "vm", CTLTYPE_NODE }, \
    { "fs", CTLTYPE_NODE }, \
    { "net", CTLTYPE_NODE }, \
    { "debug", CTLTYPE_NODE }, \
    { "hw", CTLTYPE_NODE }, \
    { "machdep", CTLTYPE_NODE }, \
    { "user", CTLTYPE_NODE }, \
}

/*
 * CTL_KERN identifiers
 */
#define    KERN_OSTYPE          1    /* string: system version */
#define    KERN_OSRELEASE          2    /* string: system release */
#define    KERN_OSREV          3    /* int: system revision */
#define    KERN_VERSION          4    /* string: compile time info */
#define    KERN_MAXVNODES          5    /* int: max vnodes */
#define    KERN_MAXPROC          6    /* int: max processes */
#define    KERN_MAXFILES          7    /* int: max open files */
#define    KERN_ARGMAX          8    /* int: max arguments to exec */
#define    KERN_SECURELVL          9    /* int: system security level */
#define    KERN_HOSTNAME        10    /* string: hostname */
#define    KERN_HOSTID        11    /* int: host identifier */
#define    KERN_CLOCKRATE        12    /* struct: struct clockrate */
#define    KERN_VNODE        13    /* struct: vnode structures */
#define    KERN_PROC        14    /* struct: process entries */
#define    KERN_FILE        15    /* struct: file entries */
#define    KERN_PROF        16    /* node: kernel profiling info */
#define    KERN_POSIX1        17    /* int: POSIX.1 version */
#define    KERN_NGROUPS        18    /* int: # of supplemental group ids */
#define    KERN_JOB_CONTROL    19    /* int: is job control available */
#define    KERN_SAVED_IDS        20    /* int: saved set-user/group-ID */
#define    KERN_BOOTTIME        21    /* struct: time kernel was booted */
#define    KERN_MAXID        22    /* number of valid kern ids */

#define CTL_KERN_NAMES { \
    { 0, 0 }, \
    { "ostype", CTLTYPE_STRING }, \
    { "osrelease", CTLTYPE_STRING }, \
    { "osrevision", CTLTYPE_INT }, \
    { "version", CTLTYPE_STRING }, \
    { "maxvnodes", CTLTYPE_INT }, \
    { "maxproc", CTLTYPE_INT }, \
    { "maxfiles", CTLTYPE_INT }, \
    { "argmax", CTLTYPE_INT }, \
    { "securelevel", CTLTYPE_INT }, \
    { "hostname", CTLTYPE_STRING }, \
    { "hostid", CTLTYPE_INT }, \
    { "clockrate", CTLTYPE_STRUCT }, \
    { "vnode", CTLTYPE_STRUCT }, \
    { "proc", CTLTYPE_STRUCT }, \
    { "file", CTLTYPE_STRUCT }, \
    { "profiling", CTLTYPE_NODE }, \
    { "posix1version", CTLTYPE_INT }, \
    { "ngroups", CTLTYPE_INT }, \
    { "job_control", CTLTYPE_INT }, \
    { "saved_ids", CTLTYPE_INT }, \
    { "boottime", CTLTYPE_STRUCT }, \
}

/*
 * CTL_HW identifiers
 */
#define    HW_MACHINE     1        /* string: machine class */
#define    HW_MODEL     2        /* string: specific machine model */
#define    HW_NCPU         3        /* int: number of cpus */
#define    HW_BYTEORDER     4        /* int: machine byte order */
#define    HW_PHYSMEM     5        /* int: total memory */
#define    HW_USERMEM     6        /* int: non-kernel memory */
#define    HW_PAGESIZE     7        /* int: software page size */
#define    HW_DISKNAMES     8        /* strings: disk drive names */
#define    HW_DISKSTATS     9        /* struct: diskstats[] */
#define    HW_MAXID    10        /* number of valid hw ids */

#define CTL_HW_NAMES { \
    { 0, 0 }, \
    { "machine", CTLTYPE_STRING }, \
    { "model", CTLTYPE_STRING }, \
    { "ncpu", CTLTYPE_INT }, \
    { "byteorder", CTLTYPE_INT }, \
    { "physmem", CTLTYPE_INT }, \
    { "usermem", CTLTYPE_INT }, \
    { "pagesize", CTLTYPE_INT }, \
    { "disknames", CTLTYPE_STRUCT }, \
    { "diskstats", CTLTYPE_STRUCT }, \
}


/*
 * CTL_USER definitions
 */
#define    USER_CS_PATH         1    /* string: _CS_PATH */
#define    USER_BC_BASE_MAX     2    /* int: BC_BASE_MAX */
#define    USER_BC_DIM_MAX         3    /* int: BC_DIM_MAX */
#define    USER_BC_SCALE_MAX     4    /* int: BC_SCALE_MAX */
#define    USER_BC_STRING_MAX     5    /* int: BC_STRING_MAX */
#define    USER_COLL_WEIGHTS_MAX     6    /* int: COLL_WEIGHTS_MAX */
#define    USER_EXPR_NEST_MAX     7    /* int: EXPR_NEST_MAX */
#define    USER_LINE_MAX         8    /* int: LINE_MAX */
#define    USER_RE_DUP_MAX         9    /* int: RE_DUP_MAX */
#define    USER_POSIX2_VERSION    10    /* int: POSIX2_VERSION */
#define    USER_POSIX2_C_BIND    11    /* int: POSIX2_C_BIND */
#define    USER_POSIX2_C_DEV    12    /* int: POSIX2_C_DEV */
#define    USER_POSIX2_CHAR_TERM    13    /* int: POSIX2_CHAR_TERM */
#define    USER_POSIX2_FORT_DEV    14    /* int: POSIX2_FORT_DEV */
#define    USER_POSIX2_FORT_RUN    15    /* int: POSIX2_FORT_RUN */
#define    USER_POSIX2_LOCALEDEF    16    /* int: POSIX2_LOCALEDEF */
#define    USER_POSIX2_SW_DEV    17    /* int: POSIX2_SW_DEV */
#define    USER_POSIX2_UPE        18    /* int: POSIX2_UPE */
#define    USER_STREAM_MAX        19    /* int: POSIX2_STREAM_MAX */
#define    USER_TZNAME_MAX        20    /* int: POSIX2_TZNAME_MAX */
#define    USER_MAXID        21    /* number of valid user ids */

#define    CTL_USER_NAMES { \
    { 0, 0 }, \
    { "cs_path", CTLTYPE_STRING }, \
    { "bc_base_max", CTLTYPE_INT }, \
    { "bc_dim_max", CTLTYPE_INT }, \
    { "bc_scale_max", CTLTYPE_INT }, \
    { "bc_string_max", CTLTYPE_INT }, \
    { "coll_weights_max", CTLTYPE_INT }, \
    { "expr_nest_max", CTLTYPE_INT }, \
    { "line_max", CTLTYPE_INT }, \
    { "re_dup_max", CTLTYPE_INT }, \
    { "posix2_version", CTLTYPE_INT }, \
    { "posix2_c_bind", CTLTYPE_INT }, \
    { "posix2_c_dev", CTLTYPE_INT }, \
    { "posix2_char_term", CTLTYPE_INT }, \
    { "posix2_fort_dev", CTLTYPE_INT }, \
    { "posix2_fort_run", CTLTYPE_INT }, \
    { "posix2_localedef", CTLTYPE_INT }, \
    { "posix2_sw_dev", CTLTYPE_INT }, \
    { "posix2_upe", CTLTYPE_INT }, \
    { "stream_max", CTLTYPE_INT }, \
    { "tzname_max", CTLTYPE_INT }, \
}

#ifdef __cplusplus
extern "C" {
#endif

int sysctl(int *name, unsigned int namelen, void *oldp, size_t *oldlenp, void *newp, size_t newlen);

int sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp, size_t newlen);

int sysctlnametomib(const char *name, int *mibp, size_t *sizep);

#ifdef __cplusplus
}
#endif


#endif    /* !_SYS_SYSCTL_H_ */
