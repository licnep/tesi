
#ifndef FFLD_EXPORT_H
#define FFLD_EXPORT_H

#ifdef FFLD_STATIC_DEFINE
#  define FFLD_EXPORT
#  define FFLD_NO_EXPORT
#else
#  ifndef FFLD_EXPORT
#    ifdef FFLD_EXPORTS
        /* We are building this library */
#      define FFLD_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define FFLD_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef FFLD_NO_EXPORT
#    define FFLD_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef FFLD_DEPRECATED
#  define FFLD_DEPRECATED __attribute__ ((__deprecated__))
#  define FFLD_DEPRECATED_EXPORT FFLD_EXPORT __attribute__ ((__deprecated__))
#  define FFLD_DEPRECATED_NO_EXPORT FFLD_NO_EXPORT __attribute__ ((__deprecated__))
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define FFLD_NO_DEPRECATED
#endif

#endif
