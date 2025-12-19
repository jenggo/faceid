#ifndef FACEDETECTION_EXPORT_H
#define FACEDETECTION_EXPORT_H

#ifdef FACEDETECTION_STATIC_DEFINE
#  define FACEDETECTION_EXPORT
#else
#  ifndef FACEDETECTION_EXPORT
#    ifdef facedetection_EXPORTS
        /* We are building this library */
#      define FACEDETECTION_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define FACEDETECTION_EXPORT __attribute__((visibility("default")))
#    endif
#  endif
#endif

#endif /* FACEDETECTION_EXPORT_H */
