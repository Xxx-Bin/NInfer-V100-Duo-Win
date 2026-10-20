# Locate an FFmpeg development installation (headers plus import/static libraries).
#
# Upstream consumes plain variables (FFMPEG_INCLUDE_DIRS, FFMPEG_LIBRARY_DIRS,
# FFMPEG_LIBRARIES) instead of imported targets, so this module only has to resolve the four
# libraries the media decoder uses. vcpkg-style prefixes, a portable toolkit and a system
# installation are all handled through FFMPEG_ROOT.

set(_ffmpeg_hints "")
if(FFMPEG_ROOT)
  list(APPEND _ffmpeg_hints "${FFMPEG_ROOT}")
endif()
if(DEFINED ENV{FFMPEG_ROOT})
  list(APPEND _ffmpeg_hints "$ENV{FFMPEG_ROOT}")
endif()
list(APPEND _ffmpeg_hints "${CMAKE_PREFIX_PATH}")

find_path(FFMPEG_INCLUDE_DIR
  NAMES libavcodec/avcodec.h
  HINTS ${_ffmpeg_hints}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
if(NOT FFMPEG_INCLUDE_DIR)
  find_path(FFMPEG_INCLUDE_DIR NAMES libavcodec/avcodec.h PATH_SUFFIXES include)
endif()

foreach(_component avformat avcodec avutil swscale)
  string(TOUPPER "${_component}" _upper)
  find_library(FFMPEG_${_upper}_LIBRARY
    NAMES ${_component}
    HINTS ${_ffmpeg_hints}
    PATH_SUFFIXES lib lib/x64
    NO_DEFAULT_PATH)
  if(NOT FFMPEG_${_upper}_LIBRARY)
    find_library(FFMPEG_${_upper}_LIBRARY NAMES ${_component})
  endif()
  if(FFMPEG_${_upper}_LIBRARY)
    get_filename_component(_dir "${FFMPEG_${_upper}_LIBRARY}" DIRECTORY)
    list(APPEND FFMPEG_LIBRARY_DIRS "${_dir}")
  endif()
  list(APPEND FFMPEG_LIBRARIES "${FFMPEG_${_upper}_LIBRARY}")
endforeach()
list(REMOVE_DUPLICATES FFMPEG_LIBRARY_DIRS)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFMPEG
  REQUIRED_VARS
    FFMPEG_INCLUDE_DIR
    FFMPEG_AVFORMAT_LIBRARY
    FFMPEG_AVCODEC_LIBRARY
    FFMPEG_AVUTIL_LIBRARY
    FFMPEG_SWSCALE_LIBRARY)

if(FFMPEG_FOUND)
  set(FFMPEG_INCLUDE_DIRS "${FFMPEG_INCLUDE_DIR}")
endif()

mark_as_advanced(
  FFMPEG_INCLUDE_DIR
  FFMPEG_AVFORMAT_LIBRARY
  FFMPEG_AVCODEC_LIBRARY
  FFMPEG_AVUTIL_LIBRARY
  FFMPEG_SWSCALE_LIBRARY)
