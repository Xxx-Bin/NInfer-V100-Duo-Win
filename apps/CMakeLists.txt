add_executable(ninfer
  cli/main.cpp
  cli/options.cpp)
target_include_directories(ninfer PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer PRIVATE
  ninfer_engine
  ninfer_product_load_progress
  ninfer_product_prompt_input)

add_executable(ninfer-serve serve/main.cpp)
target_include_directories(ninfer-serve PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
target_link_libraries(ninfer-serve PRIVATE ninfer_serve ninfer_product_load_progress)

# A media-enabled Windows build links against import libraries, so the FFmpeg and libcurl DLLs have
# to be next to the executables or the process fails to start. Staging them here keeps the build
# tree runnable without touching PATH.
if(WIN32 AND NOT NINFER_STUB_MEDIA AND FFMPEG_INCLUDE_DIR)
  get_filename_component(NINFER_FFMPEG_DLL_DIR "${FFMPEG_INCLUDE_DIR}/../bin" ABSOLUTE)
  # Only the DLLs: the same directory also holds ffmpeg/ffplay/ffprobe, which nothing here uses.
  file(GLOB NINFER_MEDIA_RUNTIME_DLLS "${NINFER_FFMPEG_DLL_DIR}/*.dll")
  foreach(_app ninfer ninfer-serve)
    if(TARGET ${_app})
      add_custom_command(TARGET ${_app} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${NINFER_MEDIA_RUNTIME_DLLS}
                "$<TARGET_FILE:${NINFER_CURL_TARGET}>"
                "$<TARGET_FILE_DIR:${_app}>"
        COMMENT "Staging media runtime DLLs next to ${_app}")
    endif()
  endforeach()
endif()
