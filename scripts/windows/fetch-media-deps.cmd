@echo off
rem Fetch and build the media dependencies for a NINFER_STUB_MEDIA=OFF build:
rem   _toolchain\media\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\   headers + import libraries
rem   _toolchain\media\curl\                                      shared libcurl (Schannel)
rem Both are development trees, not just runtimes: the linker and the compiler need the .lib/.h
rem files that the plain "ffmpeg.exe" and "curl.exe" downloads do not contain.
rem
rem Override the download locations when the upstream hosts are unreachable:
rem   set NINFER_FFMPEG_URL=https://.../ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip
rem   set NINFER_CURL_URL=https://.../curl-8.11.1.tar.gz
setlocal
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "MEDIA=%ROOT%\_toolchain\media"
set "CMAKE=%ROOT%\.tools\venv\Scripts\cmake.exe"
set "NINJA=%ROOT%\.tools\venv\Scripts\ninja.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"
if not exist "%NINJA%" set "NINJA=ninja"
if not exist "%MEDIA%" mkdir "%MEDIA%"

rem cl writes its debug record through TMP/TEMP; the directory has to exist before the first
rem compile, otherwise every translation unit fails with D8050.
set "TMP=%MEDIA%\dep-tmp"
set "TEMP=%MEDIA%\dep-tmp"
if not exist "%TMP%" mkdir "%TMP%"

if not defined NINFER_FFMPEG_URL set "NINFER_FFMPEG_URL=https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-win64-lgpl-shared-8.1.zip"
if not defined NINFER_CURL_URL set "NINFER_CURL_URL=https://curl.se/download/curl-8.11.1.tar.gz"

if not exist "%MEDIA%\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1\include\libavcodec\avcodec.h" (
  echo Downloading FFmpeg development build...
  curl.exe -fL --retry 3 -o "%MEDIA%\ffmpeg-dev.zip" "%NINFER_FFMPEG_URL%" || exit /b 1
  tar.exe -xf "%MEDIA%\ffmpeg-dev.zip" -C "%MEDIA%" || exit /b 1
)

if not exist "%MEDIA%\curl-8.11.1\CMakeLists.txt" (
  echo Downloading libcurl source...
  curl.exe -fL --retry 3 -o "%MEDIA%\curl-src.tar.gz" "%NINFER_CURL_URL%" || exit /b 1
  tar.exe -xzf "%MEDIA%\curl-src.tar.gz" -C "%MEDIA%" || exit /b 1
)

if exist "%MEDIA%\curl\include\curl\curl.h" (
  echo libcurl already installed at "%MEDIA%\curl".
  exit /b 0
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%I"
if not defined VS (
  echo Could not locate Visual Studio 2022 with the C++ toolset.
  exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

echo Building libcurl...
"%CMAKE%" -S "%MEDIA%\curl-8.11.1" -B "%MEDIA%\curl-build" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_INSTALL_PREFIX="%MEDIA%\curl" ^
  -DBUILD_SHARED_LIBS=ON ^
  -DBUILD_STATIC_LIBS=OFF ^
  -DBUILD_CURL_EXE=OFF ^
  -DCURL_USE_SCHANNEL=ON ^
  -DCURL_USE_OPENSSL=OFF ^
  -DCURL_USE_LIBPSL=OFF ^
  -DCURL_USE_LIBSSH2=OFF ^
  -DUSE_NGHTTP2=OFF ^
  -DUSE_LIBIDN2=OFF ^
  -DCURL_ZLIB=OFF ^
  -DCURL_BROTLI=OFF ^
  -DCURL_ZSTD=OFF ^
  -DUSE_LIBRTMP=OFF ^
  -DCURL_DISABLE_LDAP=ON || exit /b 1
"%CMAKE%" --build "%MEDIA%\curl-build" --target install --parallel 8 || exit /b 1
exit /b 0
