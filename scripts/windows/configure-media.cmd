@echo off
rem Configure the multimodal (media enabled) Windows build: MSVC 14.38, portable CUDA 12.8, sm_70,
rem FFmpeg decode plus libcurl URL acquisition. Output goes to build-win-media so the stubbed
rem build in build-win stays intact.
rem Usage: scripts\windows\configure-media.cmd [extra cmake args]
setlocal
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "CUDA=%ROOT%\_toolchain\cuda-12.8"
set "CMAKE=%ROOT%\.tools\venv\Scripts\cmake.exe"
set "NINJA=%ROOT%\.tools\venv\Scripts\ninja.exe"
set "MEDIA=%ROOT%\_toolchain\media"
set "FFMPEG=%MEDIA%\ffmpeg-n8.1-latest-win64-lgpl-shared-8.1"
set "VCVARS="

if not exist "%CMAKE%" set "CMAKE=cmake"
if not exist "%NINJA%" set "NINJA=ninja"
if not exist "%CUDA%\bin\nvcc.exe" (
  echo Missing portable CUDA toolkit at "%CUDA%"
  echo Run: python scripts\windows\fetch_cuda_toolkit.py
  exit /b 1
)
if not exist "%FFMPEG%\include\libavcodec\avcodec.h" (
  echo Missing FFmpeg development tree at "%FFMPEG%"
  echo See docs\windows.md, section "Media build".
  exit /b 1
)
if not exist "%MEDIA%\curl\include\curl\curl.h" (
  echo Missing libcurl development tree at "%MEDIA%\curl"
  echo Run: _toolchain\media\build-curl.cmd
  exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%I"
)
if defined VS set "VCVARS=%VS%\VC\Auxiliary\Build\vcvars64.bat"
if not defined VCVARS set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo Could not find vcvars64.bat; install Visual Studio 2022 with the C++ workload.
  exit /b 1
)

rem cl writes its debug record through TMP/TEMP; the directory has to exist before the first
rem compile or every translation unit fails with D8050.
set "NINFER_BUILD_TMP=%ROOT%\build-win-media-tmp"
if not exist "%NINFER_BUILD_TMP%" mkdir "%NINFER_BUILD_TMP%"
set "TMP=%NINFER_BUILD_TMP%"
set "TEMP=%NINFER_BUILD_TMP%"
set "VSLANG=1033"
set "VSCMD_SKIP_SENDTELEMETRY=1"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

set "CUTLASS_SOURCE="
if exist "%ROOT%\_toolchain\cutlass\CMakeLists.txt" (
  set "CUTLASS_SOURCE=-DFETCHCONTENT_SOURCE_DIR_CUTLASS=%ROOT%\_toolchain\cutlass"
)

"%CMAKE%" -S "%ROOT%" -B "%ROOT%\build-win-media" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=70 ^
  -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA%" ^
  -DNINFER_STUB_MEDIA=OFF ^
  -DFFMPEG_ROOT="%FFMPEG%" ^
  -DCMAKE_PREFIX_PATH="%MEDIA%\curl" ^
  %CUTLASS_SOURCE% ^
  %*
exit /b %errorlevel%
