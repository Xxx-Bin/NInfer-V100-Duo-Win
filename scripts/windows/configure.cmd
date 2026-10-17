@echo off
rem Configure the native Windows build: MSVC 14.38, portable CUDA 12.8, sm_70, stubbed media.
rem Usage: scripts\windows\configure.cmd [extra cmake args]
setlocal
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "CUDA=%ROOT%\_toolchain\cuda-12.8"
set "CMAKE=%ROOT%\.tools\venv\Scripts\cmake.exe"
set "NINJA=%ROOT%\.tools\venv\Scripts\ninja.exe"
set "VCVARS="

if not exist "%CMAKE%" set "CMAKE=cmake"
if not exist "%NINJA%" set "NINJA=ninja"
if not exist "%CUDA%\bin\nvcc.exe" (
  echo Missing portable CUDA toolkit at "%CUDA%"
  echo Run: python scripts\windows\fetch_cuda_toolkit.py
  exit /b 1
)

rem Locate the Visual Studio installation that carries the C++ toolset instead of hard-coding an
rem edition-specific path.
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

rem nvcc writes intermediate files through TMP/TEMP. Keep them inside the build tree: the
rem per-session temp directory can be read-only for toolchain children, and a failed compile then
rem leaves no debris in the user profile.
set "NINFER_BUILD_TMP=%ROOT%\build-win-tmp"
if not exist "%NINFER_BUILD_TMP%" mkdir "%NINFER_BUILD_TMP%"
set "TMP=%NINFER_BUILD_TMP%"
set "TEMP=%NINFER_BUILD_TMP%"
rem English tool output keeps CMake's MSVC include scanner deterministic; telemetry is off so a
rem configure never waits on a network call.
set "VSLANG=1033"
set "VSCMD_SKIP_SENDTELEMETRY=1"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

rem FetchContent clones CUTLASS and then asks git to update its (empty) submodule list. Git for
rem Windows runs that through MSYS sh, which cannot always create its shared-memory object. A
rem pre-cloned checkout at _toolchain\cutlass lets FetchContent use the source in place; when it is
rem absent the normal network download still applies.
set "CUTLASS_SOURCE="
if exist "%ROOT%\_toolchain\cutlass\CMakeLists.txt" (
  set "CUTLASS_SOURCE=-DFETCHCONTENT_SOURCE_DIR_CUTLASS=%ROOT%\_toolchain\cutlass"
)

"%CMAKE%" -S "%ROOT%" -B "%ROOT%\build-win" -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=70 ^
  -DCMAKE_CUDA_COMPILER="%CUDA%\bin\nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA%" ^
  -DNINFER_STUB_MEDIA=ON ^
  %CUTLASS_SOURCE% ^
  %*
exit /b %errorlevel%
