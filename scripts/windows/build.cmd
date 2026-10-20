@echo off
rem Build the native Windows tree configured by scripts\windows\configure.cmd.
rem Usage: scripts\windows\build.cmd [target ...]
setlocal
set "ROOT=%~dp0..\.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"
set "CMAKE=%ROOT%\.tools\venv\Scripts\cmake.exe"
set "VCVARS="

if not exist "%CMAKE%" set "CMAKE=cmake"

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

rem nvcc creates its temporaries through TMP/TEMP; keep them inside the tree, where the build
rem user's token can always write, and where a failed compile leaves no debris in the user profile.
set "NINFER_BUILD_TMP=%ROOT%\build-win-tmp"
if not exist "%NINFER_BUILD_TMP%" mkdir "%NINFER_BUILD_TMP%"
set "TMP=%NINFER_BUILD_TMP%"
set "TEMP=%NINFER_BUILD_TMP%"
set "VSLANG=1033"
set "VSCMD_SKIP_SENDTELEMETRY=1"

call "%VCVARS%" >nul
if errorlevel 1 exit /b 1

rem Each nvcc process can hold several GB while expanding CUTLASS templates. On a memory-constrained
rem machine the parallel set can push cicc.exe into an out-of-memory crash, which surfaces as a
rem failed CUDA object with a crash dump in build-win-tmp. Lower the cap with, for example,
rem   set NINFER_BUILD_JOBS=4
if not defined NINFER_BUILD_JOBS set "NINFER_BUILD_JOBS=8"

rem The media-enabled tree is configured into build-win-media by configure-media.cmd; point the
rem build at it with NINFER_BUILD_DIR instead of hard-coding a second script.
if not defined NINFER_BUILD_DIR set "NINFER_BUILD_DIR=%ROOT%\build-win"

if "%~1"=="" (
  "%CMAKE%" --build "%NINFER_BUILD_DIR%" -j %NINFER_BUILD_JOBS%
) else (
  "%CMAKE%" --build "%NINFER_BUILD_DIR%" -j %NINFER_BUILD_JOBS% --target %*
)
exit /b %errorlevel%
