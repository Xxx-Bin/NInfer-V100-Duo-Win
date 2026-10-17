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

if "%~1"=="" (
  "%CMAKE%" --build "%ROOT%\build-win" -j 8
) else (
  "%CMAKE%" --build "%ROOT%\build-win" -j 8 --target %*
)
exit /b %errorlevel%
