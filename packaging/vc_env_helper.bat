@echo on

for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do (
    if exist "%%i\VC\Auxiliary\Build\vcvarsall.bat" (
        set "VS15INSTALLDIR=%%i"
        set "VS15VCVARSALL=%%i\VC\Auxiliary\Build\vcvarsall.bat"
        goto vswhere
    )
)
:vswhere
if not defined VS15VCVARSALL (
   echo ERROR: could not locate Visual Studio via vswhere
   exit /b 1
)
if "%VSDEVCMD_ARGS%" == "" (
    call "%VS15VCVARSALL%" x64 || exit /b 1
) else (
    call "%VS15VCVARSALL%" x64 %VSDEVCMD_ARGS% || exit /b 1
)

@echo on

if "%CU_VERSION%" == "xpu" call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

set DISTUTILS_USE_SDK=1

set args=%1
shift
:start
if [%1] == [] goto done
set args=%args% %1
shift
goto start

:done
if "%args%" == "" (
    echo Usage: vc_env_helper.bat [command] [args]
    echo e.g. vc_env_helper.bat cl /c test.cpp
)

%args% || exit /b 1
