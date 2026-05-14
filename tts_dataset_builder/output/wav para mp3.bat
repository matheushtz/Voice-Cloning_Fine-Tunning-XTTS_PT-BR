@echo off
setlocal enabledelayedexpansion

REM Verifica se ffmpeg está instalado
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: ffmpeg nao encontrado no PATH.
    pause
    exit /b
)

REM Converte output.wav para output.mp3 mantendo o mesmo nome
set "input=Boa noite Fel.wav"
set "output=Boa noite Fel.mp3"

if not exist "%input%" (
    echo Arquivo %input% nao encontrado.
    pause
    exit /b
)

ffmpeg -y -i "%input%" -codec:a libmp3lame -b:a 192k "%output%"

echo Conversao concluida: %output%
pause