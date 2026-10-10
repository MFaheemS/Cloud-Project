@echo off
REM Bench360 engine launcher for Ollama (llama.cpp / GGUF backend).
REM Added for the 4 GB consumer-GPU reproduction: Bench360 ships launchers for
REM vLLM / TGI / SGLang / LMDeploy only, all of which preallocate far more VRAM
REM than a 4 GB laptop GPU has. Ollama exposes the same OpenAI-compatible
REM /v1 API on the port Bench360 polls, so the harness itself is unchanged.
REM
REM Usage (called by InferenceEngineClient.launch):
REM   launch_engine_ollama.bat --engine=ollama --model=<tag> --name=<container>
REM Runs in the foreground so the parent Popen stays alive, as Bench360 expects.

setlocal enabledelayedexpansion

set "ALL=%*"
set "MODEL="

REM cmd splits batch parameters on '=', so parse the raw command line instead
REM of using %1/shift: take everything after "--model=" up to the next space.
REM Presence check by substitution, so no external find.exe is needed
REM (a Unix `find` earlier on PATH would otherwise be picked up).
if "!ALL!"=="!ALL:--model=!" (
  echo Error: --model is required 1>&2
  exit /b 1
)
REM %VAR:*search=repl% splits on the FIRST '=', so search is "--model" and the
REM remainder still starts with the '=' from "--model=". Strip it.
set "MODEL=!ALL:*--model=!"
if "!MODEL:~0,1!"=="=" set "MODEL=!MODEL:~1!"
for /f "tokens=1 delims= " %%M in ("!MODEL!") do set "MODEL=%%M"

if "!MODEL!"=="" (
  echo Error: could not parse --model 1>&2
  exit /b 1
)

set "OLLAMA_HOST=127.0.0.1:23333"
set "OLLAMA_MODELS=D:\ollama\Models"

echo [launch_engine_ollama] model=!MODEL! host=%OLLAMA_HOST%

REM Make sure the weights are present before serving; a cached pull is a no-op.
"D:\Ollama\ollama.exe" pull !MODEL!
if errorlevel 1 (
  echo Error: failed to pull !MODEL! 1>&2
  exit /b 1
)

REM Foreground serve - Bench360 polls /v1/models until the tag appears.
"D:\Ollama\ollama.exe" serve
exit /b %errorlevel%
