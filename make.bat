@echo off
setlocal
set IMAGE_NAME=neumonia
set IMAGE_TAG=1.0.0
set MODEL_DIR=%cd%\models

if "%1"=="install" goto install
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="test" goto test
if "%1"=="test-all" goto testall
if "%1"=="smoke" goto smoke
if "%1"=="check-warnings" goto checkwarnings
if "%1"=="run" goto run
if "%1"=="docker-build" goto dockerbuild
if "%1"=="docker-run" goto dockerrun
if "%1"=="verify" goto verify
if "%1"=="clean" goto clean
goto usage

:install
uv sync
goto :eof

:lint
uv run ruff check .
goto :eof

:format
uv run ruff format .
goto :eof

:test
uv run pytest -m "not requires_model and not gui" -q
goto :eof

:testall
uv run pytest -q
goto :eof

:smoke
uv run python scripts\smoke_test.py --image tests\data\sample_synthetic.dcm --dry-run
goto :eof

:checkwarnings
uv run python scripts\check_warnings.py
goto :eof

:run
uv run python src\detector_neumonia.py
goto :eof

:dockerbuild
docker build -t %IMAGE_NAME%:%IMAGE_TAG% .
goto :eof

:dockerrun
docker run --rm -v %MODEL_DIR%:/app/models:ro %IMAGE_NAME%:%IMAGE_TAG% python scripts/smoke_test.py --image tests/data/sample_synthetic.dcm
goto :eof

:verify
call :lint
uv run ruff format --check .
call :test
call :checkwarnings
goto :eof

:clean
if exist .venv rmdir /s /q .venv
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .ruff_cache rmdir /s /q .ruff_cache
if exist htmlcov rmdir /s /q htmlcov
if exist .coverage del /f /q .coverage
goto :eof

:usage
echo Uso: make.bat [install^|lint^|format^|test^|test-all^|smoke^|check-warnings^|run^|docker-build^|docker-run^|verify^|clean]

endlocal