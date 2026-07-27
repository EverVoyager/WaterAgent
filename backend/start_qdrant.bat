@echo off
chcp 65001 >nul
title Qdrant Vector Database

REM ============================================================
REM Qdrant 启动脚本（Windows）
REM ============================================================
REM 前置条件：
REM   1. 访问 https://github.com/qdrant/qdrant/releases
REM   2. 下载最新版 qdrant-x86_64-pc-windows-msvc.zip
REM   3. 解压到 D:\AgentProject\WaterAgents\tools\qdrant\
REM      解压后目录结构：
REM         tools\qdrant\qdrant.exe
REM         tools\qdrant\config.yaml
REM
REM 启动后：
REM   - REST API: http://127.0.0.1:6333
REM   - Web UI:   http://127.0.0.1:6333/dashboard
REM   - 数据目录: tools\qdrant\storage\（自动创建）
REM ============================================================

set QDRANT_DIR=D:\AgentProject\WaterAgents\tools\qdrant
set QDRANT_EXE=%QDRANT_DIR%\qdrant.exe

if not exist "%QDRANT_EXE%" (
    echo [错误] 未找到 qdrant.exe: %QDRANT_EXE%
    echo.
    echo 请先下载 Qdrant:
    echo   1. 访问 https://github.com/qdrant/qdrant/releases
    echo   2. 下载 qdrant-x86_64-pc-windows-msvc.zip
    echo   3. 解压到 %QDRANT_DIR%\
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo 启动 Qdrant 向量数据库
echo ============================================================
echo 可执行文件: %QDRANT_EXE%
echo REST API:   http://127.0.0.1:6333
echo Web UI:     http://127.0.0.1:6333/dashboard
echo 数据目录:   %QDRANT_DIR%\storage\
echo ============================================================
echo.
echo 按 Ctrl+C 停止服务
echo.

cd /d "%QDRANT_DIR%"
"%QDRANT_EXE%"

pause
