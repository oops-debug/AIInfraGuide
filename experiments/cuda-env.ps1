# 铺好 MSVC + CUDA 编译环境（当前会话生效）
# 用法: . .\cuda-env.ps1   注意前面必须有点+空格（dot sourcing），否则环境变量不会留在当前会话
$V = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
$K = "C:\Program Files (x86)\Windows Kits\10"
$S = "10.0.26100.0"

$env:PATH += ";$V\bin\Hostx64\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
$env:INCLUDE = "$V\include;$K\Include\$S\ucrt;$K\Include\$S\um;$K\Include\$S\shared;$K\Include\$S\winrt;$K\Include\$S\cppwinrt"
$env:LIB = "$V\lib\x64;$K\Lib\$S\ucrt\x64;$K\Lib\$S\um\x64"

Write-Host "CUDA env ready: nvcc $(nvcc --version | Select-Object -Last 1 | ForEach-Object { $_.Trim() })"
