# CUDA 实验构建脚本
#
# 用法：
#   make                    # 编译全部 .cu
#   make dotProduct_timing  # 只编译某一个
#   ./dotProduct_timing     # 运行
#   make clean              # 清理产物
#
# 换显卡架构（如 4090 用 sm_89）：
#   make ARCH=sm_89

NVCC ?= nvcc
ARCH ?= sm_120

SRCS := $(wildcard *.cu)
BINS := $(SRCS:.cu=)

# Windows 原生编译需要 /utf-8，否则源码里的中文注释会让 MSVC 按 GBK 解析报错
ifeq ($(OS),Windows_NT)
  EXTRA := -Xcompiler "/utf-8"
endif

.PHONY: all clean list
all: $(BINS)

%: %.cu
	$(NVCC) -O3 -arch=$(ARCH) $(EXTRA) $< -o $@

list:
	@echo "目标: $(BINS)"

clean:
	rm -f $(BINS) *.exe *.o
