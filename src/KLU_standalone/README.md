# KLU 独立源码包 —— 稀疏线性方程组求解参考实现

本目录是从 **SuiteSparse 7.14.1** 中抽取出来的 **KLU 2.3.6** 独立源码包，只保留
求解 `Ax = b`（实数、double、int32 索引）所需的全部代码，去掉了 CMake、BLAS、
OpenMP 等依赖，可直接用 `gcc` 编译。用于给选手提供一个高精度的参考解，用来
校验自己的求解器（例如 GLU）的结果。

- KLU 官方主页：https://github.com/DrTimothyAldenDavis/SuiteSparse
- KLU 论文：T. A. Davis, E. Palamadai Natarajan, *Algorithm 907: KLU, A Direct
  Sparse Linear Solver for Circuit Simulation Problems*, ACM TOMS, 2010.

---

## 1. 目录结构

```
KLU_standalone/
├── README.md              本文件
├── Makefile               编译脚本（Linux / gcc）
├── include/               全部头文件
│   ├── klu.h              KLU 用户 API
│   ├── klu_internal.h     KLU 内部结构
│   ├── klu_version.h      版本宏
│   ├── amd.h / amd_internal.h           AMD  fill-reducing 排序
│   ├── colamd.h                         COLAMD 列近似最小度排序
│   ├── btf.h / btf_internal.h           BTF 块三角化
│   └── SuiteSparse_config.h             公共配置 / 内存函数
├── src/                   全部 .c 源码（38 个文件，默认 double/int32 精度）
├── examples/
│   └── klu_demo.c         读 Matrix Market 文件求解 Ax=b 的示例程序
└── LICENSE/               各子包许可证（BSD 3-clause）
```

KLU 内部对矩阵做 **块三角化（BTF）+ 块内 AMD 排序**，再对每个对角块做部分选主元
LU 分解。AMD / COLAMD / BTF / SuiteSparse_config 是它的直接依赖，所以一并打包。

---

## 2. 编译

环境要求：Linux + gcc（无需 CMake、BLAS、Fortran）。

```bash
cd KLU_standalone
make            # 生成 libklu.a 和 klu_demo
make clean      # 清理
```

`Makefile` 默认参数：`gcc -O2 -fPIC -w`。如需自定义：

```bash
make CC=gcc CFLAGS="-O3 -march=native -w"
```

编译产物：

| 文件 | 说明 |
|------|------|
| `libklu.a` | 静态库（KLU + AMD + COLAMD + BTF + SuiteSparse_config） |
| `klu_demo`  | 示例程序 |

在自己的程序里使用时，只需要：

```bash
gcc -O2 -Iinclude your_solver.c libklu.a -lm -o your_solver
```

---

## 3. 快速开始：`klu_demo`

批量计算 `src/matrix` 中矩阵的 CPU 参考解和耗时：

```bash
python3 src/KLU_standalone/test/run_reference.py  # 从仓库根目录运行
```

结果默认写入 `src/KLU_standalone/test_result`，包含按矩阵名保存的解向量、
分阶段耗时、完整运行耗时和汇总表，详见 [test/README.md](test/README.md)。

```bash
./klu_demo <matrix.mtx> <out_x.txt> [-expand]
```

它读入矩阵文件，构造 `b = [1, 1, ..., 1]^T`，求解 `Ax = b`，把解向量 `x`
逐行写入 `<out_x.txt>`，并把残差打印到 stderr：

```bash
$ ./klu_demo ../GLU_public/src/matrix/add32_csr.mtx x.txt
n=4960 nnz=23884 ||Ax-b||2=1.421397e-13 ||b||2=7.042727e+01 status=0
$ head -3 x.txt
428.68246991689824
961.01135527456074
926.44090422326008
```

- 输出使用 `%.17g`，保证 double 十进制往返精度无损，便于逐位比对。
- stderr 行含义：`n` 阶数、`nnz` 组装后 CSC 的非零元数、`||Ax-b||2` 残差二范数、
  `||b||2` 右端项二范数、`status`（KLU 状态码，`0` 或 `n` 表示成功）。

### 3.1 输入文件格式

支持标准 Matrix Market coordinate 格式，也兼容本目录矩阵的几种变体：

- **标准头**：`%%MatrixMarket matrix coordinate real general`
- **单百分号头**：有些文件写成 `%MatrixMarket ...`（非法但能识别）
- **完全没有头**：第一行直接是 `<行数> <列数> <非零元数>`（如 `add32_csr.mtx`）

数据行格式为 `i j v`，**1-based 索引**，行序任意；重复的 `(i,j)` 会被自动求和；
空白行与 `%` 注释行会被跳过；识别 `symmetric` 关键字。

### 3.2 `-expand` 选项

默认**不展开** symmetric 矩阵，直接使用文件中存储的元素——这与 GLU 读取矩阵的
方式一致，方便两者对同一个线性系统求解。

加 `-expand` 才会按标准 Matrix Market 语义把 `symmetric` 文件的下三角镜像成完整
矩阵。注意本目录中的 `G2_circuit_csr.mtx` 虽然标注为 `symmetric`，但它的存储
元素本身就构成完整矩阵，若加 `-expand` 展开会得到一个奇异矩阵，`klu_factor`
会失败。因此**与 GLU 对比时请使用默认（不加 `-expand`）**。

---

## 4. 与 GLU 对比

GLU 的 `lu_cmd` 对每个矩阵取 `b = ones(n)`，把解写到 `x.dat`，并打印
`||Ax-b||` 的各种范数。对比步骤：

```bash
# 1) 参考解（本包）
./klu_demo <matrix>.mtx klu_x.txt

# 2) GLU 解
cd GLU_public/src && ./lu_cmd -i matrix/<matrix>.mtx && cp x.dat /path/to/glu_x.txt

# 3) 解向量二范数相对误差，以 KLU 为参考
python3 - <<'EOF'
import math
g=[float(l) for l in open('glu_x.txt') if l.strip()]
k=[float(l) for l in open('klu_x.txt') if l.strip()]
assert len(g)==len(k)
num=sum((a-b)**2 for a,b in zip(g,k)); den=sum(b*b for b in k)
print("rel_err_2norm = %.6e" % math.sqrt(num/den))
EOF
```

注意事项：

- **`rajat25_csr.mtx`**：GLU 无法求解，不在对比范围内。
- **`G2_circuit_csr.mtx`**：用默认参数（不加 `-expand`），原因见 3.2。
- 目录中 13 个矩阵里，`twotone_csr.mtx`、`rajat26_csr.mtx`、`rajat27_csr.mtx`
  条件数很大（解的量级可达 `1e10` 以上），两个求解器各自的残差都可能大于对方，
  此时"相对误差"对小扰动非常敏感，结论需要谨慎解读。
- GLU 的 `x.dat` 用 `setprecision(10)` 输出（约 10 位有效数字），本身带有
  截断误差；若发现相对误差在 `1e-10` 量级，可能来自这里的输出精度，而不是求解器。

---

## 5. KLU API 速查

头文件：`#include "klu.h"`，所有类型都在 `klu.h` 中声明。

```c
#include "klu.h"

klu_common Common;
klu_symbolic *Sym;
klu_numeric  *Num;

klu_defaults(&Common);                          /* 1. 默认参数 */
Sym = klu_analyze(n, Ap, Ai, &Common);          /* 2. 符号分解（排序 + 前置分析） */
Num = klu_factor(Ap, Ai, Ax, Sym, &Common);     /* 3. 数值分解 */
klu_solve(Sym, Num, n, 1, b, &Common);          /* 4. 求解，结果覆盖 b */
klu_free_symbolic(&Sym, &Common);
klu_free_numeric(&Num, &Common);
```

### 5.1 CSC 格式要求（最容易出错的地方）

KLU 使用 **CSC（按列压缩）**，并且对输入的 Ap/Ai/Ax 有硬性要求：

- `Ap`：长度 `n+1`，`Ap[i]` 是第 `i` 列在 `Ai/Ax` 中的起始下标；
- `Ai`：长度 `Ap[n]`，**每列内的行下标必须严格递增**；
- **不允许重复行下标**，重复项必须在调用前自行求和；
- `Ap`、`Ai` 是 `int32`（`int`），`Ax` 是 `double`；
- 下三角 / 上三角的划分由 KLU 依据 `Ap` 自行判断，无需手动标记。

`examples/klu_demo.c` 演示了从 triplet 到合法 CSC 的完整转换过程
（计数 → 填入 → 列内插入排序 → 合并重复项），可以直接复用。

### 5.2 常用扩展调用

```c
/* 只改变矩阵数值、稀疏结构不变时，复用符号分解，比重新 factor 快 */
klu_refactor(Ap, Ai, Ax, Sym, Num, &Common);

/* 多右端项：klu_solve 的第 3 个参数是右端项个数 */
klu_solve(Sym, Num, n, nrhs, B, &Common);   /* B 为 n*nrhs 列主序 */

/* 转置系统 A^T x = b */
klu_tsolve(Sym, Num, n, nrhs, B, &Common);

/* 对角缩放（默认关闭），对病态矩阵有时有帮助 */
Common.scale = 1;   /* 或 2；在 klu_factor 之前设置 */
```

`Common.status` 是状态码：`KLU_OK`(0) 表示成功；`> 0` 是警告（`KLU_SINGULAR`
== 1，矩阵数值奇异）；`< 0` 是错误。`klu_analyze` / `klu_factor` 返回非空指针、
`klu_solve` / `klu_refactor` 返回 1 表示成功，返回 0（或空指针）表示失败，建议
每次调用后检查返回值。

### 5.3 其他可调项（`klu_common`）

| 字段 | 默认 | 说明 |
|------|------|------|
| `Common.btf` | 1 | 是否使用 BTF 块三角化 |
| `Common.ordering` | AMD | 块内排序方式（AMD / COLAMD / 用户自定义） |
| `Common.scale` | 0 | 对角缩放：0 关、1 求和缩放、2 最大值缩放 |
| `Common.tol` | 0.001 | 部分选主元阈值 |
| `Common.memgrow` | 1.2 | 内存增长因子 |

---

## 6. 精度

- 全程 **double**（IEEE-754 双精度），无 GPU，无单精度转换。
- 采用部分选主元 LU，数值稳定性好；对电路类矩阵（对角占优、非对称）尤其合适。

---

## 7. 许可证

本包代码来自 SuiteSparse 7.14.1，各子包均为 **BSD 3-clause**，详见 `LICENSE/`：

- `KLU_License.txt`
- `AMD_License.txt`
- `COLAMD_License.txt`
- `BTF_License.txt`

`SuiteSparse_config` 同样为 BSD 3-clause（见 `include/SuiteSparse_config.h` 头部
声明）。使用时请保留版权声明。
