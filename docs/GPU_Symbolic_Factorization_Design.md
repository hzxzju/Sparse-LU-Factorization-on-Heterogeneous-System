# GPU Sparse LU Symbolic Factorization：Codex 实现规格说明书

> **状态：Phase 1 已有独立原型与测试入口；真实 HTHPCC 编译及主机检查通过，GPU/CPU 一致性尚待可访问的目标 GPU 验证。Phase 2–4 仍是设计，不代表已有实现。**  
> **依据：** 当前仓库的 `symbolic.cc`（`Symbolic_Matrix::fill_in / csr / predictLU / leveling`）、Numeric 头文件、`hc_runtime.h` 和 Makefile；代码语义优先。  
> **目标：** 将符号结构预测与相关预处理迁移至 wave width = 64 的 GPU；保留后续 Numeric LU 所依赖的数组语义。  
> **硬件/编译注意：** HTHPCC、`htcc` 与厂商 GPU runtime 的具体 API 名称必须以仓库实际头文件为准；以下函数名为**建议接口**，不是已存在的 API。

## 0. Codex 接手任务：先阅读这些内容

1. 阅读本文件和工程现有的 `symbolic.cc`、`symbolic.h`、GPU kernel 头文件及 `Makefile`。
2. 找出上层调用 `Symbolic_Matrix` 的调用链，确认 `ai/ap/ax` 是不是经过排列或缩放后的 CSC，以及 Numeric LU 使用 `sym_c_ptr`、`sym_r_idx`、`l_col_ptr`、`csr_r_ptr`、`csr_c_idx`、`csr_diag_ptr`、`val`、`level_ptr`、`level_idx` 的具体方式。
3. 不要假设本文件中的 `Gpu*` 类已经存在：它们是拟议的新增设计。
4. 先实现**最小正确原型**：对固定列顺序、无动态数值选主元的输入，用 GPU 产生 `sym_c_ptr / sym_r_idx / l_col_ptr`，与 CPU `fill_in()` 比较。之后再迁移 `csr()`、`predictLU()`、`leveling()`。
5. 不允许为了并行而改变矩阵数学语义；凡是还不能保证等价的环节，先保留 CPU fallback 并明确标注。

## 1. 现有 CPU 文件分析：真正需要迁移什么

| 函数 | 输入 | 当前操作 | 必须保持的输出 |
|---|---|---|---|
| `fill_in(unsigned *ai, unsigned *ap)` | CSC 行索引、列指针 | 对列副本排序；遍历持续增长的当前列，对行号小于列号的项，合并前驱列已生成的对角及严格 L 后段；强制插入结构对角 | `sym_c_ptr[n+1]`、`sym_r_idx[nnz]`、`l_col_ptr[n]`、`nnz` |
| `csr()` | 预测 CSC | 按列扫描，并向各行 buffer 写入列号，行内天然升序 | `csr_r_ptr[n+1]`、`csr_c_idx[nnz]`、`csr_diag_ptr[n]` |
| `predictLU(ai,ap,ax)` | 原始 CSC 数值、预测 CSC | 逐个预测位置寻找原始值；不存在的 fill-in 写 0 | `val[nnz]` |
| `leveling()` | 预测 CSC/CSR 和对角偏移 | 找两种列依赖，`level[i]=1+max(level[predecessors])` | `num_lev`、`level_ptr[num_lev+1]`、`level_idx[n]` |
| `solve()` / `solve_CSR()` | LU 数值等 | 数值解三角系统 | **不属于**当前 symbolic GPU 迁移范围 |

**重要语义：** `l_col_ptr[j]` 是第 `j` 列**对角元素在整个 CSC 行索引中的绝对位置**。所以：

- `sym_c_ptr[j] ... l_col_ptr[j]-1`：`row < j`，该列 U 部分；
- `l_col_ptr[j]`：结构对角 `row == j`；
- `l_col_ptr[j]+1 ... sym_c_ptr[j+1]-1`：`row > j`，该列 L 部分。

CSR 必须行内列号升序，`csr_diag_ptr[i]` 是 CSR 第 `i` 行中对角元素的绝对位置。`leveling()` 依赖其左侧恰好都是列号小于 `i` 的元素。

**现有 CPU `fill_in()` 并非列独立：** 对列 `i` 的非零行 `k<i`，需要读取已完成 `k` 列的 `sym_r_idx[l_col_ptr[k] ... sym_c_ptr[k+1])`。直接一列一个 CTA 同时运行会产生错误的前驱结构读取。

### 1.1 与真实代码的核对及规格修正

| 核对项 | 代码事实与结论 |
|---|---|
| `fill_in` 列依赖 | 与上述依赖判断一致；原表述“严格下三角+对角后段”不够准确，实际起点包含对角。遍历的是不断变化的 `tmpcol2`，新填入的 `k<i` 也会继续触发合并，不是仅扫描原始列的 U 项。 |
| `l_col_ptr` | 与规格一致，是全 CSC 的绝对偏移。Numeric 的 `LL_updateFromLeft` 调用处以 `l_col_ptr[k]+1` 读取严格 L；RL 也直接用这个全局偏移读写主元。不是列内下标，也不是严格 L 起点。 |
| CSR 行内排序 | 与规格一致，CPU `csr()` 按列号 `i=0..n-1` 向行 buffer 追加 `i`，不调用排序。无重复坐标时严格升序，`csr_diag_ptr` 同样是绝对偏移；Leveling 和 RL 的对角前后区间依赖这一点。 |
| Leveling U 规则 | 与 §7 一致：扫描列 `i` 对角前的 `k`，且仅在 `l_col_ptr[k]+1 != sym_c_ptr[k+1]` 时使用 `level[k]`。条件是前驱列具有严格 L 项，不是只要有 `U(k,i)` 就依赖。 |
| Leveling L 规则 | 与 §7 一致：扫描 CSR 第 `i` 行对角前的所有 `k`，不增加“前驱有严格 L”过滤。两条规则共同对 `max_lv` 取最大值；CPU 没有显式依赖边去重。未来用入度 DAG 时才需要去重以正确匹配增减计数。源点层级为 0，同层 `level_idx` 按列号升序。 |

原文“严格升序”还隐含**无重复输入坐标**这一前提。CPU `sort` 和 `set_union` 不消除输入的重复项，`set_union` 保留最大重数；不能声称 CPU 自动正规化为集合。Phase 1 GPU 入口拒绝重复坐标，接受未排序列，且不改动 `ai/ap/ax` 的配对关系。非法输入直接报错，不以 CPU fallback 掩盖。

原文对 CPU 重复调用累积 `push_back` 的警告也需按当前代码更新：现有 CPU `fill_in/csr/predictLU/leveling` 已在入口清空相应输出；GPU 成功提交新结构时清空派生数据，失败则保留旧结构。测试覆盖重复调用，不能继续将“CPU 完全没有重置”作为当前事实。

调用链也有需要限定的输入语义：CLI 的 `preprocess()` / `my_DumpA()` 导出经过 NICSLU 预处理的 CSR 矩阵 `B`；`lu_cmd.cpp` 将同一组数组作为 `B^T` 的 CSC 做符号及数值分解，最后调用 `solve_CSR()`。因此这里“原始 CSC”指**符号阶段收到的固定结构**，不能拿文件中原矩阵的 CSC 与预处理后的结果直接比较。Python binding 保持现有两参数 CPU `fill_in()` 调用。

后文 `Gpu*` 对象仍是未来接口建议。Phase 1 实际只新增 `SymbolicConfig` 和三参数 `Symbolic_Matrix::fill_in(ai,ap,config)`，将三个结果数组下载到原有 host vector；没有实现这些拟议类，也不向 Numeric 暴露临时 device 指针。

## 2. 改写数学算法：原始 CSC 上的独立列图搜索

### 2.1 图定义与适用范围

建立有向图 `G`，其边 `k -> r` 等价于原始输入 `A(r,k)` 结构非零。图邻接表就是原始 CSC 第 `k` 列 `ai[ap[k] ... ap[k+1])`。

对于固定目标列 `j`，若存在一条由原矩阵原始非零边组成的路径 `j -> k1 -> ... -> r`，且**全部内部节点**满足 `< min(j,r)`，则在**固定消元顺序的标准结构消元模型**下 `(r,j)` 属于预测 LU 结构；`(j,j)` 另外强制保留。该结论不能直接覆盖带动态数值 pivot、动态 permutation 或不符合该结构消元模型的扩展逻辑。

### 2.2 GPU 友好的 minimax 路径状态

对每个目标列 `j`，设置 `best[r]` 为该路径的最小门槛：`1 + (路径所有内部节点编号的最大值)`；直接边的门槛为 0。使用 `uint32_t`，`INF=UINT32_MAX`。初始：`best[j]=0`，对每条原始直接边 `j->r` 设置 `best[r]=0`。

对于一个需要展开的中间节点 `k`（仅需 `k<j`），读取原始列 `k` 的每条边 `k->r`，计算：

```
candidate = max(best[k], k + 1)
old       = atomicMin(&best[r], candidate)
if (candidate < old && r < j): queue_next.push_once(r)
```

当队列达到不动点时，对最终行号 `r` 的合法性判定是：

```
valid(r,j) = (r == j) || (best[r] <= min(r,j))
```

注意 `candidate` 的传播要保留**所有获得更好代价的 `r<j` 的节点**，即便它们当前不是合法输出行；它们可能是后续合法路径的中间节点。`r>=j` 不可能是合法中间节点，无须展开，但仍需要记录其 `best[r]` 用来判断最终输出。

### 2.3 微型手算例子

原始 CSC 中有 `A(0,1)` 与 `A(2,0)`；对于目标列 `j=1`：

- 原始边 `1->0`，初始化 `best[0]=0`，节点 0 入队；
- 扩展原始列 0，发现 `0->2`：`candidate=max(0,0+1)=1`；
- `best[2]=1 <= min(2,1)=1`，于是 `(2,1)` 被预测为 fill-in。

这实现了对列 1 的独立推导，不需要读取此前生成的第 0 列 LU 结构。

## 3. GPU 对象与接口设计

> 类名与方法名均为**建议名**。真实工程使用何种指针包装、Stream 类型、错误码、内存管理，必须沿用已有 runtime 约定。建议指针所有权采用 RAII，并在 `getDeviceResult` 明确生命周期；不要返回将失效的裸指针。

### 3.1 `GpuOriginalMatrix`：只读矩阵数据

**输入：** CPU `n, ap[n+1], ai[nnzA], ax[nnzA]`，以及用户提供的设备与 stream。  
**成员：** `n`、`nnzA`、`d_ap`、`d_ai`、`d_ax`。  
**方法：**

- `upload(hostCSC, stream) -> Status`：分配并传输原始 CSC。
- `normalize(stream) -> Status`：可选，对每列按行号排序，**同时重排 `ax`**；处理或拒绝重复 `(r,c)`。
- `validate(stream) -> Status`：验证指针单调、`ap[0]=0`、`ap[n]=nnzA`、行号范围、数值/索引同步；若 `n` 超过 32-bit 索引能力则拒绝。
- `release()`：释放拥有的 buffer。

**注意：** 和 CPU 分解必须使用相同的排列、缩放之后的输入；否则一致性比对不成立。

### 3.2 `GpuColumnWorkspace`：每个活跃 CTA 一个 Slot

每个 Slot 的 Global Memory 包含：

| 缓冲 | 容量 | 内容 |
|---|---:|---|
| `best` | `n` | `uint32_t` 路径门槛，`INF` 表示不可达 |
| `queue_cur` | `n` | 当前 frontier 的节点编号 |
| `queue_next` | `n` | 下一 frontier 的节点编号 |
| `enqueue_epoch` | `n` | 每轮只允许相同节点入队一次 |

每 CTA 的 Shared Memory 包含 `cur_size,next_size,job_cursor,epoch`，或与厂商 runtime 兼容的等价同步控制值。CPU 端的 `WorkspacePool` 根据 `batch_cols` 提供 Slot；不同 CTA 不共享 Slot。

容量边界：若只扩展 `<j` 节点且每轮去重，则每轮队列最多 `n` 个元素。仍需在 Debug 中检查写入边界；`epoch` 回卷要处理或保证不发生。

### 3.3 `GpuSymbolicPatternBuilder`：原始图 -> LU CSC

建议对外函数：

| 方法 | 输入 | 输出 | 主要动作 |
|---|---|---|---|
| `countAllColumns(input, workspace, col_nnz, diag_local)` | 原始 CSC、批大小 | `col_nnz[n]`、`diag_local[n]` | 各列独立图搜索，统计合法行 |
| `buildColumnPointers(col_nnz)` | 列计数 | `sym_c_ptr[n+1]`、`nnz` | Exclusive Scan |
| `writeAllColumns(input, workspace, sym_c_ptr, diag_local)` | 原始 CSC、列指针 | `sym_r_idx[nnz]`、`l_col_ptr[n]` | 再次搜索、升序输出 |

内部设备例程 `searchOneColumn(j, d_ap, d_ai, slot)` 不返回容器；只在 Slot 中产生 `best[]`，供 count / write 阶段消费。

### 3.4 `GpuCscCsrConverter`

- `convert(sym_c_ptr, sym_r_idx, n, nnz) -> {csr_r_ptr,csr_c_idx,csr_diag_ptr}`。
- 内部统计每行非零数、Prefix Scan、COO 散射/排序、寻找对角，保证 CSR 行内升序。

### 3.5 `GpuValueMapper`

- `map(inputCSC, sym_c_ptr, sym_r_idx) -> d_val[nnz]`：原始 `(r,j)` 存在则复制其数值，不存在则填 0。用于 Numeric LU 输入，不是最终 LU 因子。

### 3.6 `GpuLevelBuilder`

- `build(sym_c_ptr, sym_r_idx, l_col_ptr, csr_r_ptr, csr_c_idx, csr_diag_ptr) -> {num_lev,level_ptr,level_idx}`。
- 可选中间输出 `{in_degree,out_ptr,out_idx}` 供 synchronization-free Numeric LU 复用。必须通过两种依赖规则生成边并去重（见 §7）。

### 3.7 `GpuSymbolicAnalyzer`：顶层对外适配

```
initialize(inputCSC, config, stream) -> Status
analyze() -> Status
getDeviceResult() -> GpuSymbolicResult
// 兼容既有 CPU Symbolic_Matrix 的接口，按需：
downloadResult() -> HostSymbolicResult
release()
```

`GpuSymbolicResult` 至少包含：`n,nnz,sym_c_ptr,sym_r_idx,l_col_ptr,csr_r_ptr,csr_c_idx,csr_diag_ptr,val,num_lev,level_ptr,level_idx`，全部为正确设备内存地址或元信息。

## 4. CTA、Wave、Thread：`gpu_symbolic_column_search`

### 4.1 启动映射

**初始建议：** `blockDim.x=256`，每个 CTA 4 个 Wave，每 Wave 64 个 Thread；`gridDim.x=batch_cols`；单 CTA 对应 `j=batchStart+blockIdx.x`；`slot=blockIdx.x`。

| 层级 | 任务 | 并行粒度 |
|---|---|---|
| Grid | 多目标列同时分析 | 目标列无互相依赖 |
| CTA | 某一列完整的图搜索 | 维护独立 frontier 与 best |
| Wave | 动态领取一个待展开节点 `k` | 64 条线程合作扫其邻接列 |
| Lane/Thread | 从 `ap[k]+laneId` 开始，步长 64 访问邻边 | 一次访问/更新一条 `(k,r)` |
| `atomicMin` | 同一个目标列内的 `best[r]` 合并 | 不能用无同步的普通写覆盖 |

### 4.2 初始化阶段（全 CTA 256 线程）

1. 线程 `tid=threadIdx.x` 执行 `for (r=tid; r<n; r+=256)`：`best[r]=INF; enqueue_epoch[r]=0`。
2. CTA 同步；指定一个线程设置 `best[j]=0`。
3. 全 CTA 并行扫描原始列 `j`：`for (p=ap[j]+tid;p<ap[j+1];p+=256)`，对直接邻居 `r=ai[p]` 初始化 `best[r]=0`。若 `r<j`，通过 `atomicAdd` 入 `queue_cur`。如输入有重复行，应先正规化去重。避免在未完成初始化前让 Wave 开始处理队列。
4. CTA 同步；开始搜索轮次。

### 4.3 每轮 Wave 领取工作

设 `laneId=tid%64`、`waveId=tid/64`。每个 Wave 的 Lane 0 从 CTA Shared `job_cursor` 执行 `atomicAdd(&job_cursor,1)`；结果广播给 Wave 其它 Lane（采用运行环境已有的 Wave shuffle/broadcast intrinsic）。

- `job>=cur_size`：Wave 本轮没有新任务；其他 Wave 仍可能继续，必须保证所有 CTA 线程最终到达同一个 CTA Barrier。
- `job<cur_size`：读取 `k=queue_cur[job]`；该 Wave 独立执行 `k` 的邻边遍历。

Wave 完成一个节点后继续领下一个节点，直至耗尽当前队列；这种动态分配针对稀疏列出度不平衡。

### 4.4 每 Lane 处理一条邻边

```
for (p = ap[k] + laneId; p < ap[k+1]; p += 64) {
    r = ai[p];
    candidate = max(best[k], k+1);
    old = atomicMin(&best[r], candidate);
    if (candidate < old && r < j) {
        // 在下一轮入队（需要 epoch 去重）
    }
}
```

优化：每个 Wave 可先读取一次 `best[k]` 并广播，从而减少对同一个位置的多线程读取。**正确性：** 即使不同 Wave 同时改善同一 `r`，`atomicMin` 也确保 `best[r]` 只下降不反弹。

### 4.5 Epoch 去重 + 双缓冲 Frontier

每轮以 `nextEpoch` 为标签，成功改善且 `r<j` 时：

```
previous = atomicExch(&enqueue_epoch[r], nextEpoch);
if (previous != nextEpoch) {
    position = atomicAdd(&next_size, 1);
    queue_next[position] = r;
}
```

同一节点一轮只入队一次，但如果下一轮又找到更小的 `best[r]`，**允许再次入队**，直至不动点。

**同步要求：** 一个 frontier 的所有生产者完成写入后，必须发生全 CTA Barrier，再安全读取下一轮。`atomicExch` / `atomicAdd` 和 barrier 的 acquire/release 细节遵循 HTHPCC runtime。不得让少数 Wave 提前跳出包含 CTA Barrier 的循环；所有线程必须进入相同数量的 barrier。

每轮末：

1. CTA Barrier；确保队列数据完成写入；
2. 单线程交换 `cur/next` 队列描述、设置 `cur_size=next_size;next_size=0;job_cursor=0;epoch++`；
3. CTA Barrier；全部线程使用新的队列元数据；
4. `cur_size==0` 时**所有线程一起**结束本列搜索。

### 4.6 资源预算与溢出检查

每 Slot 四个 `uint32[n]`，约 `16n` bytes；`batch_cols=128,n=100000` 时约 **204.8 MB（十进制）**，不含最终稀疏结构等其它显存。对 `best` 和当前 frontier 的批次采用 Slot 重用；同时处理多少列应按 GPU 显存、occupancy 和实际矩阵动态配置。极大 `n` 时需设计 touched list 减少 `O(n)` 初始化。

## 5. 两遍构建 CSC：Count -> Scan -> Write

### Kernel A：`kernel_count_patterns`（1 CTA / 目标列）

- **输入：** `d_ap,d_ai,n,batchStart,workspaceSlots`。
- **计算：** 执行 §4 的搜索直至 frontier 为空；随后按连续行号扫描 `best[0..n)`。
- **每 Thread：** 处理 `r=tid+256*t`；`valid=(r==j)||(best[r]<=min(r,j))`。
- **CTA Reduction：** 累加 `col_nnz[j]`，另累加 `diag_local[j]`（满足 `r<j && valid` 的数量）。
- **输出：** `col_nnz[n],diag_local[n]`。不写 `sym_r_idx`。

### Kernel B：`exclusive_scan_col_nnz`

- **输入：** `col_nnz[n]`。
- **输出：** `sym_c_ptr[n+1]`，`nnz=sym_c_ptr[n]`。
- **操作：** 并行 Exclusive Prefix Scan，分配 `sym_r_idx[nnz]`；Phase 1 不分配或生成 `val`。`nnz` 采用足够大的索引类型，并在使用 `unsigned` 旧接口前检查溢出。

### Kernel C：`kernel_write_patterns`（1 CTA / 目标列）

- **输入：** 原始 CSC、`sym_c_ptr`、`diag_local`、Slot。
- **计算：** 重新运行 §4 图搜索；按行号升序枚举合法节点。
- **线程组织：** CTA 同时处理连续 256 个行号；各 Wave 对 64 个 `valid` 做 ballot/prefix（或共享内存块内 scan），计算每个合法 `r` 的稳定写入偏移。
- **输出：** `sym_r_idx[sym_c_ptr[j] .. sym_c_ptr[j+1])`，从小到大排序；`l_col_ptr[j]=sym_c_ptr[j]+diag_local[j]`。
- **校验：** 实际写入数必须等于 `col_nnz[j]`；每列有唯一对角；所有行号严格递增。

两遍搜索牺牲部分重复计算换取受限工作空间；后续可考虑生成 touched-node 列表及其排序后缓存结果，减少第二遍图搜索。

## 6. CSC -> CSR 与 Value Mapping

### Kernel D：`count_rows`

- **输入：** `sym_r_idx[nnz]`。
- **输出：** `row_counts[n]`。
- **每 Thread：** 处理一段线性 CSC 条目；对 `row_counts[sym_r_idx[p]]` 原子加一。

### Kernel E：`scan_row_counts`

- **输入：** `row_counts[n]`。
- **输出：** `csr_r_ptr[n+1]`。
- **操作：** 并行 Exclusive Scan。

### Kernel F：`scatter_coo`

- **输入：** `sym_c_ptr,sym_r_idx,csr_r_ptr`。
- **输出：** `(row,col,cscPosition)` 临时 COO 或未排序的 `csr_c_idx`。每 Thread 扫描一批 CSC 非零：行号来自 `sym_r_idx`；列号来自当前 CSC 列；行写游标通过原子加分配位置。

### Kernel G：`sort_csr_rows`

- **输入：** COO/未排序的 CSR。
- **输出：** 按 `(row,col)` 排序的 CSR 列号。
- **重要原因：** GPU atomic scatter 的写入先后不确定，原 CPU `csr()` 却保证了 CSR 行内升序；`leveling()` 对 CSR 对角前部分的遍历依赖这个顺序。
- **建议实现：** 将 `(row,col)` 编码成 64-bit key 并用现有 runtime 可用的并行排序库；无现成库则分行排序。不要未经检查就假设 GPU 排序 primitive 可用。

### Kernel H：`find_csr_diagonal`

- **输入：** `csr_r_ptr,csr_c_idx`（行内有序）。
- **输出：** `csr_diag_ptr[n]`。
- **分工：** 一 CTA/Wave 处理一行，二分/并行归约找 `csr_c_idx[p]==row`；不存在则报错（正常应由强制结构对角保证）。

### Kernel V：`kernel_predict_values`

- **输入：** 原始 `ap,ai,ax`，预测 `sym_c_ptr,sym_r_idx`。
- **输出：** `val[nnz]`。
- **CTA：** 一 CTA 对应一预测列 `j`；**Thread：** `p=sym_c_ptr[j]+tid+256*t`；读 `r=sym_r_idx[p]`，在原始列 `j`（经规范化排序）二分查找 `r`；命中写 `ax[position]`，否则写 0。
- **注意：** `val` 是符号预测后的初始数值，不能当作数值分解完成的因子。

## 7. GPU `leveling()`：保留原 CPU 的两种依赖规则

CPU 代码中，对于目标列 `i`，前驱 `k<i` 分两类：

**规则 U：** CSC 第 `i` 列 `sym_c_ptr[i] .. l_col_ptr[i]-1` 中的行号 `k`；只有 `l_col_ptr[k]+1 != sym_c_ptr[k+1]`（第 `k` 列严格 L 非零结构存在）才建立 `k -> i`。

**规则 L：** CSR 第 `i` 行 `csr_r_ptr[i] .. csr_diag_ptr[i]-1` 中的列号 `k`，建立 `k -> i`。

重复 `k->i` 必须去重，得到 DAG（所有边指向编号更大的列）。

### Kernel I：`build_dependency_dag`

- **输入：** 预测 CSC、CSR 和对角指针。
- **CTA：** 一列 `i`；**Wave：** 两种依赖的多段扫描；**Lane：** 检查每个候选元素是否符合规则，输出候选 `(k,i)`。
- **后续：** 对 `(k,i)` 去重，统计 `in_degree[i]`、生成后继表 `out_ptr[n+1] / out_idx[E]`。
- **输出：** GPU DAG、`in_degree`。

### Kernel J：`level_frontier_step`

- **输入：** 当前入度为 0 的 frontier、DAG 后继表。
- **一 Wave：** 通过任务游标领取一个 ready 列，64 Lane 扫描它的后继；对后继入度 `atomicSub`；返回值为 1 说明最后一条前驱刚处理完，可加入**下一层** frontier。
- **同步：** 第一版推荐“一层一个 Kernel”，由 Kernel 边界同步全局 frontier。不要用简单的 CTA 自旋等待某个可能尚未被调度的前驱 CTA，可能出现饥饿或死锁。
- **输出：** `level[i]`、`num_lev`；`level[i]` 等于最长依赖路径长度（源点层级为 0）。

### Kernel K：`pack_levels`

- **输入：** `level[n]`, `num_lev`。
- **输出：** `level_ptr[num_lev+1]`、`level_idx[n]`。
- **操作：** 按 level 直方图 -> Prefix Scan -> 散射；如需与 CPU 完全相同的 `level_idx` 顺序，对每层按列号升序排序。

## 8. 顶层调用序列与接口边界

```
CPU input CSC
    |
    v
GpuOriginalMatrix.upload / validate / normalize
    |
    v
[A] gpu_count_patterns        // 列独立图搜索，统计 nnz 和 diag_local
    |
    v
[B] exclusive_scan_col_nnz   // sym_c_ptr
    |
    v
[C] gpu_write_patterns       // sym_r_idx + l_col_ptr
    |
    +----------------------+-------------------------+
    |                      |                         |
    v                      v                         |
[D..H] GPU CSC->CSR     [V] GPU Value Mapper          |
    |                      |                         |
    v                      |                         |
[I..K] GPU Leveling        |                         |
    |                      |                         |
    +----------------------+-------------------------+
                           |
                           v
              GpuSymbolicResult (device pointers)
                           |
                           v
                     GPU Numeric LU
```

`GpuValueMapper` 与 CSR 构建在得到 CSC 后可以并行安排到不同 Stream，但第一版建议先用单一 Stream，减少并发 bug。Leveling 依赖 CSC 与 CSR，不依赖 `val`。

## 9. 逐步实施与验收标准

### Phase 1 — 列搜索的正确性

- 在独立新 GPU 文件中实现 `searchOneColumn`、Count、Scan、Write。
- 只接入原 `fill_in()` 的 CPU/GPU 对比入口；先保留旧接口和完整 CPU fallback。
- 测试 `n=1,2,16,32,64,128`、随机非对称结构、复杂 fill-in 链、缺失对角、长列、空列；各列升序且对角存在。
- 逐项对比 `sym_c_ptr,sym_r_idx,l_col_ptr` 完全一致。

### Phase 2 — CSR + Val

- 迁移 D–H、V；逐项比较 `csr_r_ptr,csr_c_idx,csr_diag_ptr,val`。
- 特别测试 GPU Scatter 后必须排序，检查对角左右分割。

### Phase 3 — Leveling / DAG

- 用真实 CPU 的两类依赖边构建 GPU DAG，重复边去重。
- 对每列 `level[i]` 和 `num_lev` 验证等价，对 `level_ptr,level_idx` 验证相同层集合；如需字节一致，进一步要求层内升序。

### Phase 4 — Numeric LU 集成与性能

- 直接将 GPU result 指针喂给现有 Numeric LU（确认 API 的内存所有权）；只有校验或旧 CPU 依赖确实要求时才下载结果。
- 记录：CPU `fill_in/csr/predictLU/leveling` 分项时间；GPU A/B/C、D–H、V、I–K、H2D/D2H 和端到端时间。
- 使用项目已有的 `hcTracer/hc_profiler`，不要宣称使用未用过的 Nsight 工具。
- 变换 `blockDim=64/128/256`、batch 大小、长列负载、矩阵阶数和 nnz，查看实际收益。

### 必须关注的失败模式

1. **算法适用性：** 仅固定顺序和结构消元模型等价；动态 pivot/permutation 必须另行处理。
2. **输入合法性：** CSC 重复坐标、无序行号、异常对角、索引溢出。
3. **并发性：** 初始化与 frontier 开始之间缺同步；队列多次入队；数组容量越界；Barrier 分支分歧。
4. **数据顺序：** CSC 列内严格升序；CSR 行内严格升序；`l_col_ptr/csr_diag_ptr` 必须是**全局绝对偏移**。
5. **性能：** 两遍图搜索与 `O(batch_cols*n)` 初始化/最终扫描可能吃掉收益；高出度节点导致单 Wave 执行过长，需后续优化为多 Wave 共同处理高出度列。
6. **调用链：** `symbolic.cc` 中没有展示 `symbolic.h` 定义和工程的 Numeric LU 使用点；不要猜测不存在的字段或 API。
7. **重复调用：** 当前 CPU 各阶段已重置对应输出；GPU 成功提交时清空派生数组，失败保留旧结构。继续用重复分析测试防止状态污染。

### Phase 1 原型、构建与验收

实际代码在 `src/symbolic_gpu.cu`，不覆盖两参数 CPU `fill_in()`。默认 CPU 路径不变。GPU 默认 256 threads/CTA（4 个 64-thread Wave），也允许 64/128 threads；一 CTA 负责一个目标列。Count 与 Write 共用 minimax 搜索，双 frontier 和 epoch 去重允许节点在后续轮次重新入队；Wave 的 lane 0 动态领取工作，使用本机 SDK 的 `__shfl_sync(~0ULL, job, 0, 64)` 广播。所有 frontier 交换均有 CTA barrier。

设备选择已改为 `include/gpu_device.h` 中的共用逐卡探测：跳过返回初始化错误或不满足 kernel 要求的卡，Numeric 优先复用 Symbolic 选择的设备。它要求 SDK 先完成设备枚举；关于当前主机在 GPU 2 队列状态检查中阻塞的证据与边界，见 [初始化诊断](GPU_Device_Initialization_Diagnostics.md)。

Count 统计列长度与对角局部偏移；Scan 使用设备端分层 64-bit exclusive scan，检查总 `nnz <= UINT_MAX` 后才缩窄为旧接口；Write 再次搜索，通过共享内存块内 scan 按行号稳定升序写出，`l_col_ptr[j]=sym_c_ptr[j]+diag_local[j]`。每 Slot 占 `16*n` bytes；`batch_cols` 是 Slot 上限，实际按空闲显存一半及 grid 上限收缩，分配失败会继续减半。两遍均按批次复用 Slot，空间为 `O(batch_cols*n)`，不会为所有列分配 `n*n` 个状态。

生产代码只包含真实 `<hc_runtime.h>`，使用 `hcMalloc/hcFree`、`hcMemcpy`、`hcMemset`、`hcMemGetInfo`、设备查询、错误查询及同步。已核对 `/opt/hpcc/include/hcr/hc_runtime_api.h` 的声明（当前安装路径指向 HPCC 3.8.0）和编译器的 `__clang_hpcc_device_functions.h` / `__clang_hpcc_atomic_functions.h`；块同步沿用 Makefile 的 `-D__sync_threads=__syncthreads`。不依赖假设存在的 scan/sort runtime API。

在仓库根目录执行：

```sh
# 独立编译 Phase 1：无需编译 Numeric 或链接 NICSLU 预处理库。
make -C src symbolic_gpu_test DEBUG=0 -j2

# 主机检查；需要已安装的 HTHPCC runtime，但不要求有 GPU。
make -C src test-symbolic-host DEBUG=0

# 目标 GPU 验收：每次都禁用 fallback，逐元素对照 CPU。
timeout 300s ./src/symbolic_gpu_test
```

测试包括：`n=0/1/2/16/32/64/128/257`、非对称随机结构、无序列、缺失对角、空列、长列、经非法输出节点继续传播的路径、必须降低代价重新入队的路径；组合 `block_size=64/128/256` 与 `batch_cols=1/3/128`，反复使用同一个对象并检查输入未被修改。另用 `n=4097, block_size=64` 检查三级 scan 和尾部 tile。对 `nnz/sym_c_ptr/sym_r_idx/l_col_ptr` 逐项比较，失败报告数组与首个不一致下标；再用原有 CPU `csr/predictLU/leveling` 检查下游兼容性，这不代表这些阶段已迁移 GPU。

主机数学检查穷举 `n=1..4` 的全部 **66,066** 种结构，用独立的全对 bottleneck Floyd–Warshall 检查 §2 路径规则与真实 CPU `fill_in()` 完全一致；另检查 Leveling 两种规则、重复依赖、对角绝对偏移、CSR 升序、配置、非法输入、重复分析和空矩阵。它们只能验证主机语义，不能替代 GPU 并发正确性测试。

测试程序退出码：`0` 为本次所选检查通过；`1` 为失败；默认模式检测不到可访问 GPU 时打印 `SKIP` 并退出 **77**。`--host-only` 明确只要求主机检查通过。`make test-symbolic-gpu` 遇到 77 仍失败，防止把跳过当作 GPU 验收成功。当前环境的主机检查与 HTHPCC 编译/链接已通过，默认测试返回 77；**Phase 1 GPU 一致性验收尚未通过，不继续 Phase 2–4，也不宣称加速收益。**

现有 CLI 提供实验入口（CSR、Value Mapping、Leveling 仍在 CPU，Numeric 核心未改动）：

```sh
make -C src MAIN DEBUG=0 -j2
./src/lu_cmd -i /path/to/matrix.mtx --symbolic-backend gpu \
  --symbolic-block-size 256 --symbolic-batch-cols 128 \
  --symbolic-verify --symbolic-no-fallback
```

验收使用 `--symbolic-no-fallback`；不加时 runtime/显存错误会明确记录并回退 CPU，结构不一致不会被回退隐藏。可选下游数值回归保留在 `make -C src symbolic_gpu_numeric_test`，运行该程序会追加 RL-level、LL-level、LL-SF 对照；它不属于 Phase 1 最小构建依赖。

测试专用 CUDA adapter 仍可用于验证结构和块同步，但 CUDA 的 32-lane warp 不能直接执行 64-lane shuffle；该 adapter 分支用静态步进的逻辑 64-lane 工作分配，不能验证目标 GPU 的原生 shuffle、动态 Wave 调度或性能。
## 10. 和另一条 GPU 实施路线对比

| 路线 | 思想 | 优势 | 风险 |
|---|---|---|---|
| **A（本设计优先）独立列图搜索** | 每列从原始 CSC 图独立求路径门槛 | CTA 之间不依赖前驱 LU 列 | 重复搜索、WorkSpace 初始化开销 |
| B：依赖驱动的 GPU Symbolic | 保留 CPU 的前驱列合并语义，以 ready queue 动态调度 | 复用既有符号结构，可能减少遍历 | 需解决动态依赖、死锁、队列调度；不能简单 CTA 自旋 |

先完成 A 的正确性原型，拿到实测时间后决定继续优化 A，还是改用 B。**并行度更高不等于最终更快。**

## 11. 可以直接给 Codex 的任务指令

复制以下文字到 Codex（确保仓库内已有本文件和 `symbolic.cc`）：

> 请先阅读 `GPU_Symbolic_Factorization_Design.md` 和当前仓库中的 `symbolic.cc`、`symbolic.h`、Numeric LU 相关头文件、GPU runtime 封装以及 Makefile。文档是技术规格，而不是已经实现的代码。请先核对文件中关于 `fill_in` 列依赖、`l_col_ptr` 对角绝对偏移、CSR 行内排序要求、Leveling 两种依赖规则的解释与真实代码是否一致；如果发现不一致，列出差异并以现有代码为准。接着实现 Phase 1：新增不覆盖 CPU 路径的 GPU 列独立图搜索原型（64-thread Wave、每 CTA 1 列、4 Wave/CTA，可配置），实现 Count -> Scan -> Write，生成 `sym_c_ptr/sym_r_idx/l_col_ptr`，加 CPU/GPU 逐元素一致性测试。使用仓库真实的 HTHPCC/htcc API，不凭空创造 runtime 函数；优先提供能编译的最小增量和测试方法。Phase 1 通过前不要直接迁移 CSR、Value Mapping、Leveling，更不要修改 Numeric LU 的数值核心。

---

**设计校核结论：** 本文件描述的是一个可实现且可单元测试的设计方向，而非经过证明与完整工程测试的成品。第一步的成功标准是：**真实输入矩阵上得到与现有 CPU `fill_in()` 完全一致的有序结构输出**。
