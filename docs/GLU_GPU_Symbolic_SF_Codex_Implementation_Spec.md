# GLU：基于列级 `done` 调度的 GPU Symbolic Factorization 重构实施说明（交给 Codex）

> **任务性质**：在现有 GLU 项目中直接实施代码改造，而不是只给出方案或伪代码。  
> **基准源码**：用户提供的 `numeric(1).cu`（827 行）与 `symbolic(1).cc`（348 行）。进入仓库后应以真正对应的 `numeric.cu`、`symbolic.cc`、`left_looking_sf.cuh`、`symbolic.h`、相关调用点和构建脚本为准。  
> **目标**：把 GPU 符号分解改为与 numerical factorization 的 **column-wise / synchronization-free** 调度风格一致的分阶段实现。不要再次使用“为每个列做整图搜索 + 大量不规则 global-memory 访问”的算法。对列内符号集合执行 CPU 版相同的 **逐个前驱列合并（set union）** 语义；由 `thread 0` 管理列状态及 `done` 就绪门，block 内线程协作完成结构合并；实现有界内存、无死锁、可验证的异步列调度。

---

## 0. 要 Codex 立即执行的任务

1. **先读代码再改代码**：定位仓库里的 `symbolic_factorization` 当前 GPU 实现、CPU `Symbolic_Matrix::fill_in`、`LUonDevice`、`LL_factorizeSyncFree` 的实际定义，以及所有 `symbolic_factorization` / `fill_in` / `csr` / `predictLU` / `leveling` 的调用链。确认项目使用的 HTHPCC/hc_runtime 兼容 API，不要将 CUDA API 假定为完全兼容。
2. **保留原 CPU 实现作为 golden reference 和 fallback**，增加或替换 GPU symbolic SF 路径，默认按项目现有选项体系接入，不应无条件替换全部 CPU 路径。
3. **交付可编译实现**：host 入口、device kernel、scratch/pool、调度队列、错误处理、必要头文件与构建配置、验证测试、性能计时。最后给出修改文件列表、测试结果和已知限制。
4. **不得偷换语义**：输出必须与 `symbolic.cc::fill_in` 对合法 canonical CSC 输入的结构等价，包括 `sym_c_ptr`、`sym_r_idx`、`l_col_ptr`，并能无改动地供后续 `csr`、`predictLU`、`leveling` 和 numerical LU 使用。
5. **明确区分新增 symbolic 的 `done` 协议与 numeric 已有 `done` 协议**：用户提供的 `numeric(1).cu` 仅看到 `done_dev` 在 host 端清零，以及调用 `LL_factorizeSyncFree`；真正消费和更新它的逻辑在**未随这两份文件提供**的 `left_looking_sf.cuh`。Codex 必须打开仓库中此头文件确认 numeric 的协议，能复用则复用；如不同，使用独立名称 `sym_done` 和注明含义，严禁默认为两者语义一致。

### 核心验收目标

- 列 `i` 的符号集合以 CPU 版逐前驱 `L[j]` 并集的结果为准；**不是**独立图搜索得到的近似模式。
- GPU 任务允许并行推进不互相依赖的列；遇到未完成的前驱列时**不允许 block 一直自旋等待**，必须登记等待并让出执行资源。
- `sym_done[i] == 0` 表示**列 `i` 目前可以开始/继续推进**；`sym_done[i] != 0` 表示因尚未发布的某个前驱而暂停。列**完成**状态需由单独的 `sym_state[i] == FINISHED` 表示，不能混淆两者。
- 不预留 `n × n` 个 `unsigned` 的矩阵，也不构造 dense numerical matrix；仅使用每个活跃 worker 的紧凑 bitmap、按实际结构非零增长的稀疏缓存以及最终输出。
- `sym_r_idx` 严格递增、唯一、范围正确，所有列有结构对角；输出能够逐元素与 CPU golden reference 比较。
- benchmark 同时给出 **GPU compute-only** 和 **end-to-end symbolic**（含初始化/传输/pack）时间，并提供 GPU 结构访存/重试/休眠的观测数据。性能不达标时不可隐藏退化。

---

## 1. 从已提供源码提取的**确定事实**

### 1.1 `numeric(1).cu` 的调度风格（必须沿用其工程习惯）

从提供的文件可以直接确认：

- `LUonDevice(...)` 支持 `SchedulingStrategy::SynchronizationFree`，与 left-looking 路径一起调用 `LL_factorizeSyncFree`。
- 主机端分配 `next_col_dev`（一个 `unsigned`）和 `done_dev`（`n` 个 `unsigned`），**在每次分解前都清零**。
- SF 时使用一个固定 worker grid，形式近似：

  ```cpp
  LL_factorizeSyncFree<<<TMPMEMNUM, ll_threads>>>(
      sym_c_ptr_dev, sym_r_idx_dev, val_dev, l_col_ptr_dev,
      tmpMem, next_col_dev, done_dev, n, PERTURB, pert);
  ```

- `ll_threads` 不超过 256，并按目标设备的 **64-thread wave** 对齐；`TMPMEMNUM` 根据 SM/MP 数量、设备总显存、`sf_workers` 参数和可分配内存确定，并在分配失败时逐步缩小。
- `hcMalloc` 不是零初始化：通过 `hcMemset` 明确清零；`hcMemcpy` 负责数据流转；最终以 `hcDeviceSynchronize()` 等待完成。所有错误路径集中释放 stream、event 和 GPU buffer。
- `LL_factorizeSyncFree` 的真正 `done` 细节不能从 `numeric.cu` 推出。因此仅模仿**调度架构与资源管理**，不要猜测其 ready/complete 编码。

**实现要求**：重用设备选择 `glu_gpu::SelectDevice`、设备限制检查、HTHPCC 错误打印、波宽对齐、worker 数目和内存探测的既有机制；不要为了 symbolic 再搭一个与现有项目完全不同的设备管理系统。

### 1.2 `symbolic(1).cc::Symbolic_Matrix::fill_in` 的精确结构语义

每个目标列 `i` 按以下过程处理：

1. `tmpcol2` 初始化为原始 CSC 列 `ai[ap[i] : ap[i+1]]`，并按行排序。
2. 从当前已排序集合**最小的行号开始**遍历；如果遇到 `j < i`，读取**已经完成的**列 `j` 的 `L` 部分：
   
   ```cpp
   colhead = sym_r_idx.begin() + l_col_ptr[j];
   colend  = sym_r_idx.begin() + sym_c_ptr[j + 1];
   tmpcol2 = set_union([colhead, colend), tmpcol2);
   ```
   
   注意 `[l_col_ptr[j], sym_c_ptr[j+1])` **包含 j 的结构对角位置**，不只是对角下方。合并过程有可能产生新的 `k < i`，后续还必须继续处理这些新前驱。
3. 遍历到 `j >= i` 即停止向左查依赖；确保结构对角 `i` 存在（不存在则插入）。
4. 输出列内升序行索引到 `sym_r_idx`，以对角的全局位置记录 `l_col_ptr[i]`，累积 `sym_c_ptr[i+1]`。
5. 一个新 pattern 会清空旧 `sym_c_ptr / sym_r_idx / l_col_ptr / CSR / val / level_ptr / level_idx` 等派生状态；完成后 `nnz = sym_c_ptr[n]`。

该代码本身先 `sort`，但**未显式 `unique` 输入副本**。新 GPU 方案将唯一索引集合用作数学结构，这要求输入先满足 canonical CSC（行合法且无重复），或者提前规范化重复项；如果项目可能接受重复项，请在 GPU/CPU golden 测试中统一规范化策略，不能默默改变既有调用方可观察行为。对于不合法或无法规范化的输入，明确报错或回退 CPU。

### 1.3 后续数据布局不可更改

- `sym_c_ptr`：长度 `n+1`，第 `i` 列的区间为 `[sym_c_ptr[i], sym_c_ptr[i+1])`。
- `sym_r_idx`：列内严格升序、没有重复；包括 U 部分和 L 部分。
- `l_col_ptr[i]`：**全局 CSC 索引**，满足 `sym_r_idx[l_col_ptr[i]] == i`，左侧为上三角、右侧为下三角。
- `csr()`：将上述 CSC 转成行结构，之后生成 `csr_diag_ptr`。
- `predictLU()`：按照新 `sym_r_idx` 的顺序，把原始 `(ap, ai, ax)` 中存在的数值映射到 `val`，填充位置设零。
- `leveling()`：依赖上述 CSC/CSR，并服务于 numerical LU。

**不要在 symbolic kernel 中做浮点数值更新、pivot 扰动或构建 dense `val`。** Symbolic 只计算结构。数值准备仍走 `predictLU`（除非显式单独优化且验证一致）。

---

## 2. 正确算法：将 CPU `set_union` 换成 GPU worker 局部 bitmap 并集

### 2.1 每列的不变式

对列 `i` 定义临时结构集合 `S_i`，初始为输入列的所有行索引与 `{i}`；`next_pred` 为下一个尚未处理的前驱下界，初始为 `0`。重复执行：

```text
S_i = unique(A(:,i).row_index ∪ {i})
next_pred = 0

while true:
    j = min { r ∈ S_i | next_pred <= r < i }
    if j 不存在:
        finalize i
        break

    if column j 尚未完全发布:
        挂起 column i，等待 j 完成；不要 busy-wait
        恢复后从当前 checkpoint 继续
    else:
        S_i = S_i ∪ L_pattern(j)    // 行号来自 published col j 的 [diag, end)
        next_pred = j + 1
```

解释：在 `symbolic.cc` 中，对 `j` 的合并新增行号均满足 `row >= j`（结构对角及 L 部分）。所以已经处理过的 `< j` 行不会重新出现，`next_pred = j+1` 可以严格复现 CPU 的升序遍历，既不漏掉新增前驱，也不需要从 0 重扫整张图。所有 `j < i`，所以依赖关系无环。

**严格禁止**把“依赖是否已经完成”视为“该列当前 bitmap 是否暂时包含它”：`S_i` 可能随着前驱合并增长，必须处理动态产生的新前驱。

### 2.2 每个活跃 worker 使用**压缩 bitmap**（不是 dense value workspace）

假设 `W` 个并发 worker、`B = ceil(n/32)`，每个 worker 有：

- `work_bitmap[W][B]`：每一位表示一行是否进入 `S_i`，每个元素是 `uint32_t`。
- `dirty_bitmap[W][B]`：自上一个 checkpoint 后**新加入**的行，用于挂起时增量保存。
- 若需额外 scratch，只按 `W` 或实际 `nnz` 分配，禁止默认分配 `n × n` 个元素。

`set_bit(r)` 由整个 block 对原始行号或 `L[j]` 逐个并行执行；使用符合目标 runtime 的 32-bit 原子 OR（如无原生 atomicOr，封装基于受支持 CAS 的等价操作并单测）：

```cpp
uint32_t mask = uint32_t(1) << (r & 31);
uint32_t old  = atomicOr(&work_bitmap[worker * B + (r >> 5)], mask);
if ((old & mask) == 0)
    atomicOr(&dirty_bitmap[worker * B + (r >> 5)], mask);
```

- **只有新增的位才写入 dirty**。这保证 checkpoint 中每列行索引至多被存一次。
- 在多线程写入后做 **block barrier**（本项目优先使用已验证的 `__sync_threads()` 兼容形式），再由 thread 0 扫描候选前驱；block 内扫描/填充轮次保持所有线程控制流一致，不能让部分线程跳过 barrier。
- 用位级 `find-next-set-bit`（按 32-bit word 查询，而不是对每个可能行号发 global-memory 随机访问），找到 `>= next_pred` 的最小 `j < i`。
- 待发布 `L[j]` 保存在列完成时确定的**只读稀疏行号区间**，线程沿连续区间访问并 parallel OR 到 bitmap。该方案对已发布 predecessor 的读取量约为其实际 L 结构项数，而不是扫描整个数值矩阵。

内存量：两个 bitmap 的主 workspace 是约 `2 * W * ceil(n/32) * 4` 字节。记录在日志中。允许开发者在实测后把 `dirty` 优化成差量行表，但不得破坏 checkpoint 的唯一性与正确性。

### 2.3 挂起/恢复必须保持增量进度，不允许重跑已经处理过的 predecessor

**原因**：只在挂起后把列重新入队并从输入 `A(:,i)` 开始，每次都重新读取已处理的 `L` 列，可能再次造成用户当前指出的大量 global-memory 访问，性能无法接受。

为每列维护：

- `sym_cursor[i]`：尚未处理的最小候选前驱下界；只增不减。
- `checkpoint_head[i]`：该列已发现集合的差量 chunks 链首（或等价的 CSR 式 delta-chunk 目录）。
- `checkpoint_pool`：按实际**新增行索引**累积的压缩存储区；每个 chunk 是一组独立行号，重建 bitmap 时允许无序（bitmap 本身去重）。
- `checkpoint_chunk_start/len/next`：chunk 元数据；只有 `dirty` 非空时创建 chunk，防止大量 0 长度空块。

**首次入队**：清空 worker bitmap / dirty，写入输入列和对角，`cursor=0`。

**挂起前**：

1. 等待该列 worker 所有位图 OR 写入完成（block barrier）。
2. 对 `dirty_bitmap` 做 popcount、block 内前缀和并枚举为行索引；由 thread 0 以**有界安全预留**方式从 `checkpoint_pool` 分配连续区域。block 协作输出 delta chunk。
3. 完成 chunk 元数据链接、`sym_cursor[i]` 写入，把已成功持久化的 `dirty_bitmap` 位清零（否则 CAS 发现前驱刚完成并立即继续时，下次 checkpoint 会重复记录相同行）；block barrier 后执行设备可用的 **device-scope release publication**（见第 4 节）。
4. thread 0 将 `sym_done[i]=1`、`sym_state[i]=WAITING`，在所等待的 `j` 的 waiter 链登记；成功登记则**释放 worker**去取其他列。若 `j` 恰好已完成，禁止丢失唤醒，立即继续（不需要重新排队），当前 bitmap 可继续使用。

**被唤醒后**：新的 worker 清空 `work_bitmap` 和 `dirty_bitmap`；读取 `checkpoint_head[i]` 的所有 delta chunks，还原 `S_i`（重建时不把 checkpoint 数据再次设为 dirty），恢复 `sym_cursor[i]`，再处理尚未完成的 predecessor。**不再读先前已合并的 L[j]**。如果该列从未挂起，checkpoint 可以完全省略。

**额外约束**：checkpoint pool 的记录条目只按“此前未见过的行”增长：理论上累计行数不超过所有列最终结构非零数。合并未导致新增行时也不可反复追加重复快照。chunk 链表不能在另一个 worker 仍写入时被读取；队列状态的 release/acquire 顺序要保证保存后才唤醒。

---

## 3. 分阶段 kernels / functions：明确的实现拆分

建议新建 `symbolic_sf.cuh`（或项目惯用名称）和对应实现文件；在 `symbolic_gpu.cu`（或现有 GPU symbolic 文件）提供 host 入口。为了避免每个 predecessor 都产生一次 host kernel launch，**阶段可以是同一个持久化 worker kernel 内的 block 协作函数**；阶段划分指算法与状态职责，不强制拆成大量小 kernel。

| 阶段 | 建议名字 | 粒度 | 做什么 | 同步/输出 |
|---|---|---|---|---|
| 0 | `symbolic_sf_init` | host + init kernel | 输入校验，设备结构分配，队列和状态初始化 | 所有 init 完成后才启动 worker |
| 1 | `symbolic_sf_take_column` | worker CTA / thread 0 | 从 ready queue 取列；仅在 `sym_done[i]==0` 且状态 READY 时 claim | `READY -> RUNNING`，防止重复执行 |
| 2 | `symbolic_sf_load_pattern` | worker CTA | 新列读取原始 CSC + 对角；恢复列读取 checkpoint chunks | `work_bitmap / dirty_bitmap / cursor` |
| 3 | `symbolic_sf_expand_one_pred` | worker CTA | thread 0 找下一行 `j<i`，确认 `j` 已完成；整个 block 将 `L[j]` 并入本列位图 | 更新 `cursor=j+1`；若未完成则转挂起 |
| 4a | `symbolic_sf_checkpoint_and_park` | worker CTA + thread 0 | 差量写 checkpoint、登记到 `wait_head[j]`，释放 worker | `done=1, state=WAITING`；成功挂起后返回 scheduler |
| 4b | `symbolic_sf_finalize_column` | worker CTA | 无未处理前驱时：bitmap popcount、按行升序导出到 final arena、确定局部对角位置 | 写只读 `final_start / col_len / diag_rel` |
| 5 | `symbolic_sf_publish_and_wake` | thread 0 | release 发布完成；关闭本列 waiter 链并唤醒挂起列 | `state=FINISHED`, waiter 的 `done:1->0` 后 enqueue |
| 6 | `symbolic_sf_pack_csc` | 单独后处理 kernels | `col_len` 做 exclusive scan 生成 `sym_c_ptr`，并行将 final arena 按列号 scatter 到最终 CSC；确定 `l_col_ptr` | 得到 CPU 完全兼容 `sym_c_ptr/sym_r_idx/l_col_ptr` |
| 7 | `symbolic_sf_copy_commit` | host | 校验 nnz/边界；D2H；**成功时才覆盖** `Symbolic_Matrix` 派生数据；失败时 fallback | CPU 对象无半成品 |

### 3.1 主 worker 核心伪代码（是明确实现逻辑，不是完整可编译源码）

```cpp
__global__ void SymbolicFactorizeSyncFree(SymContext ctx) {
    const unsigned worker = blockIdx.x;
    const unsigned tid    = threadIdx.x;
    __shared__ int column;
    __shared__ int next_pred;
    __shared__ int step;

    while (true) {
        if (tid == 0) {
            // pop 必须是正确实现的并发队列，而非非原子更新的普通数组
            column = ready_queue_try_pop(ctx);  // EMPTY / STOP / column id
        }
        __sync_threads();
        if (column == STOP) break;
        if (column == EMPTY) {
            // 做有界退避并返回队列检查，不要对某个前驱 done 忙等
            continue;
        }

        // thread0 claim READY 且 done==0；其他线程只在 claim 成功后处理。
        if (!try_claim_column(ctx, column)) continue;

        load_or_restore_bitmap(ctx, worker, column, tid);
        __sync_threads();

        bool parked = false;
        for (;;) {
            if (tid == 0)
                next_pred = find_next_set_below_column(ctx, worker,
                                                      cursor[column], column);
            __sync_threads();
            if (next_pred == NONE) break;

            if (!published_acquire(ctx, next_pred)) {
                // 对当前 dirty rows 进行一次增量持久化。
                checkpoint_dirty_rows(ctx, worker, column, tid);
                __sync_threads();
                if (tid == 0)
                    step = try_park_on_predecessor(ctx, column, next_pred);
                __sync_threads();
                if (step == PARKED) { parked = true; break; }
                // predecessor 已同时发布：不可丢失唤醒，继续本地推进。
            }

            // 必须先 acquire 确认 predecessor 的 row interval 已完整发布。
            union_published_L_rows_into_bitmap(ctx, worker, column, next_pred, tid);
            __sync_threads();
            if (tid == 0) cursor[column] = next_pred + 1;
            __sync_threads();
        }

        if (parked) continue;
        pack_worker_bitmap_to_final_arena(ctx, worker, column, tid);
        __sync_threads();
        if (tid == 0) publish_column_and_wake_waiters(ctx, column);
        __sync_threads();
    }
}
```

伪代码中的 `try_claim_column / try_park... / queue` **必须**按下文锁与内存发布协议实现；不可直接用几次普通 load/store 和 `volatile` 代替。

### 3.2 可复用 numeric 的地方 / 不可复用的地方

可复用：GPU 设备选择、对齐至 64 wave 的 blockSize、`sf_workers` worker 上限、内存分配失败降级、运行计时、检查并清理 HTHPCC API 资源、队列中 `next_col` 概念（若核实兼容）。

不可机械复用：`tmpMem` dense value workspace、数值 pivot、floating-point atomicAdd、`val`、`PERTURB`、`level_idx` 对 dependency 的假设、以及未确认的 numeric `done` 取值。

---

## 4. `done`、发布、等待与无死锁调度（强制）

### 4.1 明确状态，不使用一个 `done` 兼任所有角色

```cpp
enum SymState : unsigned {
    SYM_READY    = 0, // 在 ready queue 中，且 sym_done=0
    SYM_RUNNING  = 1, // 已由唯一 worker claim
    SYM_WAITING  = 2, // 因某个更小的 predecessor 挂起
    SYM_FINISHED = 3  // 行索引/元数据已全部发布，只读
};

// sym_done: 0 = 当前可以运行/恢复; 1 = 等待一个 predecessor 的发布。
// sym_state: 用来区分 NEW/READY/RUNNING/WAITING/FINISHED。
// wait_head[j]: EMPTY=-1, CLOSED=-2, 或等待在 j 上的目标列链表头。
// wait_next[i]: 目标列 i 在一个 wait_head 链上的单链指针。
```

初始：**所有** `sym_done[i]=0`, `sym_state[i]=SYM_READY`, `wait_head[i]=EMPTY`，将每列各入队一次。初始就绪不等于所有前驱都完成；一旦发现 `j < i` 但 j 未完成，本列必须挂起。完成的列永远是 `SYM_FINISHED`，不再排队。

### 4.2 无丢失唤醒协议（需要严格实现）

在等待列 `i` 发现前驱 `j` 未发布时，thread 0：

1. 先写好 bitmap 的 checkpoint、`cursor[i]`、chunk metadata；执行可用的 **device-scope release fence**。
2. 设置 `sym_done[i]=1`、`sym_state[i]=SYM_WAITING`。
3. 以 atomic CAS 将 `i` 加入 `wait_head[j]`。若观察到 `CLOSED`，说明 `j` 刚好完成：取消挂起，恢复 `done=0` / `RUNNING`，立即继续当前本地 bitmap；**此时不得把任务同时入队**。
4. CAS 成功后停止处理 `i`，从 worker 主循环取其他 ready 列。

在列 `j` 完成时，thread 0：

1. 等所有线程写好 final rows 和 `final_start[j]/col_len[j]/diag_rel[j]`；跨 CTA 的发布使用 **release**。
2. 将 `sym_state[j]` 设置为 `SYM_FINISHED`，之后原子交换 `wait_head[j]` 到 `CLOSED`，原子取出旧等待链表。
3. 遍历取出的每个目标列 `i`，先令 `sym_done[i]=0`、`sym_state[i]=SYM_READY`，然后以 **release enqueue** 入 ready queue。一个 WAITING 列只允许被唤醒并入队一次。
4. 最后把 `finished_columns` 增加 1。

消费者在读取 FINISHED 状态或弹出队列时使用匹配的 **acquire**，确保不会读到尚未初始化的前驱稀疏区间或 checkpoint。单纯 `__sync_threads()` 只保障同 block，不能代替跨 CTA fence。**不可认为 `volatile` 足以解决跨 block 内存顺序。**

> **移植约束**：现有源码是 `hc_runtime.h` / HTHPCC 风格，不应直接假定 `cuda::atomic_ref`、`__threadfence`、`__popc`、64-bit CAS、某个 C++ memory-order intrinsic 在当前编译器均可用。Codex 应先检查仓库/工具链，封装 `atomic_load_acquire`、`atomic_store_release`、`device_fence_release`、`atomic_cas`、`popcount32`、`find_first_set`；针对该 runtime 实测通过后再用。必要时换成 kernel 边界同步的保守实现。**不允许用未定义的数据竞争换取表面性能。**

### 4.3 ready queue：优先持久化 worker + 事件唤醒

默认实现采用一个设备侧 **有界 MPMC ready queue**（或仓库中已有并经过并发测试的等价 work queue）：

- 启动前一次性投递 `0 ... n-1` 的列任务；每列在 READY 时最多存在一个 queue entry。
- 建议 ring capacity 至少 `next_power_of_two(n + W + 1)`（前提是主机/设备索引范围许可）；实现 per-slot sequence number 的并发 ring 协议，含 acquire/release 和 wrap-around 处理。不要编写没有序列号保护的简单 `atomicAdd(tail)` + `queue[tail%cap]` 版本，那会让消费者读到未写好的 slot。也不可用存在 ABA 风险的无版本 Treiber 栈。
- kernel 启动 `W` 个 worker blocks，thread 0 取任务，协作处理；取不到任务时短暂有界退避，观察 `finished_columns == n` 或 `error_code != 0` 后统一退出。
- 不依赖 blockIdx.x 的调度顺序；不依赖所有 W 个 block 同时 resident；不让高编号列占住 block 等待尚未调度的低编号列。每次等待都会 park 并释放 worker。
- `j < i` 的等待关系严格递减，最终总有最小尚未完成列可执行；在无队列丢失/内存错误的条件下不会形成依赖环。
- 需要设备侧 error code、queue overflow、arena overflow 检查。出现不可恢复错误时所有 worker 均应可在有限时间退出，并由 host 汇报，不能挂死 `hcDeviceSynchronize()`。

**保守兼容回退**：若当前 HTHPCC 不能可靠实现跨 block CAS 或 MPMC queue，先实现**有 kernel 边界的非阻塞 round scheduler**：一轮读 `ready[]`，挂起时记录阻塞前驱并退出本轮，发布者将被唤醒任务写进 `next_ready[]`，host 同步后反复 launch，直到所有列完成。可额外用 wait 链实现事件唤醒，避免每轮盲目重试全部列。必须仍采用 bitmap + checkpoint 增量算法；该模式用于验证正确性和兼容性，不作为高性能默认交付，且在性能报告中区分。

---

## 5. 输出：解决 GPU 计算完成顺序与 CSC 列顺序不一致

**绝对不要**直接对全局 `sym_c_ptr[i+1]` 在多个无序完成的 blocks 中做 `atomicAdd` 来生成最终 CSC，也不要让每列在独立、未知容量的输出空间越界写。正确策略是：

### 5.1 先写并发完成顺序的只读 final arena

按列维护：

```cpp
uint32_t* final_start;   // n；该列在 final_pool 中的区间起点
uint32_t* col_len;       // n；该列结构长度（包括结构对角）
uint32_t* diag_rel;      // n；对角在该列内部偏移
uint32_t* final_pool;    // 按实际填充结构非零增长的行索引存储
uint64_t* final_used;    // 若 runtime 可支持 64-bit CAS，或按受限 uint32_t 实现
```

列 `i` 的所有 predecessor 都处理完后：

1. 统计 bitmap 中 set bits 总数 `len`（含对角）。
2. 从 pool **原子预留连续 `len` 个 unsigned**。预留必须检查 `len <= capacity - used`；任何越界/溢出设置 `SYM_ERR_POOL_OVERFLOW` 并安全退出，不可写越界。可以 CAS 循环实现安全 reserve；不能先 atomicAdd 再希望越界回滚。
3. 逐 32-bit word 做 popcount、前缀偏移；按 word 从低位到高位枚举行号，block 线程将其写入 `final_pool[start : start+len]`，保证严格升序。
4. 统计 `diag_rel = count{r in S_i | r < i}`；检查 bitmap bit `i` 一定为 1。
5. 写 `final_start[i]` / `col_len[i]` / `diag_rel[i]`，release 后发布 `SYM_FINISHED`。

这样前驱 `j` 被后续列读取时使用 `final_pool[final_start[j] + diag_rel[j] : final_start[j]+col_len[j]]`，恰好对应 CPU `sym_r_idx[l_col_ptr[j] : sym_c_ptr[j+1]]`。

### 5.2 最后一次 scan + pack

所有列完成后，执行：

```cpp
sym_c_ptr[0] = 0;
sym_c_ptr[i+1] = sym_c_ptr[i] + col_len[i]; // exclusive/inclusive scan
l_col_ptr[i] = sym_c_ptr[i] + diag_rel[i];
for p in [0, col_len[i]):
    sym_r_idx[sym_c_ptr[i] + p] = final_pool[final_start[i] + p];
```

实现并行 prefix-scan 与 pack（可复用项目现有 scan；不能引用没有实际编译依赖的第三方库）。若项目没有可靠 GPU scan，首版可将 `col_len` D2H 后 CPU prefix sum，再 H2D `sym_c_ptr` 进行并行 pack；**必须在计时里记录这部分传输/开销**，不能把 CPU scan 伪称纯 GPU。为可扩展性，随后补齐 GPU scan。`sum(col_len)` 可能超过现有 `unsigned` 的表达上限，必须检查并报错或 CPU fallback；不要静默溢出。

**兼容点**：最终 `l_col_ptr` 是全局索引，不是 `diag_rel`。验证 `sym_c_ptr.size()==n+1`、`sym_r_idx.size()==nnz`、每列 `begin <= diag < end`、`sym_r_idx[diag] == i`。

---

## 6. 显存预算、性能优化与故障恢复

### 6.1 内存预算是核心要求

必须记录并监控：

```text
raw CSC inputs:       (n+1)*sizeof(unsigned) + nnzA*sizeof(unsigned)
worker bitmap:        W*ceil(n/32)*sizeof(uint32_t)
worker dirty bitmap:  W*ceil(n/32)*sizeof(uint32_t)
final arena:          current symbolic nnz * sizeof(unsigned), bounded capacity
checkpoint arena:     at most one stored copy of each newly discovered row per column
metadata:             O(n + number_of_nonempty_delta_chunks)
packed CSC:           (n+1 + nnzSymbolic + n) * sizeof(unsigned)
ready queue:          O(n + W)
```

**不允许**默认配置 `n*n*sizeof(REAL)` 或 `n*n*sizeof(unsigned)` 的 global-memory 中间矩阵。复杂度上，真实 fill 本身可能趋近 `O(n²)`，若输出结构确实致密，占用随真实 `nnz(L+U)` 上升是合理的；但 workspace 不能提前按 dense matrix 预分配。

### 6.2 pool 容量与 oom 处理

由于 symbolic `nnz` 事先未知，初始容量设成**可配置、预算内**的大小，可参考 `nnzA` 的若干倍作为启发式，但**不许把这个倍数当作正确性的上界**：

- 依设备实际分配能力做 `hcMalloc` 探测并缩容，复用 numeric 的现有做法。
- `final_pool`、`checkpoint_pool`、chunk metadata 各自检查容量，避免只检查某个 pool。
- 超额时 kernel 将错误码写入 `error_code` 并停止，host 可释放后**增大 pool 并从零重跑整个 symbolic**；禁止在已有前驱/状态被部分发布时直接偷偷重分配并继续使用旧 device pointer。
- 内存无法满足时可调用 CPU `fill_in` fallback，且日志显式写 `GPU symbolic fallback: ...`。fallback 不等于 GPU 性能成功。
- 所有 `size_t` 乘法、pool offset 和 `nnz` 聚合做 overflow 检查；如果本项目 CSC 用 32-bit `unsigned`，明确拒绝不能表示的结果。

### 6.3 必须实施的结构访问优化

1. **只在原始 CSC 初始化时读 `ai/ap`**。后续访问统一指向已完成 predecessor 的 packed `final_pool` 区间；不要在每次推进时扫整个全局 `ai`。
2. 候选前驱从 bitmap set bits 增量发现，用 `cursor` 记录处理位置；**禁止每次从行 0 开始重搜**。
3. checkpoint 挂起后不重复遍历已经处理过的 predecessor `L[j]`。
4. warp/wave 内连续加载 `L[j]` 的 row indices，批量位图 OR；让 thread 0 只承担控制，不负责串行遍历整个 predecessor 的 L。
5. Bitmap `word` 读、popcount、导出采用块内扫描/分块，避免 per-nonzero 的一条线程串行插入全局有序数组。
6. 根据 `n`、SM/MP 数、每 worker 位图内存和 pool 预算调 `W`；跟 numeric 一样遵守 `maxThreadsPerBlock` 和 **64 lanes/wave**。先以 `min(256, maxWaveAlignedThreads)` 为起点，不为扩大 W 牺牲关键路径内存吞吐。
7. 记录 global memory 访问量/代理计数（见测试），并比较传统图搜索路径与 CPU/新 GPU SF；不能只展示 GPU kernel 内的单段短时间。

### 6.4 额外优化应在正确实现后逐项上

按 profile 决定是否进行：对相同 bitmap word 的批量 OR/warp 聚合、shared-memory staging 小 bitmap、针对短列的单 wave 快路径、设备内 scan 优化、持久化 queue 批量 pop/push、bitmap 清零的高带宽内核。**不应**在首版引入不受控的 graph reachability、全局排序或依赖锁自旋。

---

## 7. 建议数据结构与文件改造范围

### 7.1 设备侧上下文（名称可按项目代码规范调整）

```cpp
struct SymbolicSFDeviceContext {
    const unsigned *ap, *ai;
    unsigned n, nnz_input;

    unsigned *sym_done;               // 0=can start/resume, 1=waiting
    unsigned *sym_state;              // READY/RUNNING/WAITING/FINISHED
    unsigned *sym_cursor;             // earliest candidate j not yet processed
    int      *wait_head, *wait_next;  // closed-list wake protocol
    unsigned *finished_columns;
    unsigned *error_code;

    uint32_t *work_bitmap, *dirty_bitmap;
    unsigned bitmap_words, workers;

    unsigned *checkpoint_head;
    unsigned *checkpoint_pool;
    unsigned *chunk_start, *chunk_len;
    int      *chunk_next;
    // checkpoint arena / metadata reservation counters and capacities

    unsigned *final_pool;
    unsigned *final_start, *col_len, *diag_rel;
    // final arena counter and capacity

    // Bounded correct MPMC ready queue:
    // queue slots, per-slot sequence, head/tail counters, capacity
};
```

实际实现请使用合理的类型宽度、const、restrict、地址空间限定、构造/析构包装和错误枚举。核实 HTHPCC 对 64-bit atomics 支持，不支持时选择经过验证的设备端受限索引方案和清晰的规模上限。当前上下文只是说明边界，**不要求原样复制成为 ABI**。

### 7.2 建议文件

- `symbolic_sf.cuh`：device worker stage helpers、bitmap 操作、同步/状态/waiter 协议、kernel 声明。
- `symbolic_gpu.cu` 或仓库对应文件：device kernel + `SymbolicFactorizationOnDevice(...)` host 实现，分配/初始化/调度/scan/pack/拷贝/cleanup。
- `symbolic.h`：新增**明确 opt-in** GPU symbolic 的接口/配置（保持现有 `fill_in` CPU 接口）；如项目已有策略 enum，沿用而不是再造命令行系统。
- `symbolic.cc`：保留原 CPU `fill_in`、数据生命周期；仅在确有必要时抽取 `reset_derived_structure()`，让 CPU/GPU 成功提交后使用一致清理逻辑。
- 原调用入口/构建配置：把 GPU symbolic 放在 **`csr/predictLU/leveling` 之前**，保留回退和测试选项；补齐所需 `.cu/.cuh` 构建引用。
- `tests/`：新增 deterministic pattern 对比、随机矩阵/property tests、sanitizer 或并发竞态测试；性能 benchmark 脚本。

主机入口建议 `bool SymbolicFactorizationOnDevice(Symbolic_Matrix&, const unsigned* ap, const unsigned* ai, std::ostream&, std::ostream&, const SymOptions&)`，但**必须以仓库实际输入方向与现有类 API 为准**：`fill_in` 的参数顺序为 `(ai, ap)`，容易在新包装层搞反。

### 7.3 对 `Symbolic_Matrix` 的事务式提交

- 新 GPU 计算先将结果放进临时 CPU vectors（或临时设备对象），在结构完整且验证通过后，再替换类成员。
- 替换前清理旧 `csr_r_ptr/csr_c_idx/csr_diag_ptr/val/level_ptr/level_idx/num_lev` 以及 Debug 派生缓存，避免保留上次分解的历史数据；新 `nnz` 由 `sym_c_ptr.back()` 设置。
- 如果 GPU 初始化/内存/内核/传输失败，CPU `Symbolic_Matrix` 保持原状态，或**显式**执行 CPU `fill_in` 完整重建，不能只有一部分成员被覆盖。
- 后续 `csr() -> predictLU(ai, ap, ax) -> leveling()` 保持既有项目顺序/条件，不要混入 GPU 数值计算的变化。

---

## 8. 竞态安全与并发单测（必须先于性能调优）

逐项验证以下并发场景：

1. **发布与读取**：列 j 的 `final_pool` 以及 `final_start/len/diag_rel` 在 `SYM_FINISHED` 变为可见前已完成；任意读 j 的 CTA 必须 acquire。
2. **等待登记与发布同时发生**：waiter 的 CAS 与 publisher 的 atomicExchange(`wait_head[j]`, CLOSED) 交错不会漏 wakeup，且不会重复 enqueue。
3. **多 waiter**：多个目标列同时等 j，publisher 完整遍历 waiter list；`wait_next` 的每个 i 在任何时刻只归属于一个前驱列表。
4. **不同 worker 复用 bitmap**：上一任务的 `work_bitmap/dirty_bitmap` 已完全结束，清零与新任务装载有块内同步；挂起前 checkpoint 已写完。
5. **同列不重复运行**：取任务后状态 CAS `READY -> RUNNING` 成功才允许处理，`SYM_FINISHED` 无重复入队。
6. **未知总 fill**：各 arena append 检查容量与计数器 overflow，不存在原子预留失败之后的越界写。
7. **正确退出**：当 n 列都完成时所有 worker 退出；当任何错误标记设置时所有 worker 退出并返回 host，不能永久忙等。
8. **跨 stream / 默认 stream 的初始化和使用顺序**：匹配项目 HTHPCC 规范，确认设备状态的 init、worker、pack、D2H 顺序。
9. **波宽与屏障**：`__sync_threads` 必须由 block 全部 active 线程抵达；thread 0 的分支决定共享到 `__shared__` 后再协作处理。

设备支持时运行 racecheck / sanitizer；不支持则构造大量随机小矩阵、重复 1,000 次并发调度、使用较大/较小 worker 数改变竞态时序，做到输出每次都与 golden 完全一致。

---

## 9. CPU Golden Reference、回归测试与性能验收

### 9.1 正确性测试（比较**逐元素相同**，不是只比较 nnz）

先在不修改原始 CSC 的前提下运行 CPU `fill_in` 得到 `gold`；GPU 结果要求：

```text
GPU.n == CPU.n
GPU.nnz == CPU.nnz
GPU.sym_c_ptr == CPU.sym_c_ptr
GPU.sym_r_idx == CPU.sym_r_idx
GPU.l_col_ptr == CPU.l_col_ptr
```

并逐列 assert：

```text
0 <= sym_c_ptr[i] < sym_c_ptr[i+1] <= nnz  (n>0)
sym_r_idx[sym_c_ptr[i] ... sym_c_ptr[i+1]) 严格递增
sym_r_idx[l_col_ptr[i]] == i
sym_c_ptr[i] <= l_col_ptr[i] < sym_c_ptr[i+1]
0 <= sym_r_idx[p] < n
```

以下 test matrix 均需覆盖：

- `n=0`（返回 `sym_c_ptr={0}`）、`n=1`、全空列、缺失结构对角、全对角矩阵。
- 纯上三角、纯下三角、带状矩阵、块对角、随机稀疏有向结构、接近 dense 的 fill。
- **动态新增前驱**：如列 `i=4` 原始只含行 `{0,4}`，列 `0` 的 L 带来行 `1`，列 `1` 的 L 再带来行 `3`；验证列 4 会继续合并新出现的 `1` 和 `3`，不是只扫描原始 `A(:,4)` 的前驱。
- 一个 predecessor 同时被大量列依赖；所有等待者在其发布后唤醒。
- 典型死锁触发器：worker 数为 1、2、`2 * MP`，以及 worker 数大于当前可同时 resident CTA 数；结果都必须完成。
- 大矩阵 `n` 但小 `nnzA`，验证内存与工作量随真实结构增长而不是全矩阵 values。
- 最小 pool 容量触发增长/重试、设备分配失败、队列边界、非法 row index、重复输入索引（按工程约定校验、规范化或 fallback）。
- 连续两次对同一 `Symbolic_Matrix` 使用**不同** pattern，确认缓存、`nnz`、CSR、val 和 levels 都被正确失效。

跑完 symbolic，对同样输入继续运行原 `csr()`、`predictLU()`、`leveling()` 和 numeric LU，验证其输入布局正确以及最终 LU/solve 结果与原 CPU-symbolic 版本相容；不能仅证明单独的 bitmap 测试通过。

### 9.2 数值验收

Symbolic 新旧结果完全一致后，用已有 numeric LU 进行端到端误差回归，包括存在 fill、静态 pivot 扰动（如项目支持）、正反三角求解、重复 pattern 的正常计算。浮点结果按既有 numeric 测试容差比较；**符号结构本身必须精确匹配，不采用容差**。

### 9.3 性能打点与实验格式

至少输出：

```text
symbolic_cpu_ms
symbolic_gpu_total_ms                # alloc/init + H2D + kernel + pack + D2H
symbolic_gpu_compute_ms              # only worker kernel, synchronized
symbolic_gpu_pack_ms
symbolic_gpu_h2d_ms / d2h_ms
symbolic_workers
symbolic_input_nnz / symbolic_output_nnz / fill_ratio
symbolic_bitmap_bytes
symbolic_checkpoint_pool_used / capacity
symbolic_final_pool_used / capacity
symbolic_task_parks / wakeups
symbolic_pred_merges
symbolic_predecessor_rows_read       # 主体 global-memory 稀疏读取量代理
symbolic_duplicate_rows_eliminated   # 位图 OR 已置位的条目数
symbolic_oom_retries / cpu_fallback_count
```

建议的表格列：`matrix, n, nnzA, nnzSymbolic, cpu_ms, old_gpu_graph_ms, new_gpu_kernel_ms, new_gpu_e2e_ms, speedup_e2e, workers, peak_workspace_MB, pred_rows_read, parks`。

- 对比相同设备、同一编译优化、同一个输入/预处理条件；预热后多次运行，报告 median 与 P90 或 stddev。
- 如老版 GPU 图搜索代码仍可运行，则同表比较，证明新方案相较旧版降低重复全局访存；若无法运行，报告原因，不能编造加速比。
- 观察瓶颈是否从图搜索随机读变成 bitmap 原子 OR、checkpoint reload 或 queue 自旋，随后针对实测热点优化。
- **不得设定无数据支持的强制固定加速比**；通过标准是结构正确、可复现、峰值内存有界、性能与成本透明，并针对 regressions 给出原因和可执行改进。

---

## 10. 推荐开发顺序与提交节点

按以下顺序实际落地，不要在初版一次塞进不易调试的复杂优化：

**提交 A｜基线与接口**  
阅读仓库的 `left_looking_sf.cuh`、运行时原子操作、symbolic 调用路径；记录原 numeric 与原 symbolic 接口。保存 CPU golden test cases。确定 compile target、显存上限、64-lane wave。

**提交 B｜单 worker 的 bitmap CPU 等价算法**  
先以 **W=1 的 GPU worker** 完成 `load -> find-next-predecessor -> union L -> finalize`，用发布 arena 和 `pack CSC` 证明结构正确。W=1 不依赖等待链的并发正确性，是调试 bitmap/排序/CSC 对角位置的稳定基线。

**提交 C｜多 worker + `done` + 非阻塞等待唤醒**  
引入 `sym_done/sym_state`、ready queue、`wait_head/wait_next`、checkpoint delta、release/acquire 发布；在 W=1、2、多 W 的场景跑端到端一致性与并发回归。优先保证 device kernel 不挂起。

**提交 D｜资源/错误管理与 CPU fallback**  
补齐 overflow、预留、安全退出、重试、HTHPCC error checks、cleanup、重复调用数据生命周期；接入原工程配置。用户原有 CPU symbolic 仍可显式选用。

**提交 E｜性能优化与文档**  
计时、profile、GPU scan（若可用）、wave 协作 bitmap 并集、hotspot 优化；输出性能对比及测试命令。

### 代码交付时必须回答

1. 哪些文件、接口、选项被修改？如何编译并启用 GPU symbolic？
2. `thread 0` 对每列 `sym_done` 的读/写点在哪里？列未完成 predecessor 时如何让出 worker？
3. 如何保证读取 `L[j]` 时 j 已完成、没有跨 CTA 竞态？如何避免 deadlock / lost wakeup？
4. 如何防止动态 `sym_r_idx` 的并发写入冲突，以及如何从 final arena pack 成 CSC？
5. 如何保证没有 `n×n` dense temporary 与重复 graph searching？pool 满时怎么办？
6. CPU/GPU symbolic `sym_c_ptr/sym_r_idx/l_col_ptr` 是否在所有测试上逐元素相同？
7. 列出实测的时间、workspace 和 counters；若新 GPU 仍比 CPU 慢，指出造成慢的主要 kernel/访存原因并说明下一步优化点。

---

## 11. 重要实现陷阱（绝不允许忽略）

- `symbolic.cc` 合并前驱时起点是 **`l_col_ptr[j]`（含对角）**，不能错误改为 `l_col_ptr[j] + 1`。
- 不能只合并原始输入中 `<i` 的行：**合并后新出现的 `<i` 行仍是前驱**。
- `sym_done[i] == 0` 在本方案中表示 **runnable**，并不表示列 i 已完成；真正读取前驱时必须检查 `SYM_FINISHED`。
- thread 0 只是控制者；不能让一个 thread 0 逐元素完成所有 union、全部 bitmap 枚举，造成新的串行瓶颈。
- **不要让一个 block 长时间对 `done[j]` 原地自旋**；这在 block 调度顺序不确定、CTA 非全量 resident 时可以引发 GPU 死锁。
- 不能把 `__sync_threads` 当作跨 block 同步；不能把 `volatile` 当作 release/acquire。
- 不可直接在 unordered completion 阶段写入最终 CSC `sym_c_ptr`；先写每列 immutable final arena，再 scan + pack。
- 不可在挂起后重复执行以前合并过的所有 `L[j]`，否则可能抵消性能收益。
- 不可给符号分解传入数值 `val` workspace 作为结构缓存；本优化目的恰恰是减少访问 dense numeric global memory。
- 不可仅因为同名 `done_dev` 就假设 numeric 和 symbolic 的位含义一致；确认 `left_looking_sf.cuh` 后写明实际语义。
- 若使用 `uint64_t` 原子计数器和 device fence，必须确认当前 HTHPCC 编译工具链支持；不支持要提供清晰的降级实现和测试证据。
- 如果输入存在重复行：GPU bitset 会去重，而 CPU 原 `sort` 并没有显式 `unique`，必须统一输入合法性前提或 canonicalize，不可让差异悄悄传播到 numerical。
- 必须处理 `n=0`、空输入列及缺对角的结构插入；不能对空 vector 取 `&vec[0]`。

---

## 12. Codex 最终完成判据

**Definition of Done：** 项目能够在原有构建系统下编译运行，GPU symbolic 新路径通过逐元素 CPU golden 对比，接入现有 `csr/predictLU/leveling/numerical LU` 无回归；`thread0 + done=0 runnable` 的非阻塞列级调度、稀疏 `L[j]` 逐前驱合并、bitmap/增量 checkpoint、arena publish、scan/pack 均已实现；死锁/竞态/边界/OOM/资源清理有自动化覆盖；日志有 compute/end-to-end/内存/关键稀疏访问 counters 和可复现 benchmark 命令。请直接给出代码 diff、运行测试和结果，不要只回复设计思路。

**如实现环境缺少编译器或 GPU**：至少提供完整改动与静态可检查的构建路径，运行 CPU reference 和纯 host 端算法模拟/小规模单测；清晰标注未能执行的 GPU 编译/运行/性能测试，不得宣称已经 GPU 验证成功。
