# Phase 5 · Viewer Hub（数据浏览 / 训练与实验中心 / 策划验证）

任务清单已按**当前算法与流程**调整：阶段一 = 类型条件化质量模型 + 类型内 Top-K 排序 + 解释 / 反事实；阶段二 = AR-GNN / 规则 best-of-16 自回归扩展（骨架 → 完整关键点网络）；阶段三 = **走廊自适应**（Tutte 嵌入 + 中轴引导松弛 + 平面性修复 + 走廊 / 出入口 / 竖向核 / 中庭渲染，**不再做店铺分区**）；并新增**旧商场改造流程**（同轮廓、只固定出入口、重新生长与重排、指标前后对比）。

## 架构（后端与界面解耦）

```
src/mall_space_planner/hub/           # 业务逻辑：UI 与 API 共用，不含任何 Streamlit 代码
  catalog.py      Catalog           案例库统计 / 缺失 / 异常 / 筛选 / 单案例拓扑与指标 / 真实网络与轮廓 / 对比
  workbench.py    Workbench         阶段一推荐 → 阶段二生成 → 阶段三走廊方案；候选摘要 / 约束满足 / 风险；改造流程；导出
  experiments.py  ExperimentRegistry 扫描 outputs/experiments、checkpoints、results_snapshot → 统一表；对比；CSV / MD / LaTeX
  jobs.py         JobRunner/JobSpec 后台子进程任务（Popen + 进程组）：提交 / 状态 / 日志 / 曲线解析 / 停止 / 失败返回码
  export.py                          候选 → JSON / GeoJSON / SVG / PNG
  viz.py                             共享 matplotlib 绘图（改造四联图亦被 scripts/renovate_stage3.py 复用）
  api.py          FastAPI            REST：/api/catalog/* /api/plan/* /api/renovation/* /api/jobs/* /api/experiments/*
apps/viewer_hub/                      # Streamlit 界面（仅展示与交互）
  Home.py  _common.py  pages/1_数据与拓扑案例浏览器.py  pages/2_训练与实验管理中心.py  pages/3_商场智能策划验证系统.py
```

启动：
```bash
pip install -e ".[app]"                                  # streamlit fastapi uvicorn plotly
streamlit run apps/viewer_hub/Home.py                    # 界面（进程内直接调用 hub 对象）
PYTHONPATH=src uvicorn mall_space_planner.hub.api:app --port 8000   # 可选：REST 后端（后续 React 前端直连）
```
数据自动定位：`data/processed/legacy` 存在（Mac）→ 真实数据；否则回退合成数据并在页面顶部提示。真实轮廓 / 图 CSV 通过 `configs/data/legacy.yaml`（`legacy.local.yaml` 覆盖本机路径）。

## 任务清单（✅ 已完成 · ⏳ 待做）

### UI 1 · 数据与拓扑案例浏览器
- ✅ 数据统计（楼层 / 商场 / 划分 / 类型 / 城市簇）、字段分布直方图 + 箱线（plotly，可按类型 / 簇 / 划分分组）、评分 vs 拓扑指标散点
- ✅ 缺失值报告、异常警告（常量列、>5σ 离群、负值、评分越界（按 0–5 / 0–100 量表自适应）、同商场评分方差、非连通 / 缺失拓扑、类型强转）
- ✅ 筛选（类型 / 簇 / 划分 / 评分区间 / 面积区间 / 关键字 / 是否有图）+ CSV 下载；筛选结果联动单案例下拉
- ✅ 单案例：抽象拓扑图（骨架高亮）、**真实轮廓 + 真实位置关键点网络**（图 CSV 可达时；图 CSV ↔ 掩膜像素帧自动对齐并显示对齐参数，仍越界时给出警告）、条件字段、由图重算的拓扑指标与案例表 g_* 指标
- ✅ 多案例对比：指标表、分组柱状图、并排拓扑图、CSV 下载；audit.md / manifest 查看
- ⏳ 拓扑图交互（缩放 / 悬停节点属性）—— 需 plotly 网络图或 React 前端

### UI 2 · 训练与实验管理中心
- ✅ 任务类型：阶段一训练 / 评估、消融（多 seed）、阶段二 AR-GNN 训练 / 评估、**阶段三走廊评估**、**改造批量实验**、**导出全部真实平面图**（`export_floor_plates.py`：色块+拓扑 / 轮廓+拓扑 / 缩略图 + floors.csv）、论文图表、自定义命令；配置从 `configs/` 自动列出；参数覆盖 `key=value`
- ✅ **后台子进程**运行（关闭页面不中断），任务表（状态 / 耗时 / 返回码 / pid），实时日志（自动刷新）、**停止**（SIGTERM→SIGKILL 进程组）、**失败时提取错误行**
- ✅ 训练曲线：日志中的 `epoch=… key=value` 实时解析；已完成实验读取 checkpoint `meta.json` history（AR-GNN）或 run.json history（阶段一）
- ✅ 实验结果：按族（阶段一训练 / 消融 / 阶段二评估 / 阶段二训练 / 阶段三评估 / 改造）列表，多实验对比表 + 柱状图（含 std 误差线），逐查询 / 逐样本结果；结果图两种模式：**按实验浏览**（可切换实验，分页 / 单张 + 下载）与 **同一楼层跨批次对比**（每行一楼层、每列一实验）
- ✅ Checkpoints 列表（阶段一 joblib / 阶段二 .pt，大小、epoch、超参）
- ✅ 导出：任意族 → CSV / Markdown / LaTeX（选列、小数位）；`data/results_snapshot` 论文表格浏览下载；`outputs/figures` 图表
- ⏳ 任务队列并发上限 / 排队（当前提交即运行）；MLflow 可选接入

### UI 3 · 商场智能策划验证系统
**新建策划**
- ✅ 条件输入（手动 / 从案例复制，全部 query 字段 + 偏好类型）
- ✅ 轮廓：手绘顶点 / 上传 JSON·GeoJSON / 数据库楼层（真实掩膜）/ 按面积匹配真实轮廓 / 矩形；轮廓预览与面积
- ✅ 阶段一：类型排名（期望评分、CI、可比案例、经验均分）→ 类型内 Top-K（得分 / 置信 / 质量 / 相似度）→ 原型解释（摘要、因子、相似案例证据、拓扑推理、风险、**反事实**）+ 原型拓扑图与指标
- ✅ 约束：目标节点数、候选数、出入口上下限、中庭上限、走廊占比、目标平均连接度、seed
- ✅ 阶段二生成 → 阶段三走廊方案（主 / 次走廊、出入口、竖向核、中庭、关键点网络叠加）多候选并排
- ✅ 每候选：约束满足表（平面 / **节点与走廊均在轮廓内**（非凸 L / U 轮廓越界修复）/ 出入口 / 无内部断头 / 锐角率 / 走廊占比 / 中庭）、**阶段一预测评分**、风险提示
- ✅ **一键重新生成**（每候选独立按钮：同原型 / 轮廓 / 约束，换 seed）
- ✅ 导出：单候选 JSON / GeoJSON / SVG / PNG；全部候选 ZIP + summary.csv
- ⏳ 修改条件后自动重跑阶段一（当前需再点按钮）；画布拖拽绘制轮廓（需自定义组件）

**旧商场改造**
- ✅ **全部可达楼层**列表（评分 / 类型 / 楼层号 / 节点数 / 面积 / 出入口数，可排序）+ **现状缩略图画廊**（固定 360 px 方形画布、6 列、分页；底图 = 功能色块平面 `clean_img`（可达时），叠加全部 M 节点网络与骨架；优先读取 `outputs/floor_plates/thumbs` 预渲染集，否则即时绘制并缓存到 `outputs/cache/thumbnails`）；点「选择」直接切换下方选定楼层；推荐筛选为可选开关
- ✅ 运行：保留轮廓与出入口位置 → 骨架为原型重新生长（AR-GNN 或规则 best-of-16）→ 重排全部节点 → 走廊方案；内部多候选按改造目标（断头、回路、ASPL、连通性、预测评分）择优
- ✅ 四联图（现状真实网络 | 更新网络 | 走廊方案 | 指标表）+ 指标改善 / 变差计数 + 前后行数据；PNG / JSON 导出
- ✅ **一键换 seed 重新生成**
- ✅ 回路数为中性指标（只报告，不评判改善）
- ⏳ 同一楼层多次结果并排比较 / 收藏

### 通用
- ✅ FastAPI 全量路由（TestClient 验证 30+ 路由）；Streamlit AppTest 验证四页及全部交互流程；`tests/unit/test_hub.py`
- ⏳ React 前端（直接消费 `/api/*`）；鉴权；候选会话持久化（当前内存）

## 校验（沙盒）
```
pytest -q tests/                     -> 58 passed
Streamlit AppTest：Home / UI1 / UI2 / UI3 渲染 OK；UI3 推荐→生成 2 候选（25 s）→重新生成候选 2（12 s）OK；
改造 B000A0E928_1（27 s）→ 换 seed（37 s）OK；UI2 自定义任务提交→运行→曲线→停止 OK；UI1 对比 / 单案例 OK
FastAPI TestClient：catalog / plan(generate, regenerate, 4 种导出) / renovation(run, png) / jobs / experiments(export csv|md|tex) 全部 200
```
