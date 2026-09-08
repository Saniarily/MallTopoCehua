"""Viewer Hub – landing page.

Run from the repository root:  ``streamlit run apps/viewer_hub/Home.py``
Optional FastAPI backend:      ``PYTHONPATH=src uvicorn mall_space_planner.hub.api:app --port 8000``
"""
from __future__ import annotations

import streamlit as st

from _common import ROOT, data_banner, get_catalog, get_jobs, get_registry, setup_page

setup_page("MallTopoCehua · Viewer Hub", "🏬")

cat = get_catalog()
data_banner(cat)
ov = cat.overview()

c1, c2, c3, c4 = st.columns(4)
c1.metric("楼层案例", ov["n_floors"])
c2.metric("商场", ov["n_malls"])
c3.metric("拓扑图", ov["n_graphs"], f"缺失 {ov['n_missing_graphs']}" if ov["n_missing_graphs"] else None)
reg = get_registry()
c4.metric("已登记实验", len(reg.records))

jobs = get_jobs().list(5)
running = [j for j in jobs if j.status == "running"]
if running:
    st.warning(f"后台任务运行中：{', '.join(j.label or j.kind for j in running)}", icon="⏳")

st.markdown(
    """
### 三个界面

| 页面 | 用途 | 对应算法 / 流程 |
|---|---|---|
| **1 · 数据与拓扑案例浏览器** | 数据统计、字段分布、缺失值、筛选、拓扑图 / 轮廓 / 评分 / 拓扑指标、多案例对比、异常警告 | `CaseDatabase`、`graph_metric_row`、审计报告 |
| **2 · 训练与实验管理中心** | 选阶段 / 模型 / 配置 / 参数，后台启动训练与评估（状态、日志、停止、错误日志）、损失曲线、checkpoint、消融、实验对比、导出 CSV / MD / LaTeX / PNG | `scripts/train_*`、`evaluate_*`、`run_ablation`、`evaluate_stage3`、`renovate_stage3` 经 `JobRunner` 子进程运行 |
| **3 · 商场智能策划验证系统** | **新建**：条件 → 轮廓 → 阶段一类型与 Top-K 原型（分数、相似案例、解释、反事实）→ 约束 → 阶段二生成多候选 → 阶段三走廊方案 → 约束满足 / 评分 / 风险 → 一键重新生成 → 导出 JSON / GeoJSON / SVG / PNG。<br>**改造**：选择既有楼层 → 保留轮廓与出入口 → 重新生长与重排 → 现状 / 更新 / 走廊方案 / 指标对比 → 一键重新生成 | `PlanningService`、AR-GNN / 规则生成器、`CorridorFitter`、`render_corridors`、`renovate_floor` |

后端业务逻辑全部在 `src/mall_space_planner/hub/`（Catalog / Workbench / ExperimentRegistry / JobRunner），本界面只做展示；同一套对象亦由 FastAPI (`mall_space_planner.hub.api`) 提供 REST 接口，可替换为 React 前端。
""",
    unsafe_allow_html=True,
)

with st.expander("环境 / 路径"):
    st.code(f"repo: {ROOT}\ndata: {ov['path']}\njobs: {get_jobs().dir}", language="text")
