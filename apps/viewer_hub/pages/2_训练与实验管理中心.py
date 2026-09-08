"""UI 2 – training & experiment centre (background jobs via JobRunner, results via ExperimentRegistry)."""
from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from _common import ROOT, get_jobs, get_registry, list_configs, setup_page

from mall_space_planner.hub.experiments import ExperimentRegistry
from mall_space_planner.hub.jobs import JobSpec

setup_page("2 · 训练与实验管理中心", "🧪")
jobs = get_jobs()

tab_new, tab_jobs, tab_results, tab_ckpt, tab_export = st.tabs(["新建任务", "任务队列 / 日志 / 曲线", "实验结果与对比", "Checkpoints", "导出论文表格"])

# --------------------------------------------------------------------------------------------------------- new job
with tab_new:
    st.markdown("所有任务以 **后台子进程** 运行（`outputs/jobs/<id>/log.txt`），关闭页面不会中断；可在下一页查看状态、日志、曲线，或停止。")
    kind = st.selectbox("任务类型", ["train_stage1", "ablation", "evaluate_stage1", "train_stage2", "evaluate_stage2", "evaluate_stage3", "renovate", "figures", "custom"],
                        format_func=lambda k: {"train_stage1": "阶段一 训练排序器", "ablation": "阶段一 消融 / 模型对比（多 seed）", "evaluate_stage1": "阶段一 评估 checkpoint", "train_stage2": "阶段二 训练 AR-GNN",
                                               "evaluate_stage2": "阶段二 评估生成器", "evaluate_stage3": "阶段三 走廊自适应评估", "renovate": "阶段三 旧商场改造批量实验", "figures": "生成论文图表", "custom": "自定义命令"}[k])
    args: dict = {}
    label = st.text_input("任务标签", value=kind)
    ov_text = ""
    if kind in ("train_stage1", "evaluate_stage1"):
        args["config"] = st.selectbox("配置", list_configs("stage1"), index=max(0, list_configs("stage1").index("configs/stage1/extra_trees.yaml")) if "configs/stage1/extra_trees.yaml" in list_configs("stage1") else 0)
        if kind == "train_stage1":
            args["seed"] = st.number_input("seed", 0, 10_000, 42)
        else:
            cks = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "outputs/experiments/stage1").glob("*/checkpoint")) if (ROOT / "outputs/experiments/stage1").exists() else []
            args["checkpoint"] = st.selectbox("checkpoint 目录", cks) if cks else st.text_input("checkpoint 目录")
        ov_text = st.text_area("参数覆盖（每行 key=value，例：stage1.ranker.params.n_estimators=300）", height=80)
    elif kind == "ablation":
        args["config"] = st.selectbox("消融配置", list_configs("ablations"))
        args["out_dir"] = st.text_input("输出目录", "outputs/experiments/ablation_ui")
        ov_text = st.text_area("参数覆盖（每行 key=value，例：seeds=[42,43]）", height=80)
    elif kind == "train_stage2":
        s2 = [c for c in list_configs("stage2") if "ar_gnn" in c]
        args["config"] = st.selectbox("配置", s2)
        args["corpus"] = st.text_input("语料路径（留空用配置默认）", "") or None
        args["limit"] = st.number_input("样本上限（0 = 全部）", 0, 100_000, 0) or None
        args["force"] = st.checkbox("强制重建语料", False)
        ov_text = st.text_area("参数覆盖（每行 key=value，例：generator.params.epochs=20  generator.params.d_model=96）", height=80)
    elif kind == "evaluate_stage2":
        args["config"] = st.selectbox("配置", list_configs("stage2"))
        args["limit"] = st.number_input("样本数", 1, 5000, 200)
        args["seed"] = st.number_input("seed", 0, 10_000, 0)
        args["ground_truth"] = st.checkbox("含真值参考 (ref_ground_truth)", True)
        args["corpus"] = st.text_input("语料路径（留空用配置默认）", "") or None
    elif kind == "evaluate_stage3":
        args["config"] = st.selectbox("数据配置", list_configs("data"), index=list_configs("data").index("configs/data/legacy.yaml") if "configs/data/legacy.yaml" in list_configs("data") else 0)
        args["split"] = st.selectbox("划分", ["test", "val", "train"])
        args["limit"] = st.number_input("楼层数", 1, 2000, 50)
        args["out"] = st.text_input("输出目录", "outputs/experiments/stage3_eval_ui")
        args["resume"] = st.checkbox("断点续跑", True)
    elif kind == "renovate":
        args["config"] = st.selectbox("数据配置", list_configs("data"), index=list_configs("data").index("configs/data/legacy.yaml") if "configs/data/legacy.yaml" in list_configs("data") else 0)
        args["split"] = st.selectbox("划分", ["test", "val", "train"])
        args["limit"] = st.number_input("楼层数", 1, 2000, 40)
        ls = st.number_input("仅低分商场（评分 ≤，0 = 不限）", 0.0, 100.0, 4.5, 0.1)
        args["low_score"] = ls if ls > 0 else None
        args["no_filter"] = st.checkbox("关闭楼层筛选（面积 / 节点数 / 出入口 / 楼层）", False)
        args["out"] = st.text_input("输出目录", "outputs/experiments/renovation_ui")
    elif kind == "figures":
        only = st.text_input("只生成（空格分隔，如 F09 R15；留空 = 全部）", "")
        args["only"] = only.split() or None
    else:
        args["command"] = st.text_input("命令（在仓库根目录执行，PYTHONPATH=src 已设置）", "python scripts/run_e2e.py --n-candidates 2")
    if ov_text.strip():
        args["override"] = dict(line.split("=", 1) for line in ov_text.strip().splitlines() if "=" in line)
    try:
        st.code(" ".join(JobSpec(kind, args, label).command()), language="bash")
        can = True
    except Exception as e:  # noqa: BLE001
        st.error(f"参数不完整：{e}"); can = False
    if st.button("▶ 启动后台任务", type="primary", disabled=not can):
        j = jobs.submit(JobSpec(kind, args, label))
        st.success(f"已启动 `{j.job_id}`（pid {j.pid}）— 到「任务队列」查看日志。")
        st.session_state["ui2_job"] = j.job_id

# --------------------------------------------------------------------------------------------------------- jobs
with tab_jobs:
    lst = jobs.list(100)
    if not lst:
        st.info("暂无任务。")
    else:
        tbl = pd.DataFrame([{"job_id": j.job_id, "状态": j.status, "标签": j.label, "类型": j.kind, "开始": pd.Timestamp(j.started_at, unit="s").strftime("%m-%d %H:%M:%S"),
                             "耗时 s": round((j.finished_at or time.time()) - j.started_at), "返回码": j.returncode, "pid": j.pid} for j in lst])
        st.dataframe(tbl, width="stretch", height=min(400, 40 + 35 * len(tbl)))
        ids = [j.job_id for j in lst]
        jid = st.selectbox("查看任务", ids, index=ids.index(st.session_state["ui2_job"]) if st.session_state.get("ui2_job") in ids else 0)
        j = jobs.refresh(jid)
        c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
        c1.metric("状态", j.status)
        if c2.button("⏹ 停止", disabled=j.status != "running"):
            jobs.stop(jid); time.sleep(0.5); st.rerun()
        auto = c3.toggle("自动刷新 (3 s)", value=j.status == "running")
        c4.code(" ".join(j.command), language="bash")
        n = st.slider("日志行数", 20, 2000, 200)
        log = jobs.tail(jid, n)
        if j.status == "failed":
            st.error(f"任务失败（返回码 {j.returncode}）。错误日志（末尾）：")
            err = [ln for ln in log.splitlines() if "Error" in ln or "Traceback" in ln or "error" in ln.lower()]
            if err:
                st.code("\n".join(err[-15:]), language="text")
        st.code(log or "(空)", language="text")
        curves = jobs.parse_curves(jid)
        ykeys = [k for k in curves if k != "epoch"]
        if ykeys:
            st.markdown("**训练曲线**（从日志解析 `epoch=… key=value`）")
            x = curves.get("epoch") or list(range(len(curves[ykeys[0]])))
            fig = go.Figure()
            for k in ykeys:
                fig.add_trace(go.Scatter(x=x[: len(curves[k])], y=curves[k], mode="lines+markers", name=k))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch")
            st.plotly_chart(fig, width="stretch")
        if auto and j.status == "running":
            time.sleep(3); st.rerun()

# --------------------------------------------------------------------------------------------------------- results
with tab_results:
    reg = get_registry()
    fams = reg.families()
    if not fams:
        st.info("尚无实验结果（outputs/experiments 为空）。")
    else:
        fam = st.selectbox("实验族", fams, format_func=lambda f: {"stage1": "阶段一 训练结果", "stage1_ablation": "阶段一 消融 / 模型对比", "stage2_eval": "阶段二 生成器评估", "stage2_train": "阶段二 AR-GNN 训练", "stage3_eval": "阶段三 走廊评估", "renovation": "阶段三 改造实验"}.get(f, f))
        split = st.selectbox("划分", ["test", "val"]) if fam in ("stage1", "stage1_ablation") else None
        tbl = reg.table(fam, split)
        st.dataframe(tbl.drop(columns=["path"], errors="ignore"), width="stretch")
        names = tbl["name"].tolist()
        sel = st.multiselect("对比实验", names, default=names[: min(4, len(names))])
        if sel:
            cmp = reg.compare(sel, split or "test")
            st.dataframe(cmp, width="stretch")
            num = cmp.dropna(how="all")
            metric = st.selectbox("柱状图指标", [m for m in num.index if not str(m).endswith("_std") and m not in ("seeds",)])
            fig = go.Figure(go.Bar(x=list(num.columns), y=num.loc[metric].astype(float).tolist(), error_y=dict(type="data", array=num.loc[f"{metric}_std"].astype(float).tolist()) if f"{metric}_std" in num.index else None))
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=20, b=10), yaxis_title=metric)
            st.plotly_chart(fig, width="stretch")
            one = st.selectbox("训练曲线 / 逐样本结果", sel)
            cv = reg.curves(one)
            if cv:
                figc = go.Figure()
                for k, v in cv.items():
                    figc.add_trace(go.Scatter(y=v, mode="lines+markers", name=k))
                figc.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="epoch")
                st.plotly_chart(figc, width="stretch")
            pq = reg.per_query(one, split or "test")
            if pq is not None:
                with st.expander(f"逐查询 / 逐样本结果（{len(pq)} 行）"):
                    st.dataframe(pq, width="stretch")
        rec = reg.get(sel[0]) if sel else None
        if rec is not None:
            pngs = sorted(rec.path.glob("*.png"))[:12]
            if pngs:
                st.markdown("**结果图**")
                cols = st.columns(min(3, len(pngs)))
                for i, p in enumerate(pngs):
                    cols[i % len(cols)].image(str(p), caption=p.name)

# --------------------------------------------------------------------------------------------------------- checkpoints
with tab_ckpt:
    reg = get_registry()
    ck = reg.checkpoints()
    st.dataframe(ck, width="stretch") if len(ck) else st.info("未发现 checkpoint。")
    st.caption("阶段一：outputs/experiments/stage1/<name>/checkpoint · 阶段二：outputs/checkpoints/stage2/<name> 与 data/results_snapshot/stage2/checkpoints_r5")

# --------------------------------------------------------------------------------------------------------- export
with tab_export:
    reg = get_registry()
    fam = st.selectbox("导出实验族", reg.families() or ["-"], key="exp_fam")
    split = st.selectbox("划分", ["test", "val"], key="exp_split") if fam in ("stage1", "stage1_ablation") else None
    tbl = reg.table(fam, split) if fam != "-" else pd.DataFrame()
    keep = st.multiselect("列", list(tbl.columns), default=[c for c in tbl.columns if c not in ("path", "modified", "family")])
    sub = tbl[keep] if keep else tbl
    digits = st.number_input("小数位", 0, 6, 3)
    c1, c2, c3 = st.columns(3)
    c1.download_button("CSV", sub.to_csv(index=False).encode("utf-8-sig"), f"{fam}_{split or 'all'}.csv", "text/csv")
    c2.download_button("Markdown", ExperimentRegistry.to_markdown(sub, digits).encode("utf-8"), f"{fam}_{split or 'all'}.md", "text/markdown")
    c3.download_button("LaTeX", ExperimentRegistry.to_latex(sub, digits, caption=f"{fam} ({split or 'all'})", label=f"tab:{fam}").encode("utf-8"), f"{fam}_{split or 'all'}.tex", "text/plain")
    st.code(ExperimentRegistry.to_markdown(sub, digits), language="markdown")
    st.markdown("**论文快照表格** (`data/results_snapshot`)")
    for p in reg.paper_tables():
        with st.expander(str(p.relative_to(ROOT))):
            if p.suffix == ".md":
                st.markdown(p.read_text(encoding="utf-8"))
            else:
                st.dataframe(pd.read_csv(p), width="stretch")
            st.download_button("下载", p.read_bytes(), p.name, key=f"dl_{p}")
    figs = sorted((ROOT / "outputs/figures").glob("*.png"))[:30] if (ROOT / "outputs/figures").exists() else []
    if figs:
        st.markdown("**已生成图表** (`outputs/figures`)")
        cols = st.columns(3)
        for i, p in enumerate(figs):
            cols[i % 3].image(str(p), caption=p.name)
