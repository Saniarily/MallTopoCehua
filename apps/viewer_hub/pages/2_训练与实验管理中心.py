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
    kind = st.selectbox("任务类型", ["train_stage1", "ablation", "evaluate_stage1", "train_stage2", "evaluate_stage2", "evaluate_stage3", "renovate", "floor_plates", "figures", "custom"],
                        format_func=lambda k: {"train_stage1": "阶段一 训练排序器", "ablation": "阶段一 消融 / 模型对比（多 seed）", "evaluate_stage1": "阶段一 评估 checkpoint", "train_stage2": "阶段二 训练 AR-GNN",
                                               "evaluate_stage2": "阶段二 评估生成器", "evaluate_stage3": "阶段三 走廊自适应评估", "renovate": "阶段三 旧商场改造批量实验", "floor_plates": "导出全部真实平面图（色块+拓扑 / 轮廓+拓扑 / 缩略图）", "figures": "生成论文图表", "custom": "自定义命令"}[k])
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
    elif kind == "floor_plates":
        args["config"] = st.selectbox("数据配置", list_configs("data"), index=list_configs("data").index("configs/data/legacy.yaml") if "configs/data/legacy.yaml" in list_configs("data") else 0)
        args["out"] = st.text_input("输出目录", "outputs/floor_plates")
        args["workers"] = st.number_input("并行进程", 1, 16, 4)
        args["limit"] = st.number_input("楼层上限（0 = 全部）", 0, 5000, 0) or None
        args["force"] = st.checkbox("覆盖已有图片", False)
        st.caption("生成 plan_topo/（功能色块平面 + 全部 M 节点拓扑）、outline_topo/（轮廓 + 拓扑）、thumbs/（360 px 缩略图，改造页直接读取）与 floors.csv；可断点续跑。")
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
        if sel:
            st.markdown("---")
            recs = {n: reg.get(n) for n in sel}
            png_sets = {n: {p.name: p for p in sorted(r.path.glob("*.png"))} for n, r in recs.items() if r is not None}
            png_sets = {n: v for n, v in png_sets.items() if v}
            if png_sets:
                mode = st.radio("结果图", ["按实验浏览", "同一楼层跨批次对比"], horizontal=True, key="res_img_mode")
                if mode == "按实验浏览":
                    which = st.selectbox("实验", list(png_sets), index=list(png_sets).index(one) if one in png_sets else 0, key="res_img_exp")
                    pngs = list(png_sets[which].values())
                    st.markdown(f"**{which}**（{len(pngs)} 张）")
                    g1, g2, g3 = st.columns([1, 1, 3])
                    per_page = g1.selectbox("每页", [6, 12, 24, 48], index=1)
                    n_pages = max(1, (len(pngs) + per_page - 1) // per_page)
                    page = g2.number_input("页", 1, n_pages, 1) if n_pages > 1 else 1
                    pick = g3.selectbox("或直接查看单张", ["（全部）", *[p.name for p in pngs]])
                    if pick != "（全部）":
                        p1 = png_sets[which][pick]
                        st.image(str(p1), caption=f"{which} / {p1.name}", width="stretch")
                        st.download_button("下载", p1.read_bytes(), f"{which.replace('/', '_')}_{p1.name}", "image/png", key=f"dlpng_{which}_{p1.name}")
                    else:
                        ncol = 2 if per_page <= 6 else 3
                        sub = pngs[(page - 1) * per_page: page * per_page]
                        for r0 in range(0, len(sub), ncol):
                            cols = st.columns(ncol)
                            for col, p1 in zip(cols, sub[r0:r0 + ncol]):
                                col.image(str(p1), caption=p1.name, width="stretch")
                else:
                    common = sorted(set.intersection(*[set(v) for v in png_sets.values()])) if len(png_sets) > 1 else sorted(next(iter(png_sets.values())))
                    if not common:
                        st.info("所选实验之间没有同名结果图（不同楼层集合）。")
                    else:
                        st.caption(f"{len(common)} 个楼层在所选 {len(png_sets)} 个实验中都有结果图；每行一个楼层，每列一个实验（按上方「对比实验」顺序）")
                        fl = st.selectbox("楼层图", ["（逐页浏览）", *common], key="res_cmp_floor")
                        show = [fl] if fl != "（逐页浏览）" else None
                        if show is None:
                            per_page = st.selectbox("每页楼层数", [3, 5, 10], index=0, key="res_cmp_pp")
                            n_pages = max(1, (len(common) + per_page - 1) // per_page)
                            page = st.number_input("页", 1, n_pages, 1, key="res_cmp_page") if n_pages > 1 else 1
                            show = common[(page - 1) * per_page: page * per_page]
                        for name in show:
                            st.markdown(f"**{name}**")
                            cols = st.columns(len(png_sets))
                            for col, (exp, files) in zip(cols, png_sets.items()):
                                col.image(str(files[name]), caption=exp, width="stretch")

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
