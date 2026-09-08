"""UI 1 – data & topology case browser."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from _common import data_banner, get_catalog, setup_page, show_fig

setup_page("1 · 数据与拓扑案例浏览器", "📚")
cat = get_catalog()
data_banner(cat)
db = cat.db
df = cat.cases
ov = cat.overview()

tab_stats, tab_filter, tab_case, tab_compare, tab_quality = st.tabs(["数据统计", "筛选案例", "单案例：拓扑 / 轮廓 / 指标", "多案例对比", "缺失值与异常"])

# ------------------------------------------------------------------------------------------------------------ stats
with tab_stats:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("楼层", ov["n_floors"]); c2.metric("商场", ov["n_malls"]); c3.metric("拓扑图", ov["n_graphs"]); c4.metric("缺失拓扑", ov["n_missing_graphs"])
    a, b, c = st.columns(3)
    with a:
        st.markdown("**数据划分（按商场分组，无泄漏）**")
        st.dataframe(pd.Series(ov["splits"], name="楼层数"), width="stretch")
    with b:
        st.markdown("**布局类型**")
        st.dataframe(pd.Series(ov["layout_types"], name="楼层数"), width="stretch")
    with c:
        st.markdown("**城市簇**")
        st.dataframe(pd.Series(ov["city_clusters"], name="楼层数"), width="stretch")

    st.markdown("**字段分布**")
    num_cols = cat.numeric_columns()
    col = st.selectbox("字段", num_cols, index=num_cols.index(db.label_col) if db.label_col in num_cols else 0)
    hue = st.selectbox("分组", [None, "layout_type", "city_cluster", "split"], format_func=lambda x: x or "（不分组）")
    fig = px.histogram(df, x=col, color=hue, nbins=30, marginal="box", barmode="overlay", opacity=0.75)
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("**评分 vs 拓扑指标**")
    gm = [c for c in db.graph_metric_cols if c in df]
    if gm:
        x = st.selectbox("横轴（拓扑指标）", gm, index=0)
        fig2 = px.scatter(df, x=x, y=db.label_col, color="layout_type" if "layout_type" in df else None, hover_name=db.id_col, trendline=None)
        fig2.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, width="stretch")
    with st.expander("字段描述统计"):
        st.dataframe(df[num_cols].describe().T, width="stretch")
    with st.expander("数据审计报告 (audit.md) / manifest"):
        st.markdown(cat.audit_markdown() or "_无 audit.md_")
        st.code(cat.manifest_json(), language="json")

# ------------------------------------------------------------------------------------------------------------ filter
with tab_filter:
    f1, f2, f3 = st.columns(3)
    lts = f1.multiselect("布局类型", sorted(df["layout_type"].dropna().unique()) if "layout_type" in df else [])
    cls = f2.multiselect("城市簇", sorted(df["city_cluster"].dropna().unique()) if "city_cluster" in df else [])
    sps = f3.multiselect("划分", sorted(df["split"].dropna().unique()) if "split" in df else [])
    g1, g2, g3 = st.columns(3)
    lo, hi = float(df[db.label_col].min()), float(df[db.label_col].max())
    sr = g1.slider("评分范围", lo, hi, (lo, hi))
    if "total_area" in df:
        alo, ahi = float(df["total_area"].min()), float(df["total_area"].max())
        ar = g2.slider("总面积范围 (m²)", alo, ahi, (alo, ahi))
    else:
        ar = None
    txt = g3.text_input("ID / 商场 关键字")
    hg = st.checkbox("仅含拓扑图", value=False)
    res = cat.filter(lts or None, cls or None, sr, ar, sps or None, True if hg else None, txt or None)
    st.caption(f"{len(res)} 条")
    show_cols = [c for c in [db.id_col, db.mall_id_col, "layout_type", "city_cluster", "split", db.label_col, "total_area", *db.graph_metric_cols[:6]] if c in res]
    st.dataframe(res[show_cols], width="stretch", height=420)
    st.download_button("下载筛选结果 CSV", res.to_csv(index=False).encode("utf-8-sig"), "cases_filtered.csv", "text/csv")
    st.session_state["ui1_filtered_ids"] = res[db.id_col].astype(str).tolist()

# ------------------------------------------------------------------------------------------------------------ case
with tab_case:
    ids = st.session_state.get("ui1_filtered_ids") or df[db.id_col].astype(str).tolist()
    fid = st.selectbox("楼层案例", ids, key="ui1_case")
    row = cat.case(fid)
    topo = cat.topology(fid)
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(f"**{fid}** · 商场 `{row.get(db.mall_id_col)}` · 类型 **{row.get('layout_type')}** · 评分 **{row.get(db.label_col)}**")
        if topo is None:
            st.warning("该案例无拓扑图")
        else:
            sk = {n for n, a in topo.node_attrs.items() if a.get("is_skeleton") or a.get("skeleton")} or None
            from mall_space_planner.hub.viz import draw_case_topology

            fig, ax = plt.subplots(figsize=(6, 5))
            draw_case_topology(ax, topo, sk, title=f"{fid}：{topo.num_nodes} 节点 / {topo.num_edges} 连接")
            show_fig(fig)
        rn = cat.real_network(fid)
        if rn is not None:
            from mall_space_planner.hub.viz import draw_network_in_outline

            st.markdown("**真实轮廓与关键点网络**（骨架边加粗）")
            fig, ax = plt.subplots(figsize=(6, 5))
            sk_nodes = set(rn["skeleton"].nodes) if rn["skeleton"] is not None else set()
            draw_network_in_outline(ax, rn["full"], rn["positions"], rn["outline"], sk_nodes, title=f"{fid}（真实位置）")
            show_fig(fig)
            al = rn["outline"].extra.get("csv_align")
            if al:
                st.caption(f"图 CSV → 掩膜像素对齐：{al['mode']}，平移 {al['shift_px']} px，缩放 {al['scale']}，IoU {al['coverage_identity']} → {al['coverage_best']}")
            from shapely.geometry import Point
            outside = [v for v, pxy in rn["positions"].items() if not rn["outline"].polygon.buffer(1.0).covers(Point(pxy))]
            if outside:
                st.warning(f"{len(outside)} 个关键点仍在轮廓外（{', '.join(outside[:6])}…）——该楼层的图 CSV 与掩膜可能来自不同区域 / 图幅", icon="⚠️")
        else:
            o = cat.outline(fid)
            if o is None:
                st.caption("轮廓：不可用（需要 legacy 掩膜 / total.csv；Mac 上可用）")
    with right:
        st.markdown("**条件字段**")
        st.dataframe(pd.Series({k: row.get(k) for k in db.query_cols + ["city_cluster"] if k in row}, name="值").astype(str), width="stretch")
        st.markdown("**拓扑指标（由图重新计算）**")
        tm = cat.topology_metrics(fid)
        st.dataframe(pd.Series({k: v for k, v in tm.items() if v is not None}, name="值").astype(str), width="stretch")
        gmv = {k: row.get(k) for k in db.graph_metric_cols if k in row}
        if gmv:
            st.markdown("**案例表中的 g_* 指标**")
            st.dataframe(pd.Series(gmv, name="值").astype(str), width="stretch")
        st.session_state.setdefault("ui1_compare", [])
        if st.button("加入对比"):
            if fid not in st.session_state["ui1_compare"]:
                st.session_state["ui1_compare"].append(fid)
            st.success(f"已加入：{st.session_state['ui1_compare']}")

# ------------------------------------------------------------------------------------------------------------ compare
with tab_compare:
    default = st.session_state.get("ui1_compare", [])
    sel = st.multiselect("选择 2–6 个案例", df[db.id_col].astype(str).tolist(), default=default[:6])
    if len(sel) >= 2:
        cmp = cat.compare(sel)
        st.dataframe(cmp.astype(str), width="stretch")  # transposed -> mixed types per column
        metric_rows = [m for m in db.graph_metric_cols if m in cmp.index]
        if metric_rows:
            long = cmp.loc[metric_rows].astype(float).reset_index().melt(id_vars="index", var_name="案例", value_name="值").rename(columns={"index": "指标"})
            figc = px.bar(long, x="指标", y="值", color="案例", barmode="group")
            figc.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(figc, width="stretch")
        n = len(sel)
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
        from mall_space_planner.hub.viz import draw_case_topology

        for ax, f in zip([axes] if n == 1 else axes, sel):
            t = cat.topology(f)
            if t is None:
                ax.axis("off"); ax.set_title(f"{f}\n(无拓扑)")
                continue
            sk = {nn for nn, a in t.node_attrs.items() if a.get("is_skeleton") or a.get("skeleton")} or None
            draw_case_topology(ax, t, sk, title=f"{f}\n评分 {cat.case(f).get(db.label_col)}")
        show_fig(fig)
        st.download_button("下载对比表 CSV", cmp.to_csv().encode("utf-8-sig"), "compare.csv", "text/csv")
    else:
        st.info("至少选择两个案例。")

# ------------------------------------------------------------------------------------------------------------ quality
with tab_quality:
    st.markdown("**缺失值**")
    mr = cat.missing_report()
    st.dataframe(mr, width="stretch") if len(mr) else st.success("无缺失值")
    st.markdown("**异常警告**（常量列、>5σ 离群、负值、评分越界、同商场评分方差、非连通 / 缺失拓扑、类型强转）")
    an = cat.anomalies()
    if len(an):
        st.dataframe(an, width="stretch")
    else:
        st.success("未发现异常")
