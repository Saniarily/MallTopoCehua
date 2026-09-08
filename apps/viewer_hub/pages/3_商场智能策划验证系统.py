"""UI 3 – planning validation workbench: new-build (Stage 1 → 2 → 3) and renovation workflows with one-click regenerate."""
from __future__ import annotations

import json
import time
import zipfile
import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from _common import data_banner, get_catalog, get_workbench, list_configs, setup_page, show_fig

from mall_space_planner.hub.export import candidate_to_geojson, candidate_to_json, candidate_to_svg, draw_candidate
from mall_space_planner.hub.viz import draw_renovation
from mall_space_planner.schemas import PlanningCondition

setup_page("3 · 商场智能策划验证系统", "🧭")
cat = get_catalog()
data_banner(cat)
db = cat.db

with st.sidebar:
    st.markdown("**模型配置**")
    s1c = list_configs("stage1"); s2c = list_configs("stage2")
    s1 = st.selectbox("阶段一排序器", s1c, index=s1c.index("configs/stage1/extra_trees.yaml") if "configs/stage1/extra_trees.yaml" in s1c else 0)
    s2 = st.selectbox("阶段二服务配置", s2c, index=s2c.index("configs/stage2/search_baseline.yaml") if "configs/stage2/search_baseline.yaml" in s2c else 0)
    wb = get_workbench(s1, s2)
    st.caption(f"生成器：{wb.generator_name}")
    restarts = st.slider("阶段三 重启次数", 1, 8, 3)
    iters = st.slider("阶段三 迭代数", 40, 200, 80, 20)

tab_new, tab_reno = st.tabs(["新建策划（阶段一 → 二 → 三）", "旧商场改造（同轮廓 · 保留出入口 · 重新生长）"])
S = st.session_state

# ==================================================================================================== NEW BUILD
with tab_new:
    st.markdown("#### ① 输入条件")
    src = st.radio("条件来源", ["手动输入", "从案例复制"], horizontal=True)
    if src == "从案例复制":
        ref = st.selectbox("案例", db.cases[db.id_col].astype(str).tolist())
        base = wb.condition_from_case(ref)
    else:
        cl = st.selectbox("城市簇", sorted(db.cases["city_cluster"].dropna().unique().astype(int)) if "city_cluster" in db.cases else [2])
        base = wb.default_condition(int(cl))
    cols = st.columns(4)
    vals = {}
    for i, c in enumerate(db.query_cols):
        v = getattr(base, c, None)
        vals[c] = cols[i % 4].number_input(c, value=float(v) if v is not None else 0.0, format="%.3f")
    pref = st.selectbox("偏好类型（可选）", [None, *sorted(db.cases["layout_type"].dropna().unique())], format_func=lambda x: x or "（无）")
    cond = PlanningCondition(city_cluster=base.city_cluster, preferred_layout=pref, **vals)

    st.markdown("#### ② 场地轮廓")
    osrc = st.radio("轮廓来源", ["手绘顶点", "上传 JSON", "数据库楼层", "按面积匹配真实轮廓", "矩形（按总面积）"], horizontal=True)
    pts = None; ofid = None; oarea = None
    if osrc == "手绘顶点":
        txt = st.text_area("顶点坐标（米），每行 x,y，逆时针", "0,0\n140,0\n140,90\n70,90\n70,55\n0,55", height=140)
        try:
            pts = [tuple(map(float, ln.replace("，", ",").split(","))) for ln in txt.strip().splitlines() if ln.strip()]
        except ValueError:
            st.error("坐标格式错误"); pts = None
    elif osrc == "上传 JSON":
        up = st.file_uploader("JSON：{\"exterior\": [[x,y],...]} 或 GeoJSON Polygon", type=["json", "geojson"])
        if up:
            d = json.load(up)
            if "exterior" in d:
                pts = [tuple(p) for p in d["exterior"]]
            elif d.get("type") == "Polygon":
                pts = [tuple(p) for p in d["coordinates"][0]]
            elif d.get("type") == "Feature":
                pts = [tuple(p) for p in d["geometry"]["coordinates"][0]]
            st.success(f"读取 {len(pts or [])} 个顶点")
    elif osrc == "数据库楼层":
        ofid = st.selectbox("楼层", db.cases[db.id_col].astype(str).tolist(), key="ol_fid")
    elif osrc == "按面积匹配真实轮廓":
        oarea = st.number_input("单层面积 m²", 2000.0, 200000.0, 15000.0, 500.0)
    else:
        oarea = st.number_input("单层面积 m²（矩形）", 2000.0, 200000.0, float(vals.get("total_area", 60000.0)) / 4, 500.0, key="rect_area")
        pts = None
    outline, osrc_name = wb.outline_from(pts, ofid, oarea)
    oc1, oc2 = st.columns([1, 2])
    with oc1:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        from mall_space_planner.hub.viz import poly
        poly(ax, outline.polygon, fc="#f4f4f4", ec="#333", lw=1.2); ax.set_aspect("equal"); ax.autoscale(); ax.axis("off"); ax.set_title(f"{osrc_name} · {outline.polygon.area:,.0f} m²", fontsize=9)
        show_fig(fig, dpi=110)

    st.markdown("#### ③ 阶段一：类型推荐 → Top-K 原型")
    top_k = st.slider("Top-K", 3, 10, 5)
    if st.button("运行阶段一推荐", type="primary"):
        with st.spinner("推荐中 …"):
            tr = wb.recommend_types(cond)
            S["ui3_types"] = tr
            S["ui3_cond"] = cond.model_dump()
            S["ui3_recs"] = {r.layout_type: wb.recommend(cond, r.layout_type, top_k) for r in tr.recommendations[:4]}
            S["ui3_recs"]["（不限类型）"] = wb.recommend(cond, None, top_k)
    if "ui3_types" in S:
        tr = S["ui3_types"]
        st.markdown(f"仅条件预测评分 **{tr.conditions_only_score:.2f}** · 城市簇 {tr.cluster}")
        st.dataframe(pd.DataFrame([{"排名": r.rank, "布局类型": r.layout_type, "期望评分": round(r.expected_score, 2), "CI 低": round(r.ci_low, 2), "CI 高": round(r.ci_high, 2), "可比案例": r.n_comparable_cases, "经验均分": None if r.empirical_mean_score is None else round(r.empirical_mean_score, 2)} for r in tr.recommendations]), width="stretch", hide_index=True)
        lt = st.selectbox("查看类型内 Top-K", list(S["ui3_recs"]))
        recs = S["ui3_recs"][lt]
        st.dataframe(pd.DataFrame([{"排名": r.rank, "原型": r.prototype_id, "得分": round(r.score, 3), "置信": None if r.confidence is None else round(r.confidence, 2), "质量分": None if r.quality_score is None else round(r.quality_score, 2), "相似度": None if r.similarity is None else round(r.similarity, 3), "类型": r.layout_type} for r in recs]), width="stretch", hide_index=True)
        pid = st.selectbox("选择原型", [r.prototype_id for r in recs])
        rsel = next(r for r in recs if r.prototype_id == pid)
        e = rsel.explanation
        c1, c2 = st.columns([1, 1])
        with c1:
            if e:
                st.markdown(f"**解释**：{e.recommendation_summary}")
                if e.top_factors:
                    st.dataframe(pd.DataFrame(e.top_factors), width="stretch", hide_index=True)
                if e.topology_reasoning:
                    st.markdown("\n".join(f"- {t}" for t in e.topology_reasoning))
                if e.matched_case_evidence:
                    st.markdown("**相似案例证据**"); st.dataframe(pd.DataFrame(e.matched_case_evidence), width="stretch", hide_index=True)
                if e.risks:
                    st.warning("\n".join(f"- {r}" for r in e.risks))
                if e.counterfactuals:
                    st.markdown("**反事实（修改条件后的变化）**"); st.dataframe(pd.DataFrame(e.counterfactuals), width="stretch", hide_index=True)
        with c2:
            topo = cat.topology(pid)
            if topo is not None:
                from mall_space_planner.hub.viz import draw_case_topology
                fig, ax = plt.subplots(figsize=(4.5, 4))
                draw_case_topology(ax, topo, None, title=f"原型 {pid}：{topo.num_nodes} 节点 / {topo.num_edges} 连接")
                show_fig(fig, dpi=110)
                st.dataframe(pd.Series({k: v for k, v in cat.topology_metrics(pid).items() if v is not None}, name="原型拓扑指标").astype(str), width="stretch")

        st.markdown("#### ④ 约束 → 阶段二生成多候选 → 阶段三走廊方案")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        n_target = k1.number_input("目标节点数", 8, 150, max(24, topo.num_nodes + 12) if topo is not None else 30)
        n_cand = k2.number_input("候选数量", 1, 8, 3)
        min_ent = k3.number_input("最少出入口", 1, 10, 2)
        max_ent = k4.number_input("最多出入口", 1, 12, 6)
        max_atria = k5.number_input("中庭数上限", 0, 6, 2)
        cr = k6.number_input("走廊占比目标", 0.10, 0.35, 0.18, 0.01)
        tad = st.slider("目标平均连接度（0 = 用原型语料默认）", 0.0, 4.0, 0.0, 0.1)
        seed0 = st.number_input("起始 seed", 0, 100000, 0)
        if st.button("▶ 生成候选方案", type="primary"):
            sk = cat.topology(pid)
            kw = dict(layout_type=rsel.layout_type, target_avg_degree=tad or None, corridor_ratio=cr, min_entrances=int(min_ent), max_entrances=int(max_ent), max_atria=int(max_atria), score_fn=wb.score_fn(cond), restarts=restarts, iters=iters)
            prog = st.progress(0.0, "生成中 …")
            cands = []
            for i in range(int(n_cand)):
                cands.append(wb.generate_candidate(sk, pid, outline, int(n_target), int(seed0) + i, index=i, **kw))
                prog.progress((i + 1) / n_cand, f"候选 {i + 1}/{n_cand} 完成（{cands[-1].elapsed_s:.0f}s）")
            prog.empty()
            S["ui3_gen"] = {"cands": cands, "kw": kw, "sk": sk, "pid": pid, "outline": outline, "n_target": int(n_target), "min_ent": int(min_ent)}

    g = S.get("ui3_gen")
    if g:
        cands = g["cands"]
        st.markdown("#### ⑤ 候选方案：可视化 · 约束满足 · 评分与风险 · 一键重新生成")
        summary = pd.DataFrame([c.summary() for c in cands])
        st.dataframe(summary, width="stretch", hide_index=True)
        show_net = st.checkbox("叠加关键点网络", True)
        cols = st.columns(len(cands))
        for i, (col, c) in enumerate(zip(cols, cands)):
            with col:
                fig, ax = plt.subplots(figsize=(4.6, 4.6))
                draw_candidate(ax, c, show_network=show_net, show_plan=True, title=f"候选 {i + 1} · seed {c.seed}" + (f" · 预测 {c.pred_score:.3f}" if c.pred_score is not None else ""))
                show_fig(fig, dpi=110)
                if st.button("🔄 重新生成", key=f"regen_{i}", help="不满意？同一原型 / 轮廓 / 约束，换 seed 重新生成本候选"):
                    with st.spinner(f"重新生成候选 {i + 1} …"):
                        new_seed = c.seed + 1000 + int(time.time()) % 997
                        g["cands"][i] = wb.generate_candidate(g["sk"], g["pid"], g["outline"], g["n_target"], new_seed, index=i, **g["kw"])
                    st.rerun()
                rep = wb.constraint_report(c, min_entrances=g["min_ent"])
                ok = sum(r["满足"] for r in rep)
                st.markdown(f"**约束满足 {ok}/{len(rep)}**")
                st.dataframe(pd.DataFrame(rep).assign(满足=lambda d: d["满足"].map({True: "✅", False: "❌"})), width="stretch", hide_index=True)
                rk = wb.risks(c)
                if rk:
                    st.warning("\n".join(f"- {r}" for r in rk))
                else:
                    st.success("未发现风险")
                with st.expander("导出"):
                    st.download_button("JSON", json.dumps(candidate_to_json(c), ensure_ascii=False, indent=1).encode("utf-8"), f"candidate_{i + 1}.json", "application/json", key=f"dj{i}")
                    st.download_button("GeoJSON", json.dumps(candidate_to_geojson(c), ensure_ascii=False).encode("utf-8"), f"candidate_{i + 1}.geojson", "application/geo+json", key=f"dg{i}")
                    st.download_button("SVG", candidate_to_svg(c).encode("utf-8"), f"candidate_{i + 1}.svg", "image/svg+xml", key=f"ds{i}")
                    fig, ax = plt.subplots(figsize=(7, 7)); draw_candidate(ax, c, title=f"candidate {i + 1}")
                    from _common import fig_png
                    st.download_button("PNG", fig_png(fig, 160), f"candidate_{i + 1}.png", "image/png", key=f"dp{i}")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, c in enumerate(cands):
                z.writestr(f"candidate_{i + 1}.json", json.dumps(candidate_to_json(c), ensure_ascii=False, indent=1))
                z.writestr(f"candidate_{i + 1}.geojson", json.dumps(candidate_to_geojson(c), ensure_ascii=False))
                z.writestr(f"candidate_{i + 1}.svg", candidate_to_svg(c))
            z.writestr("summary.csv", summary.to_csv(index=False))
        st.download_button("⬇ 全部候选打包 ZIP（JSON + GeoJSON + SVG + summary.csv）", buf.getvalue(), "candidates.zip", "application/zip")

# ==================================================================================================== RENOVATION
with tab_reno:
    st.markdown("保留既有商场 **轮廓与出入口位置**，以阶段一骨架为原型重新生长完整关键点网络并重排所有节点，得到走廊方案；对比现状 / 更新的关键拓扑指标与阶段一预测评分。")
    all_ids = wb.renovation_floor_ids()
    if not all_ids:
        st.warning("没有可达的真实图 CSV（需要 legacy graph_dir，Mac 上可用）；沙盒仅含测试夹具楼层。")
        floors = []
    else:
        st.markdown(f"**候选楼层：全部 {len(all_ids)} 个可达楼层**（可选按条件筛选推荐）")
        use_filter = st.toggle("只显示推荐楼层（节点数 / 面积 / 有出入口 / 首层或二层 / 低分）", value=False)
        if use_filter:
            r1, r2, r3, r4 = st.columns(4)
            low = r1.number_input("仅低分楼层（评分 ≤，0 = 不限）", 0.0, 100.0, 0.0, 0.1)
            min_nodes = r2.number_input("最少节点数", 4, 100, 12)
            min_area = r3.number_input("最小面积 m²", 500.0, 50000.0, 6000.0, 500.0)
            pf = r4.multiselect("楼层号", [1, 2, 3, 4, 5, 6], default=[1, 2])
            floors = wb.renovation_floors(low or None, 500, int(min_nodes), float(min_area), prefer_floors=tuple(pf) or None, require_entrance=True)
            st.caption(f"推荐 {len(floors)} 个楼层")
            if not floors:
                st.info("没有楼层满足推荐条件；关闭开关可浏览全部楼层。")
        else:
            floors = all_ids
        # metadata table + thumbnail gallery so the designer sees the plan, not only an id
        key_tbl = ("ui3_floor_tbl", tuple(floors))
        if S.get("ui3_floor_tbl_key") != key_tbl:
            S["ui3_floor_tbl"] = pd.DataFrame(wb.renovation_floor_table(floors)); S["ui3_floor_tbl_key"] = key_tbl
        tbl = S["ui3_floor_tbl"]
        if len(tbl):
            sort_by = st.selectbox("排序", [c for c in ["score", "area_m2", "n_nodes", "n_entrances", "floor_id"] if c in tbl.columns], index=0)
            tbl = tbl.sort_values(sort_by, ascending=sort_by != "area_m2").reset_index(drop=True)
            with st.expander(f"楼层列表（{len(tbl)}）", expanded=False):
                st.dataframe(tbl, width="stretch", hide_index=True, height=min(360, 40 + 35 * len(tbl)))
            page_n = 12
            n_pages = max(1, (len(tbl) + page_n - 1) // page_n)
            pg = st.number_input("缩略图页", 1, n_pages, 1) - 1 if n_pages > 1 else 0
            sub = tbl.iloc[pg * page_n:(pg + 1) * page_n]
            cols = st.columns(6)
            for i, (_, rr) in enumerate(sub.iterrows()):
                with cols[i % 6]:
                    thumb = wb.floor_thumbnail(rr["floor_id"])
                    if thumb:
                        st.image(thumb, width="stretch")
                    sc = f"评分 {rr['score']:.2f} · " if pd.notna(rr.get("score")) else ""
                    st.caption(f"**{rr['floor_id']}**  \n{sc}{rr.get('layout_type') or ''}  \n{int(rr['n_nodes']) if pd.notna(rr.get('n_nodes')) else '?'} 节点 · {int(rr['area_m2']) if pd.notna(rr.get('area_m2')) else '?'} m² · {int(rr['n_entrances']) if pd.notna(rr.get('n_entrances')) else '?'} 出入口")
                    if st.button("选择", key=f"pick_{rr['floor_id']}"):
                        S["ui3_reno_fid"] = rr["floor_id"]
        floors = tbl["floor_id"].tolist() if len(tbl) else floors
    if floors:
        default_fid = S.get("ui3_reno_fid") if S.get("ui3_reno_fid") in floors else floors[0]
        fid = st.selectbox("楼层", floors, index=floors.index(default_fid), key="ui3_reno_select")
        S["ui3_reno_fid"] = fid
        pc1, pc2 = st.columns([1, 3])
        with pc1:
            th = wb.floor_thumbnail(fid, size=3.0, dpi=110)
            if th:
                st.image(th, caption=f"{fid} 现状", width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        seed = c1.number_input("seed", 0, 100000, 0, key="reno_seed")
        ncand = c2.number_input("内部候选数（按改造目标择优）", 1, 16, 6)
        use_score = c3.checkbox("目标含阶段一预测评分", True)
        keep_pos = c4.checkbox("保留原型节点位置（对照：几乎不变）", False)
        run = st.button("▶ 运行改造", type="primary")
        regen = st.button("🔄 不满意，换 seed 重新生成", help="同一楼层、同一约束，换 seed 重新生长并重排")
        if regen and "ui3_reno" not in S:
            st.info("请先运行一次改造。"); regen = False
        if run or regen:
            sd = int(seed) if run else S["ui3_reno"]["seed"] + 1000 + int(time.time()) % 997
            with st.spinner(f"改造 {fid}（seed {sd}）…"):
                t0 = time.time()
                r = wb.renovate(fid, seed=sd, n_candidates=int(ncand), use_score=use_score, keep_skeleton_positions=keep_pos, restarts=restarts, iters=iters)
                r["seed"] = sd; r["elapsed"] = time.time() - t0
                S["ui3_reno"] = r
    r = S.get("ui3_reno")
    if r:
        st.caption(f"楼层 {r['floor_id']} · seed {r['seed']} · 生成器 {r['generator_name']} · 保留出入口 {len(r['anchored'])} 个 · {r['elapsed']:.0f}s")
        fig = draw_renovation(r)
        show_fig(fig, dpi=120)
        t = pd.DataFrame(r["table"]).rename(columns={"": "变化"})
        imp = int((t["变化"] == "▲").sum()); wor = int((t["变化"] == "▼").sum())
        st.markdown(f"**指标改善 {imp} 项 · 变差 {wor} 项**（▲ 改善 / ▼ 变差）")
        st.dataframe(t, width="stretch", hide_index=True)
        row = r["row"]
        st.dataframe(pd.DataFrame([{k: v for k, v in row.items() if k.startswith("after_") or k.startswith("before_")}]).T.rename(columns={0: "值"}).astype(str), width="stretch")
        fig2 = draw_renovation(r)
        from _common import fig_png
        e1, e2 = st.columns(2)
        e1.download_button("PNG（四联图）", fig_png(fig2, 160), f"renovation_{r['floor_id']}.png", "image/png")
        e2.download_button("指标 JSON", json.dumps({"floor_id": r["floor_id"], "seed": r["seed"], "before": r["ind_b"], "after": r["ind_a"], "row": row}, ensure_ascii=False, indent=1, default=str).encode("utf-8"), f"renovation_{r['floor_id']}.json", "application/json")
