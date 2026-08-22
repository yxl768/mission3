"""
从 DBAASP 数据库下载真实抗菌肽数据
1. 批量获取短肽列表（长度5-50）
2. 逐条获取详细信息（靶菌活性、MIC、溶血活性）
3. 筛选革兰氏阴性菌活性肽作为正样本
4. 获取溶血/毒性实验数据
5. 保存为CSV
"""
import os
import sys
import json
import time
import re
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MIN_SEQ_LEN, MAX_SEQ_LEN, AMINO_ACIDS

DBAASP_API = "https://dbaasp.org/peptides"
DETAIL_API = "https://dbaasp.org/peptides/{pid}"
BATCH_SIZE = 100
MAX_DETAIL_FETCH = 3000  # 最多获取详细信息的肽数
REQUEST_TIMEOUT = 30
RETRY_TIMES = 3

# 革兰氏阴性菌常见物种名（用于筛选）
GRAM_NEGATIVE_SPECIES = [
    "escherichia coli", "pseudomonas aeruginosa", "klebsiella pneumoniae",
    "salmonella", "shigella", "enterobacter", "proteus", "acinetobacter",
    "neisseria", "helicobacter", "vibrio", "campylobacter", "legionella",
    "serratia", "morganella", "citrobacter", "haemophilus", "bordetella",
    "brucella", "yersinia", "erwinia", "xanthomonas", "chromobacterium",
    "aeromonas", "pseudomonas", "burkholderia", "stenotrophomonas",
]

# 标准氨基酸集合
STD_AA = set(AMINO_ACIDS)


def create_session():
    """创建请求session，跳过系统代理"""
    s = requests.Session()
    s.trust_env = False
    s.headers.update({"Accept": "application/json"})
    return s


def fetch_peptide_list(session, limit=BATCH_SIZE, offset=0, seq_len_range="5-50"):
    """批量获取肽列表"""
    params = {"limit": limit, "offset": offset, "sequenceLength.value": seq_len_range}
    for attempt in range(RETRY_TIMES):
        try:
            r = session.get(DBAASP_API, params=params, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < RETRY_TIMES - 1:
                time.sleep(2)
            else:
                raise
    return None


def fetch_peptide_detail(session, pid):
    """获取单条肽的详细信息"""
    url = DETAIL_API.format(pid=pid)
    for attempt in range(RETRY_TIMES):
        try:
            r = session.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < RETRY_TIMES - 1:
                time.sleep(1)
            else:
                return None
    return None


def is_gram_negative(species_name):
    """判断靶菌物种是否为革兰氏阴性菌"""
    if not species_name:
        return False
    s = species_name.lower().strip()
    for gn in GRAM_NEGATIVE_SPECIES:
        if gn in s:
            return True
    return False


def parse_mic_value(concentration):
    """
    解析MIC浓度值，返回浮点数（μg/mL或μM）
    可能格式: "5", "1.58-25.33", ">=64", "<=2", "0.5-2"
    取范围的几何平均值或边界值
    """
    if not concentration:
        return None
    c = str(concentration).strip()
    # 去除单位
    c = re.sub(r'[^\d.\-<>~=]', '', c)
    if not c:
        return None
    # 范围 "1.58-25.33"
    range_match = re.match(r'^<?~?(\d+\.?\d*)\s*-\s*>?~?(\d+\.?\d*)', c)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        if low > 0 and high > 0:
            return float(np.sqrt(low * high))  # 几何平均
        return (low + high) / 2
    # >=64, <=2 等
    bound_match = re.match(r'^[<>~>=]+(\d+\.?\d*)', c)
    if bound_match:
        return float(bound_match.group(1))
    # 单值
    single_match = re.match(r'^(\d+\.?\d*)', c)
    if single_match:
        return float(single_match.group(1))
    return None


def is_standard_sequence(seq):
    """检查是否只含标准20种氨基酸"""
    if not seq or not isinstance(seq, str):
        return False
    return all(c in STD_AA for c in seq)


def extract_gram_negative_activities(target_activities):
    """
    从targetActivities中提取革兰氏阴性菌的活性数据
    返回: list of dict {species, measure, concentration, unit, activity, is_gram_neg}
    """
    results = []
    for ta in target_activities:
        species_info = ta.get("targetSpecies", {})
        species_name = species_info.get("name", "") if species_info else ""
        is_gn = is_gram_negative(species_name)

        measure_info = ta.get("activityMeasureGroup", {})
        measure = measure_info.get("name", "") if measure_info else ""
        concentration = ta.get("concentration", "")
        unit_info = ta.get("unit", {})
        unit = unit_info.get("name", "") if unit_info else ""
        activity = ta.get("activity")

        mic_val = parse_mic_value(concentration)

        results.append({
            "species": species_name,
            "measure": measure,
            "concentration": concentration,
            "unit": unit,
            "mic_value": mic_val,
            "activity": activity,
            "is_gram_negative": is_gn,
        })
    return results


def extract_hemolytic_activities(hemo_activities):
    """提取溶血活性数据"""
    results = []
    for ha in hemo_activities:
        measure_info = ha.get("activityMeasureGroup", {})
        measure = measure_info.get("name", "") if measure_info else ""
        concentration = ha.get("concentration", "")
        unit_info = ha.get("unit", {})
        unit = unit_info.get("name", "") if unit_info else ""
        activity = ha.get("activity")
        # 溶血活性通常用HC50, LC50等
        results.append({
            "measure": measure,
            "concentration": concentration,
            "unit": unit,
            "activity": activity,
        })
    return results


def download_dbaasp_dataset():
    """主下载流程"""
    session = create_session()

    # Step 1：获取肽列表
    print("[DBAASP] Step 1: 获取肽列表...")
    all_peptides = []
    offset = 0
    total_count = None

    while True:
        if total_count and offset >= min(total_count, MAX_DETAIL_FETCH + BATCH_SIZE):
            break
        batch = fetch_peptide_list(session, limit=BATCH_SIZE, offset=offset)
        if not batch:
            break
        if total_count is None:
            total_count = batch.get("totalCount", 0)
            print(f"  总肽数(len 5-50): {total_count}")

        items = batch.get("data", [])
        if not items:
            break

        # 只保留有序列的单体肽
        for item in items:
            seq = item.get("sequence", "")
            if seq and item.get("complexity") == "monomer":
                all_peptides.append({
                    "id": item["id"],
                    "dbaasp_id": item.get("dbaaspId", ""),
                    "name": item.get("name", ""),
                    "sequence": seq,
                    "sequence_length": item.get("sequenceLength", len(seq)),
                    "synthesis_type": item.get("synthesisType", ""),
                })

        offset += BATCH_SIZE
        if offset >= MAX_DETAIL_FETCH + BATCH_SIZE:
            break
        time.sleep(0.3)  # 礼貌限速

    # 过滤长度和标准氨基酸
    filtered = [p for p in all_peptides
                if MIN_SEQ_LEN <= p["sequence_length"] <= MAX_SEQ_LEN
                and is_standard_sequence(p["sequence"])]
    print(f"  列表获取完成: {len(all_peptides)} 单体肽, 过滤后(8-30aa, 标准AA): {len(filtered)}")

    # Step 2：获取详细信息
    print(f"[DBAASP] Step 2: 获取详细信息(最多{MAX_DETAIL_FETCH}条)...")
    records = []
    n_gn_active = 0
    n_with_hemo = 0

    for p in tqdm(filtered[:MAX_DETAIL_FETCH], desc="Downloading details", ncols=100):
        detail = fetch_peptide_detail(session, p["id"])
        if not detail:
            continue
        time.sleep(0.15)  # 礼貌限速

        seq = detail.get("sequence", "")
        if not is_standard_sequence(seq):
            continue

        # 提取靶菌活性
        target_acts = detail.get("targetActivities", [])
        gn_activities = extract_gram_negative_activities(target_acts)

        # 提取溶血活性
        hemo_acts = detail.get("hemoliticCytotoxicActivities", [])
        hemo_data = extract_hemolytic_activities(hemo_acts)

        # 判断是否有革兰氏阴性菌活性
        gn_acts_only = [a for a in gn_activities if a["is_gram_negative"]]
        has_gn_activity = len(gn_acts_only) > 0

        # 获取最佳MIC值（最低MIC，即最强活性）
        best_mic = None
        best_mic_unit = ""
        best_mic_species = ""
        if gn_acts_only:
            mic_acts = [a for a in gn_acts_only if a["measure"] == "MIC" and a["mic_value"] is not None]
            if mic_acts:
                best = min(mic_acts, key=lambda x: x["mic_value"])
                best_mic = best["mic_value"]
                best_mic_unit = best["unit"]
                best_mic_species = best["species"]

        # 溶血数据
        hemo_value = None
        hemo_measure = ""
        if hemo_data:
            # 取第一个有activity值的
            for h in hemo_data:
                if h["activity"] is not None:
                    hemo_value = h["activity"]
                    hemo_measure = h["measure"]
                    break

        # targetGroups
        target_groups = detail.get("targetGroups", [])
        tg_names = [tg.get("name", "") for tg in target_groups if tg] if target_groups else []

        record = {
            "dbaasp_id": p["dbaasp_id"],
            "name": p["name"],
            "sequence": seq,
            "length": len(seq),
            "synthesis_type": p["synthesis_type"],
            "has_gram_negative_activity": has_gn_activity,
            "gn_species_count": len(gn_acts_only),
            "best_gn_mic": best_mic,
            "best_gn_mic_unit": best_mic_unit,
            "best_gn_mic_species": best_mic_species,
            "total_activities": len(target_acts),
            "has_hemolytic_data": len(hemo_data) > 0,
            "hemo_activity_value": hemo_value,
            "hemo_measure": hemo_measure,
            "target_groups": ";".join(tg_names),
        }
        records.append(record)

        if has_gn_activity:
            n_gn_active += 1
        if hemo_data:
            n_with_hemo += 1

    df = pd.DataFrame(records)
    print(f"\n[DBAASP] 下载完成:")
    print(f"  总肽数: {len(df)}")
    print(f"  革兰氏阴性菌活性肽: {n_gn_active}")
    print(f"  有溶血数据: {n_with_hemo}")

    # 保存原始数据
    raw_path = os.path.join(DATA_DIR, "dbaasp_raw.csv")
    df.to_csv(raw_path, index=False, encoding="utf-8")
    print(f"  原始数据已保存: {raw_path}")

    return df


def build_train_dataset_from_dbaasp(df_raw):
    """
    从DBAASP原始数据构建训练数据集
    - 正样本：有革兰氏阴性菌活性记录的肽
    - 负样本：有序列但无任何革兰氏阴性菌活性记录的肽
    - MIC标签：MIC ≤ 16 μg/mL 或 ≤ 16 μM 作为强阳性
    """
    print("\n[DBAASP] 构建训练数据集...")

    # 正样本：有G-活性
    df_pos = df_raw[df_raw["has_gram_negative_activity"] == True].copy()
    df_pos["label"] = 1
    df_pos["source"] = "DBAASP_GN_active"

    # 负样本：无G-活性记录（可能是测试过但无活性，或未测试G-菌）
    # 为了更严格，选择有活性记录但非G-的肽，或无任何活性记录的肽
    df_neg = df_raw[df_raw["has_gram_negative_activity"] == False].copy()
    df_neg["label"] = 0
    df_neg["source"] = "DBAASP_non_GN"

    # 去重
    df_pos = df_pos.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    df_neg = df_neg.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    # 平衡正负样本（负样本取正样本的1倍或全部）
    n_pos = len(df_pos)
    n_neg = min(len(df_neg), n_pos)
    df_neg = df_neg.sample(n=n_neg, random_state=42).reset_index(drop=True)

    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"  正样本(G-活性): {n_pos}")
    print(f"  负样本(非G-): {n_neg}")
    print(f"  总计: {len(df_all)}")

    # 统计MIC分布
    mic_vals = df_pos["best_gn_mic"].dropna()
    if len(mic_vals) > 0:
        print(f"  MIC分布: min={mic_vals.min():.2f}, median={mic_vals.median():.2f}, "
              f"max={mic_vals.max():.2f}")
        print(f"  MIC ≤ 16: {(mic_vals <= 16).sum()} 条")
        print(f"  MIC ≤ 8: {(mic_vals <= 8).sum()} 条")

    # 保存
    out_path = os.path.join(DATA_DIR, "dbaasp_dataset.csv")
    df_all.to_csv(out_path, index=False, encoding="utf-8")
    print(f"  数据集已保存: {out_path}")

    return df_all


if __name__ == "__main__":
    df_raw = download_dbaasp_dataset()
    df_dataset = build_train_dataset_from_dbaasp(df_raw)
    print("\n[DBAASP] 完成！")
    print(df_dataset[["sequence", "label", "best_gn_mic", "best_gn_mic_species"]].head(10).to_string())
