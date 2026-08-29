"""compare_phases.py —— Phase1 vs Phase2 HD 结果对比。"""
import sys
import pandas as pd
from collections import Counter

P1  = "res/sanity_hd.xlsx"
P2  = "res/phase2_hd.xlsx"
P2B = "res/phase2b_hd.xlsx"
P2C = "res/phase2c_hd.xlsx"
TEXT_COLS = ["text1", "text2", "text3", "text4", "text5", "text6"]
GOLD = {"text1": "Real", "text2": "FaiHal", "text3": "FaiHal",
        "text4": "FacHal", "text5": "FacHal", "text6": "FacHal"}
LEGACY = {"Real": "Real", "Faithfulness": "FaiHal", "Factuality": "FacHal"}


def load(path):
    df = pd.read_excel(path, engine="openpyxl")
    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].map(lambda x: LEGACY.get(str(x).strip(), str(x).strip()) if pd.notna(x) else "")
    return df


def accuracy(df, col):
    gold = GOLD[col]
    valid = df[df[col] != ""]
    if len(valid) == 0:
        return 0.0, 0
    correct = (valid[col] == gold).sum()
    return correct / len(valid) * 100, len(valid)


def print_comparison(p1, p2):
    print(f"{'col':<8} {'P1 acc':>8} {'P2 acc':>8} {'delta':>8}  "
          f"{'P1 FaiHal':>10} {'P2 FaiHal':>10}  {'P1 FacHal':>10} {'P2 FacHal':>10}")
    print("-" * 80)
    for col in TEXT_COLS:
        a1, n1 = accuracy(p1, col)
        a2, n2 = accuracy(p2, col)
        fai1 = (p1[col] == "FaiHal").sum()
        fai2 = (p2[col] == "FaiHal").sum()
        fac1 = (p1[col] == "FacHal").sum()
        fac2 = (p2[col] == "FacHal").sum()
        delta = a2 - a1
        sign = "+" if delta >= 0 else ""
        print(f"{col:<8} {a1:>7.1f}% {a2:>7.1f}% {sign}{delta:>6.1f}%  "
              f"{fai1:>10} {fai2:>10}  {fac1:>10} {fac2:>10}")

    # efi_text selection
    print()
    e1 = (p1["efi_text"] == "text1").sum()
    e2 = (p2["efi_text"] == "text1").sum()
    n = len(p1)
    print(f"efi_text→text1:  Phase1 {e1}/{n} ({e1/n*100:.1f}%)  "
          f"Phase2 {e2}/{n} ({e2/n*100:.1f}%)")

    # overall label distribution
    print()
    for phase, df in [("Phase1", p1), ("Phase2", p2)]:
        all_labels = []
        for c in TEXT_COLS:
            all_labels += [v for v in df[c] if v]
        ct = Counter(all_labels)
        total = sum(ct.values())
        print(f"{phase}: Real {ct['Real']}({ct['Real']/total*100:.1f}%)  "
              f"FaiHal {ct['FaiHal']}({ct['FaiHal']/total*100:.1f}%)  "
              f"FacHal {ct['FacHal']}({ct['FacHal']/total*100:.1f}%)")


def print_three_way(p1, p2, p2b):
    print(f"{'col':<8} {'P1':>7} {'P2':>7} {'P2b':>7}  {'P1→P2':>7} {'P2→P2b':>7} {'P1→P2b':>8}")
    print("-" * 65)
    for col in TEXT_COLS:
        a1, _ = accuracy(p1, col)
        a2, _ = accuracy(p2, col)
        a2b, _ = accuracy(p2b, col)
        def fmt(d): return f"{'+'if d>=0 else ''}{d:.1f}%"
        print(f"{col:<8} {a1:>6.1f}% {a2:>6.1f}% {a2b:>6.1f}%  "
              f"{fmt(a2-a1):>7} {fmt(a2b-a2):>7} {fmt(a2b-a1):>8}")
    e1 = (p1["efi_text"]=="text1").sum()
    e2 = (p2["efi_text"]=="text1").sum()
    e2b = (p2b["efi_text"]=="text1").sum()
    n = len(p1)
    print(f"\nefi→text1: P1={e1}/{n}({e1/n*100:.0f}%)  P2={e2}/{n}({e2/n*100:.0f}%)  P2b={e2b}/{n}({e2b/n*100:.0f}%)")


if __name__ == "__main__":
    files = {"P1": P1, "P2": P2, "P2b": P2B, "P2c": P2C}
    dfs = {}
    for name, path in files.items():
        try:
            dfs[name] = load(path)
        except FileNotFoundError:
            print(f"跳过 {name}（文件不存在）")

    common = set.intersection(*[set(df["id"].astype(str)) for df in dfs.values()])
    for name in dfs:
        dfs[name] = dfs[name][dfs[name]["id"].astype(str).isin(common)].reset_index(drop=True)
    print(f"对比文档数: {len(common)}\n")

    versions = list(dfs.keys())
    print(f"{'col':<8} " + " ".join(f"{v:>7}" for v in versions) + f"  {'best':>5}")
    print("-" * (8 + 8 * len(versions) + 8))
    col_avgs = {v: [] for v in versions}
    for col in TEXT_COLS:
        accs = {v: accuracy(dfs[v], col)[0] for v in versions}
        best_v = max(accs, key=accs.get)
        row = f"{col:<8} " + " ".join(f"{accs[v]:>6.1f}%" for v in versions)
        row += f"  {best_v:>5}"
        print(row)
        for v in versions:
            col_avgs[v].append(accs[v])

    print("-" * (8 + 8 * len(versions) + 8))
    avgs = {v: sum(col_avgs[v]) / len(col_avgs[v]) for v in versions}
    print(f"{'avg':<8} " + " ".join(f"{avgs[v]:>6.1f}%" for v in versions) +
          f"  {max(avgs, key=avgs.get):>5}")

    print()
    for v in versions:
        e = (dfs[v]["efi_text"] == "text1").sum()
        n = len(dfs[v])
        print(f"efi→text1 {v}: {e}/{n} ({e/n*100:.0f}%)")
