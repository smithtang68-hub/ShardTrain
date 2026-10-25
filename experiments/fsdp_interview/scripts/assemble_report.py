"""Assemble reviewed per-question sections without changing their evidence."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


def rebase_links(content, source_parent, destination_parent):
    def replace(match):
        target = match.group(1)
        angled = target.startswith("<") and target.endswith(">")
        value = target[1:-1] if angled else target
        if value.startswith(("/", "#")) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", value):
            return match.group(0)
        path, separator, fragment = value.partition("#")
        relocated = os.path.relpath(os.path.join(source_parent, path), destination_parent)
        relocated = relocated.replace("\\", "/") + separator + fragment
        if angled or " " in relocated:
            relocated = "<" + relocated + ">"
        return "](" + relocated + ")"

    return re.sub(r"\]\((<[^>]+>|[^)]+)\)", replace, content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intro", type=Path, required=True)
    parser.add_argument("--sections", type=Path, required=True)
    parser.add_argument("--timing-csv", type=Path, required=True)
    parser.add_argument("--delivery-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.delivery_audit.read_text(encoding="utf-8"))
    if audit["status"] != "PASS":
        raise ValueError("Final report requires completed artifact verification")
    with args.timing_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6 or any(row["status"] != "PASS" for row in rows):
        raise ValueError("Expected six independently repeated primary experiment groups")
    pieces = [args.intro.read_text(encoding="utf-8").rstrip(), "## 主负载正式计时总表\n"]
    pieces.append(
        "每组为3次独立进程、每次10 warmup+30 measured，共90个全局 optimizer step。"
        "P50/P95按90步合并计算；step取四rank最大值，有效label每步只计一次。"
        "下表不使用任何profile/shape诊断step。通信清单及对照意义见Q5/Q6。\n"
    )
    table_lines = [
        "| 组别 | P50 / P95 ms | Mean ms | 有效labels/s | Peak allocated / reserved GiB | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['series']} | {float(row['p50_step_ms']):.2f} / "
            f"{float(row['p95_step_ms']):.2f} | {float(row['mean_step_ms']):.2f} | "
            f"{float(row['training_tokens_per_s']):.2f} | "
            f"{float(row['peak_allocated_GiB']):.3f} / "
            f"{float(row['peak_reserved_GiB']):.3f} | PASS |"
        )
    pieces.append("\n".join(table_lines))
    pieces.append(
        "\n原始统计：[timing_aggregate.csv](../experiments/fsdp_interview/results/20261014/"
        "analysis/q05_timing_v1/timing_aggregate.csv)，逐run/step同目录。\n\n"
        "## 七题实验与证据\n"
    )
    for question in ["Q1", "Q3", "Q5", "Q6", "Q9", "Q10", "Q12"]:
        path = args.sections / f"{question}.md"
        content = path.read_text(encoding="utf-8").strip()
        if not content.startswith(f"### {question}："):
            raise ValueError(f"Unexpected report section: {path}")
        pieces.append(rebase_links(content, path.parent, args.output.parent))
    for extra in ["GEMM_C0_C1.md", "GEMM_three_way.md"]:
        path = args.sections / extra
        if path.is_file():
            pieces.append(
                rebase_links(
                    path.read_text(encoding="utf-8").strip(), path.parent, args.output.parent
                )
            )
    pieces.append(
        "## 原始数据交付与可复核范围\n\n"
        f"完整服务器返回目录：`{audit['canonical_root']}`。远端清单与传输包、"
        f"{audit['remote_manifest_files']}条远端清单记录逐文件SHA256核验；"
        "清单文件本身由全归档SHA256锁定，"
        f"共返回{audit['verified_regular_files']}个regular files；"
        f"{audit['original_symlinks']}个软链接只保存目标元数据，未跟随外部库目录。\n\n"
        "本轮21份`.nsys-rep`、21份SQLite、150份allocator snapshot已保留。"
        "工作镜像位于`D:/ShardTrainArtifacts/fsdp_interview_20261014`，"
        "仓库`experiments/fsdp_interview/results/20261014`通过NTFS junction指向它。"
        "实际分析的工作镜像trace/snapshot已与完整服务器返回清单核验。"
        "早期失败、下载partial、中断derived目录均保留，正式报告只引用注明版本的有效派生数据。\n\n"
        "[交付核验记录](../runs/fsdp_interview_20261014/delivery_audit.json)、"
        "[生产代码范围核验](../runs/fsdp_interview_20261014/production_code_scope_audit_final.json)、"
        "[21份trace索引](../runs/fsdp_interview_20261014/trace_inventory.csv)。"
        "[全部阶段清单](../runs/fsdp_interview_20261014/stage_execution_ledger.csv)"
        "记录147个owned阶段：144 passed、3个开发期failed均保留；"
        "[阶段退出审计](../runs/fsdp_interview_20261014/suite_execution_audit.json)"
        "确认全部阶段无遗留owned进程/ownership错误。失败分别为v2 runner缺失digest依赖、"
        "v3 Q9 observer校验及R4首轮device混用；后续独立修复运行通过，失败未改写为PASS。"
        "服务器精确命令、进程归属/清理及NVML记录在完整返回目录的`logs/<stage>/`；"
        "环境/API/拓扑与精确官方源码比对在`environment/`，输入缓存在`data/`。\n\n"
        "主性能使用冻结source v6，后续rank-selector/GEMM诊断使用v8，"
        "R4验证器修复及minimal capture使用v9。所有源码归档/manifest保留；"
        "最终Windows代码含离线解析/报告修订，不能把最终版本误称为全部GPU阶段同一版本。\n\n"
        "## 未证实事项与最短下一步\n\n"
        "1. SM/DRAM竞争归因：当前V100与Nsight版本不支持这些counter。"
        "在可支持的GPU上先做隔离NCCL/GEMM资源测量，再决定是否运行可选C4 CTA调节。\n"
        "2. 完整critical path与profile开销分解：本轮用独立minimal capture检验采集扰动，"
        "仍不把kernel交集当节省时间。若需精确分解，逐项开关emit_nvtx、"
        "allocator traceback与CUPTI，重复独立测量。\n"
        "3. 可选动态token不均衡L3与optimizer跨world恢复：本轮未执行；"
        "Q9 R4只验证FULL model state跨world portability，不代表optimizer续训。\n"
        "4. 原生最优性能：本次比较安装版官方FSDP2在固定matched精度/TP/SP路径的表现，"
        "未搜索torch.compile、attention替代或全部原生调优组合，不宣称FSDP2性能上限。\n"
    )
    args.output.write_text("\n\n".join(pieces).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output), "sections": 7}))


if __name__ == "__main__":
    main()
