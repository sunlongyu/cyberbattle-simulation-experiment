# Chapter 3 Signaling-Game Simulation 2

本目录是第三章多阶段信号博弈的正式实验结果目录，用于保存论文侧采用的 Chapter 3 代码、JSON/CSV 结果和图表。

## 目录说明

- `cyber_simulation_core/`: 核心模型、策略和实验逻辑
- `run_experiments.py`: 生成实验结果 JSON 和 CSV
- `plot_results.py`: 根据结果文件生成 PNG 图片
- `results/`: 已生成的实验结果与图片
- `results/figures/drafts/`: 从临时目录迁移过来的草稿图版本，用于追溯图形调整过程，不替代正式图

## 运行方式

在仓库根目录执行：

```bash
python sg_deception_simulation2/run_experiments.py
python sg_deception_simulation2/plot_results.py
```

维护规则：`sg_deception_simulation2/` 与 `sg_deception_simulation/` 是两个独立实验目录，不要互相覆盖结果文件。
