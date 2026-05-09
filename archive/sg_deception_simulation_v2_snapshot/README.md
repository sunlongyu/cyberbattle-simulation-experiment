# sg_deception_simulation_v2_snapshot

本目录保存第三章多阶段信号博弈实验的第二版历史快照与对应结果文件，仅用于追溯，不作为当前默认运行目录。

## 目录说明

- `cyber_simulation_core/`: 核心模型、策略和实验逻辑
- `run_experiments.py`: 生成实验结果 JSON 和 CSV
- `plot_results.py`: 根据结果文件生成 PNG 图片
- `results/`: 已生成的实验结果与图片

## 运行方式

如需复查该历史版本，可在仓库根目录执行：

```bash
python archive/sg_deception_simulation_v2_snapshot/run_experiments.py
python archive/sg_deception_simulation_v2_snapshot/plot_results.py
```

当前正式 Chapter 3 实验路径为 `sg_deception_simulation/`。
