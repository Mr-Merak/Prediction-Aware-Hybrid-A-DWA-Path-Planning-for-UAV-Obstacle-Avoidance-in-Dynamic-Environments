# Drones 重投文件包

本文件夹包含论文《Prediction-Aware Hybrid A*–DWA Path Planning for UAV Obstacle Avoidance in Dynamic Environments》大修重投所需的全部材料。

## 目录结构

```
drones/
├── v8_paper.tex                      # 论文源文件（已重写）
├── v8_paper.pdf                      # 编译好的 PDF
├── figure1.png ... figure8.png       # 论文引用的 8 张图片（与 tex 同级，保证可编译）
├── Definitions/                      # MDPI 模板（mdpi.cls 等，编译必需）
├── cover letter - Drones - resubmission.docx   # 重投 cover letter
├── response_to_reviewers.md          # 逐条回复四个审稿人（投稿时附上）
├── revision_plan.md                  # 修订计划 + KF bug 记录（工作记录）
├── code/                             # 全部代码
│   ├── 11.py                         # 原始算法实现（未改动，被 benchmark 导入）
│   ├── fixes.py                      # FIX 1–7 七个 bug 修复
│   ├── tuning.py                     # 规范对齐（自适应启发式公式）
│   ├── benchmark.py                  # 实验框架（10 个配置 + 统计 + 机动场景 + 加速度约束）
│   ├── dstar_lite.py                 # D* Lite 基线
│   ├── sensitivity.py                # 参数敏感性扫描
│   ├── make_figures.py               # 图片生成脚本（figure3–8）
│   ├── diagnose_kf.py                # KF 预测误差诊断
│   ├── patchtest.py                  # 补丁测试
│   └── 路径规划.bib                   # 参考文献库（备用）
└── data/
    ├── experiment_data_nominal_rerun.xlsx      # 名义场景（100 种子，含原始运行）
    ├── experiment_data_maneuver_rerun.xlsx     # 机动场景（100 种子）
    ├── experiment_data_crossing.xlsx           # 交叉/迎头场景
    ├── experiment_data_density.xlsx            # 障碍密度扫描
    ├── experiment_data_structured.xlsx         # 窄通道/瓶颈/死端
    ├── experiment_data_structured_failure.xlsx # 结构化场景 outcome 分解
    ├── experiment_data_sensitivity100.xlsx     # 100 种子敏感性
    ├── experiment_data_alpha_modes.xlsx        # sigmoid/linear/threshold
    ├── experiment_data_failure.xlsx            # 碰撞几何分类
    ├── experiment_data_learned_predictor.xlsx  # GRU vs KF vs CV
    ├── experiment_data_3d6dof.xlsx             # 6-DOF 3D（100 种子）
    ├── experiment_data_factorial.xlsx          # nominal/structured/maneuver 2×2 汇总
    └── experiment_data_factorial_structured_astarpred.xlsx # 结构化 A*+Pred raw（补缺口）
```

## 核心改动（相对原稿）

1. **修复致命 bug**：`Environment.update_kf_trackers` 只做测量更新、缺时间更新（predict），导致 KF 退化为静态平滑器、风险场整体偏移 ~12 m。修复后预测误差从 ~12 m 降到 ~0.1 m（见 `fixes.py` FIX 4，`diagnose_kf.py` 可复现）。
2. **预测感知 DWA**（核心新贡献）：DWA 前向仿真传播预测的障碍物位置，而非把障碍当静止。
3. **7 个 bug 修复 + 4 个新基线**（D\* Lite、朴素 A\*+DWA、DWA-only、预测感知 DWA）。
4. **统计严谨性**：100 配对种子、Wilson 95% CI、精确 McNemar、配对 t/Wilcoxon、失败原因分解、完整运行时、跟踪误差。
5. **约束运动学 UAV 模型**：速度/角速度/加速度限制。
6. **术语**：experiments → numerical simulations；补 Kalman 1960 / de Boor / SciPy 奠基引用。

## 最终结果（名义场景，100 配对种子，含加速度约束）

| 方法 | 成功率 [95% CI] | vs A\*+DWA (McNemar) |
|---|---|---|
| 传统 A* | 0% | p < 0.001 |
| D\* Lite | 2% [0.6, 7.0] | p < 0.001 |
| A\* + DWA（基线） | 70% [60.4, 78.1] | — |
| Proposed | 77% [67.8, 84.2] | p = 0.210 |
| **完整框架 + 预测感知 DWA** | **87% [79.0, 92.2]** | **p = 0.0005** |

机动场景：完整框架 80% vs A\*+DWA 71%（p = 0.093）；预测感知 DWA 在机动下仅 +2 pp（p = 0.500）。
敏感性：成功率全程 68–88%（100 配对种子/设置），无崩溃点。
6-DOF 3D 验证：预测感知 DWA 在高度窗口和 3D 交叉场景分别为 90% 和 94%（100 配对种子）。

## 上传前请手动收尾

1. **检查 figure1、figure2**（环境建模、KF 流程概念图，未重新生成）。
2. **核对文献对比表**（`v8_paper.tex` 中 Table `tab:literature`）里各文献的 ✓/— 标记。
3. **更新 GitHub 仓库**（论文 data availability 指向的仓库需同步 `code/` 里的新代码）。
4. **最后通读**论文，确认作者信息、致谢（DeepSeek 声明）、基金号等元信息。

## 编译方式

在 `drones/` 目录下运行 `pdflatex v8_paper.tex`（需 TeX Live，已含 MDPI 模板 `Definitions/mdpi.cls`）。需编译两遍以解析交叉引用。
