# XGBoost market timing

## 实现和公平比较

`timing/models/xgboost.py` 实现 `BaseModel.fit/predict`，由现有注册机制自动加载。
预测目标仍是下一开盘到再下一开盘的 SPY 超额收益，输出一维数组。
使用 XGBoost 3.1.2 的 `XGBRegressor`、平方误差、CPU histogram 树和单线程。
初始预测为本次训练窗口的收益均值；每次 fit 都创建新模型。
API 参考：https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor

| 项目 | 设定 |
|---|---|
| 特征 | 与 Ridge、LASSO、Random Forest 相同的 14 个 core 特征 |
| 窗口 | 252 / 1260 个已实现标签 |
| 重训 | 每 21 个交易日，与其他模型一致 |
| 预处理 | none，与 Random Forest 一致；线性模型保留训练窗口标准化 |
| 参数网格 | depth ∈ {2,3}，learning_rate ∈ {0.03,0.1}，两个窗口，共 8 组 |
| 固定参数 | 200 棵树，min_child_weight=20，reg_lambda=1，完整行列采样 |
| 调参 | 相同验证日期和扣费后 Sharpe 选择规则，无随机交叉验证或 early stopping |
| 策略 | 相同历史预测标准化、仓位限制、no-trade band、交易成本及评价指标 |

参数在查看 XGBoost 结果前预设。XGBoost、LASSO 和 Random Forest 均为 8 组，
Ridge 原配置为 10 组；这是共享评价协议，而非完全相等的搜索预算。
不要根据测试期表现反复修改网格。比较真实结果时，应使用同一个 dataset 和 data_hash
重新运行各模型验证，不能把新下载数据的 XGBoost 指标与不同数据快照的旧指标混用。

## 时间序列检查

- 特征使用当日及之前数据，缺失对齐只做前向填充。
- 标签在 t+2 才实现，滚动引擎只使用 `label_end <= fit_date` 的完整训练窗口。
- XGBoost 只接收引擎给定的训练数据，不访问全量数据或测试集。
- 验证入口先剔除标签结束时间晚于验证末日的行。
- 仓位信号的均值和标准差只使用之前的预测（`shift(1)`）。
- 自动测试扰动未来特征及尚未实现标签，检查历史预测不变；另检查完整运行与
  截断运行的预测前缀一致、每个训练标签已实现、重复 fit 不继承旧树。

这些检查验证代码层面的时间边界；Yahoo 后续修订和复权历史不是 point-in-time
数据库，不能由合成数据测试证明不存在供应商历史修订影响。

## 运行

在仓库根目录使用 Python 3.11 或更新版本：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

如 macOS 报缺少 `libomp.dylib`，安装 OpenMP runtime（有 Homebrew 时：
`brew install libomp`）。若 `python3.12` 不存在，先安装 Python 3.12 或用 uv 建环境。

本机已建好 `.venv`（Python 3.12.16）并安装依赖；无需重新建环境。
本机测试使用 scikit-learn 附带的 OpenMP 库，在 VS Code 的终端执行：

```bash
source .venv/bin/activate
export DYLD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/sklearn/.dylibs${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
python -m pytest -q
```

2026-11-10：完整测试 **76 passed，18 warnings**。警告来自 LASSO 在未来数据极端
扰动后的收敛检查及现有 Matplotlib 布局。真实市场数据未下载，真实验证回测未运行；
原有 `results/` 和测试锁未更改。自动模型契约测试使用生产配置的 200 棵树；
端到端合成数据测试将树数降到 5，以检查流程并控制运行时间，生产 YAML 不受影响。

有团队共享的本地数据时直接使用；否则运行数据构建命令。

```bash
python -m timing.cli data
python -m timing.cli validate --model all
python -m timing.cli report
```

只调试 XGBoost 接入时可使用 `python -m timing.cli validate --model xgboost`。
该命令也运行基准模型，但不会重新生成其他候选模型的验证结果。

仓库已经包含 `results/test/lock.json`，XGBoost 是锁定后新增模型。
在验证选定参数后，如需测试期探索性比较，使用：

```bash
python -m timing.cli test --exploratory xgboost
python -m timing.cli report
```

结果写入 `results/test_exploratory/xgboost/`。保留测试锁和原冻结结果，报告中注明
探索性比较；不要把新增模型称为原 one-shot 测试选出的最终模型。
本次开发的合成数据测试不构成真实市场收益结果。

## 提交 feature/xgboost 和 PR

在 GitHub Desktop 选择此仓库和 `feature/xgboost`。检查 Changes 中仅包含本次代码、
配置、依赖、测试和文档；不要选中虚拟环境、市场数据或历史结果变动。
填写提交标题 `Add XGBoost market timing model`，提交后 Push origin，再创建面向
团队目标分支（通常为 main）的 PR。提交和推送由你执行。

也可以用终端（先查看状态，确认分支）：

```bash
git status --short --branch
git diff --check
git add timing/models/xgboost.py configs/models/xgboost.yaml requirements.txt README.md docs/xgboost.md tests/test_xgboost.py tests/test_pipeline.py tests/conftest.py
git commit -m "Add XGBoost market timing model"
git push origin feature/xgboost
```

建议 PR 标题：`Add XGBoost market timing model with temporal leakage tests`。
描述说明新增模型采用共享窗口、特征、策略和成本；8 组预设验证参数；没有 early
stopping；新增未来扰动与重训检查；集成测试支持动态模型名单。
填写 `76 passed (18 warnings)`，并注明尚未生成真实市场验证结果、测试期须走 exploratory。
