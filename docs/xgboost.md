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

环境和完整命令见根目录 `README.md`。只调试 XGBoost 时：

```bash
python -m timing.cli validate --model xgboost    # 也会运行基准模型
python -m timing.cli report
```

如 macOS 报缺少 `libomp.dylib`，安装 OpenMP runtime（有 Homebrew 时：`brew install libomp`）。

测试期结果由 `python -m timing.cli test --confirm-final` 统一生成，全组冻结所有模型后再运行。

自动测试中，模型契约测试使用生产配置的 200 棵树；端到端合成数据测试将树数降到 5，
以检查流程并控制运行时间，生产 YAML 不受影响。
