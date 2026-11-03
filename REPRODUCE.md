# 安装与复现

所有命令都在仓库根目录运行。正式数值核验使用 Python 3.12 和 `requirements.lock.txt`；`requirements.txt` 仅描述兼容范围，使用浮动版本不能保证逐项相同。整理时在 Windows、Python 3.12.14 上完成全部回放；其他系统需实际运行校验后判断。

## 1. 安装

```bash
git clone https://github.com/zjw-buaa/Data-Mining-sol.git
cd Data-Mining-sol
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
```

若激活脚本被系统策略阻止，无需修改系统策略；使用 `.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt`，后续同样用此解释器替换 `python`。macOS/Linux 使用 `source .venv/bin/activate` 后安装。

## 2. 测试与主实验离线回放

```bash
python -m unittest discover -s tests -v
python run.py --replay
```

回放核验原始 CSV、六份核心源码哈希、真实调用日志、已执行策略和模型读到的质量统计，然后从原始数据重新运行三组处理与聚类。完整核对 26 个 CSV、8 个 JSON，只忽略 `seconds / processing_seconds / clustering_seconds` 耗时字段。程序在回放期间阻止网络连接。

结果写入新的 `results/replay/<时间戳>/`。末行应含 `status: verified`、`new_api_calls: 0`、`csvs: 26`、`jsons: 8`；目录中的 `replay_verification.json` 记录来源和检查范围。缺文件、核心源码变化、策略与响应不符或数值变化均会报错，不会静默跳过。

离线回放验证的是“该次模型计划交给本地工具后能否重现结果”，不代表重新调用模型还能得到同一决策。

## 3. 三次补充实验离线回放

```bash
python run.py --replay --recorded-run results/stability/runs/02
python run.py --replay --recorded-run results/stability/runs/03
python run.py --replay --recorded-run results/stability/runs/04
```

编号 02、03、04 沿用原始运行编号，对应补充实验第 1、2、3 次。每次仍重新计算三组的 34 个规定结果文件，核对原始完整运行生成的标准化 SHA-256 指纹。回放成功后会输出完整大表，仓库不用重复保存三套可再生成的大表。`result_fingerprints.json` 还记录原文件字节哈希和来源 ZIP 哈希，详见[来源说明](SOURCES.md)。

`python -m unittest discover -s tests -v` 同时核对补充统计中的逐次记录与原始运行摘要，以及 n=3、样本标准差等统计口径。

## 4. 只运行本地基线

```bash
python run.py --smoke --agent off
python run.py --agent off --no-figures
```

第一条用真实数据的首尾小样本检查流程，不生成正式图；第二条完整运行两组基线。去掉 `--no-figures` 可生成图表。没有智能体组时，输出会如实标注状态。

## 5. 重新调用真实模型

真实调用会使用自己的 API 额度，也可能产生与保存结果不同的策略。Windows 可执行：

```powershell
.\scripts\run_deepseek.ps1
```

启动脚本优先使用项目 `.venv`，隐藏输入密钥，仅在当前进程中使用。跨平台可在终端安全设置 `LLM_BASE_URL=https://api.deepseek.com`、`LLM_MODEL=deepseek-flash` 和自己的 `LLM_API_KEY`，再运行：

```bash
python run.py --agent required
```

模型名称与服务可用性以自己的账号为准；`deepseek-flash` 是保存实验使用的名称。程序不会自动读取 `.env`，`.env.example` 仅是变量说明。可选单价变量只有在填写实际报价时才用于计算费用；保存实验的账单费用未知。

## 输出与常见问题

| 情况 | 处理方式 |
| --- | --- |
| `--out` 目录非空 | 换一个新的空目录；默认会自动新建 |
| 输入/源码哈希不一致 | 确认使用本仓库的原始 CSV 和 `src`，不要为通过检查而改哈希 |
| 数值核验不一致 | 先核对 Python、锁定依赖和运行平台；查看异常指出的文件 |
| 模型 `not_configured / api_error / invalid_plan` | 查看 `agent_status.json`；这不算完成真实智能体实验 |
| CSV 用 Excel 打开乱码 | 本仓库结果和复核 CSV 使用 UTF-8 BOM；原始数据保持原字节 |

新运行位于 `results/run_/`、`results/smoke_/` 或 `results/replay/`，均已忽略。核心流程保留实验对应版本，因此新实验自动生成的 `首轮结果说明.md` 仍含历史交接模板；当前完成状态以本仓库 README、复核材料和补充实验说明为准。

## 2026-11-02理由检查增强

新实验入口现启用理由与策略的有限规则检查和一次澄清；历史四次实验当时仅做结构校验。原始结果仍可离线复算，不能声称所有自然语言理由已获得验证。详见[增强说明](audit/理由一致性增强_1102.md)。
