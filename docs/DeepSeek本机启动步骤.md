# DeepSeek本机启动步骤

本轮通过安全提示输入新密钥，直接启动真实模型实验。密钥只放在本次子进程环境，不写入源码、日志或配置文件；关闭后不需要重启Codex。

1. 登录DeepSeek开放平台，进入API keys页面，撤销曾经贴到聊天中的密钥并创建新密钥；确认账户有可用额度。不要把新密钥再贴到聊天或GitHub。
2. 打开Windows PowerShell，执行下面两行。本机已具备本轮需要的Python和依赖，脚本会自动使用它们。

```powershell
cd 'C:\Users\user\Documents\Codex\2026-10-26\google-play-26-27-25-codex\outputs\google-play-project'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run_deepseek.ps1
```

3. 出现`API key:`提示时，粘贴刚生成的新密钥并按回车。输入不显示明文，这是正常现象。不要把密钥拼接在命令后面。
4. 等待程序结束。脚本使用DeepSeek官方API基础地址和deepseek-flash模型，关闭思考模式并将输出限制为4096 tokens；只发质量统计和少量样例。终端应依次出现三组运行信息，最后输出本次目录和模型状态。
5. 成功标志为最后的JSON含`"groups": ["minimum", "fixed", "agent"]`及`"agent_status": "completed"`。将最后的output目录发给Codex即可，不要发送密钥；随后复核真实响应、三组结果和图表并更新交付包。

如果失败，只提供错误码或最后的状态。401通常为密钥认证失败；402为余额不足；429为限流；500/503为服务端错误。本机脚本会清理或恢复临时环境变量。不要反复运行以免产生不必要调用；保留本次失败目录用于核查。

该命令的ExecutionPolicy参数只作用于这一个PowerShell子进程，不修改全局执行策略。迁移到其他电脑时，先按README建立.venv并安装requirements.lock.txt；当前机器无需重复安装。

来源：https://api-docs.deepseek.com/ 、https://api-docs.deepseek.com/guides/json_mode/ 、https://api-docs.deepseek.com/guides/thinking_mode/ 、https://api-docs.deepseek.com/quick_start/error_codes/ 。
