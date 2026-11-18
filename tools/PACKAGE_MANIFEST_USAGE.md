# 最终整包清单

工具：`tools/package_manifest.py`（仅 Python 标准库，不导入模型）。

先等待所有训练、推理、评分和文档写入结束，完成 `PACKAGE_STATUS.json`、结果与验证记录，再执行：

```bash
python3 tools/package_manifest.py generate
python3 tools/package_manifest.py verify
```

默认根目录由脚本位置确定，迁移整个文件夹后仍可使用。`--root` 可指定小测试目录。默认生成、校验均对每个不同的设备号/inode 直接计算一次 SHA256，同一文件的多个硬链接复用该次计算。统计同时报告逻辑文件大小与去重 inode 大小；不把符号链接目标重复计入大小。

唯一排除项是根目录 `PACKAGE_MANIFEST.json` 自身。隐藏文件、环境依赖、Git 来源记录、历史结果、所有实验输出及空目录均纳入。符号链接记录原始目标与解析后的相对目标，必须有效且处于所选根目录内。

若已有清单，会先逐字节备份为 `provenance/PACKAGE_MANIFEST_before_<时间戳>.json`，备份也纳入新清单。默认不使用历史哈希缓存。如确实需要增量生成，可显式运行：

```bash
python3 tools/package_manifest.py generate --reuse-prior-stat-cache
```

缓存仅来自本工具此前的 COMPLETE 清单，并要求设备号、inode、大小、mtime_ns、ctime_ns 全部不变。使用依据及复用计数写入清单。不仅凭文件名或大小采用其他来源清单的哈希。`verify` 始终重新计算每个当前 inode 的 SHA256，不使用上一次哈希缓存。

前后两遍完整扫描检测操作期间的文件变化；这不是文件系统快照或写入锁，不能替代“所有写入已结束”这个前提。生成失败时原清单保留，但可能已产生备份，需要下次生成将它一起纳入。

完成生成后，**不要再往包内写日志或验证报告**，也不要将这两个命令的 stdout 重定向至包内文件，否则会改变被校验集合。校验结果打印到调用方即可。若后续编辑文件，重新生成并校验；旧清单会自动保留。

小临时树验证记录见 `validation/MANIFEST_TOOL_CHECK.json`，包含硬链接、旧清单备份、复制后校验、缓存使用、内容变化、坏链接、外部链接及并发写入拒绝等检查。该记录不表示最终整包清单已经生成。
