# 全部历史材料

用户要求将本项目数据、代码、结果、权重全部放到统一文件夹。旧项目根目录已迁入此处，旧宿主机路径不再存在；没有删除实验结果或权重。

- `infrared_problem_screen_20261009/`：运行用完整归档。包含所有历史阶段的数据、结果、报告、权重、预测与源代码；原符号链接已展开，不依赖外部旧路径。排除Git内部、缓存和另行收录的第三方依赖。
- `original_project/`：原项目目录的整体迁移，保留Git来源、原目录结构和来源凭据。43个原符号链接已调整为包内目标。迁移为同文件系统目录rename，没有删除文件。
- `../data/stage1/`：冻结训练／开发／确认输入的直达入口。
- `../environment/site/`：已验证的第三方Python依赖。

两棵历史树的大多数实体共享原文件inode，不是重新复制一份几十GB的数据。历史文件不原地修改；新实验写入`../experiments/`。

收录与逐文件SHA见[ALL_LOCAL_COLLECTION.json](../provenance/ALL_LOCAL_COLLECTION.json)，迁移见[PROJECT_RELOCATION.json](../provenance/PROJECT_RELOCATION.json)。历史报告里的绝对旧路径作为当时来源记录保留；当前执行入口通过容器将本目录映射到这些兼容路径，宿主机不需要原项目目录。
