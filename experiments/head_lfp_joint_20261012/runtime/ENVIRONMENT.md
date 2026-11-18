# H/HL 本地环境与独立执行核验

复用固定镜像 `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92`，软件版本及已验证kernel不变；不安装升级。新运行挂载、协议范围已登记，规范化spec SHA256：`011f48e3d303459771d27e2afda2f09b1afc5260c1467cafb7fe3abceafdeab0`。CPU线程2，容器4CPU、20GiB、网络关闭，GPU只用已授权且空闲的1/2/3。

所有实验数据与源文件从统一包读取：包内archive映射到历史代码约定路径，只读；不挂载原项目。E可写。依赖复用包内`environment/site`。

fresh代理应在主机逐字执行：

```bash
python3 /home/ly/DeepProPlus_TLFP_Code/experiments/head_lfp_joint_20261012/code/launch.py --kind preflight --gpu 3 --tag doc_follow_v2
```

预期exit0与`checks/GPU_INTEGRATION_doc_follow_v2.json`为PASS。验证实际模型/发布权重、4×40×128×128训练来源批次、273/373逐名范围、head更新/其余原参数及BN冻结、梯度穿过TPro、分块一致性和AMP诊断。每组一次丢弃的集成更新，不保存正式训练权重；所有正式训练重新从发布权重初始化。若GPU被占或指令失败，原样报告，不停止外部任务、不自行修改。

正式执行另由主代理在代码与协议freeze通过后启动。CPU测试已在实际模型完成，不把附件声称的31合成测试当作本机证据。

首次doc_follow在Docker挂载阶段exit125：只读archive内无被排除的依赖挂载点。尚未启动Python/训练。v2删除多余嵌套挂载，PYTHONPATH直接指向包内environment/site，依赖字节和镜像不变。原失败日志保留。
