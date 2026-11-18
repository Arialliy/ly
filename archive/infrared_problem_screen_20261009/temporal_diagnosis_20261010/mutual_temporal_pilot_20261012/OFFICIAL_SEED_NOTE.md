# 官方源码与本轮seed的区别

核对日期：2026-10-12。范围为本项目冻结的本地官方源码，不代表作者训练发布权重时的完整私有运行环境。

- 官方[train.py第33行](../../official_deeppro/train.py:33)定义`seed_everything(seed=46)`，内部设置Python、NumPy、PyTorch及CUDA种子。
- 全文AST中没有对`seed_everything`的直接调用；主入口仅解析参数并调用`main(args)`。全仓文本检索亦未找到其他调用。函数默认46不等于运行时已启用。
- [第122行](../../official_deeppro/train.py:122)的DataLoader worker使用`np.random.seed(worker_id + int(time.time()))`，引入当前时间。
- 官方test.py未找到设置随机种子的语句。
- 本轮[train_mutual.py第155行](code/train_mutual.py:155)明确执行Python/NumPy/PyTorch/CUDA的seed46；三组实际输入与标签哈希一致，E1/E2新增与原参数初值均一致，见[训练审计](TRAINING_AUDIT.md)。

因此：**源码种子函数的默认参数是46；官方发布权重实际训练seed未知；本轮实验固定seed46已落实。** 不将本轮固定种子解释为作者完整训练复现，不新增其他seed。
