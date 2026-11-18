# 训练／推理实现独立核对

2026-10-12。**PASS：31项核对通过，无待解决的实现阻断。** 只读检查源码，并使用实际发布权重完成CPU初始化/输出核对；未运行正式训练或GPU。

实际检查：E0为70,913参数、无hook；E1/E2均72,194参数、各一个layer1输出hook。全部原参数保持不变；E1/E2完整初始状态逐位相同；40帧小裁剪上零初始化输出与E0逐位相同。

- **E0 has no added parameters or hook**：attach_aggregation returns None before constructing any module; CPU actual-load check confirms70,913parameters/zerohooks
- **E1 and E2 use one module class**：Only VARIANTS mode argument changes; architecture and parameter capacity are identical
- **E1/E2 bitwise initial state**：Independent resetseed46 actualcheckpoint loads and attaches give exact fullstate equality
- **Original state unmodified**：Before/afterattach actualreleased tensors identical for all3variants
- **Insertion before TPro**：layer1 forwardhook transforms result before original forward permutes and calls TPro
- **Initial model output equivalent**：ActualCPU model input1x1x40x9x10; E1/E2 outputs bitwise equalE0, maxerror0
- **Positive window loss unchanged**：common_loss calls unchanged officialcriterion on logits/target validgather perwindow
- **Only empty labels trigger top1**：torch.count_nonzero gatheredtarget decidesbranch; nonempty takes originalcriterion
- **Empty negative operator unchanged**：Imports oldbackground_top1pct_mean, applies to softplus(backgroundlogits), no sigmoid/binarythreshold
- **Empty valid mask equality enforced**：Emptywindow asserts torch.equal(valid,background) beforechangingloss
- **Four-window outer reduction**：torch.stack(applied).mean() retains old equalwindowweight
- **Identical masks and exposure**：Reuses build_batch/build_targets and compares each input/labelSHA to512oldlogrows
- **Frozen dataset identities**：previous.validate_inputs hashesplan,selection,release,originalmodel/helper; validatesdatafingerprints
- **Frozen optimizer and precision**：FreshAdam1e-4wd1e-4; seed46; FP32; deterministic; TF32off; no scheduler/clip/AMP
- **All parameter adaptation**：Every original/new parameter requiresgradTrue and includedAdam
- **BN statistics frozen**：model.train then everyBN.eval; eachstep comparesallBNrunningbuffers toinitialsnapshot
- **Fixed endpoint512**：Exactly512validatedplanentries; snapshots0/128/256/384/512; hardlinked final512
- **No checkpoint selection**：No developmentmetric read from train; no best selection
- **Source and checkpoint provenance**：Snapshot recordsinitialreleaseSHA,plan,labelconfig,current sourcehashes; end verifies sourcesunchanged
- **Inference strict state**：load_state_dict strictTrue, variant/step/radius/sourceSHA/snapshotSHA/plan/labelconfig validated
- **Diagnostics restricted**：mass/current onlyE2 step512 and never accepted in formaltrain variants
- **No training in inference**：eval plus torch.inference_mode; no optimizer
- **Frozen five-window inference**：Eightstartunion O/S,probabilitymax,core128halo20; originalnormalization
- **No GT influence on model output**：OptionalGTqueries only passive sparsehook; denseforward path unchanged
- **Inference stats policy common**：collect_statsFalse for E1/E2/all diagnostic modes
- **Module gradient and boundary tests**：137modulechecks including knownscoreoracle, outputprojectionstart, descriptorsafterupdate, 2rtiling
- **Loss numerical tests**：13TRAIN_LOSS_CHECK entries pass, including exactoldpositivevalue/gradient and emptynegative support
- **radius_cli_resolution**：args.radius !=2 rejected beforetrain.
- **effective_supervision_metadata_resolution**：EFFECTIVE_SUPERVISION storedin RUN andeverycheckpoint, explicitlysupersedes historicalempty-loss note; inferchecks equality.
- **prior_train_dependency_resolution**：Inference expected_sources includes prior_train_source_sha256 andchecks run/checkpoint.
- **telemetry_cost_scope_resolution**：Inference metadata peak_memory_scope explicitlystates E2provenanceinclusion; cleanpreflightmodelpeak shouldbeusedforstrictmodelcomparison.

已完成的元数据细节修正：训练半径CLI固定2；有效监督说明明确覆盖历史LABEL_CONFIG中的空窗旧说明；推理补核对导入的旧训练助手SHA；E2稀疏来源记录下的显存峰明确标记包含记录开销。

核对绑定的源码SHA和CPU证据见[JSON](RUNNER_SOURCE_REVIEW.json)。本核对不能替代GPU资源预检、完整非零分块检验或正式训练后审计，也不证明检测收益。
