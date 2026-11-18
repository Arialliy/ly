# Isolated spatial-reference checks

This environment performs finite operator/interface checks. It does not train a detector or score development detections.

Base image: `sha256:eb956873870d737f2b68dff0724b8f6f56f778829d119b0938d29b7abeaa2f92` (Python 3.11.12, Torch 2.7.0+cu128, NumPy 2.2.5). Original project environment is unchanged.

The six packages in `../dependencies/requirements.txt` were downloaded as pinned binary wheels and installed with `--no-deps --target dependencies/site`. `INSTALL_MANIFEST.json` includes wheel and installed-file hashes. The launcher sets PYTHONPATH to this overlay. It uses the image digest, disables network, mounts the original project read-only and only this package writable, and limits CPU threads. The package overlay was not installed into the host or the base image.

Environment spec: `ENV_SPEC.json`, canonical JSON SHA256 `5188ef2ba4bab7919ffe44d1701a6d7be390cda9601308258e1917bc3aa0a9dc`. Provider ledger: `/home/ly/small_object_ideas_20260928/.aris/compute/local_docker.md`.

Run commands on the HOST with `python3`; numerical Python runs inside Docker. Receipt paths must not already exist. For repeats, use a new receipt filename. Only GPUs 1, 2, 3 are authorized. The launcher checks the selected GPU immediately before launch and refuses occupancy >=500 MiB; do not stop other jobs. GPU3 is the intended card for the documented finite witness. If blocked, report the receipt rather than changing cards silently.

Fresh documentation-follow invocation (execute verbatim):

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/route_closeout_20261012/run_reference_checks.py witness --gpu 3 --receipt runtime/DOC_FOLLOW_FINAL_GPU3.launch.json
```

Expected exit 0 and a line beginning `WITNESS`, result status PASS, shape `[2,4,5,17,19]`, finite nonzero gradients for the added parameters. Seed46, deterministic FP32, no AMP/TF32. This verifies CUDA forward/backward through the self-written spatial reference and adapter, not detection benefit. Report any divergence without altering source, environment or invocation.

Other finite checks (separate runs):

```bash
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/route_closeout_20261012/run_reference_checks.py tests --receipt reference_checks/CPU_SUITE_v2.launch.json
python3 /home/ly/small_object_ideas_20260928/idea-stage/infrared_problem_screen_20261009/temporal_diagnosis_20261010/route_closeout_20261012/run_reference_checks.py integration --gpu 3 --receipt reference_checks/DEEPPRO_GPU3_v3.launch.json
```

The official-class check intentionally has no automatic fix. The source snapshot is pinned in `upstream/PROVENANCE.json`; AST-selected class bodies are unchanged. Current official roundtrip failures must remain failures in their receipts; they do not make the checker itself broken and do not establish author-model performance. Its first CPU check is `reference_checks/UPSTREAM_CPU.launch.result.json` (exit1 CHECK_FAILED, no execution errors). No author whole-model checkpoint has been tested in this package.
