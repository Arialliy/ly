"""CPU-only import/constructor validation; not a CUDA numerical test.

Run this script directly in the pinned image without exposed GPU devices.
"""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    experiment = Path(__file__).resolve().parents[2]
    repository = experiment.parents[1]
    original_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("fresh_pda_import_audit", experiment / "code/pda.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "torch" not in sys.modules
    assert sys.path == original_path
    first = module.build_pda()
    second = module.build_pda()
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_available(), "Run the import check without GPU devices exposed."
    assert first.conv_l2_1.in_channels == 16
    assert first.dcn_pack["l1"].deformable_groups == 4
    assert set(p.data_ptr() for p in first.parameters()).isdisjoint(p.data_ptr() for p in second.parameters())
    assert not torch.equal(first.conv_l2_1.weight, second.conv_l2_1.weight)
    assert sys.path == original_path
    assert "DCN2" not in sys.modules
    native = sys.modules["_deeppro_m1_source_relations_rfr.dcn._rfr_dcn"]
    assert Path(native.__file__).parent == experiment / "upstream/rfr/dcn"
    assert not any("m1_recurrent" in str(getattr(item, "__file__", "")) for item in list(sys.modules.values()))
    try:
        first(torch.randn(1, 16, 8, 8), torch.randn(1, 16, 8, 8))
    except RuntimeError as exc:
        assert "not implement" in str(exc).lower() and "cpu" in str(exc).lower(), str(exc)
        cpu_error = str(exc).splitlines()[0]
    else:
        raise AssertionError("The official CPU stub unexpectedly returned an alignment.")
    report = {
        "status": "passed", "python": sys.version.split()[0],
        "torch": torch.__version__, "cuda_toolkit": torch.version.cuda,
        "lazy_import_without_torch": True, "sys_path_unchanged": True,
        "fresh_parameter_storage_and_initialization": True,
        "previous_m1_imported": False, "global_DCN2_imported": False,
        "native_extension": str(Path(native.__file__).relative_to(repository)),
        "pda_parameters": sum(p.numel() for p in first.parameters()),
        "cpu_forward_error": cpu_error, "cuda_forward_backward_tested": False,
    }
    (experiment / "upstream/rfr/IMPORT_CHECK.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
