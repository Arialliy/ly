"""Source-selection checks; import isolation must not rewrite author classes."""
import ast

import pytest

from load_official_lfp import CLASS_NAMES, DEFAULT_SOURCE, load_official_lfp, selected_class_tree


def test_selected_class_nodes_are_unchanged():
    source = DEFAULT_SOURCE.read_text(encoding="utf-8")
    original = ast.parse(source)
    selected = selected_class_tree(source)
    actual = {node.name: node for node in selected.body}
    expected = {node.name: node for node in original.body
                if isinstance(node, ast.ClassDef) and node.name in CLASS_NAMES}
    assert set(actual) == set(expected) == set(CLASS_NAMES)
    for name in CLASS_NAMES:
        assert ast.dump(actual[name], include_attributes=True) == ast.dump(expected[name], include_attributes=True)


def test_selected_tree_has_no_imports_or_other_top_level_statements():
    selected = selected_class_tree(DEFAULT_SOURCE.read_text(encoding="utf-8"))
    assert len(selected.body) == len(CLASS_NAMES)
    assert all(isinstance(node, ast.ClassDef) for node in selected.body)


def test_missing_or_duplicate_classes_fail():
    source = "\n".join(f"class {name}:\n    pass\n" for name in CLASS_NAMES)
    with pytest.raises(ValueError, match="exactly one"):
        selected_class_tree(source + "\nclass ConvDWT:\n    pass\n")
    with pytest.raises(ValueError, match="exactly one"):
        selected_class_tree("class ConvDWT:\n    pass\n")


def test_checksum_failure_precedes_dependency_import():
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_official_lfp(DEFAULT_SOURCE, expected_sha256="0" * 64)
