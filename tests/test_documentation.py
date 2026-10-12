import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/source"
REFERENCE = re.compile(r"\{([^{}]+)\}")
DECLARATION = re.compile(r"^\s*(--iwes-[\w-]+):\s*(.+);\s*$", re.MULTILINE)


def flatten_tokens(node, path=()):
    if "$value" in node:
        return {path: node["$value"]}
    values = {}
    for name, child in node.items():
        if not name.startswith("$"):
            values.update(flatten_tokens(child, (*path, name)))
    return values


def resolve_token(value, tokens, resolving=()):
    if not isinstance(value, str):
        return str(value)

    def replace_reference(match):
        path = tuple(match.group(1).split("."))
        assert path in tokens, f"Unknown design-token reference: {path}"
        assert path not in resolving, f"Circular design-token reference: {path}"
        return resolve_token(tokens[path], tokens, (*resolving, path))

    return REFERENCE.sub(replace_reference, value)


def variable_name(path):
    parts = []
    for part in path:
        kebab = re.sub(r"(?<!^)(?=[A-Z])", "-", part).lower()
        parts.append(re.sub(r"(?<=[a-z])(?=\d)", "-", kebab))
    return "--iwes-" + "-".join(parts)


def test_sphinx_design_tokens_match_authoritative_source():
    source = json.loads(
        (ROOT / "docs/fraunhofer-design/design-tokens.json").read_text()
    )
    tokens = flatten_tokens(source)
    expected = {
        variable_name(path): resolve_token(value, tokens)
        for path, value in tokens.items()
    }
    declarations = DECLARATION.findall((SOURCE / "_static/iwes-tokens.css").read_text())
    actual = dict(declarations)
    assert len(declarations) == len(actual), "Duplicate CSS token declarations"
    assert actual == expected


def test_sphinx_styles_use_defined_tokens_and_readable_text():
    tokens = dict(DECLARATION.findall((SOURCE / "_static/iwes-tokens.css").read_text()))
    css = (SOURCE / "_static/iwes.css").read_text()
    for name in re.findall(r"var\((--iwes-[\w-]+)\)", css):
        assert name in tokens, f"Undefined corporate token: {name}"
    assert tokens["--iwes-typography-font-size-base"] == "16px"
    for selector in (".md-typeset", ".md-nav", ".md-typeset code", ".md-typeset pre"):
        assert selector in css
    assert "font-size: var(--iwes-typography-font-size-base)" in css
    assert "prefers-reduced-motion: reduce" in css
    assert ":focus-visible" in css


def test_logo_is_present_and_catalog_links_api_instead_of_embedding_classes():
    index = (SOURCE / "index.rst").read_text()
    readme = (ROOT / "README.md").read_text()
    assert ".. image:: ../../Logo_IWOPY.svg" in index
    assert ":alt: IWOPY logo" in index
    assert ":width: 100%" in index
    assert 'src="Logo_IWOPY.svg"' in readme
    assert 'src="Logo_IWOPY.svg" width="100%"' in readme
    assert 'srcset="Logo_IWOPY_white.svg"' in readme
    assert (ROOT / "Logo_IWOPY.svg").is_file()
    assert (ROOT / "Logo_IWOPY_white.svg").is_file()
    css = (SOURCE / "_static/iwes.css").read_text()
    logo_styles = re.search(r"\.md-typeset \.iwopy-logo\s*\{([^}]+)\}", css).group(1)
    assert re.search(r"\bwidth:\s*100%;", logo_styles)
    assert re.search(r"\bheight:\s*auto;", logo_styles)
    catalog = (SOURCE / "optimizer_catalog.rst").read_text()
    assert ".. autoclass::" not in catalog
    for backend in (
        "scipy/optimizer",
        "scipy/de",
        "cma/optimizer",
        "pymoo/optimizer",
        "pygmo/optimizer",
        "nevergrad/optimizer",
        "optuna/optimizer",
        "deap/optimizer",
    ):
        assert f"<_autoapi/iwopy/interfaces/{backend}/index>" in catalog


def test_vectorized_interface_overview_covers_all_adapters_and_public_benchmarks():
    catalog = (SOURCE / "optimizer_catalog.rst").read_text()
    assert "New Vectorized Interfaces" not in catalog
    section = catalog.split("\nVectorized Interfaces\n", 1)[1].split(
        "\nShared Problem Definitions\n", 1
    )[0]
    headings = re.findall(r"^([^\n]+)\n~+\n", section, re.MULTILINE)
    assert headings == [
        "SciPy Minimization",
        "SciPy Differential Evolution",
        "pycma",
        "pymoo",
        "PyGMO",
        "Nevergrad",
        "Optuna",
        "DEAP",
    ]
    assert section.count("for constructor parameters and methods.") == 8
    assert "vectorize=False" in section and "vectorize=True" in section
    assert "from iwopy.benchmarks import ParetoProblem" in catalog
    assert "<_autoapi/iwopy/benchmarks/quadratic/index>" in catalog


@pytest.mark.parametrize("path", sorted(SOURCE.glob("*.rst")))
def test_literalinclude_targets_exist(path):
    for include in re.findall(
        r"^\s*\.\. literalinclude::\s+(.+)$", path.read_text(), re.MULTILINE
    ):
        assert (path.parent / include.strip()).is_file(), (
            f"Missing include in {path.name}: {include}"
        )
