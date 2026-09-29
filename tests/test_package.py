import iwopy


def test_package_version_is_nonempty_string():
    assert isinstance(iwopy.__version__, str)
    assert iwopy.__version__