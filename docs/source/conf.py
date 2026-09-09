import pathlib
import sys

repository_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))

from iwopy import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "iwopy"
copyright = "2026, Fraunhofer IWES"
author = "Fraunhofer IWES"
version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "numpydoc",
    "sphinx_immaterial",
    "autoapi.extension",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "myst_nb",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
}

# Source file types handled by Sphinx and MyST-NB.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

master_doc = "index"
language = "en"
exclude_patterns = [
    "build",
    "_iwopy",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]
pygments_style = None

numpydoc_use_rtype = False
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
autosectionlabel_prefix_document = True

# -- Autodoc options ---------------------------------------------------------

autodoc_typehints = "signature"
autodoc_class_signature = "separated"

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_immaterial"
html_theme_options = {
    "site_url": "https://fraunhoferiwes.github.io/iwopy.docs/index.html",
    "repo_url": "https://github.com/FraunhoferIWES/iwopy",
    "icon": {"repo": "fontawesome/brands/github", "edit": "material/file-edit-outline"},
    "palette": {"primary": "teal"},
    "toc_title_is_page_title": True,
}
htmlhelp_basename = "iwopydoc"

latex_documents = [
    (master_doc, "iwopy.tex", "iwopy Documentation", "Fraunhofer IWES", "manual"),
]
man_pages = [(master_doc, "iwopy", "iwopy Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "iwopy",
        "iwopy Documentation",
        author,
        "iwopy",
        "Optimization tools in Python",
        "Miscellaneous",
    ),
]
epub_title = project
epub_exclude_files = ["search.html"]

# -- AutoAPI configuration --------------------------------------------------

autoapi_dirs = [str(repository_root / "iwopy")]
autoapi_root = "_autoapi"
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "inherited-members",
]
autoapi_python_class_content = "both"
autoapi_member_order = "groupwise"
autoapi_python_use_implicit_namespaces = False
autoapi_keep_files = False
autoapi_ignore = ["*/tests/*", "*/__pycache__/*"]

# -- Notebook configuration -------------------------------------------------

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]
nb_execution_mode = "auto"
nb_execution_timeout = 300
nb_ipywidgets_js = {
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js": {
        "integrity": "sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=",
        "crossorigin": "anonymous",
    },
    "https://cdn.jsdelivr.net/npm/@jupyter-widgets/html-manager@*/dist/embed-amd.js": {
        "data-jupyter-widgets-cdn": "https://cdn.jsdelivr.net/npm/",
        "crossorigin": "anonymous",
    },
}

suppress_warnings = ["mystnb.unknown_mime_type"]
