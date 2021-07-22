extensions = [
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.imgmath',  # Render math as images
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.intersphinx'
]
master_doc = 'index'

intersphinx_mapping = {#"nifty8": ("https://ift.pages.mpcdf.de/nifty/", None),
                       "numpy": ("https://numpy.org/doc/stable/", None),
                       "ducc0": ("https://mtr.pages.mpcdf.de/ducc/", None),
                       }

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_references = True
napoleon_include_special_with_doc = True

project = u'resolve'
copyright = u'2019-2021, Max-Planck-Society'
author = u'Philipp Arras'

# FIXME release = resolve.version.__version__
# FIXME version = release[:-2]

language = None
exclude_patterns = []
add_module_names = False

html_theme = "pydata_sphinx_theme"
html_theme_options = {"gitlab_url": "https://gitlab.mpcdf.mpg.de/ift/resolve"}
