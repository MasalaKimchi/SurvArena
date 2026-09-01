from pygments.lexers.special import TextLexer
from sphinx.highlighting import lexers

project = "SurvArena"

extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "alabaster"
nitpicky = True

myst_heading_anchors = 6

lexers["mermaid"] = TextLexer()
