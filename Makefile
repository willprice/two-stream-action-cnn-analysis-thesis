PANDOC:=pandoc
LATEX:=pdflatex
GPP:=gpp
INKSCAPE:=inkscape

BIBLIOGRAPHY:=$(HOME)/references.bib
THESIS:=thesis
POSTER_PDF:=poster.pdf
INCLUDE_PATHS:=fragments\
			  diagrams
SVG_IMAGES := $(wildcard media/images/*.svg)
IMAGES := $(SVG_IMAGES:.svg=.png)
TIKZ_SRC := $(shell find diagrams -type f -name '*.tikz')

BUILD_DIR:=build
TIKZ_DEST := $(patsubst %.tikz,$(BUILD_DIR)/%.png,$(TIKZ_SRC))

MARKDOWN_EXTENSIONS=link_attributes footnotes definition_lists
PANDOC_OPTIONS:=--latex-engine=xelatex\
			   --filter=pandoc-crossref\
			   --filter=pandoc-citeproc --csl computer.csl\
			   --bibliography=$(BIBLIOGRAPHY)
# Macros are defined in a style like HTML mode for GPP
# e.g. `<##define note|\textcolor{red}{#1}>`
# and usage `<##note Help>` yields `\textcolor{red}{Help}`

# GPP_OPTION Cheat sheet
# -x     Enable `#exec` macro for running arbitrary commands
# -Idir  Use `dir` as include directory
# -T     TeX-like mode (\define{x}{y})
# -H     HTML-like mode (<#define x|y>)
GPP_MACRO_START:='<\#\#'
GPP_MACRO_END_WITHOUT_ARGS:='>'
GPP_ARG_START:='\B'
GPP_ARG_SEP:='|'
GPP_ARG_END:='>'
GPP_ARG_STACK:='<'
GPP_ARG_UNSTACK:='>'
GPP_ARG_NUM_REF:='\#'
GPP_QUOTE:=''
GPP_OPTIONS:=$(foreach fragment,$(INCLUDE_PATHS), -I$(fragment)) \
			-x \
			-U $(GPP_MACRO_START) $(GPP_MACRO_END_WITHOUT_ARGS) \
         $(GPP_ARG_START) $(GPP_ARG_SEP) $(GPP_ARG_END) \
         $(GPP_ARG_STACK) $(GPP_ARG_UNSTACK) $(GPP_ARG_NUM_REF) \
         $(GPP_QUOTE)


# See http://gnu-make.2324884.n4.nabble.com/concatenate-words-with-no-space-td9268.html
# for details on how this trick for concatenating strings without spaces works
_SPACE := #empty var
SPACE := $(_SPACE) $(_SPACE)
all: $(BUILD_DIR)/$(THESIS).pdf $(BUILD_DIR)/$(THESIS).tex $(BUILD_DIR)/$(POSTER_PDF) $(TIKZ_DEST) $(IMAGES) $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p "$@"

$(BUILD_DIR)/%.md.pp: %.md
	$(GPP) -O "$@"\
		$(GPP_OPTIONS) \
		$< >/dev/null

$(BUILD_DIR)/%.tex: $(BUILD_DIR)/%.md.pp $(BUILD_DIR) $(IMAGES) $(TIKZ_DEST)
	$(PANDOC) \
		$(PANDOC_OPTIONS)\
		$<\
		--standalone\
		--from=markdown$(subst $(SPACE),,$(foreach ext,$(MARKDOWN_EXTENSIONS),+$(ext)))\
		--to=latex\
		--output=$@\

$(BUILD_DIR)/%.pdf: $(BUILD_DIR)/%.md.pp $(BUILD_DIR) $(IMAGES) $(TIKZ_DEST)
	$(PANDOC) \
		$(PANDOC_OPTIONS)\
		$<\
		--standalone\
		--from=markdown$(subst $(SPACE),,$(foreach ext,$(MARKDOWN_EXTENSIONS),+$(ext)))\
		--to=latex\
		--output=$@\

$(BUILD_DIR)/%.png: %.tikz $(BUILD_DIR)
	mkdir -p "$(dir $@)"
	$(GPP) $(GPP_OPTIONS) -O $(basename $@).tex -D"FILE=$<" diagrams/tikz-document-template.tex
	cd "$(dir $@)" && $(LATEX) -halt-on-error -shell-escape "$(notdir $(basename $@)).tex"

%.png: %.svg
	$(INKSCAPE) \
		--without-gui \
		--export-png="$@" \
		--export-width=1024 \
		$<

$(BUILD_DIR)/$(POSTER_PDF): poster.tex $(IMAGES) $(TIKZ_DEST)
	$(LATEX) -halt-on-error -shell-escape "$<"
	mv $(POSTER_PDF) "$@"

.PHONY: all
