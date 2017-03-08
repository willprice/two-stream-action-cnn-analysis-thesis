PANDOC:=pandoc
LATEX:=pdflatex
GPP:=gpp
INKSCAPE:=inkscape

BIBLIOGRAPHY:=$(HOME)/references.bib
THESIS_PDF:=thesis.pdf
POSTER_PDF:=poster.pdf
INCLUDE_PATHS:=fragments\
			  diagrams
BUILD_DIR:=build
SVG_IMAGES := $(wildcard media/images/*.svg)
IMAGES := $(SVG_IMAGES:.svg=.png)
TIKZ_SRC := $(shell find diagrams -type f -name '*.tikz')
TIKZ_DEST := $(patsubst %.tikz,$(BUILD_DIR)/%.png,$(TIKZ_SRC))

MARKDOWN_EXTENSIONS=link_attributes
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


all: $(BUILD_DIR)/$(THESIS_PDF) $(BUILD_DIR)/$(POSTER_PDF) $(TIKZ_DEST) $(IMAGES) $(BUILD_DIR)

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.pdf: %.md $(BUILD_DIR) $(IMAGES) $(TIKZ_DEST)
	$(GPP) -O $(BUILD_DIR)/$<.pp\
		$(GPP_OPTIONS) \
		$< >/dev/null

	$(PANDOC) \
		$(PANDOC_OPTIONS)\
		$(BUILD_DIR)/$<.pp\
		--standalone\
		--from=markdown$(foreach ext,$(MARKDOWN_EXTENSIONS),+$(ext))\
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
	mkdir -p "$(dir $@)"
	$(LATEX) -halt-on-error -shell-escape "$(notdir $(basename $@)).tex"

.PHONY: all
