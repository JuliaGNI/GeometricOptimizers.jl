# The figures on the manifold and optimizer pages, compiled from their TikZ sources.
#
# They came here from `GeometricMachineLearning`'s `docs/src/tikz/Makefile` along with the chapters
# that use them, trimmed to the six sources those chapters reference. Each figure exists twice, once
# for each Documenter theme; the pages include both and CSS shows the right one.
#
# `all` is what the documentation workflow runs before `make.jl`. The PNGs are build products and
# are not committed — `.gitignore` covers them — so a local `julia --project=docs docs/make.jl`
# without this step will report six missing images and nothing else.
#
# Needs `xelatex` (texlive-xetex, texlive-science) and `pdftocairo` (poppler-utils).

MYDIR = .

all: pdf
	$(MAKE) png res=150
	$(MAKE) clean

# `latex` renders at a higher resolution, for the PDF build rather than the HTML one.
latex: pdf
	$(MAKE) png res=500
	$(MAKE) clean

pdf: $(MYDIR)/*.tex
	for file in $^ ; do \
		xelatex -shell-escape $${file} ; \
		xelatex -shell-escape $${file} ; \
	done

# `-transp` keeps the background transparent, which is what lets the dark variant sit on a dark page
png: $(MYDIR)/*.pdf
	for file in $^ ; do \
		pdftocairo -png -r $(res) -transp -singlefile $${file} $${file%.*} ; \
	done

clean:
	rm -f *.aux
	rm -f *.fdb_latexmk
	rm -f *.fls
	rm -f *.log
	rm -f *.xdv
	rm -f *.pdf

empty: clean
	rm -f *.png
