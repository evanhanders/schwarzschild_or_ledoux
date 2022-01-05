.PHONY: main.pdf all clean

all: ms.pdf

ms.pdf: ms.tex intro.tex theory.tex results.tex conclusions.tex  biblio.bib appendix_table.tex appendix_simulations.tex appendix_model.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make ms.tex

clean:
	latexmk -CA
