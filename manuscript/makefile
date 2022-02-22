.PHONY: main.pdf all clean

all: ms.pdf

ms.pdf: ms.tex intro.tex theory.tex results.tex conclusions.tex  biblio.bib appendix_simulations.tex appendix_model.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make ms.tex; texcount -v3 -merge -incbib -dir -sub=none -utf8 -sum ms.tex

clean:
	latexmk -CA
