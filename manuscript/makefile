.PHONY: main.pdf all clean

all: ms.pdf

ms.pdf: ms.tex intro.tex results.tex  biblio.bib appendix_table.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make ms.tex

clean:
	latexmk -CA
