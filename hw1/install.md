install latexmk extension
use configs files in .vscode 

sudo apt update && sudo apt install -y \
  texlive texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
  texlive-bibtex-extra biber latexmk texlive-xetex

  sudo apt install -y

  sudo apt install -y python3-pygments

  cd /home/ant/projects/Electrotechnics_electronics_homeworks/hw1/Latex
latexmk -pdf Main.tex
# If using minted/includesvg:
latexmk -pdf -shell-escape Main.tex

pdflatex --version
xelatex --version
biber --version

sudo apt update
sudo apt install -y inkscape

sudo apt update
sudo apt install -y texlive-lang-cyrillic cm-super