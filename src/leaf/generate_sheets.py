#!/usr/bin/env python3


# This is a modified version of https://apsjournals.apsnet.org/doi/suppl/10.1094/PHYTO-01-16-0018-R/suppl_file/PHYTO-01-16-0018-R.sf2.txt
# from: 
# An Improved Method for Measuring Quantitative 
# Resistance to the Wheat Pathogen Zymoseptoria tritici 
# Using High-Throughput Automated Image Analysis

from string import Template
import os

def generate_sheets():
    begin_block = '''
    \\documentclass[a4paper]{minimal}
    \\usepackage[margin=0cm]{geometry}
    \\usepackage{auto-pst-pdf,pst-barcode}
    \\begin{document}'''

    page_block = Template('''
    \\begin{center}
    \\begin{pspicture}[shift=*](20,29)
    \\psline(0,29)(20,29) % top line
    \\psline(0,0)(0,29) % vertical left line
    \\psline(0,0)(20,0) % Bottom line
    \\psline(20,0)(20,29) % vertical right line
    \\psline(1.5,0)(1.5,28) % vertical left line
    \\psline(18.5,0)(18.5,28) % vertical right line
    \\pscircle*(1.5,28.5){0.25} % Circle
    \\pscircle*(18.5,28.5){0.25} % Circle
    \\psline(0,3.5)(20,3.5) % Horizontal lines
    \\psline(0,7)(20,7)
    \\psline(0,10.5)(20,10.5)
    \\psline(0,14)(20,14)
    \\psline(0,17.5)(20,17.5)
    \\psline(0,21)(20,21)
    \\psline(0,24.5)(20,24.5)
    \\psline(0,28)(20,28)
    \\rput(10,28.5){Page {$current_n} of {$total_n}}''')

    qr_block = Template('''
    \\rput(1.6, $loc1){\psbarcode{$name}{width=0.5 height=0.5}{qrcode}} % Top QR Code
    \\rput[Bl](4, $loc2){$name2}''')

    end_block = '''
    \\end{pspicture}
    \\end{center}
    \\end{document}
    '''

    # Get the input file name
    print("Enter the name of the file containing your sample names, followed by [ENTER]:")
    input_file = input()

    # Add .txt extension if not already present
    if not input_file.endswith('.txt'):
        input_file += '.txt'

    # Check if input file exists
    if not os.path.isfile(input_file):
        directory = os.getcwd()
        print(f'File "{input_file}" not found. Check if the file is present in your working directory: {directory}')
        exit()

    # Read input file and create a list of names
    with open(input_file, 'r') as file:
        names = file.read().replace('\n', ' ').split()

    length = len(names)

    # Confirm the contents of the input file
    print(f'Your file contains {length} samples and looks like:')
    print(*names, sep='\n')

    response = input('Does this look correct? Type "n" to exit or "y" to continue:')

    if response.lower() == 'n':
        print('Fix your file and try again')
        exit()

    # Get the output file name
    print("Enter the name of your output file, followed by [ENTER]:")
    output_file = input()

    # Add .tex extension if not already present
    if not output_file.endswith('.tex'):
        output_file += '.tex'

    per_page = 8  # Number of leaves per page
    pages = (length + (per_page - 1)) // per_page  # Round up to the nearest whole page

    # Generate the LaTeX script
    tex_script = begin_block
    # Generate QR codes and names for each page
    num = 0
    for i in range(pages):
        # This block is the general page design 
        tex_script += page_block.substitute(current_n=i+1, total_n=pages)
        # This block is specific for QR codes
        for j in range(per_page):
            if num < length:
                tex_script += qr_block.substitute(name=names[num], 
                                                name2=names[num].replace('_','\_'), 
                                                loc1=26.6-j*3.5, 
                                                loc2=27.5-j*3.5)
    tex_script += end_block

    # Save the .tex script
    with open(output_file, 'w') as file:
        file.write(tex_script)

    # Run pdflatex to generate the PDF
    os.system(f'pdflatex -shell-escape {output_file}')
