"""
This script was used to convert pdf file specific pages into cropped png
"""

import PyPDF2
from wand.image import Image
from wand.color import Color
import os
from math import floor
import io


def pdf_page_to_png(src_pdf, pagenum = 0, resolution = 72,):
    """
    Returns specified PDF page as wand.image.Image png.

    :param PyPDF2.PdfFileReader src_pdf: PDF from which to take pages.
    :param int pagenum: Page number to take.
    :param int resolution: Resolution for resulting png in DPI.
    """
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))

    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    img = Image(file = pdf_bytes, resolution = resolution)
    img.convert("png")
    img.background_color = Color('white')
    img.alpha_channel = 'remove'

    return img


# Main
# ==== change here for different pdf file
#src_filename = "lot_plan_01A001.pdf"

#src_pdf = PyPDF2.PdfFileReader(src_filename, "rb")

# What follows is a lookup table of page numbers within sample_log.pdf and the corresponding filenames.
##pages = [{"pagenum": 22,  "filename": "samplelog_jrs0019_p1"},
##         {"pagenum": 23,  "filename": "samplelog_jrs0019_p2"},
##         {"pagenum": 124, "filename": "samplelog_jrs0075_p3_2011-02-05_18-55"},]

# Convert each page to a png image.
##for page in pages:
##    big_filename = page["filename"] + ".png"
##    small_filename = page["filename"] + "_small" + ".png"
##
##    img = pdf_page_to_png(src_pdf, pagenum = page["pagenum"], resolution = 300)
##    img.save(filename = big_filename)
##
##    # Ensmallen
##    img.transform("", "200")
##    img.save(filename = small_filename)

path = '/Users/rz2333/Downloads/Study/BU/Fall_2018/CS542_ml/Final_project/Data/Eko'
pngpath = '/Users/rz2333/Downloads/Study/BU/Fall_2018/CS542_ml/Final_project/Data/Eko_png'

def main():
    allfile = os.listdir(path)
    for file in allfile:
        if file != '.DS_Store':
        #big_filename = "lot_plan_01A001"+ ".png"
            pdf_filename = os.path.join(path, file)
            png_filename = pdf_filename.split('.')[0]
            #small_filename = "lot_plan_01A001"+ "_small" + ".png"
            src_pdf = PyPDF2.PdfFileReader(pdf_filename, "rb")
    
            img = pdf_page_to_png(src_pdf, pagenum = 0, resolution = 300)
            w,h = img.size   ###crop the image propotionally
            #Crop
            #img.crop(left = 100, right = 3500,bottom = 3300)
            img.crop(left = floor(w*1/15), right = floor(w*13/15), top = floor(h*1/10), bottom = floor(h*9/10))
            img.save(filename = os.path.join(pngpath, png_filename+'.png'))
    
        # Ensmallen
        #img.transform("", "200")
        #img.save(filename = big_filename)

# Deal with the cropping for JRS0070.
##jrs0070 = {"pagenum": 109, "filename": "samplelog_jrs0070_p1"}
##
##img = pdf_page_to_png(src_pdf, pagenum = jrs0070["pagenum"], resolution = 300)
##
##big_filename = jrs0070["filename"] + ".png"
##small_filename = jrs0070["filename"] + "_small" + ".png"
##
### Crop
##img.crop(bottom = 1000)
##
### Save
##img.save(filename = big_filename)
##
### Ensmallen
##img.transform("", "200")
##img.save(filename = small_filename)
if __name__=='__main__':
    main()
