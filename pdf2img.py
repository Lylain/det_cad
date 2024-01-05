import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import os
path = "../pdf/"
outpath = "../images/"
out_format = "png"
dpi = 1200
# 打开PDF文件
zoom = dpi / 72
mat = fitz.Matrix(zoom, zoom)
for pdf_file in os.listdir(path):
    
    if pdf_file[-3:] == "pdf":
        print(pdf_file)
    #     doc = fitz.open(path + pdf_file)

    #     # 遍历PDF的每一页
    #     for page_num in range(len(doc)):
    #         # 获取页面
    #         page = doc.load_page(page_num)

    #         # 将页面渲染为图像
    #         pix = page.get_pixmap(matrix = mat, alpha=False, colorspace="RGB")
    #         print(pix.width, pix.height)
    #         img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    #         # 保存图像
    #         # img.save(outpath + pdf_file[:-3] + out_format)
    #         img.save('tmp.png')
    #     # 关闭PDF文件
    #     doc.close()
    # break
        images = convert_from_path(poppler_path = r"E:\ChromeDownload\Release-23.11.0-0\poppler-23.11.0\Library\bin", pdf_path= path + pdf_file, dpi=600)
        for i, image in enumerate(images):
            image.save(outpath + pdf_file[:-3] + out_format)
