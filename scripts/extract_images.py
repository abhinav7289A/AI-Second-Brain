import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_folder):
    """
    Extracts all images from the PDF and saves them to output_folder.
    Filenames: <pdf_name>_page<page>_img<index>.<ext>
    """
    doc = fitz.open(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_data = base_image["image"]
            img_ext = base_image.get("ext", "png")
            out_name = f"{pdf_name}_page{page_index+1}_img{img_index}.{img_ext}"
            out_path = os.path.join(output_folder, out_name)
            with open(out_path, "wb") as img_file:
                img_file.write(img_data)
            print(f"[extract_images] Saved {out_name}")

def main():
    base_dir = os.path.dirname(__file__)
    input_dir = os.path.join(base_dir, "..", "data", "raw_pdfs")
    output_dir = os.path.join(base_dir, "..", "data", "diagrams")
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(input_dir, fname)
        extract_images_from_pdf(pdf_path, output_dir)

if __name__ == "__main__":
    main()
