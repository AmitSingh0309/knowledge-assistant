import PyPDF2


with open("Profile Maharath Infracon.pdf", "rb") as f:
    pdf = PyPDF2.PdfReader(f)

    print(f"Total pages: {len(pdf.pages)}")

    first_page = pdf.pages[0].extract_text()
    print("\n=== FIRST PAGE TEXT ===")
    print(first_page[:500])

    if len(pdf.pages) > 1:
        second_page = pdf.pages[1].extract_text()
        print("\n=== SECOND PAGE TEXT ===")
        print(second_page)

