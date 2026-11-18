from pypdf import PdfReader

reader = PdfReader("data/test.pdf")
found = False

for i, page in enumerate(reader.pages):
    text = page.extract_text()
    
    # 检查有没有完整的“第十一条”
    if "第十一条" in text:
        print(f"🎉 找到了！在第 {i+1} 页，内容包含'第十一条'。")
        found = True
    # 检查有没有断开的情况
    elif "十一条" in text:
        print(f"⚠️ 发现线索：有'十一条'，但没有完整的'第十一条'。在第 {i+1} 页。")
        found = True

if not found:
    print("❌ 整个 PDF 都没找到'十一条'这几个字。")