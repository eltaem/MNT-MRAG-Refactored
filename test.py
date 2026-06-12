def test_image_captioning():
    from src.models import TransformerImageCaptioningModel
    from src.configs import TransformerImageCaptioningConfig
    captioning_config = TransformerImageCaptioningConfig(model_name="Qwen/Qwen3-VL-2B-Instruct", device="cuda", max_new_tokens=512, backend={"trust_remote_code": True})
    captioning_model = TransformerImageCaptioningModel(captioning_config)
    captioning_model.load_pretrained()

    caption = captioning_model.caption(image_path="data/images/test/page40_xref3460.png", prompt="Describe the image in indonesian language")

    print(caption)

def test_table_captioning():
    from src.models import OllamaTableCaptioningModel
    from src.configs import OllamaTableCaptioningConfig
    table = [
            [
            "NAMA : An. NCN\nNAMA KELUARGA: R\nUSIA : 2 th\nBANGSAL : RII BED 10\nNO. RM : 1230XXXX\nJENIS KELAMIN : Perempuan",
            None,
            None,
            "Tanggal",
            ""
            ],
            [
            None,
            None,
            None,
            "TTD\npetugas",
            ""
            ],
            [
            None,
            None,
            None,
            "BB",
            "9 kg"
            ],
            [
            None,
            None,
            None,
            "TB",
            "79,3 cm"
            ],
            [
            None,
            None,
            None,
            "BMI",
            "14,3 kg/m2"
            ],
            [
            "LANGKAH\n1",
            "Apakah BMI berada di\nbawah acuan?",
            "Tidak",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya (LILA kurang dari\nacuan)",
            "1",
            "1"
            ],
            [
            "LANGKAH\n2",
            "Apakah akhir – akhir\nini anak mengalami\npenurunan berat\nbadan ?",
            "Tidak",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya\n- Kehilangan BB\nyang tidak\ndiharapkan\n- Baju terasa\nlonggar\n- Penambahan BB\nyang tidak\nadekuat (jika < 2\ntahun)",
            "1",
            "1"
            ],
            [
            "LANGKAH\n3",
            "Apakah satu minggu\nterakhir anak\nmengalami penurunan\nasupan makan?",
            "Tidak\n(asupan makan\nseperti biasa)",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya\n(Mengalami\npenurunan asupan\nmakan untuk satu\nminggu terakhir)",
            "1",
            "1"
            ],
            [
            None,
            None,
            "Ya\n(tidak ada asupan\natau asupan sangat\nsedikit untuk satu\nminggu terakhir)",
            "2",
            ""
            ],
            [
            "LANGKAH\n4",
            "Akankah kebutuhan\ngizi anak dipengaruhi\noleh kondisi anak\n(untuk kurang lebih\nsatu minggu ke\ndepan)?",
            "Tidak",
            "0",
            ""
            ],
            [
            None,
            None,
            "Ya\n(untuk satu minggu ke\ndepan) :\n- Mengalami\npenurunan\nasupan\ndan/atau\n- Mengalami\npeningkatan\nkebutuhan\ndan/atau\n- Mengalami\npeningkatan\nkehilangan",
            "1",
            "1"
            ],
            [
            None,
            None,
            "Ya",
            "2",
            ""
            ],
            [
            "",
            "",
            "Tidak ada asupan\n(atau asupan sangat\nsedikit) untuk satu\nminggu ke depan",
            "",
            ""
            ],
            [
            "LANGKAH\n5",
            "Jumlahkan skor\nkeseluruhan (total dari\nlangkah 1 hingga\nlangkah 4)",
            "TOTAL SKOR PYMS",
            None,
            "4"
            ]
        ]
    captioning_config = OllamaTableCaptioningConfig(model_name="gpt-oss", device="cuda", max_new_tokens=512)
    captioning_model = OllamaTableCaptioningModel(captioning_config)
    captioning_model.load_pretrained()
    caption = captioning_model.caption(input_data=table, prompt="Analyze this table and provide a concise one paragraph description in indonesian")

    print(caption)
    
def check_len():
    text = "Tabel tersebut memuat rekaman klinis seorang anak perempuan 2 tahun bernama An. NCN (RM 1230XXXX, bangsal RII BED 10) dengan berat badan 9 kg, tinggi badan 79,3 cm"
    print(len(text))

def test_hashing():
    from src.utility import generate_document_id
    a_id = generate_document_id("data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf")
    b_id = generate_document_id("data/PDFs/ANAK DENGAN TB PARU , PNEUMONIA , GIZI BURUK MARASMUS.pdf")
    print(a_id)
    print(b_id)
    print(a_id == b_id)

def test_image_extractor():
    from src.configs import ImageExtractorConfig
    from src.extractors import ImageExtractor
    from src.utility import generate_document_id
    import fitz
    config = ImageExtractorConfig(image_dir="data/images")
    extracttt = ImageExtractor(config)
    doc_name = "data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    doc = fitz.open(doc_name)
    page_number = 33
    page = doc.load_page(page_number)
    print(generate_document_id(doc_name))
    nodes = extracttt.extract(page, generate_document_id(doc_name))
    for node in nodes:
        print(node)

def test_text_extractor():
    from src.configs import TextExtractorConfig
    from src.extractors import TextExtractor
    from src.utility import generate_document_id
    import fitz
    config = TextExtractorConfig()
    extracttt = TextExtractor(config)
    doc_name = "data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    doc = fitz.open(doc_name)
    page_number = 33
    page = doc.load_page(page_number)
    print(generate_document_id(doc_name))
    nodes = extracttt.extract(page, generate_document_id(doc_name))
    for node in nodes:
        print(node)

def test_table_extractor():
    from src.configs import TableExtractorConfig
    from src.extractors import TableExtractor
    from src.utility import generate_document_id
    import fitz
    config = TableExtractorConfig()
    extracttt = TableExtractor(config)
    doc_name = "data/PDFs/(HIL) DEXTRA INKASERATA + MECKEL’S DIVERTICUM POST EXPLORASI .pdf"
    doc = fitz.open(doc_name)
    page_number = 29
    page = doc.load_page(page_number)
    print(generate_document_id(doc_name))
    nodes = extracttt.extract(page, generate_document_id(doc_name))
    for node in nodes:
        print(node)

if __name__ == "__main__":
    test_table_extractor()