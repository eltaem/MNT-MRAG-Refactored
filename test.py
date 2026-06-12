from src.models import TransformerImageCaptioningModel, OllamaTableCaptioningModel
from src.configs import BaseCaptioningConfig, OllamaCaptioningConfig

def test_image_captioning():
    captioning_config = BaseCaptioningConfig(model_name="Qwen/Qwen3-VL-2B-Instruct", device="cuda", trust_remote_code=True, max_new_tokens=512)
    captioning_model = TransformerImageCaptioningModel(captioning_config)
    captioning_model.load_pretrained()

    caption = captioning_model.caption(image_path="data/images/test/page40_xref3460.png", prompt="Describe the image in indonesian language")

    print(caption)

def test_table_captioning():
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
    captioning_config = OllamaCaptioningConfig(model_name="gpt-oss", device="cuda", trust_remote_code=True, max_new_tokens=512)
    captioning_model = OllamaTableCaptioningModel(captioning_config)
    captioning_model.load_pretrained()
    caption = captioning_model.caption(input_data=table, prompt="Analyze this table and provide a concise one paragraph description in indonesian")

    print(caption)
    
def check_len():
    text = "Tabel tersebut merupakan formulir penilaian risiko gizi (PYMS) untuk seorang anak perempuan berusia 2 tahun (An. NCN) dengan berat badan 9 kg, tinggi badan 79,3 cm, dan BMI 14,3 kg/m², yang terdaftar di rumah sakit dengan nomor RM 1230XXXX. Pada langkah 1 dijelaskan bahwa BMI berada di bawah acuan, namun tidak ada penurunan berat badan. Langkah 2 menegaskan tidak ada penurunan berat badan akhir‑akhir. Langkah 3 menunjukkan asupan makan tetap normal satu minggu terakhir, dan langkah 4 menyatakan tidak ada risiko gizi yang dipengaruhi kondisi anak dalam satu minggu ke depan. Dengan skor 0 pada setiap langkah, total skor PYMS yang tercatat adalah 4, yang menandakan tingkat risiko gizi yang sangat rendah."
    print(len(text))
if __name__ == "__main__":
    check_len()