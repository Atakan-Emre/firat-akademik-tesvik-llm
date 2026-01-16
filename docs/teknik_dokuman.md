# Firat Universitesi Akademik Tesvik Asistani (LLM Fine-Tuning) - Teknik Dokuman

## 1. Proje Ozeti ve Amac
Bu proje, Firat Universitesi 2025-2026 Akademik Tesvik surecleri (yonetmelik, basvuru rehberi ve takvim) konusunda uzmanlasmis bir yapay zeka asistani gelistirmeyi hedefler. Amaç:

- Yonetmelik ve resmi dokumanlara dayali, tutarli ve resmi dilde yanitlar verebilen bir Soru-Cevap (QA) asistani olusturmak.
- Uctan uca bir LLM fine-tuning is akisinin (veri uretimi, on isleme, egitim, arayuz) tekrar edilebilir sekilde tanimlanmasi.
- Duset bellekli egitim icin QLoRA yaklasimindan faydalanmak.

Proje, Qwen-2.5-3B-Instruct temel modeli uzerinde LlamaFactory altyapisi ile fine-tuning yapar ve Gradio tabanli bir demo arayuz ile servis eder.

## 2. Kapsam ve Sinirlar

- Kapsam: Akademik Tesvik yonetmeligi, basvuru rehberi ve takvimine dayali QA bilgisi.
- Sinirlar:
  - Model yalnizca egitim verisindeki resmi dokuman icerikleriyle uyumlu bilgi uretmek uzere tasarlanmistir.
  - Resmi kararlar icin tek referans kaynagi resmi dokumanlardir; model yanitlari bilgilendirme amaclidir.
  - Egitim ve demo altyapisi Google Colab uzerinde kurgulanmistir.

## 3. Kaynak Dokumanlar
Veri uretiminin dayandigi ham kaynaklar `docs/` klasorundedir:

- `docs/Akademik_Tesvik_Yonetmeligi_Basvuru_Rehberi.docx`
- `docs/Akademik_Tesvik_Odenegi_Yonetmeligi.pdf`
- `docs/Akademik_Tesvik_Takvimi_2026.pdf`

Bu dokumanlar NotebookLM ile QA veri setine donusturulur.

## 4. Veri Uretimi (NotebookLM)

### 4.1. Veri Uretim Stratejisi
NotebookLM uzerine dokumanlar yuklenir ve dogrudan dokuman icerigine bagli QA verisi uretilir. Uretimde hedef 1000 soru-cevap kaydidir.

### 4.2. Prompt Spesifikasyonu
NotebookLM promptu, cikti formatini ve kurallari belirler:

```
KOLONLAR: "id";"question";"answer";"source_doc";"source_loc";"tags";"difficulty"
KURALLAR:
1) 1000 adet kayit uret.
2) question: 8–25 kelime.
3) answer: 40–140 kelime; kisa, net.
4) source_doc: [Basvuru Rehberi] | [Yonetmelik] | [Takvim].
5) source_loc: Madde X, Tablo Y, Tarih araligi vb.
6) tags: 3–6 anahtar kelime (orn: basvuru|yoksis).
7) difficulty: easy / med / hard.
8) Asla dokumanda olmayan bilgi ekleme.
```

### 4.3. Ham Veri Ciktisi
Ham cikti CSV formatindadir:

- Dosya: `data/dataset.csv`
- Ayirici: `;`
- Baslik satiri mevcuttur
- Ornek sutunlar: `id`, `question`, `answer`, `source_doc`, `source_loc`, `tags`, `difficulty`

## 5. Veri On Isleme (CSV -> Alpaca JSON)

### 5.1. Amaç
LlamaFactory ile egitim icin veri, Alpaca formatinda olmalidir:

- `instruction`: soru
- `input`: (bos)
- `output`: cevap

### 5.2. Donusum Akisi (Training_Colab.ipynb)
On isleme adimi, Colab notebook icinde bir kod hucresi olarak gerceklesir:

- CSV dosyasi farkli encoding'lerle (utf-8, cp1254, latin5, vb.) okunur.
- Soru ve cevap sutunlari isimlerine gore bulunur; bulunamazsa 2. ve 3. sutun varsayilir.
- Bos ya da `nan` degerler filtrelenir.
- Alpaca JSON formatina donusturulur.

### 5.3. Cikti

- Dosya: `data/atakan_qa.json`
- Kayit sayisi: 1000
- Ornek anahtarlar: `instruction`, `input`, `output`

## 6. Egitim Altyapisi ve Model Yapilandirmasi

### 6.1. Platform

- Ortam: Google Colab
- GPU: T4
- Framework: LlamaFactory
- Yontem: QLoRA (4-bit quantization)
- Temel model: `Qwen/Qwen2.5-3B-Instruct`

### 6.2. Kurulum ve Bagimliliklar
Notebook icindeki kurulum adimlari:

```
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
pip install -e ".[torch,metrics]"
pip install bitsandbytes pandas matplotlib
```

WandB tamamen devre disi birakilir:

```
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
```

### 6.3. Egitim Konfigrasyonu
`train_config.yaml` iceriginden ozet ayarlar:

- `stage`: sft
- `finetuning_type`: lora
- `lora_target`: all
- `lora_rank`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.05
- `cutoff_len`: 1024
- `max_samples`: 10000
- `learning_rate`: 2e-4
- `num_train_epochs`: 3
- `per_device_train_batch_size`: 1
- `gradient_accumulation_steps`: 8
- `lr_scheduler_type`: cosine
- `warmup_ratio`: 0.1
- `quantization_bit`: 4
- `quantization_method`: bitsandbytes
- `val_size`: 0.1
- `eval_strategy`: steps
- `eval_steps`: 100

### 6.4. Cikti ve Loglama
Egitim ciktisi Google Drive uzerinde tutulur:

- Calisma dizini: `/content/drive/MyDrive/llamafactory_runs`
- Adapter dizini: `/content/drive/MyDrive/llamafactory_runs/atakan_qa_qwen2.5_lora`
- Log: `training_log.txt`
- Ek loglar: `trainer_log.jsonl`, `training_loss.png`

Egitim komutu:

```
llamafactory-cli train /content/train_config.yaml 2>&1 | tee "/content/drive/MyDrive/llamafactory_runs/training_log.txt"
```

## 7. Degerlendirme ve Izleme

Notebook, `trainer_log.jsonl` dosyasindan kayip degerlerini okuyup matplotlib ile grafiklendirir. Loglardan gelen kayip degerleri adim bazinda gorsellestirilir. Egitim tamamlandiginda `training_loss.png` ve `training_eval_loss.png` gibi grafikleri incelemek mumkundur.

## 8. Model Yukleme ve Gradio Arayuzu

### 8.1. Model Yukleme
Egitilmis LoRA adapteri, temel Qwen modeli uzerine yuklenir:

- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Adapter: `/content/drive/MyDrive/llamafactory_runs/atakan_qa_qwen2.5_lora`

Model 4-bit olarak yuklenir:

```
AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)
```

### 8.2. Sistem Mesaji
Chat fonksiyonunda sistem rolu:

```
Sen Firat Universitesi akademik tesvik yonetmeligi konusunda uzman bir asistansın.
Cevapları sadece yonetmelik ve basvuru rehberine dayanarak, resmi ve net bir dille ver.
```

### 8.3. Gradio UI
Gradio arayuzunde su girdiler vardir:

- Soru metni (Textbox)
- Temperature (0.1 - 1.0)
- Maksimum token (64 - 1024)

Arayuz basligi: "Akademik Tesvik Asistani"

## 9. Veri ve Artefaktlar

### 9.1. Veri Dosyalari

- `data/dataset.csv`: NotebookLM uretimi ham veri
- `data/atakan_qa.json`: Alpaca formatli egitim verisi (1000 kayit)
- `data/dataset_info.json`: LlamaFactory dataset konfigurasyonu

### 9.2. Model Artefaktlari
Egitim sonrasi uretilecek ornek artefaktlar:

- LoRA adapter agirliklari
- training_log.txt
- trainer_log.jsonl
- training_loss.png

## 10. Dizin Yapisinin Ayrintili Aciklamasi

```
firat-akademik-tesvik-llm/
├── docs/                          # Kaynak dokumanlar ve teknik dokuman
├── data/                          # Ham ve islenmis datasetler
├── notebooks/                     # Colab egitim notebooku
├── readme.md                      # Proje ozet dokuman
├── test.jpg                       # Gradio arayuz gorseli
```

## 11. Guvenlik, Etik ve Uyumluluk

- Yanitlar resmi dokumana dayandirilmalidir.
- Model, dogrudan hukuki ya da idari karar araci degildir.
- Kullanici verisi ile egitim sureci yoktur; sadece dokuman tabanli QA seti kullanilir.

## 12. Tekrar Edilebilirlik ve Calistirma Adimlari

1. Kaynak dokumanlari `docs/` altina yerlestir.
2. NotebookLM ile `data/dataset.csv` uret.
3. `Training_Colab.ipynb` uzerinden CSV -> JSON donusumunu calistir.
4. `train_config.yaml` ayarlarini olustur.
5. LlamaFactory ile QLoRA egitimini calistir.
6. LoRA adapter ile Gradio arayuzunu baslat.

## 13. Bilinen Riskler ve Sorun Giderme

- GPU bulunamazsa egitim baslatilamaz (Colab runtime GPU ayari kontrol edilmeli).
- CSV encoding uyusmazligi verinin okunmasini engeller; UTF-8 veya CP1254 ile tekrar kaydetmek gerekir.
- Drive yazma izinleri yoksa log ve model ciktilari kaydedilemez.
- Model yanitlari dokuman disina tasabilir; prompt sistem rolu ile sinirlandirilir.

## 14. Gelecek Iyilestirmeler

- Daha buyuk ve cesitli QA veri seti.
- Degerlendirme icin otomatik metrikler ve insan dogrulama akisi.
- Modeli bir REST API olarak servis etme.
- Kurumici kimlik dogrulamasi veya yetkilendirme eklenmesi.

## 15. Lisans ve Atif

- Lisans: MIT (detaylar `LICENSE` dosyasinda)
- Atif önerisi: `readme.md` icindeki BibTeX kaydi.
