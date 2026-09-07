# Pre-work maili — 4 Ekim'de gönderilecek (eğitimden 3 gün önce)

> Bu bir şablon. Göndermeden önce: linkleri tıkla, komutları temiz bir makinede kopyala-yapıştır
> dene, ve kendi telefonundan maili aç — kod blokları düzgün görünüyor mu bak.

---

**Konu:** RAG eğitimi 7 Ekim — 15 dakikalık hazırlık (lütfen ofiste değil evde yapın)

Merhaba,

7 Ekim Çarşamba günkü RAG eğitimi tamamen **kendi laptopunuzda, çevrimdışı** çalışacak. API
key yok, cloud hesabı yok, hiçbir yere giriş yok. Bunun tek bedeli: modelleri **önceden**
indirmiş olmanız gerekiyor.

**Lütfen bunu evde yapın, ofis ağında değil.** Toplam 3 GB indiriliyor ve yirmi kişi aynı anda
yaparsa sabah kimse başlayamaz.

## 1. Ollama kurun — 2 dakika

[ollama.com](https://ollama.com) → indir → kur.
Windows'ta yükleyici kendi kullanıcı klasörünüze kuruyor, admin sormuyor.

## 2. İki model indirin — 10 dakika, 3.1 GB

Terminal (Windows'ta PowerShell) açıp:

```
ollama pull qwen2.5:3b
ollama pull bge-m3
```

Laptopunuz yavaşsa şunu da ekleyin (yedek, 1 GB):
```
ollama pull qwen2.5:1.5b
```

## 3. Python + repo — 3 dakika

Python 3.10 veya üzeri gerekiyor. Yoksa [python.org](https://python.org) →
**"Add python.exe to PATH" kutusunu işaretleyin** → "Install for me only" seçin (admin istemez).

```
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
pip install -r requirements.txt
```

İki paket kuruyor. Yaklaşık 400 MB — yine, evde.

## 4. Yeşil ışık — 30 saniye

```
python scripts/verify_setup.py
```

**`READY` görüyorsanız işiniz bitti.** Kapatın, unutun, çarşamba görüşürüz.

`NOT READY` görüyorsanız script size tam olarak neyin eksik olduğunu ve ne yapmanız gerektiğini
yazıyor. Takılırsanız bana çıktının ekran görüntüsünü atın — sabah 09:10'da yanınıza oturmaktan
iyidir.

Windows'ta Python'ı hiç kuramadıysanız önce şunu koşturun, o da eksikleri söyler:
```
powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
```

## İsteğe bağlı: Podman

Günün bir modülünde ChromaDB'yi container'da servis olarak çalıştıracağız.
[podman.io](https://podman.io) kurup `podman compose up -d` diyebiliyorsanız o modülü izlemek
yerine kendiniz koşturursunuz. **Kuramazsanız hiçbir şey bozulmaz** — diğer sekiz notebook
container'sız çalışıyor.

Windows'ta Podman altta WSL2 çalıştırıyor, yani admin ve bir yeniden başlatma gerekiyor.
Zorlanmayın, opsiyonel.

## Gün hakkında

09:00–15:00, öğle arası var. Notebook'lar sizde kalıyor ve pazartesi sabahı da,
internetsiz, aynı laptopta çalışmaya devam ediyor.

Bütün veriler sentetik — kurgusal bir havayolu (Helios Air) üzerinden gidiyoruz, hiçbir
Amadeus sistemi veya verisi kullanılmıyor.

Materyal: https://amadeus-rag-training.vercel.app

Görüşmek üzere,
Kuthay
