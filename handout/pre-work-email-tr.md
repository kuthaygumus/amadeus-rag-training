# Pre-work maili — 4 Ekim'de gönderilecek (eğitimden 3 gün önce)

> Bu bir şablon. Göndermeden önce: linkleri tıkla, komutları temiz bir makinede kopyala-yapıştır
> dene, ve kendi telefonundan maili aç — kod blokları düzgün görünüyor mu bak.

---

**Konu:** RAG eğitimi 7 Ekim — hazırlık: 10 dakika iş, ~3.9 GB (lütfen ofiste değil evde)

Merhaba,

7 Ekim Çarşamba günkü RAG eğitimi tamamen **kendi laptopunuzda, çevrimdışı** çalışacak. API
key yok, cloud hesabı yok, hiçbir yere giriş yok. Bunun tek bedeli: modelleri **önceden**
indirmiş olmanız gerekiyor.

**Lütfen bunu evde yapın, ofis ağında değil.** Hepsi bittiğinde diskte ~3.9 GB yer tutuyor:
3.4 GB model, ~400 MB kurulmuş paket, 83 MB ONNX embedder, 11.6 MB MNIST. Yavaş makineler için
yedek modeli de alırsanız ~4.9 GB. Komutları yazmak on dakika sürüyor; indirmelerin bitmesi
bağlantınıza göre 25 dakikayı buluyor.

**Eğitim günü salonda model indirmek yok.** Yirmi laptop aynı anda birkaç GB çekmeye
kalkarsa sabahı kaybediyoruz. Aşağıdaki adımlar bunun için var.

## 1. Ollama kurun — 2 dakika

[ollama.com](https://ollama.com) → indir → kur.
Windows'ta yükleyici kendi kullanıcı klasörünüze kuruyor, admin sormuyor.

Kurulumdan sonra açık olan terminal pencerelerini kapatıp yenisini açın: `ollama` komutu ancak
yeni pencerede tanınıyor.

## 2. Üç model indirin — 3.4 GB, bağlantınıza göre 10-25 dakika

Terminal (Windows'ta PowerShell) açıp:

```
ollama pull qwen2.5:3b
ollama pull bge-m3
ollama pull nomic-embed-text
```

`ollama list` bunları 1.9 GB, 1.2 GB ve 274 MB olarak gösteriyor. Üçü de gerekiyor —
`nomic-embed-text` küçük ama modül 6 ve 9'da `bge-m3` ile karşılaştırdığımız model, yani
atlanamıyor.

Laptopunuz yavaşsa şunu da ekleyin (yedek, 986 MB):
```
ollama pull qwen2.5:1.5b
```

Gerekip gerekmediğini 6. adımdaki script söylüyor; emin değilseniz önce onu koşturun.

## 3. Python + repo — 3 dakika

Python 3.10 veya üzeri gerekiyor. Yoksa [python.org](https://python.org) →
**"Add python.exe to PATH" kutusunu işaretleyin** → "Install for me only" seçin (admin istemez).

```
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
python -m pip install -r requirements.txt
```

**git yoksa gerek de yok.** Şu ZIP'i indirip çıkarın, sonra çıkardığınız klasöre `cd` yapın:
https://github.com/kuthaygumus/amadeus-rag-training/archive/refs/heads/main.zip

Çıplak `pip` değil `python -m pip` yazın: Windows'ta ikisi farklı Python'lara denk gelebiliyor ve
paketler notebook'ların göremediği bir yere kuruluyor. Kabuğunuz sadece `py` tanıyorsa
`py -m pip install -r requirements.txt` kullanın (ve aşağıda `python` yazan her yerde `py`).

İki paket kuruyor: bağımlılıklarıyla birlikte diskte yaklaşık 400 MB yer kaplıyor — ağdan
inen bayt sayısı bundan az. Yine, evde.

## 4. Notebook'ları açacağınız editör — 3 dakika

Notebook sunucusu kurmuyoruz. Notebook'lar `notebooks/` altında `.py` dosyası olarak da duruyor
ve VS Code onları hücre hücre çalıştırıyor.

- [code.visualstudio.com](https://code.visualstudio.com) → Windows'ta **User Installer**
  (kendi profilinize kuruyor, admin istemiyor).
- VS Code içinde Extensions (`Ctrl+Shift+X`) → **Python** ara → Microsoft'unkini kur.
- `File → Open Folder` ile **`amadeus-rag-training/notebooks`** klasörünü açın (repo kökünü
  değil — notebook'lar `../corpus` ve `../eval` okuyor).
- `Ctrl+Shift+P` → `Python: Select Interpreter` → paketleri kurduğunuz Python'ı seçin.
- Bir `.py` dosyası açıp imleci bir bloğun içine koyun, `Shift+Enter`. Altta bir Python
  terminali açılıp o bloğu çalıştırıyor. Çalışması gereken tek şey bu.

## 5. Model olmayan iki dosyayı indirin — 1 dakika + ~95 MB

```
python scripts/seed_offline_assets.py
```

İki şey daha var ve ikisi de salonda inmesin: modül 2'nin üzerinde eğitim yaptığı MNIST veri
seti (11.6 MB) ve ChromaDB'nin kendi varsayılan embedder'ı (83 MB, bir AWS adresinden) — bu
ikincisini modül 6'nın bake-off'u da, modül 8'in notebook'u da kullanıyor. Bu script ikisini de
evde indiriyor. İki kez çalıştırmak sorun değil, diskte olanı atlıyor.

## 6. Yeşil ışık — 30 saniye

```
python scripts/verify_setup.py
```

**`READY` görüyorsanız işiniz bitti.** Kapatın, unutun, çarşamba görüşürüz.

`NOT READY` görüyorsanız script size tam olarak neyin eksik olduğunu ve ne yapmanız gerektiğini
yazıyor. Takılırsanız bana çıktının ekran görüntüsünü atın — sabah 09:10'da yanınıza oturmaktan
iyidir.

Bazı satırlar `warn` yazıyor (`helios-q2` kurulu değil, VS Code bulunamadı gibi). Uyarılar
`READY`'yi bozmuyor; okuyup geçebilirsiniz.

Windows'ta Python'ı hiç kuramadıysanız önce şunu koşturun, o da eksikleri söyler:
```
powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
```

Bu komut `running scripts is disabled on this system` diyorsa makinenizin script politikası
Group Policy ile kilitlenmiş demektir ve `-ExecutionPolicy` onu geçemez. Admin hakkı da
gerekmiyor, şunu deneyin:
```
Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -
```

## İndirme tutmazsa: USB var

Evdeki bağlantı yarıda pes ederse ya da `ollama pull` bir türlü bitmezse **bana cevap yazın**,
size USB hazırlayayım. Yanımda iki stick olacak ama günden önce haber vermeniz, sabah
öğrenmemden iyi.

Model dosyaları diskte sıradan dosyalar:

| İşletim sistemi | Dizin |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

USB'den kopyalarken iki şeye dikkat:

1. **Önce Ollama'yı tamamen kapatın** — macOS'ta menü çubuğundaki ikon → Quit, Windows'ta
   sistem tepsisindeki ikon → Quit Ollama. Windows'ta sunucu oturumla birlikte açılıyor ve o
   dosyaları açık tutuyor.
2. **`blobs/` ve `manifests/` içeriğini birleştirin, klasörü komple değiştirmeyin.** Üstüne
   kopyalamak, makinede zaten olan modellerin kaydını siliyor.

Sonra Ollama'yı açıp `ollama list` deyin.

Modül 3'te kullandığımız `helios-q2` fine-tune modeli zaten hiçbir registry'de yok, onu ben
getiriyorum. Elinizde olmaması bir sorun değil: o modülün fine-tune'a soru soran **iki probe
hücresi** `(skipped — helios-q2 not installed)` yazıp geçiyor. Yerine kayıtlı bir çıktı da
oynatılmıyor, çünkü o notebook'un kaydı yok. Modülün asıl konusu olan kısım — Q2 ve Q3 fare
sheet'lerini diskten okuyup iki `| K |` satırını yan yana basan hücreler — hiçbir model
gerektirmiyor ve çalışmaya devam ediyor. `verify_setup.py` bunu `warn` olarak yazıyor,
`FAIL` olarak değil.

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

Materyal: https://amadeus-rag-training.vercel.app/tr/

Görüşmek üzere,
Kuthay
