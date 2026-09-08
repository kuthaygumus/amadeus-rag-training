---
title: "0. Kurulum — Gelmeden Önce"
description: "Ollama kur, üç model indir, iki dosyayı önden indir, tek script çalıştır. Yarım saat, bir önceki akşam — sabah ofis ağında değil."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

Bu modülün gate sorusu yok. Burası ön hazırlık: bir önceki akşam, yirmi kişiyle paylaşmadığın bir ağda yapılır.

## Zaten patlamış olan şey

Böyle bir eğitimi kurmanın normal yolu şu: `pip install transformers`, bir model hub'ından weight çek, başla. Bu yol eğitim hazırlanırken denendi ve bırakıldı. Tek soru cevaplanmadan önce her laptopa birkaç gigabyte'lık bir Python yığını kuruyor, ve günün sayılarını her makinenin hangi kütüphanenin hangi sürümünü çözdüğüne bağlı bırakıyor. Ollama'nın maliyeti tek bir indirme ve tek bir process.

Günün bütün şekli bundan çıktı. Bu eğitimdeki her model Ollama üzerinden çekiliyor ve kendi makinende çalışıyor. API key yok, cloud hesabı yok, sign-in yok, masraf yok — ve alıştırmalarda ürettiğin sayılar sonuç sayfasındaki sayılarla aynı, çünkü bugünden ekime kadar kimse altından hosted bir modeli sessizce güncellemiyor.

## Ollama tam olarak nedir

Bir framework değil, lokal bir model sunucusu. Model adını veriyorsun, weight'lerin quantize edilmiş bir kopyasını indiriyor ve `http://localhost:11434` üzerinde küçük bir HTTP API açıyor: `/api/tags` elindekileri listeliyor, `/api/embed` metni vektöre çeviriyor, `/api/chat` cevap üretiyor. Notebook'lar buraya standart kütüphanedeki `urllib` ile bağlanıyor. Kurulacak bir SDK, yanlış anda internete çıkmaya kalkacak bir import yok.

Burada olmasının tek sebebi bu: tek indirme, tek process, ve ekimde senin laptopunda bu sitedeki sayıları üreten makinedekiyle aynı cevabı veren bir model.

<div class="presenter-note">
Şunu bir kez söyle ve ağzında gevelemeden söyle: <strong>her şey önündeki laptopta çalışıyor, ve bu bir ideoloji değil ölçüm kararı.</strong> İlk on dakikada birisi "neden API kullanmıyoruz?" diye soracak. Cevap "lokal daha iyi" değil — cevap "sabitlenmiş lokal bir model eylülde de ekimde de aynı sayıyı veriyor, yani bir sayı oynadığında onu neyin oynattığını biliyoruz. Ve kimsenin oturmak için hesaba, key'e ya da onaya ihtiyacı olmuyor."
</div>

## Ollama kurulumu

**macOS.** [ollama.com](https://ollama.com) üzerinden indir, uygulamayı Applications'a sürükle, bir kez aç. Menü çubuğunda ikon ve çalışan bir sunucu görüyorsun. Terminalden doğrula:

```bash
ollama --version
```

**Windows, yönetici hakkı olmadan.** `OllamaSetup.exe` per-user bir kurulum. Binary'leri `%LOCALAPPDATA%\Programs\Ollama` altına koyuyor ve elevation istemiyor. Sunucu senin oturumunla birlikte başlıyor.

`UNVERIFIED: per-user kurulum yolu installer'ın tasarımı; yönetilen bir Windows laptopunda doğrulanmadı.` Kurulum senden yönetici parolası isterse orada dur. Direnme, günün sabahı 09:00'da IT'yi ikna etmeye çalışma — bir önceki akşam haber ver, makinesi yeşil olan biriyle eşleştirilirsin.

Installer Ollama'yı kullanıcı PATH'ine ekliyor, ama o sırada zaten açık olan bir terminal penceresi bunu görmüyor. `ollama --version` "komut bulunamadı" diyorsa, bir sonuç çıkarmadan önce o pencereyi kapat ve yenisini aç.

## Üç model indir

```bash
ollama pull qwen2.5:3b          # generation ve rerank
ollama pull bge-m3              # embedding, ve Türkçeyi kaldırıyor
ollama pull nomic-embed-text    # modül 6 ve 9'un bge-m3'ü karşılaştırdığı zayıf embedder
```

`ollama list`'in bildirdiği boyutlar: 1.9 GB, 1.2 GB ve 274 MB — toplam yaklaşık 3.4 GB.

Üçünü de indir. `nomic-embed-text`, `bge-m3`'ün alternatifi değil, kontrol grubu: modül 6 ikisini yan yana koyuyor, modül 9'un notebook'u ikisini birden yüklüyor. Gün içinde salonda model çekmek, bu sayfanın önlemek için var olduğu tek şey.

Bunu evde yap, en azından eğitim sabahı ofis wifi'sinde yapma. 3.4 GB tek laptop için sakin bir akşam. On beş laptop saat 09:00'da aynı pull'u başlattığında hattı her biri ayrı ayrı almıyor — bölüşüyorlar.

Seçimler ölçüldü, blog yazısından alınmadı. Aynı ücret tablosuna sorulan üç kaynaklı soruda `qwen2.5:3b` 3'te 3 doğru verdi, soru başına 0.9 s, ve denenen hiçbir model bunu yakalayamadı. `llama3.2`, tablo context'in **içindeyken** bile Türkçe soruya EUR 70 dedi; tablonun K satırı EUR 90 diyor — bu eğitimin dışında, yerine koyma. `bge-m3` burada çünkü structure-aware chunk'lar üzerinde, Türkçe soru / İngilizce doküman olan altı soruda hit@1 0.667 alıyor, `nomic-embed-text` ise 0.000.

### Fine-tune edilmiş model

Modül 3, Helios corpus'unun Q2 sürümüyle eğitilmiş `helios-q2` modelini kullanıyor. Hiçbir registry'de yok, yani `ollama pull helios-q2` onu bulamaz — bir kez başka bir yerde GPU üzerinde üretiliyor ve günden önce USB ile dağıtılıyor. Modül 3'ün kendi makinende canlı çalışmasını istiyorsan iste; kurulumu "Dosyalar gerçekte nerede duruyor" bölümündeki kopyala-ve-birleştir adımlarının aynısı.

Elinde yoksa günün geri kalanında hiçbir şey değişmiyor, ama tam olarak neyi kaybettiğini bilmek işe yarar. Fine-tune'a soru soran iki probe hücresi `(skipped — helios-q2 not installed)` yazıp geçiyor. Etraflarındaki her şey yine çalışıyor: Q2 ve Q3 ücret sayfalarını diskten okuyup iki `| K |` satırını yan yana basan hücrelerin modele hiç ihtiyacı yok, ve modül 3'ün asıl konusu zaten o fark. `verify_setup.py` eksik modeli hata değil, uyarı olarak raporluyor.

## Python ve repo

Python 3.10 veya üzeri — kontrol scripti `dict | None` tip sözdizimi kullanıyor, yani eski bir Python günün ortasında değil ilk satırda patlar.

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
python -m pip install -r requirements.txt
```

**git yok mu?** Gerek de yok. [ZIP'i indir](https://github.com/kuthaygumus/amadeus-rag-training/archive/refs/heads/main.zip), çıkart ve çıkardığın klasöre `cd` yap. Sonrası birebir aynı.

**Çıplak `pip` değil, `python -m pip`.** Windows'ta ikisi farklı yorumlayıcılara denk gelebiliyor; geldiğinde paketler notebook'ların göremediği bir yere kuruluyor — akşam READY veren laptopta, günün ortasında `ModuleNotFoundError`. Kabuğun sadece `py` launcher'ını tanıyorsa her yerde `py -m pip install -r requirements.txt` ve `py scripts/verify_setup.py` kullan.

İki paket. Modül 2'nin ağı eğittiği `numpy`, ve modül 8'in kullandığı vektör veritabanı `chromadb`. Bağımlılıklarıyla birlikte kurulduktan sonra diskte kabaca 400 MB tutuyorlar. Bunu evde yap, ofis ağında aynı anda on dokuz kişiyle birlikte değil.

`chromadb-client` değil **tam `chromadb`** olmak zorunda, ve sebebi modül 6'nın tamamı. İnce istemci metni lokalde embed edemiyor, dolayısıyla sessizce yanlış modele uzanmak yerine hata fırlatıyor. Gürültülü bir hata aslında iyilik olurdu. Tam pakette gerçekten olan şey ise görülmeye değer.

## Model de paket de olmayan iki dosyayı indir

```bash
python scripts/seed_offline_assets.py
```

Notebook'ların okuduğu iki şey var ki kurulumun geri kalanı onları getirmiyor, ve ikisi de aksi halde bir modülün ortasında iniyor:

- **MNIST**, dört arşiv halinde ölçülen 11.6 MB, `notebooks/mnist_data/` altına. Modül 2 ağını bunun üzerinde eğitiyor. Klasör gitignore'da, yani taze bir clone'da yok.
- **Chroma'nın varsayılan embedder'ı.** Modül 8 bir collection açarken metni vektöre nasıl çevireceğini hiç söylemiyor, Chroma da senin yerine bir model seçiyor. Bunu ilk kez gerektiren çağrı, `~/.cache/chroma/onnx_models/` altına bir `all-MiniLM-L6-v2` ONNX arşivi çekiyor — ölçüldü: 83 MB.

Bunu `pip install` sonrasında çalıştır; ikinci yarısı `chromadb`'nin import edilebilmesini gerektiriyor. İki kez çalıştırmak sorun değil: diskte olan her şey raporlanıp atlanıyor. `verify_setup.py` ikisini de kontrol ediyor ve biri eksikse NOT READY veriyor, çünkü burada eksik bir dosya salonda canlı bir indirme demek.

## Notebook'ları neyle açıyorsun

Bu eğitimde notebook sunucusu yok ve bir editör dışında kurulacak bir şey de yok. Her notebook iki kere var: `notebooks/NN_ad.ipynb` olarak ve percent formatında `notebooks/NN_ad.py` olarak — aynı hücreler, JSON yerine `# %%` işaretleriyle. Çalıştırdığın dosya `.py` olanı.

**VS Code'u ve Python eklentisini kur.** [code.visualstudio.com](https://code.visualstudio.com). Windows'ta **User Installer**'ı al: kendi profiline kuruyor ve yönetici hakkı istemiyor. Sonra Extensions panelini aç (`Ctrl+Shift+X`, macOS'ta `Cmd+Shift+X`), **Python** ara ve Microsoft'un yayınladığını kur.

**Repo kökünü değil, `notebooks` klasörünü aç.** `File → Open Folder →` `amadeus-rag-training/notebooks`. Notebook'lar `../corpus` ve `../eval` okuyor, yani çalışma dizininin `notebooks` olması gerekiyor.

**Yorumlayıcıyı seç.** `Ctrl+Shift+P` / `Cmd+Shift+P` → `Python: Select Interpreter` → paketleri kurduğun aynı Python 3.10+ sürümünü seç. Seçtiğin sürüm sağ altta durum çubuğunda görünüyor.

**Bir bloğu çalıştır.** İmleci bir bloğun içine koy ve `Shift+Enter`'a bas. Python eklentisi o bloğu pencerenin altındaki bir Python terminaline gönderiyor ve imleci bir sonrakine taşıyor. Durum basışlar arasında korunuyor — bir bloktaki değişkenler bir sonrakinde hâlâ orada — bunu bir script koşusundan ayıran şey de bu.

**"Çalıştı" neye benziyor.** `notebooks/00_bare_llm_fails.py` dosyasını aç, imleci böyle bir satıra koy ve `Shift+Enter`'a bas:

```python
import os, sys; print(sys.version); print(os.getcwd())
```

Altta bir terminal açılıyor; 3.10 veya üstü bir sürüm ve `notebooks` ile biten bir yol yazıyor. Yol `notebooks` ile bitmiyorsa notebook'lardaki göreli yollar tutmaz; o terminalde bir kez `import os; os.chdir(r"<notebooks klasörünün tam yolu>")` yazıp düzelt.

### Bir hücre makinende çalışmazsa

Ollama kapanmış, hiç çekmediğin bir model, uzun bir hücreyi beklemeyecek kadar yavaş bir laptop — ölçüm hücreleri yeniden hesaplanmak yerine kayıttan oynatılabiliyor. Notebook'un ilk bloğundan önce ortamda `USE_CACHED=1` ayarla:

```bash
USE_CACHED=1 python 05_chunking_and_noise.py
```

`notebooks/_cached.py` o zaman modeli çağırmak yerine o hücrenin sonucunu `notebooks/cached_runs.json` içinden okuyor — editörde açıp okuyabileceğin sıradan bir JSON dosyası. Bu hiçbir zaman sessiz değil: oynatılan her hücre, kaydın tarihini, makinesini ve tam mı yoksa kısaltılmış bir koşu mu olduğunu yazan bir `[CACHED]` bandı basıyor. Kaydı olan notebook'lar 03'ten 08'e kadar olanlar. Notebook 02'nin kaydı yok; `helios-q2` eksik olduğunda probe'larının oynatılmayıp atlanmasının sebebi bu. Bu, oturumu yürütmeye yarar; günün tamamını offline koşturmanın yolu değildir.

## Podman gerekli mi?

Notebook'ları çalıştırmak için hayır. Notebook 00'dan 07'ye kadar hepsi Ollama ve Python kullanıyor, başka hiçbir şey değil; hiçbir container runtime kurulu olmayan bir makinede sorunsuz koşarlar.

İstisna modül 8, ve yapabiliyorsan kendin koşturmaya değer. ChromaDB'yi kütüphane olarak değil servis olarak çalıştırıyor:

```bash
podman compose up -d          # ya da: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

Podman macOS ve Windows'ta native değil — altta küçük bir Linux VM çalıştırıyor, ki Windows'ta bu WSL2 demek. Admin yetkisi ve bir reboot gerektiriyor. Eğitim gününe kadar makinende olmayacaksa hiçbir şey bozulmaz: o modülü izle, sonra kendin koşturursun. Bu eğitimde ölçülen her şey Podman olmadan ölçüldü.

Compose dosyası repo kökünde. Modellerle aynı tavsiye: image'ı evde indir.

## Ne çalıştırıyorsun

Bu akşam notebook yok. Tek script: `scripts/verify_setup.py`. Sadece `localhost` ile konuşuyor ve `urllib`'e bu çağrıları sistem proxy'sine yönlendirmemesini söylüyor, yani ofiste de evde de uçakta da aynı davranıyor.

```bash
python scripts/verify_setup.py
```

**Ne görmen gerekiyor.** Bir sütun dolusu `ok` satırı, sonra tek başına duran bir satırda **READY** kelimesi.

**Ne kadar sürüyor.** Ollama modeli belleğe almışsa bir iki saniye. Reboot sonrası ilk koşu, model yüklenirken daha uzun sürer.

**Windows'ta Python hiç kurulu değilse.** Asıl kontrol onsuz başlayamıyor, o yüzden önce PowerShell bootstrap'ini koştur. Hiçbir şey kurmuyor, sadece eksikleri söylüyor:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
```

Bu komut `running scripts is disabled on this system` diyorsa, o makinenin execution policy'si Group Policy ile ayarlanmış demektir ve `-ExecutionPolicy` anahtarı onu geçersiz kılamaz. Yönetici hakkına da ihtiyacın yok — dosyayı pipe ile içeri ver, çünkü pipeline'dan gelen metin bir script dosyası değil:

```powershell
Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -
```

`UNVERIFIED: PowerShell scripti yönetilen bir Windows laptopunda henüz koşturulmadı. Önce nelerin test edilmesi gerektiği script'in başlık yorumunda yazıyor.`

Python scripti zincirin tamamını kontrol edip tek kelime yazıyor.

**READY** şu demek: Python 3.10+, `numpy` ve `chromadb` import ediliyor, MNIST ve Chroma'nın varsayılan embedder'ı zaten diskte, Ollama 11434 portunda cevap verdi, üç model tag'i de yerinde, embedding çağrısı gerçek vektörlerle döndü ve generation çağrısı bir tablonun doğru kolonunu okudu. Terminali kapat, unut.

**NOT READY** ise hangi satırın düştüğünü ve onu düzelten komutu yazıyor. Oku, komutu çalıştır, script'i tekrar çalıştır.

Bazı satırlar `ok` veya `FAIL` yerine `warn` yazıyor — VS Code alışılmış yerlerde bulunamadı, `helios-q2` kurulu değil. Bir uyarı READY'yi hiçbir zaman NOT READY'ye çevirmiyor. Bilinmesi iyi olan bir şey, günü durduran bir şey değil.

Script'te iki detay bilerek böyle. Modelleri **tam tag** ile eşleştiriyor: elinde `qwen2.5:1.5b` olması `qwen2.5:3b` şartını karşılamıyor, çünkü zayıf modelle gelen birine "hazırsın" demek buradaki en kötü sonuç. Ve son kontrol bir smoke test değil, doğruluk testi: modele iki ceza kolonlu bir tablo verilip K sınıfının iptal cezası soruluyor. 90 demesi gerekiyor, 70 değil. Kontrol, cevabın tamamını taramak yerine ilk sayıyı okuyor, çünkü 70 de tabloda var ve "EUR 90 (değişiklik cezası EUR 70)" doğru bir cevap. 70 diyen bir model, Modül 7'deki chunking alıştırmasını gayet düzgün çalışırken bozukmuş gibi gösterir.

Script makineni de ölçüp söylüyor: cevap başına 3 saniyenin altı rahat, 3-8 saniye idare eder ama alıştırmalar demodan yavaş hissettirir, 8 saniyenin üstü yavaş. "Yavaş" aldıysan yedek modeli de indir:

```bash
ollama pull qwen2.5:1.5b
```

Bu da 986 MB. Bedeli konusunda kendine dürüst ol: 1.5B model aynı test sorularında 3'te 2 aldı ve multi-hop soruda yanlış tablo satırını seçti. Günü onun üzerinde geçirirsen, pipeline'ın değil modelin suçu olan bir yanlış cevapla karşılaşacaksın. Hangisi olduğunu bil, debug etmeye kalkma.

## Dosyalar gerçekte nerede duruyor

Weight'ler diskte sıradan dosyalar ve taşınabilir.

| İşletim sistemi | Dizin |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

İçinde `blobs/` ve `manifests/` var. Modelleri makineler arasında taşımak için — evdeki bağlantısı pes eden herkes için offline çözüm, ve `helios-q2`'yi almanın tek yolu:

1. **Önce iki makinede de Ollama'yı tamamen kapat.** macOS: menü çubuğundaki ikon → Quit. Windows: sistem tepsisindeki ikon → Quit Ollama. Windows'ta sunucu oturumunla başlıyor ve o dosyaları açık tutuyor.
2. **Değiştirme, birleştir.** `blobs/` ve `manifests/` klasörlerinin *içeriğini* hedef makinedeki aynı adlı klasörlere kopyala. `models` dizinini komple üstüne kopyalamak, o makinede zaten bulunan modellerin `manifests/` kayıtlarını siler ve onları sessizce kayıt dışı bırakır.
3. Ollama'yı tekrar başlat ve `ollama list` çalıştır. Gelen her şey listede görünür.

<div class="presenter-note">
<strong>09:10, on dakika.</strong> Herkes aynı anda <code>verify_setup.py</code> çalıştırıyor, sen salonda dolaşıp ekranları okuyorsun — sorarak değil, okuyarak. Enter'a basmadan önce salona tahmin ettir: "kaçımız yeşil çıkacak?" Sesli bir sayı al. İki şey oluyor: insanlar bir tahmine bağlanıyor, ve akşam hiç çalıştırmamış olanlar kendiliğinden ortaya çıkıyor.
<br /><br />
<strong>Kırmızı laptop triyajı, bu sırayla.</strong> Ollama çalışmıyor → uygulamayı aç. <code>ollama</code> komutu tanınmıyor ama script sunucunun cevap verdiğini söylüyor → terminal penceresi eski, yenisini açtır. Model yok ama makine hızlı → pull'u şimdi başlat, Modül 1 sırasında biter. Model yok ve ağ sürünüyor → USB, yanında iki tane taşıyorsun. Kurulum yönetici hakkına takılmış → dur, hemen eşleştir, salonun sabahını buna harcama.
<br /><br />
<strong>Eşleştirme bir ceza değil, geçerli bir plan.</strong> Bunu yüksek sesle söyle: "iki kişiye bir laptop bu işin normal hali — biri yazar, biri çıktıyı okuyup itiraz eder." Kırmızı laptopu olan kişinin altı saat sessizce oturmasına izin verme.
<br /><br />
<strong>On dakika, on dakika demek.</strong> 09:20'de yeşil olmayan her makine bir tamir işi değil, bir eşleştirmedir. Bu süreyi günün başka hiçbir yerinden geri alamıyorsun.
</div>

## Sayılar ne dedi

<div class="measured">

| Ön hazırlığın diskte bıraktığı | Boyut |
|---|---|
| `qwen2.5:3b` | 1.9 GB |
| `bge-m3` | 1.2 GB |
| `nomic-embed-text` | 274 MB |
| `numpy` + `chromadb` ve bağımlılıkları | yaklaşık 400 MB |
| Chroma'nın `all-MiniLM-L6-v2` ONNX arşivi | 83 MB |
| MNIST, dört arşiv | 11.6 MB |
| **Toplam** | **yaklaşık 3.9 GB** |
| `qwen2.5:1.5b`, sadece script makineni yavaş bulduysa | +986 MB |

Model boyutları `ollama list`'in kurulu tag'ler için bildirdiği değerler. Python satırı paketlerin kurulduktan sonra diskte tuttuğu yer; ağdan geçen byte sayısı bundan az.

| Generation modeli | Doğru | Cevap başına ortalama |
|---|---|---|
| `qwen2.5:3b` | 3/3 | 0.9 s |
| `qwen2.5:1.5b` | 2/3 — M satırı EUR 120 derken EUR 90 dedi | 0.6 s |
| `gemma3:4b` | 2/3 | 2.1 s |
| `qwen3:4b` | 2/3 — reasoning token'larından sonra boş cevap | 10.0 s |
| `llama3.2` | 1/3 — K satırı EUR 90 derken EUR 70 dedi | 1.0 s |

Aynı ücret tablosuna sorulan üç kaynaklı soru, temperature 0, tek bir M-series Mac. Üç soru bir benchmark değil, bir smoke test: bir modeli elemeye yeter, ayakta kalanları sıralamaya yetmez.

| Embedder | Türkçe soru / İngilizce doküman olan altı soruda hit@1 |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

İki embedder de aynı structure-aware chunk'lar üzerinde skorlandı; modül 6'nın koşturduğu koşul bu. Fark o ölçümün bir özelliği, iki modelin sabit bir özelliği değil.

</div>

## Daha derine

Ollama quantize edilmiş GGUF weight'leri sunuyor; 3 milyar parametreli bir modelin ~2 GB indirme olması ve GPU'suz bir laptopta çalışması bu yüzden. Quantization her weight'i 16 bit yerine kabaca 4 bitte saklıyor, karşılığında küçük bir kalite kaybı veriyor. Bu kayıp bu eğitimdeki işlerde görünmüyor; uzun bir akıl yürütme zincirinde görünürdü. Günün hiçbir yerinde 3B'lik bir modelden tek çağrıda zekice bir şey istenmemesinin bir sebebi bu.

Bu odanın dışında seni asıl yakacak seçim embedder, çünkü sessizce patlıyor. `nomic-embed-text` ve `all-MiniLM-L6-v2` İngilizce öncelikli modeller. Türkçe bir soru ve onu cevaplayan İngilizce dokümanı verdiğinde hata da vermiyorlar, boş da dönmüyorlar — kendinden emin biçimde yanlış dokümanı döndürüyorlar. `bge-m3` çok dilli eğitilmiş, dolayısıyla Türkçe bir cümle ile İngilizce karşılığı vektör uzayında birbirine yakın düşüyor. Structure-aware chunk'lar üzerinde, aynı altı soruda fark 0.000'a karşı 0.667 ve bunu kodu okuyarak göremiyorsun. Sadece skorlayarak.

Her şeyin lokal çalışması ideoloji değil, ölçüm meselesi. Hosted bir endpoint'te eylülde aldığın sayı ile ekimde aldığın sayı karşılaştırılabilir değil ve hangi değişikliğin neye yol açtığını hiç öğrenemiyorsun. Sabitlenmiş lokal bir model günü kontrollü bir deneye çeviriyor; burada bu, frontier bir modelin getireceği ekstra kaliteden daha değerli.

On milyon dokümanda bu kurulumun çoğu ayakta kalmaz ve önce hangi parçanın kırıldığını bilmek işe yarar. Generation kabaca aynı kalır, yine tek tek sorulara cevap veriyorsun. Embedding kalmaz — on milyon chunk'ı laptopta embed etmek haftalar süren bir iş, o yüzden batch'lenmiş GPU inference'a ya da hosted bir embedding endpoint'ine taşınır, vektörler de in-process bir yapı yerine gerçek bir vector database'e. Asıl canını yakan kısım şu: embedder bir migration maliyetine dönüşüyor. Onu değiştirmek bütün corpus'u yeniden embed etmek demek. Yani ilk gün gelişigüzel verdiğin karar, geri dönmesi en pahalı karar oluyor — ki bu da onu on milyon doküman üstünde değil, şimdi yirmi soru üstünde ölçmek için iyi bir gerekçe.

## Çıkış cümlesi

> Her şey kurulu ve henüz hiçbir şey birbirine bağlı değil. Yarın modelin tamamen yalnız haliyle başlıyoruz — `notebooks/00_bare_llm_fails.py` ve hiç görmediği bir Helios Air ücret kuralı hakkında tek bir soru.
