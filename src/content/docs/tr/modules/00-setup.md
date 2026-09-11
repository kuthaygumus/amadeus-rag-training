---
title: "0. Kurulum — Gelmeden Önce"
description: "Ollama kur, üç model indir, iki dosyayı önden indir, tek script çalıştır — bir önceki akşam yarım saat. Ve bu sitedeki her komutu çalıştıran kural: durduğun yer repo kökü."
---

Bu modülün gate sorusu yok. Burası ön hazırlık: bir önceki akşam, yirmi kişiyle paylaşmadığın bir ağda yapılır. Yaklaşık yarım saat, neredeyse tamamı indirme.

## Bu sitedeki her komut nereye yazılıyor

Bu bölümü bir kez oku, günün geri kalanında tahmin etmen gereken hiçbir şey kalmasın. Tam olarak **iki yüzey** var ve ikisinin de çıpası, birazdan clone'layacağın klasör: içinde `corpus/`, `notebooks/`, `eval/` ve `exercises/` bulunan klasör.

1. **Terminal, repo kökünde.** O klasöre `cd` yapılmış tek bir terminal penceresi, gün boyunca açık kalıyor. Her `ollama …` komutu ve `python scripts/…`, `python exercises/…` ya da `python eval/…` diye yazılan her şey burada çalışıyor.
2. **VS Code, açık klasör repo kökü olacak şekilde.** Notebook'lar `notebooks/` altında percent formatında `.py` dosyaları. Birini aç, imleci bir `# %%` bloğunun içine koy, `Shift+Enter`'a bas; çıktı Interactive penceresinde görünüyor.

Bu eğitimde Jupyter sunucusu, tarayıcı notebook'u ya da bulut konsolu hiçbir yerde yok. Bu sitenin her sayfasında her komut bloğunun hemen üstünde, kalın harflerle, hangi yüzeye ait olduğu yazıyor; yani hiçbir zaman kendin çıkarmak zorunda kalmıyorsun.

**Doğru terminaldesin** dediğin an, `ls` — Windows'ta `dir` — çıktısında `corpus`, `notebooks`, `eval` ve `exercises` göründüğü an. Bir komut `No such file or directory` diyorsa önce buna bak: bu, bozuk bir kurulumdan çok daha sık yanlış pencerede olmaktan kaynaklanıyor.

<div class="presenter-note">
09:10'da tahtaya yaz ve orada bırak: <strong>iki yüzey, tek çıpa — repo kökü.</strong> Bir salonun on dakikayı kaybetmesinin en yaygın yolu şu: biri <code>python scripts/…</code> komutunu <code>notebooks/</code> içinden çalıştırıyor ve herkes gayet sağlam bir kurulumu debug etmeye çalışıyor. Bir laptop <code>No such file or directory</code> dediğinde, başka hiçbir şey sormadan önce "o terminal hangi klasörde?" diye sor.
</div>

## Ollama kurulumu

**macOS.** [ollama.com](https://ollama.com) üzerinden indir, uygulamayı Applications'a sürükle, bir kez aç. Menü çubuğunda ikon ve çalışan bir sunucu görüyorsun.

**Terminal (herhangi bir yerde — repo henüz yok):**

```bash
ollama --version
```

**Windows, yönetici hakkı olmadan.** `OllamaSetup.exe` per-user bir kurulum: binary'ler `%LOCALAPPDATA%\Programs\Ollama` altına gidiyor, elevation istemiyor, sunucu senin oturumunla başlıyor.

`UNVERIFIED: per-user kurulum yolu installer'ın tasarımı; yönetilen bir Windows laptopunda doğrulanmadı.` Kurulum senden yönetici parolası isterse orada dur — günün sabahı 09:00'da IT'yle tartışmak yerine bir önceki akşam haber ver, makinesi yeşil olan biriyle eşleştirilirsin.

Installer Ollama'yı kullanıcı PATH'ine ekliyor ama o sırada zaten açık olan bir terminal bunu görmüyor. `ollama --version` "komut bulunamadı" diyorsa, bir sonuç çıkarmadan önce o pencereyi kapat ve yenisini aç.

## Üç model indir

**Terminal (herhangi bir yerde):**

```bash
ollama pull qwen2.5:3b          # generation ve rerank
ollama pull bge-m3              # embedding; Türkçede de iş görüyor
ollama pull nomic-embed-text    # modül 6 ve 9'un karşısına koyduğu zayıf embedder
```

`ollama list`'in bildirdiği boyutlar: 1.9 GB, 1.2 GB ve 274 MB — toplam yaklaşık 3.4 GB.

Üçünü de indir. `nomic-embed-text`, `bge-m3`'ün alternatifi değil, kontrol grubu: modül 6 ikisini yan yana koyuyor, modül 9'un notebook'u ikisini birden yüklüyor. Gün içinde salonda model çekmek, bu sayfanın önlemek için var olduğu tek şey. On beş laptop saat 09:00'da aynı 3.4 GB'lık pull'u başlattığında hattı her biri ayrı ayrı almıyor — bölüşüyorlar. Bağlantın pes ederse weight'ler sıradan dosyalar ve onlara sahip bir makineden kopyalanabiliyor: [Modelleri makineler arasında taşımak](#modelleri-makineler-arasında-taşımak).

**Generation modelini değiştirme.** Bu sitedeki kayıtlı her sayı `qwen2.5:3b` ile üretildi. Aynı ücret tablosuna sorulan, context'i verilmiş üç soruda dört model daha denendi:

| Generation modeli | Doğru | Cevap başına ortalama |
|---|---|---|
| `qwen2.5:3b` | 3/3 | 0.9 s |
| `qwen2.5:1.5b` | 2/3 — M satırı EUR 120 derken EUR 90 dedi | 0.6 s |
| `llama3.2` | 1/3 — K satırı EUR 90 derken EUR 70 dedi | 1.0 s |

`UNVERIFIED: bu üç soruluk karşılaştırmanın repoda ne çalıştıran bir scripti ne de kayıtlı bir çıktısı var. Bunu bir ölçüm değil, modelin neden seçildiğinin gerekçesi olarak oku. Aşağıdaki "Sayılar ne dedi" kutusunun içindeki her şeyin kaynağı var.`

Embedder seçimi ise ölçülmüş bir seçim ve aşağıdaki kutuda duruyor.

### Fine-tune edilmiş model

Modül 3, Kraken Air corpus'unun Q2 sürümüyle eğitilmiş `kraken-q2` modelini kullanıyor. Hiçbir registry'de yok, yani `ollama pull kraken-q2` onu bulamaz: bir kez başka bir yerde GPU üzerinde üretiliyor ve günden önce USB ile dağıtılıyor.

`UNVERIFIED: kraken-q2 henüz üretilmedi. 7 Ekim'e kadar ortaya çıkmazsa modül 3, hiçbir modele ihtiyaç duymayan corpus hücreleriyle koşuyor.`

USB eline geçerse içinde bir `.gguf` ve bir Modelfile var. İkisini de `notebooks/` içine kopyala ve Modelfile'ın kendi build satırını çalıştır. Bu, sitedeki repo kökünden **çalıştırılmayan** tek komut, çünkü Modelfile içindeki `FROM ./kraken-q2.gguf` göreli bir yol:

**Terminal (`notebooks/` içinde — tek istisna):**

```bash
ollama create kraken-q2 -f kraken-q2.Modelfile
```

Elinde yoksa başka hiçbir şey değişmiyor. Fine-tune'a soru soran iki probe hücresi `(skipped — kraken-q2 not installed)` yazıp geçiyor; Q2 ve Q3 ücret sayfalarını diskten okuyup iki `| K |` satırını yan yana basan hücrelerin modele ihtiyacı yok ve modül 3'ün asıl konusu zaten o fark. `verify_setup.py` eksik modeli hata değil, uyarı olarak raporluyor.

## Python ve repo

Python 3.10 veya üzeri. Kontrol scriptinin ilk kontrolü Python sürümünün kendisi, yani eski bir yorumlayıcı günün ortasında patlamak yerine tek bir net NOT READY satırı ve onu düzelten komutu alıyor.

**Terminal (herhangi bir yerde — repo kökü buradan çıkıyor):**

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
python -m pip install -r requirements.txt
```

O `cd` repo kökü. Bu sitede *repo kökü* yazan her yer bu klasörü kastediyor.

**git yok mu?** [ZIP'i indir](https://github.com/kuthaygumus/amadeus-rag-training/archive/refs/heads/main.zip), çıkart ve çıkardığın klasöre `cd` yap. Sonrası birebir aynı.

**Çıplak `pip` değil, `python -m pip`.** Windows'ta ikisi farklı yorumlayıcılara denk gelebiliyor ve o zaman paketler notebook'ların göremediği bir yere kuruluyor — akşam READY veren laptopta, günün ortasında `ModuleNotFoundError`. Shell'in sadece `py` launcher'ını tanıyorsa her yerde `py -m pip` ve `py scripts/verify_setup.py` kullan.

İki paket: modül 2'nin ağı eğitmek için kullandığı `numpy` ve modül 8'in vektör veritabanı `chromadb`. `requirements.txt`'te ölçüldü: 79 paket içinde 86 MB wheel, diskte kabaca 400 MB'a açılıyor — bunu da evde yap. `chromadb-client` değil **tam `chromadb`** olmak zorunda: ince istemci metni lokalde embed edemiyor, dolayısıyla sessizce yanlış modele uzanmak yerine hata fırlatıyor; tam paketin bunun yerine yaptığı şey ise modül 6'nın tamamı.

## Model de paket de olmayan iki dosyayı indir

**Terminal (repo kökü):**

```bash
python scripts/seed_offline_assets.py
```

Notebook'ların okuduğu ama kurulumun geri kalanının getirmediği iki dosya var; ikisi de aksi halde bir modülün ortasında iniyor:

- **MNIST**, dört arşiv halinde ölçülen 11.6 MB, `notebooks/mnist_data/` altına. Modül 2 ağını bunun üzerinde eğitiyor. Klasör gitignore'da, yani taze bir clone'da yok.
- **Chroma'nın varsayılan embedder'ı.** Modül 8 bir collection açarken metni vektöre nasıl çevireceğini hiç söylemiyor, Chroma da senin yerine bir model seçiyor. Bunu ilk kez gerektiren çağrı, `~/.cache/chroma/onnx_models/` altına bir `all-MiniLM-L6-v2` ONNX arşivi çekiyor — ölçüldü: 83 MB.

Bunu `pip install` sonrasında çalıştır, çünkü ikinci yarısı `chromadb`'nin import edilebilmesini gerektiriyor; iki kez çalıştırmak da sorun değil, diskte olan her şey raporlanıp atlanıyor. `verify_setup.py` biri eksikse NOT READY veriyor, çünkü burada eksik bir dosya salonda canlı bir indirme demek.

## Notebook'ları VS Code'da açmak

Her notebook iki kere var: `notebooks/NN_ad.ipynb` olarak ve percent formatında `notebooks/NN_ad.py` olarak — aynı hücreler, JSON yerine `# %%` işaretleriyle. **Çalıştırdığın dosya `.py` olanı.** `.ipynb` ikizlerinde kayıtlı çıktı yok; birini açarsan VS Code sana Jupyter kurmayı önerecek — hayır de, kapat, aynı adın `.py` dosyasını aç.

**VS Code'u ve Python eklentisini kur.** [code.visualstudio.com](https://code.visualstudio.com). Windows'ta **User Installer**'ı al: kendi profiline kuruyor ve yönetici hakkı istemiyor. Sonra Extensions panelini aç (`Ctrl+Shift+X`, macOS'te `Cmd+Shift+X`), **Python** ara ve Microsoft'un yayımladığını kur.

**Açık klasör repo kökü olsun.** `File → Open Folder →` `amadeus-rag-training` — yani `corpus/`, `notebooks/`, `eval/` ve `exercises/` klasörlerini içeren klasör. `notebooks/` değil. VS Code'un gömülü terminali hangi klasörü açtıysan orada başlıyor; kökü açtığında `` Ctrl+` `` ile iki yüzeyi tek pencerede elde ediyorsun. Notebook'lar kendi çalışma dizinlerini kendileri hallediyor: her birinin ilk bloğu `_preflight`'ı çağırıyor, o da `../corpus` ve `../eval` çözülsün diye `notebooks/` içine geçiyor ve geçtiğinde yolu yazıyor.

**Yorumlayıcıyı seç.** `Ctrl+Shift+P` / `Cmd+Shift+P` → `Python: Select Interpreter` → paketleri kurduğun aynı Python 3.10+ sürümü. Seçtiğin sürüm sağ altta durum çubuğunda görünüyor.

**Bir bloğu çalıştır.** İmleç bloğun içinde, `Shift+Enter`. Durum basışlar arasında korunuyor; bunu bir script koşusundan ayıran şey de bu.

**"Çalıştı" neye benziyor.**

**VS Code — `notebooks/00_bare_llm_fails.py`, ilk `# %%` bloğu:**

```text
working directory set to .../amadeus-rag-training/notebooks
ready: models qwen2.5:3b
model: qwen2.5:3b
ready
```

İlk satır yalnızca zaten `notebooks/` içinde değilsen çıkıyor; çıkmaması da normal. Son iki satır Ollama'nın cevap vermesi. Bunun yerine başlığı `NOT READY` olan bir blok görüyorsan oku: `_preflight` eksik olan şeyi ve onu düzelten komutu adıyla yazıyor.

### Bir hücre makinende çalışmazsa

Ollama kapanmış, hiç çekmediğin bir model, uzun bir hücreyi beklemeyecek kadar yavaş bir laptop — ölçüm hücreleri yeniden hesaplanmak yerine kayıttan oynatılabiliyor. Notebook'un ilk bloğundan önce `USE_CACHED=1` ayarla.

**Terminal (repo kökü):**

```bash
USE_CACHED=1 python notebooks/05_chunking_and_noise.py
```

PowerShell'de önce kendi satırında `$env:USE_CACHED=1`, sonra `python` satırı.

`notebooks/_cached.py` o zaman modeli çağırmak yerine o hücrenin sonucunu `notebooks/cached_runs.json` içinden okuyor — editörde açıp okuyabileceğin sıradan bir JSON dosyası. Bu hiçbir zaman sessiz değil: oynatılan her hücre, kaydın tarihini, makinesini ve kaydın tam mı yoksa kısaltılmış bir koşu mu olduğunu yazan bir `[CACHED]` bandı basıyor. Kaydı olan notebook'lar 03'ten 08'e kadar olanlar; notebook 02'nin kaydı yok, `kraken-q2` eksik olduğunda probe'larının oynatılmayıp atlanmasının sebebi bu. Bu, oturumu yürütmeye yarar; günün tamamını offline koşturmanın yolu değildir.

## Ne çalıştırıyorsun

Bu akşam notebook yok. Tek script. Sadece `localhost` ile konuşuyor ve `urllib`'e bu çağrıları sistem proxy'sine yönlendirmemesini söylüyor, yani ofiste de evde de uçakta da aynı davranıyor.

**Terminal (repo kökü):**

```bash
python scripts/verify_setup.py
```

**Ne görmen gerekiyor.** Bir sütun dolusu `ok` satırı, sonra tek başına duran bir satırda **READY** kelimesi.

**Ne kadar sürüyor.** Ollama modeli belleğe almışsa bir iki saniye; reboot sonrası ilk koşu daha uzun.

**READY** şu demek: Python 3.10+, `numpy` ve `chromadb` import ediliyor, MNIST ve Chroma'nın varsayılan embedder'ı diskte, Ollama 11434 portunda cevap verdi, üç model tag'i de yerinde, embedding çağrısı gerçek vektörlerle döndü ve generation çağrısı bir tablonun doğru kolonunu okudu. Terminali kapat, unut.

**NOT READY** hangi satırın düştüğünü ve onu düzelten komutu yazıyor. Oku, komutu çalıştır, scripti tekrar çalıştır.

Bazı satırlar `ok` veya `FAIL` yerine `warn` yazıyor — VS Code alışılmış yerlerde bulunamadı, `kraken-q2` kurulu değil. Bir uyarı READY'yi hiçbir zaman NOT READY'ye çevirmiyor.

Script makineni de ölçüyor: cevap başına 3 saniyenin altı rahat, 3-8 saniye idare eder ama demodan yavaş, 8 saniyenin üstü yavaş. "Yavaş" aldıysan yedek modeli de indir:

**Terminal (repo kökü):**

```bash
ollama pull qwen2.5:1.5b
```

Bu da 986 MB ve sana bir cevaba mal oluyor: yukarıdaki üç test sorusunda 1.5B model ücret tablosunun yanlış satırını okudu. Yanlış cevabın hangisi olduğunu bil; onu pipeline'ın hatasıymış gibi debug etmeye kalkma.

**Windows'ta Python hiç kurulu değilse.** Asıl kontrol onsuz başlayamıyor, o yüzden önce bootstrap'i koştur. Hiçbir şey kurmuyor, sadece eksikleri adıyla söylüyor.

**PowerShell (repo kökü):**

```powershell
powershell -ExecutionPolicy Bypass -File scripts\verify_setup.ps1
```

Bu komut `running scripts is disabled on this system` diyorsa, execution policy'yi Group Policy ayarlamıştır ve `-ExecutionPolicy` onu geçersiz kılamaz. Yönetici hakkına ihtiyacın yok — dosyayı pipe ile içeri ver, çünkü pipeline'dan gelen metin bir script dosyası değil:

**PowerShell (repo kökü):**

```powershell
Get-Content scripts\verify_setup.ps1 | powershell -NoProfile -Command -
```

`UNVERIFIED: PowerShell scripti yönetilen bir Windows laptopunda henüz koşturulmadı. Önce nelerin test edilmesi gerektiği scriptin başlık yorumunda yazıyor.`

## Podman gerekli mi?

Notebook'ları çalıştırmak için hayır: 00'dan 07'ye kadar hepsi Ollama ve Python kullanıyor, başka hiçbir şey değil. İstisna modül 8 — ChromaDB'yi kütüphane olarak değil servis olarak çalıştırıyor — ve eğitim gününe kadar makinende bir container runtime olmayacaksa hiçbir şey bozulmaz: o modülü izle, sonra kendin koştur. Bu eğitimde ölçülen her şey Podman olmadan ölçüldü. İstiyorsan image'ı evde indir.

**Terminal (repo kökü — compose dosyası orada):**

```bash
podman compose up -d          # ya da: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

## Modelleri makineler arasında taşımak

Weight'ler diskte sıradan dosyalar ve taşınabilir — evdeki indirmesi pes eden herkes için çözüm bu.

| İşletim sistemi | Dizin |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

1. **Önce iki makinede de Ollama'yı tamamen kapat.** macOS: menü çubuğundaki ikon → Quit. Windows: sistem tepsisindeki ikon → Quit Ollama. Windows'ta sunucu oturumunla başlıyor ve o dosyaları açık tutuyor.
2. **Değiştirme, birleştir.** `blobs/` ve `manifests/` klasörlerinin *içeriğini* hedef makinedeki aynı adlı klasörlere kopyala. `models` dizinini komple üstüne kopyalamak, o makinede zaten bulunan modellerin `manifests/` kayıtlarını siler ve onları sessizce kayıt dışı bırakır.
3. Ollama'yı tekrar başlat ve `ollama list` çalıştır. Gelen her şey listede görünür.

<div class="presenter-note">
<strong>09:10, on dakika.</strong> Herkes aynı anda <code>verify_setup.py</code> çalıştırıyor, sen salonda dolaşıp ekranları okuyorsun — sorarak değil, okuyarak. Enter'a basmadan önce salona tahmin ettir: "kaçımız yeşil çıkacak?" Sesli bir sayı al. İki şey oluyor: insanlar bir tahmine bağlanıyor ve akşam hiç çalıştırmamış olanlar kendiliğinden ortaya çıkıyor.
<br /><br />
<strong>Kırmızı laptop triyajı, bu sırayla.</strong> Terminal yanlış klasörde → repo köküne <code>cd</code> yaptır ve bunu yaparken iki yüzey kuralını bütün salona söyle. Ollama çalışmıyor → uygulamayı aç. <code>ollama</code> komutu tanınmıyor ama script sunucunun cevap verdiğini söylüyor → terminal penceresi eski, yenisini açtır. Model yok ama makine hızlı → pull'u şimdi başlat, Modül 1 sırasında biter. Model yok ve ağ sürünüyor → USB, yanında iki tane taşıyorsun. Kurulum yönetici hakkına takılmış → dur, hemen eşleştir, salonun sabahını buna harcama.
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

Model boyutları `ollama list`'in kurulu tag'ler için bildirdiği değerler; Python satırı paketlerin kurulduktan sonra diskte tuttuğu yer, yani ağdan geçen byte sayısı o satırın ima ettiğinden az. Aynı toplam `scripts/verify_setup.py` içinde de hesaplanıyor, yani script ile bu sayfa birbirinden kayamaz.

| Embedder | Türkçe soru / İngilizce doküman olan altı soruda hit@1 |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

İki embedder de aynı structure-aware chunk'lar üzerinde skorlandı; modül 6'nın koşturduğu koşul bu. Altı soru bir rakam değil bir yön verir — ama 0.000'a karşı 0.667 bir yuvarlama tartışması değil. Fark o ölçümün bir özelliği, iki modelin sabit bir özelliği değil.

</div>

## Neden Ollama, neden model hub değil

Böyle bir eğitimi kurmanın normal yolu şu: `pip install transformers`, bir model hub'ından weight çek, başla. Bu yol denendi ve bırakıldı: tek soru cevaplanmadan önce her laptopa birkaç gigabyte'lık bir Python yığını kuruyor ve günün sayılarını her makinenin hangi kütüphanenin hangi sürümünü çözdüğüne bağlı bırakıyor.

Ollama bir framework değil, lokal bir model sunucusu. Model adını veriyorsun, weight'lerin quantize edilmiş bir kopyasını indiriyor ve `http://localhost:11434` üzerinde küçük bir HTTP API açıyor: `/api/tags` elindekileri listeliyor, `/api/embed` metni vektöre çeviriyor, `/api/chat` cevap üretiyor. Notebook'lar buraya standart kütüphanedeki `urllib` ile bağlanıyor. Tek indirme, tek process, SDK yok — ve ekimde senin laptopunda bu sitedeki sayıları üreten makinedekiyle aynı cevabı veren, sabitlenmiş bir model.

<div class="presenter-note">
Ağzında gevelenmemesi gereken cümle: <strong>her şey önündeki laptopta çalışıyor ve bu bir ideoloji değil, ölçüm kararı.</strong> İlk on dakikada birisi "neden API kullanmıyoruz?" diye soracak. Cevap "lokal daha iyi" değil — cevap "sabitlenmiş lokal bir model eylülde de ekimde de aynı sayıyı veriyor, yani bir sayı oynadığında onu neyin oynattığını biliyoruz. Ve kimsenin oturmak için hesaba, key'e ya da onaya ihtiyacı olmuyor."
</div>

## Daha derine

Ollama quantize edilmiş GGUF weight'leri sunuyor; 3 milyar parametreli bir modelin ~2 GB indirme olması ve GPU'suz bir laptopta çalışması bu yüzden. Quantization her weight'i 16 bit yerine kabaca 4 bitte saklıyor, karşılığında küçük bir kalite kaybı veriyor — bu eğitimdeki işlerde görünmeyen, uzun bir akıl yürütme zincirinde görünecek bir kayıp. Günün hiçbir yerinde 3B'lik bir modelden tek çağrıda zekice bir şey istenmemesinin bir sebebi bu.

Bu odanın dışında seni asıl yakacak seçim embedder, çünkü sessizce patlıyor. `nomic-embed-text` ve `all-MiniLM-L6-v2` İngilizce öncelikli modeller: Türkçe bir soru ve onu cevaplayan İngilizce dokümanı verdiğinde hata da vermiyorlar, boş da dönmüyorlar — kendinden emin biçimde yanlış dokümanı döndürüyorlar. `bge-m3` çok dilli eğitilmiş, dolayısıyla Türkçe bir cümle ile İngilizce karşılığı vektör uzayında birbirine yakın düşüyor. Bu farkı kodu okuyarak göremezsin. Sadece skorlayarak — yukarıdaki kutu tam olarak o.

## Çıkış cümlesi

> Her şey kurulu ve henüz hiçbir şey birbirine bağlı değil. Yarın modelin tamamen yalnız haliyle başlıyoruz — `notebooks/00_bare_llm_fails.py` ve hiç görmediği bir Kraken Air ücret kuralı hakkında tek bir soru.
