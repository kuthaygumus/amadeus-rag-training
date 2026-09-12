---
title: "0. Kurulum — Gelmeden Önce"
description: "Ollama, Podman Desktop ve Bruno kur, iki model indir, iki container ayağa kaldır, tek bir health isteği çalıştır — bir önceki akşam bir saatten az, neredeyse tamamı indirme. Ve bu sitedeki her komutu çalıştıran kural: durduğun yer repo kökü."
---

Bu modülün gate sorusu yok. Burası ön hazırlık: bir önceki akşam, yirmi kişiyle paylaşmadığın bir ağda yapılır. Bir saatten az, neredeyse tamamı indirme. Bu sayfadaki hiçbir şey Python, notebook ya da herhangi bir yerde hesap gerektirmiyor.

## Bu sitedeki her komut nereye yazılıyor

Bu bölümü bir kez oku, günün geri kalanında tahmin etmen gereken hiçbir şey kalmasın. Tam olarak **üç yüzey** var ve üçünün de çıpası, birazdan clone'layacağın klasör: `amadeus-rag-lab`, yani içinde `compose.yaml`, `bruno/`, `corpus/` ve `src/` bulunan klasör.

1. **Bruno.** Tıklayarak ilerlediğin istek koleksiyonu. Her modül bir klasör ve bir istek adı veriyor — **Bruno — `01-bare-llm` › `ask-about-kraken`** — sen sol panelden seçip oka basıyorsun. Her isteğin cevapta neye bakacağını söyleyen bir *Docs* sekmesi var.
2. **Terminal, repo kökünde.** `amadeus-rag-lab` içine `cd` yapılmış tek bir terminal penceresi, gün boyunca açık kalıyor. Bu sitedeki her `ollama …`, `podman compose …` ve `curl …` burada çalışıyor.
3. **Podman Desktop.** İki container'ı, loglarını ve durumlarını gösteren pencere. Tıkladığından çok bakıyorsun.

Bu sitenin her sayfasında her komut bloğunun hemen üstünde, kalın harflerle, hangi yüzeye ait olduğu yazıyor; yani hiçbir zaman kendin çıkarmak zorunda kalmıyorsun.

**Doğru terminaldesin** dediğin an, `ls` — Windows'ta `dir` — çıktısında `compose.yaml`, `bruno`, `corpus` ve `src` göründüğü an. `podman compose` compose dosyasını bulamadığını söylüyorsa önce buna bak: bu, bozuk bir kurulumdan çok daha sık yanlış pencerede olmaktan kaynaklanıyor.

<div class="presenter-note">
09:10'da tahtaya yaz ve orada bırak: <strong>üç yüzey, tek çıpa — repo kökü.</strong> Bir salonun on dakikayı kaybetmesinin en yaygın yolu şu: biri <code>podman compose up</code> komutunu kendi home dizininden çalıştırıyor ve herkes gayet sağlam bir kurulumu debug etmeye çalışıyor. Bir laptop <code>compose.yaml</code> bulunamadı dediğinde, başka hiçbir şey sormadan önce "o terminal hangi klasörde?" diye sor.
</div>

## Üç şey kur

**Ollama** modelleri makinende çalıştırıyor. **Podman Desktop** ChromaDB'yi ve api'yi container olarak çalıştırıyor. **Bruno** istek koleksiyonu. Başka hiçbir şey yok — runtime, SDK ya da editör gerekmiyor.

### Ollama

**macOS.** [ollama.com/download](https://ollama.com/download) üzerinden indir, uygulamayı Applications'a sürükle, bir kez aç. Menü çubuğunda bir ikon ve `http://localhost:11434` üzerinde dinleyen bir sunucu görüyorsun.

**Terminal (herhangi bir yerde — repo henüz yok):**

```bash
ollama --version
```

**Windows, yönetici hakkı olmadan.** `OllamaSetup.exe` per-user bir kurulum: binary'ler `%LOCALAPPDATA%\Programs\Ollama` altına gidiyor, elevation istemiyor, sunucu senin oturumunla başlıyor.

`UNVERIFIED: per-user kurulum yolu installer'ın tasarımı; yönetilen bir Windows laptopunda doğrulanmadı.` Kurulum senden yönetici parolası isterse orada dur — günün sabahı 09:00'da IT'yle tartışmak yerine bir önceki akşam haber ver, makinesi yeşil olan biriyle eşleştirilirsin.

Installer Ollama'yı kullanıcı PATH'ine ekliyor ama o sırada zaten açık olan bir terminal bunu görmüyor. `ollama --version` "komut bulunamadı" diyorsa, bir sonuç çıkarmadan önce o pencereyi kapat ve yenisini aç.

### Podman Desktop

[podman-desktop.io](https://podman-desktop.io) üzerinden indir. Bir kez aç; ilk açılış ekranı Podman'ın kendisini kurmayı öneriyor — kabul et. macOS ve Windows'ta container'lar bu adımın oluşturduğu küçük bir Linux VM'inin içinde çalışıyor ve o VM'i oluşturmak da kendi başına bir indirme; evde yapılmasının sebebi bu.

`UNVERIFIED: Podman Desktop'ın yönetilen bir Windows laptopunda yönetici hakkı olmadan kurulup kurulmadığı. Windows'ta WSL 2 gerektiriyor ve WSL 2'yi etkinleştirmek çoğunlukla elevation isteyen bir adım. Bu, günün 1 numaralı riski. Seni engelliyorsa bir önceki akşam söyle — eşleştirilirsin.`

Elinde zaten Docker Desktop ve çalışan bir `docker compose` var mı? O da çalışıyor, tek bir farkla; [Evde clone'la ve build et](#evde-clonela-ve-build-et) bölümünde yazıyor.

### Bruno

[usebruno.com/downloads](https://www.usebruno.com/downloads) üzerinden indir. Masaüstü uygulaması, per-user, hesap yok. Clone'ladığın reponun bir klasörünü koleksiyon olarak açacaksın — import edilecek, senkronlanacak bir şey yok.

## İki model indir

**Terminal (herhangi bir yerde):**

```bash
ollama pull gemma3:4b     # soruları cevaplıyor        (3.3 GB)
ollama pull bge-m3        # metni vektöre çeviriyor    (1.2 GB)
```

Toplam yaklaşık 4.5 GB. İkisini de indir. Gün içinde salonda model çekmek, bu sayfanın önlemek için var olduğu tek şey: bir kurumsal laptopta ölçüldü, ofis hattı `registry.ollama.ai` için tek akışta 3.7–7.8 MB/s veriyor. Yirmi laptop saat 09:00'da aynı 4.5 GB'lık pull'u başlattığında hattı her biri ayrı ayrı almıyor — bölüşüyorlar. Evdeki bağlantın pes ederse weight'ler sıradan dosyalar ve onlara sahip bir makineden kopyalanabiliyor: [Modelleri makineler arasında taşımak](#modelleri-makineler-arasında-taşımak).

**Modeli değiştirme.** Bu sitedeki her sayı `gemma3:4b` cevap verirken ve `bge-m3` embed ederken üretildi; ikisinden birini değiştirirsen ekranındaki cevaplar projektördekilerle örtüşmeyi bırakır ve günü bug olmayan bir farkı debug etmekle geçirirsin.

## Evde clone'la ve build et

Aynı hat hızı iki container image'ı için de geçerli: `chromadb/chroma:1.5.9` `docker.io`'dan çekiliyor, api image'ı ise repodan build ediliyor ve içindeki `npm ci`, kurumsal proxy'nin ulaşamayabileceği **Podman VM'inin içinde** çalışıyor. Bir kez evde build et. Image makinende kalıyor ve eğitim günü `--build` olmadan `podman compose up` ile başlıyor.

**Terminal (herhangi bir yerde — repo kökü buradan çıkıyor):**

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-lab.git
cd amadeus-rag-lab
podman pull docker.io/chromadb/chroma:1.5.9
podman compose up --build
```

O `cd` repo kökü. Bu sitede *repo kökü* yazan her yer bu klasörü kastediyor.

**Ne görmen gerekiyor.** İki container ayağa kalkıyor: önce 8000 portunda `chroma` — compose, ikinciyi başlatmadan önce onun healthcheck'ini, `/api/v2/heartbeat`'i bekliyor — sonra 3000 portunda `api`; son log satırları `amadeus-rag-lab listening on http://localhost:3000` ve `start with GET /health`. İlk `up --build` build artı image pull için yaklaşık bir dakika sürüyor; sonraki her `up` saniyeler.

Ollama bir container **değil**. Makinende kalıyor, GPU'nun olduğu ve modelleri az önce indirdiğin yerde; api ona `http://host.containers.internal:11434` üzerinden ulaşıyor. Docker altında değişen tek satır da bu: `OLLAMA_URL=http://host.docker.internal:11434 docker compose up --build`.

Bir sonraki bölüm için çalışır bırak — `up` ön planda kalıyor ve iki container'ın loglarını akıtıyor. Health isteği yeşil olduğunda `Ctrl+C` onu durduruyor; sonra container'ları temiz kapat:

**Terminal (`amadeus-rag-lab` repo kökü):**

```bash
podman compose down
```

`down` container'ları siliyor; image'ları ve `chroma-data` volume'unu tutuyor. Bu akşam `-v` ekleme; o volume'u siler ve bu akşam içinde bir şey olmasa da eğitim günü olacak.

**git yok mu?** [ZIP'i indir](https://github.com/kuthaygumus/amadeus-rag-lab/archive/refs/heads/main.zip), çıkart ve çıkardığın klasöre `cd` yap. Sonrası birebir aynı.

## Ne çalıştırıyorsun

Tek istek. Bruno'yu aç → *Open Collection* → clone'ladığın reponun **içindeki** `bruno/amadeus-rag-lab` klasörünü seç — repo kökünü değil, `bruno/`'yu değil. Sağ üstteki environment seçicisinden `local`'i seç; `baseUrl`'i `http://localhost:3000` yapıyor, başka hiçbir şey yapmıyor. Sonra:

**Bruno — `00-health` › `health`**

```text
GET {{baseUrl}}/health
```

Body yok. Oka bas.

**Ne görmen gerekiyor.** Status `200`, cevapta `"status": "ready"` ve `checks` altında dört `"ok"`: `ollama`, `chatModel`, `embedModel`, `chroma`. `collections` boş bir liste, `currentCollection` ise `null` — doğru, henüz hiçbir şey ingest edilmedi. *Tests* sekmesindeki iki assertion yeşile dönüyor. Bruno'yu kapat, unut.

**Bunun dışındaki her şey eksik parçayı adıyla söylüyor.** Cevap `503` ve `"status": "not-ready"`; düşen check `"ok"` yerine ne yapman gerektiğini yazıyor:

| Check | Ne yazıyor | Ne demek |
|---|---|---|
| `ollama` | `unreachable at http://host.containers.internal:11434 — start Ollama on your machine` | Container, host'taki Ollama'yı göremiyor. Ollama çalışıyor mu (menü çubuğu / tepsi ikonu)? Windows'ta aşağıdaki nota bak. |
| `chatModel` | `missing — run: ollama pull gemma3:4b` | Tam olarak onu çalıştır, evde. |
| `embedModel` | `missing — run: ollama pull bge-m3` | Aynısı. |
| `chatModel` / `embedModel` | `skipped` | Sadece `ollama` kendisi düştüğünde. Önce onu düzelt; iki model check'i bir sonraki istekte çalışıyor. |
| `chroma` | `unreachable at http://chroma:8000 — is the chroma container running?` | Podman Desktop'a bak: `chroma` yeşil değilse repo kökünde `podman compose down`, sonra `podman compose up`. |
| Bruno: connection refused | — | api hiç ayakta değil. Terminalde `podman compose up` çalışıyor mu? Environment `local` seçili mi? |

**Windows ve `OLLAMA_HOST`.** `UNVERIFIED: Windows'ta Ollama yalnızca 127.0.0.1'e bağlanabiliyor ve Podman VM'i oraya ulaşamıyor — terminalinde ollama list çalışırken container unreachable diyor. Belgelenen çözüm, kullanıcı ortam değişkeni OLLAMA_HOST=0.0.0.0 ayarlamak (Settings → System → Environment variables, kullanıcı kapsamı, yönetici gerekmiyor), sonra Ollama'yı tepsiden kapatıp yeniden açmak. Yönetilen bir Windows laptopunda henüz doğrulanmadı.` Buna evde çarparsan evde düzelt; 09:10'da çarparsan eşleşirsin.

## Modelleri makineler arasında taşımak

Weight'ler diskte sıradan dosyalar ve taşınabilir — evdeki indirmesi pes eden herkes için çözüm bu. İki model birlikte yaklaşık 4.5 GB; bu bir indirme değil, bir USB bellek.

| İşletim sistemi | Dizin |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

1. **Önce iki makinede de Ollama'yı tamamen kapat.** macOS: menü çubuğundaki ikon → Quit. Windows: sistem tepsisindeki ikon → Quit Ollama. Windows'ta sunucu oturumunla başlıyor ve o dosyaları açık tutuyor.
2. **Değiştirme, birleştir.** `blobs/` ve `manifests/` klasörlerinin *içeriğini* hedef makinedeki aynı adlı klasörlere kopyala. `models` dizinini komple üstüne kopyalamak, o makinede zaten bulunan modellerin `manifests/` kayıtlarını siler ve onları sessizce kayıt dışı bırakır.
3. Ollama'yı tekrar başlat ve `ollama list` çalıştır. `gemma3:4b` ve `bge-m3` ikisi de listede görünüyor — ya da burada durup eşleşiyorsun.

Container image'ları da aynı şekilde taşınabilir — yeşil makinede `podman save`, kırmızıda `podman load` — ama 09:10'da bu bir eşleştirme değil, bir tamir. Belleği modeller için taşı; image'lar eğitmenin derdi olsun.

<div class="presenter-note">
<strong>09:10, on dakika.</strong> Herkes aynı anda <strong>00-health</strong> çalıştırıyor, sen salonda dolaşıp ekranları okuyorsun — sorarak değil, okuyarak. Oka basmadan önce salona tahmin ettir: "kaçımız yeşil çıkacak?" Sesli bir sayı al. İki şey oluyor: insanlar bir tahmine bağlanıyor ve akşam hiç çalıştırmamış olanlar kendiliğinden ortaya çıkıyor. Yirmi laptopun kendi <code>localhost:3000</code>'ine vurması ağa hiçbir şeye mal olmuyor; ağa mal olan tek şey bir pull ve 09:10'daki bir pull zaten bir eşleştirme.
<br /><br />
<strong>Kırmızı laptop triyajı, bu sırayla.</strong> Bruno connection refused diyor → "o terminal hangi klasörde?", sonra repo kökünde <code>podman compose up</code> ve bunu yaparken üç yüzey kuralını bütün salona söyle. <code>chroma</code> kırmızı → Podman Desktop'a onunla birlikte bak, <code>down</code> sonra <code>up</code>. Mac'te <code>ollama</code> kırmızı → uygulama çalışmıyor, aç. Windows'ta <code>ollama list</code> çalışırken <code>ollama</code> kırmızı → <code>OLLAMA_HOST</code>; buna beş dakikan var, on beş değil. Model yok ama makine hızlı → pull'u şimdi başlat, modül 1 sırasında biter — ama bunu salonda yalnızca tek bir laptop yapabilir. Model yok ve ağ sürünüyor → USB bellek, yanında iki tane taşıyorsun. Podman yönetici hakkına takılmış → dur, hemen eşleştir, salonun sabahını buna harcama.
<br /><br />
<strong>Eşleştirme bir ceza değil, geçerli bir plan.</strong> Bunu yüksek sesle söyle: "iki kişiye bir laptop bu işin normal hali — biri Bruno'yu sürer, biri cevabı okuyup itiraz eder." Kırmızı laptopu olan kişinin altı saat sessizce oturmasına izin verme.
<br /><br />
<strong>On dakika, on dakika demek.</strong> 09:20'de yeşil olmayan her makine bir tamir işi değil, bir eşleştirmedir. Bu süreyi günün başka hiçbir yerinden geri alamıyorsun. Ve Ollama <em>senin</em> makinende çökmüşse: gate'ler sayfadan da okunuyor, projektörün modül 1'i yeşil olan bir laptoptan koşturuyor.
</div>

## Daha derine

Ollama bir framework değil, lokal bir model sunucusu. Model adını veriyorsun, weight'lerin quantize edilmiş bir GGUF kopyasını indiriyor ve `http://localhost:11434` üzerinde küçük bir HTTP API açıyor — api container'ının onu herhangi bir servis gibi görmesi ve kendisinin container olmasına gerek kalmaması bu yüzden. Quantization her weight'i 16 bit yerine kabaca 4 bitte saklıyor, karşılığında küçük bir kalite kaybı veriyor: bu eğitimdeki işlerde görünmeyen, uzun bir akıl yürütme zincirinde görünecek bir kayıp. Günün hiçbir yerinde 4B'lik bir modelden tek çağrıda zekice bir şey istenmemesinin bir sebebi bu.

İlk on dakikada birisi "neden API kullanmıyoruz?" diye soracak. Cevap "lokal daha iyi" değil. Cevap şu: **sabitlenmiş lokal bir model eylülde de ekimde de aynı cevabı veriyor, yani bir cevap oynadığında onu neyin oynattığını biliyoruz** — ve kimsenin oturmak için hesaba, key'e ya da onaya ihtiyacı olmuyor. Bu sitedeki her şey önündeki laptopta çalışıyor ve bu bir ideoloji değil, ölçüm kararı.

İki container aynı kararın öteki yarısı. ChromaDB `1.5.9`'a sabitlenmiş, api bir lockfile'dan build ediliyor; compose dosyası deployment'ın tamamı, `podman compose down -v` de reset'in tamamı. Modül 6 o kutuyu açıyor.

## Çıkış cümlesi

> Her şey kurulu ve henüz hiçbir şey birbirine bağlı değil. Yarın modelin tamamen yalnız haliyle başlıyoruz — **Bruno — `01-bare-llm` › `ask-about-kraken`**, hiç görmediği bir Kraken Air ücret kuralı hakkında tek bir soru.
