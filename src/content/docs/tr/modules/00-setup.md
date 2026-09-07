---
title: "0. Kurulum — Gelmeden Önce"
description: "Ollama kur, iki model indir, tek script çalıştır. On dakika, bir önceki akşam — sabah ofis ağında değil."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

Bu modülün gate sorusu yok. Burası ön hazırlık: bir önceki akşam, yirmi kişiyle paylaşmadığın bir ağda yapılır.

## Zaten patlamış olan şey

Böyle bir eğitimi kurmanın normal yolu şu: `pip install transformers`, HuggingFace'ten model çek, başla. Bu, buradaki kurumsal ağda denendi. Çalışmıyor. HuggingFace üzerinden model weight indirmeleri policy tarafından bloklanıyor — yavaş değil, throttle değil, reddediliyor.

Blok boyuta değil pattern'e bakıyor. 17 KB'lık bir weight dosyası da tıpkı 4 GB'lık biri gibi reddediliyor. Yani akla ilk gelen çözüm — "o zaman minik bir model kullanırız" — çözüm değil.

`registry.ollama.ai` bloklu değil. Aynı laptopta, aynı ağda, aynı öğleden sonra ölçüldü: 1.16 GB'lık bir model blob'u 2.36 MB/s ile indi ve tamamlandı.

Günün bütün şekli bu iki gerçekten çıktı. Bu eğitimdeki her model Ollama'dan çekiliyor ve kendi makinende çalışıyor. API key yok, cloud hesabı yok, sign-in yok, masraf yok — ve alıştırmalarda ürettiğin sayılar sonuç sayfasındaki sayılarla aynı, çünkü eylüle kadar kimse altından hosted bir modeli sessizce güncellemiyor.

## Ollama tam olarak nedir

Bir framework değil, lokal bir model sunucusu. Model adını veriyorsun, weight'lerin quantize edilmiş bir kopyasını indiriyor ve `http://localhost:11434` üzerinde küçük bir HTTP API açıyor: `/api/tags` elindekileri listeliyor, `/api/embed` metni vektöre çeviriyor, `/api/chat` cevap üretiyor. Notebook'lar buraya standart kütüphanedeki `urllib` ile bağlanıyor. Kurulacak bir SDK, yanlış anda internete çıkmaya kalkacak bir import yok.

Burada olmasının tek sebebi bu: ağdan sağ çıkan ve günü tekrarlanabilir tutan tek altyapı parçası.

<div class="presenter-note">
Şunu bir kez söyle ve ağzında gevelemeden söyle: <strong>burada HuggingFace weight'leri bloklu, Ollama registry'si değil, Ollama kullanmamızın tek sebebi bu.</strong> İlk on dakikada birisi "neden API kullanmıyoruz?" diye soracak. Cevap "lokal daha iyi" değil — cevap "bu kararı ağ zaten vermiş, ben de ölçtüm."
</div>

## Ollama kurulumu

**macOS.** [ollama.com](https://ollama.com) üzerinden indir, uygulamayı Applications'a sürükle, bir kez aç. Menü çubuğunda ikon ve çalışan bir sunucu görüyorsun. Terminalden doğrula:

```bash
ollama --version
```

**Windows, yönetici hakkı olmadan.** `OllamaSetup.exe` per-user bir kurulum. Binary'leri `%LOCALAPPDATA%\Programs\Ollama` altına koyuyor ve elevation istemiyor. Sunucu senin oturumunla birlikte başlıyor.

`UNVERIFIED: bu, bir Amadeus Windows imajında henüz test edilmedi. Per-user kurulum yolu installer'ın tasarımı, burada ölçülmüş bir sonuç değil.` Kurulum senden yönetici parolası isterse orada dur. Direnme, günün sabahı 09:00'da IT'yi ikna etmeye çalışma — bir önceki akşam haber ver, makinesi yeşil olan biriyle eşleştirilirsin.

## İki model indir

```bash
ollama pull qwen2.5:3b     # generation ve rerank
ollama pull bge-m3         # embedding, ve Türkçeyi kaldırıyor
```

İkisi birlikte yaklaşık 3.1 GB. Bunu evde yap, en azından eğitim sabahı ofis wifi'sinde yapma. Bu ağda ölçülen 2.36 MB/s ile 3.1 GB tek laptop için kabaca yirmi iki dakika — ve on beş kişi saat 09:00'da aynı anda başlarsa 2.36 MB/s'i paylaşmıyorsunuz, bölüyorsunuz.

İki seçim de ölçüldü, blog yazısından alınmadı. `qwen2.5:3b` test sorularının 3'ünde 3'ünü doğru cevapladı, soru başına 0.9 s. `llama3.2:3b`, tablo başlığı context'in **içindeyken** bile doğru cevap EUR 90 iken EUR 70 dedi — bu eğitimde yasaklı, yerine koyma. `bge-m3` burada çünkü Türkçe soru / İngilizce doküman olan altı soruda 0.667 alıyor, `nomic-embed-text` ise 0.000.

## Python ve repo

Python 3.10 veya üzeri — kontrol scripti `dict | None` tip sözdizimi kullanıyor, yani eski bir Python günün ortasında değil ilk satırda patlar.

```bash
git clone https://github.com/kuthaygumus/amadeus-rag-training
cd amadeus-rag-training
pip install -r requirements.txt
python scripts/verify_setup.py
```

İki paket. Modül 2'nin ağı eğittiği `numpy`, ve modül 5 ile 8'in kullandığı vektör veritabanı `chromadb`. Chromadb yaklaşık 400 MB wheel indiriyor, çoğu onnxruntime — bunu evde yap, ofis ağında aynı anda on dokuz kişiyle birlikte değil.

`chromadb-client` değil **tam `chromadb`** olmak zorunda, ve sebebi modül 6'nın tamamı. İnce istemci metni lokalde embed edemiyor, dolayısıyla sessizce yanlış modele uzanmak yerine hata fırlatıyor. Gürültülü bir hata aslında iyilik olurdu. Tam pakette gerçekten olan şey ise görülmeye değer.

## Podman gerekli mi?

Notebook'ları çalıştırmak için hayır. Notebook 00'dan 07'ye kadar hepsi Ollama ve Python kullanıyor, başka hiçbir şey değil; hiçbir container runtime kurulu olmayan bir makinede sorunsuz koşarlar.

İstisna modül 8, ve yapabiliyorsan kendin koşturmaya değer. ChromaDB'yi kütüphane olarak değil servis olarak çalıştırıyor:

```bash
podman compose up -d          # ya da: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

Podman macOS ve Windows'ta native değil — altta küçük bir Linux VM çalıştırıyor, ki Windows'ta bu WSL2 demek. Admin yetkisi ve bir reboot gerektiriyor. Eğitim gününe kadar makinende olmayacaksa hiçbir şey bozulmaz: o modülü izle, sonra kendin koşturursun. Bu eğitimde ölçülen her şey Podman olmadan ölçüldü.

Compose dosyası repo kökünde, container image'ı yaklaşık 650 MB. Aynı tavsiye: evde indir.

## Ne çalıştırıyorsun

Bu akşam notebook yok. Tek script: `scripts/verify_setup.py`. Sadece `localhost` ile konuşuyor, yani ofiste de evde de uçakta da aynı davranıyor.

```bash
python scripts/verify_setup.py
```

Beş şeyi kontrol edip tek kelime yazıyor.

**READY** şu demek: Python 3.10+, Ollama 11434 portunda cevap verdi, iki model tag'i de yerinde, embedding çağrısı gerçek vektörlerle döndü ve generation çağrısı bir tablonun doğru kolonunu okudu. Terminali kapat, unut.

**NOT READY** ise hangi satırın düştüğünü ve onu düzelten komutu yazıyor. Oku, komutu çalıştır, script'i tekrar çalıştır.

Script'te iki detay bilerek böyle. Modelleri **tam tag** ile eşleştiriyor: elinde `qwen2.5:1.5b` olması `qwen2.5:3b` şartını karşılamıyor, çünkü zayıf modelle gelen birine "hazırsın" demek buradaki en kötü sonuç. Ve son kontrol bir smoke test değil, doğruluk testi: modele iki ceza kolonlu bir tablo verilip K sınıfının iptal cezası soruluyor. 90 demesi gerekiyor, 70 değil. 70 diyen bir model, Modül 7'deki chunking alıştırmasını gayet düzgün çalışırken bozukmuş gibi gösterir.

Script makineni de ölçüp söylüyor: cevap başına 3 saniyenin altı rahat, 3-8 saniye idare eder ama alıştırmalar demodan yavaş hissettirir, 8 saniyenin üstü yavaş. "Yavaş" aldıysan yedek modeli de indir:

```bash
ollama pull qwen2.5:1.5b
```

Bunun bedeli konusunda kendine dürüst ol. 1.5B model aynı test sorularında 3'te 2 aldı ve multi-hop soruda yanlış tablo satırını seçti. Günü onun üzerinde geçirirsen, pipeline'ın değil modelin suçu olan bir yanlış cevapla karşılaşacaksın. Hangisi olduğunu bil, debug etmeye kalkma.

## Dosyalar gerçekte nerede duruyor

Weight'ler diskte sıradan dosyalar ve taşınabilir.

| İşletim sistemi | Dizin |
|---|---|
| macOS / Linux | `~/.ollama/models` |
| Windows | `%USERPROFILE%\.ollama\models` |

İçinde `blobs/` ve `manifests/` var. Modelleri olan bir makineden `models` dizinini komple USB'ye kopyala, olmayan makinede aynı yere bırak, Ollama'yı yeniden başlat, `ollama list` onları gösterir. Evdeki bağlantısı 3 GB'da pes eden herkes için offline çözüm bu.

<div class="presenter-note">
<strong>09:10, on dakika.</strong> Herkes aynı anda <code>verify_setup.py</code> çalıştırıyor, sen salonda dolaşıp ekranları okuyorsun — sorarak değil, okuyarak. Enter'a basmadan önce salona tahmin ettir: "kaçımız yeşil çıkacak?" Sesli bir sayı al. İki şey oluyor: insanlar bir tahmine bağlanıyor, ve akşam hiç çalıştırmamış olanlar kendiliğinden ortaya çıkıyor.
<br /><br />
<strong>Kırmızı laptop triyajı, bu sırayla.</strong> Ollama çalışmıyor → uygulamayı aç. Model yok ama makine hızlı → pull'u şimdi başlat, Modül 1 sırasında biter. Model yok ve ağ sürünüyor → USB, yanında iki tane taşıyorsun. Kurulum yönetici hakkına takılmış → dur, hemen eşleştir, salonun sabahını buna harcama.
<br /><br />
<strong>Eşleştirme bir ceza değil, geçerli bir plan.</strong> Bunu yüksek sesle söyle: "iki kişiye bir laptop bu işin normal hali — biri yazar, biri çıktıyı okuyup itiraz eder." Kırmızı laptopu olan kişinin altı saat sessizce oturmasına izin verme.
</div>

## Sayılar ne dedi

<div class="measured">

| Kurumsal ağda ölçülen | Sonuç |
|---|---|
| HuggingFace model weight'leri | policy ile bloklu — 17 KB'lık weight dosyası dahil |
| `registry.ollama.ai`, 1.16 GB blob | 2.36 MB/s ile indi |

| Generation modeli | Doğru | Cevap süresi |
|---|---|---|
| `qwen2.5:3b` | 3/3 | 0.9 s |
| `gemma3:4b` | 2/3 | 1.9 s |
| `qwen2.5:1.5b` | 2/3 — multi-hop'ta yanlış satır | — |
| `qwen3:4b` | doğru, ama reasoning token'ları yüzünden yavaş | 11.6 s |
| `llama3.2:3b` | başlık context'teyken EUR 70 dedi | yasaklı |

| Embedder | Türkçe soru / İngilizce doküman hit@1 (6 soru) |
|---|---|
| `nomic-embed-text` | 0.000 |
| `bge-m3` | 0.667 |

</div>

## Daha derine

Ollama quantize edilmiş GGUF weight'leri sunuyor; 3 milyar parametreli bir modelin ~2 GB indirme olması ve GPU'suz bir laptopta çalışması bu yüzden. Quantization her weight'i 16 bit yerine kabaca 4 bitte saklıyor, karşılığında küçük bir kalite kaybı veriyor. Bu kayıp bu eğitimdeki işlerde görünmüyor; uzun bir akıl yürütme zincirinde görünürdü. Günün hiçbir yerinde 3B'lik bir modelden tek çağrıda zekice bir şey istenmemesinin bir sebebi bu.

Bu odanın dışında seni asıl yakacak seçim embedder, çünkü sessizce patlıyor. `nomic-embed-text` ve `all-MiniLM-L6-v2` İngilizce öncelikli modeller. Türkçe bir soru ve onu cevaplayan İngilizce dokümanı verdiğinde hata da vermiyorlar, boş da dönmüyorlar — kendinden emin biçimde yanlış dokümanı döndürüyorlar. `bge-m3` çok dilli eğitilmiş, dolayısıyla Türkçe bir cümle ile İngilizce karşılığı vektör uzayında birbirine yakın düşüyor. Aynı altı soruda fark 0.000'a karşı 0.667 ve bunu kodu okuyarak göremiyorsun. Sadece skorlayarak.

Her şeyin lokal çalışması ideoloji değil, ölçüm meselesi. Hosted bir endpoint'te eylülde aldığın sayı ile ekimde aldığın sayı karşılaştırılabilir değil ve hangi değişikliğin neye yol açtığını hiç öğrenemiyorsun. Sabitlenmiş lokal bir model günü kontrollü bir deneye çeviriyor; burada bu, frontier bir modelin getireceği ekstra kaliteden daha değerli.

On milyon dokümanda bu kurulumun çoğu ayakta kalmaz ve önce hangi parçanın kırıldığını bilmek işe yarar. Generation kabaca aynı kalır, yine tek tek sorulara cevap veriyorsun. Embedding kalmaz — on milyon chunk'ı laptopta embed etmek haftalar süren bir iş, o yüzden batch'lenmiş GPU inference'a ya da hosted bir embedding endpoint'ine taşınır, vektörler de in-process bir yapı yerine gerçek bir vector database'e. Asıl canını yakan kısım şu: embedder bir migration maliyetine dönüşüyor. Onu değiştirmek bütün corpus'u yeniden embed etmek demek. Yani ilk gün gelişigüzel verdiğin karar, geri dönmesi en pahalı karar oluyor — ki bu da onu on milyon doküman üstünde değil, şimdi yirmi soru üstünde ölçmek için iyi bir gerekçe.

## Çıkış cümlesi

> Her şey kurulu ve henüz hiçbir şey birbirine bağlı değil. Yarın modelin tamamen yalnız haliyle başlıyoruz: hiç görmediği bir Helios Air ücret kuralı hakkında tek bir soru.
