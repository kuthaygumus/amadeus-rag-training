---
title: "8. ChromaDB: Embedded ve Server"
description: "Bunu production'da nasıl çalıştıracağım?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Bunu production'da nasıl çalıştıracağım?**

Kernel'i restart et.

Hata bu kadar basit ve iki saniyede üretiliyor. Bugün kurduğun her retriever, vektörleri tek bir
notebook process'inin içindeki Python listesinde tuttu. `DenseRetriever(doc_ids, documents)`
corpus'u constructor'ında embed ediyor. Process ölünce corpus hâlâ diskte, model hâlâ diskte, ama
son bir saatte iyileştirdiğin 154 structure-aware chunk vektörü yok oldu. Onları geri getirmenin
tek yolu 154 chunk'ı yeniden embed etmek — her seferinde.

28 doküman ve 78,310 karakterde bu birkaç saniye ve hiçbir şey gibi hissettiriyor. İki şey bu
hissi bozuyor. Birincisi boyut: aynı döngü gerçek bir kural corpus'unda kahve molası uzunluğunda
ve bunu her deploy'da, her crash'te, her autoscale olayında yeniden ödüyorsun. İkincisi şekil.
Aslında yazacağın şey bir servis — bir Angular front end, bir endpoint'i çağıracak, o endpoint de
retriever'ı. Liste bir Python process'inde duruyor. Request başka bir process'e geliyor. Bu
cümlenin notebook'un cevap olduğu bir versiyonu yok.

Yani soru "hangi vector database daha iyi" değil. Soru şu: vektörleri üreten process gittiğinde
vektörler nerede yaşıyor.

<div class="presenter-note">
Bunu slayttan önce yap. Salonun önünde notebook kernel'ini restart et, retrieval hücresini
yeniden çalıştır, patlamasını izlet. Sonra sor: "bu rebuild benim corpus'umda değil, sizinkinde
kaç saniye sürer?" İki üç kişi sesli olarak bir sayı söylesin. Bu modül öğle yemeğinin hemen
arkasında, salonun enerjisi günün en düşük noktasında; 20 saniyelik canlı bir hata, herhangi bir
diyagramdan daha çok dikkat satın alır. 2 dakika.
</div>

## Embedded mod: iki satırlık versiyon

Hemen herkesin ilk yazdığı şey bu ve anında çalışıyor.

```python
import chromadb
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection("helios")
collection.add(ids=ids, documents=texts)          # dikkat: embeddings= argümanı yok
print(f"{collection.count()} documents indexed")
```

```text
28 documents indexed
```

Konfigürasyon yok, daemon yok, port yok. `EphemeralClient` collection'ı bellekte tutuyor;
notebook bunu kullanıyor ki bir günlük demo diskte hiçbir şey bırakmasın.

Şimdi **yapmadığımız** şeye dikkat et. Metni vektöre nasıl çevireceğimizi hiç söylemedik. Chroma
bizim yerimize bir şey seçti, sessizce, sormadan.

## Kimsenin seçmediği default

Sorguları çalıştırmadan önce tahmin et: ne seçti ve Türkçe doküman tutan, gün boyu Türkçe soru
alan bir corpus için doğru seçim mi?

```python
DEFAULT_QUERIES = [
    ("CLASSIC K sinifi iptal cezasi ne kadar?", "fare_classic_shorthaul"),
    ("Baglantisini kaciran yolcuya yemek fisi ne kadar?", "sop_misconnect_v4"),
    ("what is the cancellation penalty for CLASSIC class K", "fare_classic_shorthaul"),
]
for question, should_be in DEFAULT_QUERIES:
    top = collection.query(query_texts=[question], n_results=3)["ids"][0]
    print(f"{'right' if top[0] == should_be else 'WRONG':>5}  {question[:50]:<52} -> {top[0]}")
```

```text
embedding function: DefaultEmbeddingFunction
WRONG  CLASSIC K sinifi iptal cezasi ne kadar?              -> macro_tr_noshow
WRONG  Baglantisini kaciran yolcuya yemek fisi ne kadar?    -> macro_tr_noshow
WRONG  what is the cancellation penalty for CLASSIC class   -> fare_classic_longhaul
```

Üç soru, üç yanlış doküman. İki Türkçe soru da aynı çağrı merkezi macro'suna düşüyor. İngilizce
olan daha yakın ama yine yanlış: soru short-haul derken long-haul ücret sayfasını döndürüyor.
Exception yok, uyarı yok, çıktıda bir şeyin ters gittiğini ima eden hiçbir şey yok.

Chroma'nın seçtiği model `all-MiniLM-L6-v2`: 384 boyut, sadece İngilizce ve ilk `add()`
çağrında indiriliyor — internet üzerinden 83,178,821 byte, makine başına bir kez, hiç sormadan.
Bu, modül 6'nın ölçtüğü modelin ta kendisi. Structure-aware chunk'lar üzerinde yirmi altın soruda
hit@1 **0.350** alıyor, `bge-m3` ise **0.800**; Türkçe soru / İngilizce doküman olan altı soruda
ise **0.000** alıyor, `bge-m3` **0.667**.

`pip install chromadb`, dokümanlarını ekle, sorgula — ve Türkçe bir corpus üzerinde sadece
İngilizce bilen bir embedding modeli çalıştırıyorsun. Veritabanına taşınmak bunu ne düzeltir ne de
sana haber verir. Sistem patlamıyor. Sessizce, kimse kontrol etmediği sürece, yanlış cevap
veriyor.

<div class="presenter-note">
Modülün tamamı bu. İki satırlık versiyonu canlı çalıştır, üç WRONG satırının oturmasını bekle ve
açıklamadan önce sus. Biri "ama hata vermedi ki" diyecek — istediğin cümle tam olarak bu, geri
tekrarla. İlk <code>add()</code> takılırsa o makinede default model seed edilmemiştir; notebook
kayıtlı çalıştırmayı oynatır ve bunu yaptığını yazar, devam et. 4 dakika.
</div>

## Aynı collection, senin seçtiğin embedder ile

Chroma kendi hesapladığın vektörleri kabul eder. Tek bir ek argüman:

```python
better = client.get_or_create_collection("helios_bge_m3")
better.add(ids=ids, documents=texts, embeddings=R.embed(texts))   # bge-m3, Ollama üzerinden, bizim çağrımız
```

Geri kalan her şey aynı — aynı client, aynı `query`, aynı sonuç şekli. İngilizce soru artık ilk
sırada `fare_classic_shorthaul` döndürüyor. Daha iyi, ama doküman seviyesinde hâlâ kusursuz değil;
zaten modül 6 ve 7'nin bütün mesaisi orada. Buradaki nokta daha dar: **embedding modeli bir
karardır ve sen vermezsen Chroma senin yerine verir.**

## Restart'tan sağ çıkmak

Gate kaybolan bir index'ti ve onu kapatan tek bir kelime var:

```python
client = chromadb.PersistentClient(path="./chroma-local")     # EphemeralClient() yerine
```

`PersistentClient` yine senin process'inin içinde çalışır — daemon yok, port yok, başlatılacak bir
şey yok. `./chroma-local` dizinine bir SQLite dosyası ve index dosyaları yazar; o dizinin kendisi
veritabanıdır. Kopyalayabilirsin, release artefact'ına koyabilirsin, sıfırdan başlamak için
silebilirsin. Bir process'te yazılıp ikinci bir process'te açıldığında `collection.count()` hiçbir
şey embed etmeden **0.09 s** içinde **28** döndürüyor.

Pek çok iç araç için vermen gereken son deployment kararı budur.

## Metadata filtreleme: listenin yapamadığı şey

Retrieval nadiren "her şeyde ara" demektir. Genelde "bu çeyreğin ücret kurallarında ara, yürürlükten
kalkmış SOP'lerde değil" demektir.

```python
kinds = {i: ("fare" if i.startswith("fare") else "sop" if i.startswith("sop")
             else "bulletin" if i.startswith("bulletin") else "other") for i in ids}
tagged = client.get_or_create_collection("helios_tagged")
tagged.add(ids=ids, documents=texts, embeddings=R.embed(texts),
           metadatas=[{"kind": kinds[i]} for i in ids])

everything  = tagged.query(query_embeddings=R.embed([q]), n_results=3)["ids"][0]
fares_only  = tagged.query(query_embeddings=R.embed([q]), n_results=3,
                           where={"kind": "fare"})["ids"][0]
```

Aramadan sonra değil aramadan önce filtrelemek, bir veritabanı ile bir döngü arasındaki farktır.
"Sadece güncel SOP" cümlesini bir cosine benzerliğinde ifade edemezsin.

## Server: aynı collection, farklı deployment

Şimdi aynı şeyi bir portun arkasına koy. Compose dosyası repo kökünde.

```bash
podman compose up -d          # ya da: docker compose up -d
```

```bash
# macOS / Linux
curl -s http://localhost:8000/api/v2/heartbeat

# Windows (PowerShell) — oradaki düz `curl`, Invoke-WebRequest alias'ıdır ve -s almaz
Invoke-RestMethod http://localhost:8000/api/v2/heartbeat
```

Image 649 MB ve image makinede hazırken server ilk heartbeat'ine başlangıçtan yaklaşık yarım
saniye sonra cevap verdi. Compose yerine elle çalıştırıyorsan registry prefix'ine dikkat: Podman,
Docker'ın yaptığı gibi Docker Hub'ı varsaymaz, dolayısıyla `podman pull chromadb/chroma` durup sana
hangi registry'yi kastettiğini sorar. `docker.io/` yaz ve devam et.

**API path'i `/api/v2/`, `/api/v1/` değil.** Bunu iki kez yüksek sesle söyle. İnternetteki ChromaDB
cevaplarının çoğu v1'e göre yazıldı ve her aramada ilk sonuç onlar. Güncel bir server'a karşı v1
path'i **`410 Gone`** döndürüyor:

```json
{"error":"Unimplemented","message":"The v1 API is deprecated. Please use /v2 apis"}
```

Tek başına status satırı, tam olarak ayağa kalkmamış bir container gibi görünür. Tek sorun bir
URL'deki versiyon numarasıyken insanlar yirmi dakika Podman debug ediyor; gövdeyi oku.

Client değişimi tek satır:

```python
client = chromadb.HttpClient(host="localhost", port=8000)
col = client.get_or_create_collection("helios_remote")
res = col.query(query_embeddings=R.embed([q]), n_results=5, where={"kind": "fare"})
```

O ilk satırın altındaki her şey embedded versiyonla harfi harfine aynı. Aynı `add`, aynı `query`,
aynı `where` filtresi, aynı sonuç şekli. Bu bir tesadüf değil, Chroma'yı öğretmeye değer kılan
ürün kararı: bir dizine karşı geliştirip bir porta karşı ship ediyorsun, retrieval kodunu yeniden
yazmadan.

<div class="presenter-note">
Eğitim günü yirmi laptop'a aynı anda 649 MB çektirme — o pull evde yapılır, kurulum sayfası da
bunu söylüyor. <code>podman images</code> ile image'ın zaten orada olduğunu göster; progress bar
izlemek isteyen olursa pull'u sadece kendi makinende yap. Container kalkmazsa: 8000 doluysa
<code>-p 8001:8000</code>; macOS'te önce <code>podman machine start</code>. Notebook'un server
hücresi try/except içinde, yani container yoksa "no server running" yazıp devam ediyor — sorun
değil, düzeltmek için durma. Yanlış söylenmemesi gereken cümle: <strong>path /api/v2/ —
internette bulacağın eski cevapların hepsi v1 diyor ve 410 Gone alıyor.</strong> 2 dakika.
</div>

## Ne gerçekten değişiyor

API değişmiyor. Collection semantiği değişmiyor. Embedding seçimin değişmiyor, retrieval kalitesi
de değişmiyor — veritabanı bir depolama kararıdır, relevance iyileştirmesi değil. Değişen şey
kodun etrafındaki her şey.

**Failure mode'lar.** Embedded, senin process'in öldüğünde ölür; listenin tamamı bu. Server bunun
üstüne bir network hop'u, bir port, bir container yaşam döngüsü ve artık birinin volume'ü olan bir
disk ekler. İki process'in aynı embedded dizini açması cluster değildir, corruption yoludur; ikinci
bir okuyucu istediğin anda aslında server'ı istiyorsundur.

**Operasyonel sahiplik.** Embedded, uygulamayı deploy edenin sorunudur. Server ise saat 03:00'te
restart edilen, yedeklenen, upgrade edilen ve zaman zaman diskteki index formatını değiştiren bir
upgrade'le bozulan bir şeydir. Bunun bir sahibi var. Kimin olduğuna server'ı başlattıktan sonra
değil, önce karar ver.

**Açıklık.** Auth'suz `-p 8000:8000`, makinenin ağ arayüzünde açık bir veritabanı demektir.
Laptop'ta sorun değil, paylaşılan bir host'ta sorun. Ya localhost'a bağla ya önüne bir token koy.

## Ne çalıştırıyorsun

**Notebook.** `notebooks/08_chromadb.py` dosyasını VS Code'da aç ve blokları Shift+Enter ile
çalıştır.

- **Ne görmelisin:** `28 documents indexed`, ardından Chroma'nın default embedder'ından üç `WRONG`
  satırı, sonra aynı corpus'un kendi verdiğin `bge-m3` embedding'leriyle yeniden sorgulanması.
- **Ne kadar sürer:** yaklaşık 2 dakika, çoğu `bge-m3` embedding çağrısı.

**Server, opsiyonel.** Repo kökünden tek komut:

```bash
podman compose up -d                            # ya da: docker compose up -d
```

```bash
# macOS / Linux
curl -s http://localhost:8000/api/v2/heartbeat

# Windows (PowerShell)
Invoke-RestMethod http://localhost:8000/api/v2/heartbeat
```

- **Ne görmelisin:** `{"nanosecond heartbeat": ...}`. `/api/v1/` path'i `410 Gone` döner.
- **Ne kadar sürer:** image zaten çekilmişse birkaç saniye; pull'un kendisi 649 MB.

```bash
podman logs helios-chroma
podman compose down
```

Notebook'un iki yarısı arasında farklı olan tek Python:

```python
client = chromadb.EphemeralClient()                           # bellekte, process ile ölür
client = chromadb.PersistentClient(path="./chroma-local")     # embedded, diskte
client = chromadb.HttpClient(host="localhost", port=8000)     # server
```

## Sayılar ne dedi

<div class="measured">

| | ölçülen |
|---|---|
| iki satırlık versiyonun indexlediği doküman | 28 |
| Chroma'nın default embedder'ı, yirmi altın soru | `all-MiniLM-L6-v2` hit@1 0.350, `bge-m3` 0.800 |
| aynı default, Türkçe soru / İngilizce doküman | 0.000, `bge-m3` 0.667 |
| default modelin ilk `add()` indirmesi | 83,178,821 byte |
| ikinci bir process'te açılan `PersistentClient` | `count()` 0.09 s'de 28, hiçbir şey embed edilmeden |
| `docker.io/chromadb/chroma` image | 649 MB |
| başlangıçtan sonra ilk v2 heartbeat, image hazırken | yaklaşık 0.5 s |
| `/api/v1/heartbeat` | `410 Gone` — "The v1 API is deprecated. Please use /v2 apis" |

</div>

Yirmi soru, iki tasarım arasında seçim yapmaya yeter, bir sonuç yayımlamaya yakınından yetmez;
0.05'in altındaki bir fark bu örneklemin gürültüsünün içindedir. 0.350'ye karşı 0.800 farkı değil.

20 soruluk benchmark'ı Chroma üzerinden yeniden çalıştırmadık. 154 vektörde arama pratikte
tükenmeli olduğu için sıralamanın in-memory retriever ile aynı çıkması ve hit@1'in yine **0.800**
olması gerekir — ama "gerekir" bir ölçüm değildir. Bunu kesinleştirecek şey:
`eval/run_benchmark.py` dosyasını listeye değil collection'a bağlayıp iki hit@1 değerini doğrudan
karşılaştırmak.

## Daha derine

**Neden Docker değil Podman.** Üç sebep, hiçbiri ideolojik değil. Rootless: container senin
kullanıcınla çalışır, dolayısıyla bir escape root'a değil senin uid'ine düşer. Daemonless:
yetkili bir arka plan servisi yok, `podman run` sadece shell'inin sahibi olduğu bir process
ağacıdır ve ikinci bir supervisor'a ihtiyaç duymadan systemd unit'lerine oturur. Ve lisans
konuşması yok — Docker Desktop belirli bir şirket büyüklüğünün üstünde ücretli bir abonelik,
kurumsal ortamda bu satın alma demek, satın alma da eğitim gününün ertelenmesi demek. CLI aynı
flag'leri alıyor, yani bildiğin hiçbir şey boşa gitmiyor.

**Altta HNSW ne yapıyor.** Chroma'nın index'i bir Hierarchical Navigable Small World graph'ı. Her
vektör bir düğüm ve birkaç yakın komşusuna kenarları var; graph katmanlar hâlinde kuruluyor: en
üstte uzun mesafeli bağlantıları olan seyrek bir katman, altında gittikçe sıklaşanlar. Bir sorgu
en üstten giriyor, hangi komşu query'ye daha yakınsa açgözlü şekilde oraya yürüyor, daha iyisini
bulamadığında bir katman aşağı iniyor ve tekrarlıyor. On milyon düğüm yerine birkaç yüz düğüme
dokunuyorsun; arama collection büyüdükçe bu yüzden hızlı kalıyor.

**Neden "approximate".** Açgözlü yürüyüşün garantisi yok. Her komşunun daha kötü olduğu bir yerel
minimuma yerleşebilir, gerçek en yakın vektör ise graph'ın adım olarak hiç önermediği bir yerde
durabilir. Sub-linear zamanı, bazen en iyisi yerine ikinci en iyiyi almayı kabul ederek satın
alıyorsun. Ayar düğmesi, yürüyüş sırasında tutulan aday listesinin boyutu — liste büyüdükçe
graph'ın daha fazlası geziliyor, recall artıyor, latency de kabaca doğrusal artıyor. Build
tarafının kendi düğmeleri var: düğüm başına kenar sayısı ve builder'ın ekleme sırasında ne kadar
sıkı arayacağı; bunlar bellek ve index kurma süresini graph kalitesine karşı takas ediyor.
Chroma'da bunlar collection metadata'sı, oluştururken veriliyor ve sonradan değiştirmesi zahmetli.

Buradaki recall, aynı vektörler üzerinde exact brute-force aramaya göre recall demek, senin gold
set'ine göre değil. Bunlar farklı sorular ve insanlar sürekli birbirine karıştırıyor. Birincisini
bir öğleden sonrada ölçebilirsin: ikisini de çalıştır, top-k kümelerinin kaç kez uyuştuğunu say.
28 dokümanda ölçülecek bir şey yok — graph aramadan küçük — ve hissettiğin gecikme index değil,
embedding çağrısıdır.

**Vector database ne zaman yerini hak ediyor.** Burada değil. Yirmi sekiz vektör, cache'e sığan bir
matrise karşı bir numpy dot product'ı: exact, mikrosaniye, sıfır bağımlılık. Bu corpus üzerinde bir
ölçüm değil, bir pratik kural olarak: brute force yüz bin vektör civarına kadar meşru bir
production cevabı olarak kalıyor. Onun üstünde ilk soru şu: zaten Postgres çalıştırıyor musun.
Çalıştırıyorsan `pgvector` sana tek bir yedekleme hikâyesi, tek bir transaction sınırı ve vektör
aramasını yanındaki iş kolonlarına join etme imkânı veriyor — benzerlik aramasını yapan aynı
sorguda fare family ve effective quarter'a göre filtreleyebilirsin, iki sistemi senkron tutup
uyuşmalarını ummak yerine. Ayrı bir vektör deposu, vektör iş yükü kendi başına bir operasyon
problemine dönüştüğünde hak ediyor: on milyonlarca vektör, embedder değiştirdiğin için rutinleşen
yeniden indeksleme, tenant başına collection, sharding ya da index ile filtrenin birlikte
planlanmak zorunda olduğu ölçekte filtre artı vektör sorguları.

Açıkça ayarlanmaya değen bir detay: distance function. Bir collection bir space ayarı taşır — L2,
cosine, inner product — ve bu her zaman varsaydığın şey değildir. Vektörlerin birim uzunluktaysa
L2 ve cosine aynı sıralamayı üretir, fark etmez. Değilse farklı sıralamalar üretir ve bu farkı bir
saat kovalarsın. `hnsw:space`'i kendin ver, tahmin etmeyi bırak.

<div class="presenter-note">
Süre: toplam 10 dakika — kernel restart'a 2, üç WRONG satırına 4, senin seçtiğin embedder ile
<code>PersistentClient</code> tek satırına 2, server değişimi ve <code>/api/v2/</code>'ye 2. Günün
en kısa slotu olması bilinçli: salon yeni yemek yedi ve yükü bir tuzak ile iki client satırı.
<br /><br />
<strong>Geç kalırsan: bu modül kesme listesinde ikinci sırada, 10 dakikadan 5'e.</strong> Server
yarısını, metadata filtrelemeyi ve "Daha derine" bölümünü tamamen kes — kernel restart'ı ve üç
WRONG satırını tut, server değişimini üç client satırını yan yana gösterip tek cümlede söyle.
Buradaki sessiz default, kimsenin dokümantasyondan kendi başına çıkaramayacağı tek şey.
<br /><br />
Biri "peki production'da Chroma mı pgvector mı?" diye sorarsa ürün adıyla değil soruyla cevap ver:
kaç vektör, ve zaten Postgres çalıştırıyor musunuz. Onların corpus'unda ölçmediğin bir veritabanını
tavsiye etmeyi reddet.
</div>

## Çıkış cümlesi

> Aynı API, farklı deployment. Depolama çözüldü ama retrieval hâlâ 0.800'de — beş sorudan biri ilk
> sırada yanlış dokümanla dönüyor. Peki semantik arama yetmiyorsa?
