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
son bir saatte iyileştirdiğin 152 structure-aware chunk vektörü yok oldu. Onları geri getirmenin
tek yolu 152 chunk'ı yeniden embed etmek — her seferinde.

28 doküman ve 75 KB'de bu birkaç saniye ve hiçbir şey gibi hissettiriyor. İki şey bu hissi
bozuyor. Birincisi boyut: aynı döngü gerçek bir kural corpus'unda kahve molası uzunluğunda ve
bunu her deploy'da, her crash'te, her autoscale olayında yeniden ödüyorsun. İkincisi şekil.
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

## Embedded: tek process, tek dizin

Bunu düzelten en küçük şey bir server değil.

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma-local")
col = client.get_or_create_collection("iris_q3", metadata={"hnsw:space": "cosine"})
```

`PersistentClient` senin process'inin içinde çalışır. Daemon yok, port yok, başlatılacak bir şey
yok. `./chroma-local` dizinine bir SQLite dosyası ve index dosyaları yazar; o dizinin kendisi
veritabanıdır. Kopyalayabilirsin, release artefact'ına koyabilirsin, sıfırdan başlamak için
silebilirsin.

Chunk'ları bir kere yaz, üstelik zaten güvendiğin embedding'lerle:

```python
col.add(
    ids=chunk_ids,
    documents=texts,
    embeddings=embed(texts),                      # bge-m3, Ollama üzerinden, bizim çağrımız
    metadatas=[{"doc": p, "quarter": "2026-Q3"} for p in parents],
)
```

Sonra kernel'i restart et, aynı path'i aç, `col.count()` hiçbir şey embed etmeden **152** yazsın.
Bu deployment'ın kazancı tam olarak bu kadar — ve pek çok iç araç için ihtiyacın olan son
deployment budur.

`embeddings=` argümanına dikkatli bak, çünkü bu sayfada ölçülmüş bir sonuç taşıyan tek satır o.
Onu vermez, sadece `documents=` geçersen Chroma senin yerine bir embedding function seçer ve
seçtiği şey `all-MiniLM-L6-v2` olur: ilk `add()` çağrında sessizce indirilir, sadece İngilizce
bilir ve on dokümanlık probe corpus'ta `bge-m3`'ün **4/5**'ine karşı **2/5** aldı. Modül 6 tam da
bu hataydı. Veritabanına taşınmak onu ne düzeltir ne de sana haber verir. Ya kendi vektörünü ver
ya kendi embedding function'ını ver; kararı asla default'a bırakma.

## Server: aynı collection, farklı deployment

Şimdi aynı şeyi bir portun arkasına koy.

```bash
podman pull docker.io/chromadb/chroma      # 649 MB, ölçüldü, makine başına bir kez
podman run -d -p 8000:8000 --name chroma docker.io/chromadb/chroma
curl -s http://localhost:8000/api/v2/heartbeat
```

Pull 649 MB ve server `run`'dan yaklaşık bir saniye sonra cevap veriyor. Registry prefix'ine
dikkat: Podman, Docker'ın yaptığı gibi Docker Hub'ı varsaymaz, dolayısıyla `podman pull
chromadb/chroma` durup sana hangi registry'yi kastettiğini sorar. `docker.io/` yaz ve devam et.

**API path'i `/api/v2/`, `/api/v1/` değil.** Bunu iki kez yüksek sesle söyle. İnternetteki
ChromaDB cevaplarının çoğu v1'e göre yazıldı, her aramada ilk sonuç onlar ve güncel bir server'a
karşı 404 dönüyorlar. Heartbeat'te alınan bir 404, tam olarak "container ayağa kalkmadı" gibi
görünür; insanlar tek sorun bir URL'deki versiyon numarasıyken yirmi dakika Podman debug eder.

Client değişimi tek satır:

```python
client = chromadb.HttpClient(host="localhost", port=8000)
col = client.get_or_create_collection("iris_q3", metadata={"hnsw:space": "cosine"})
res = col.query(query_embeddings=embed([q]), n_results=5, where={"quarter": "2026-Q3"})
```

O ilk satırın altındaki her şey embedded versiyonla harfi harfine aynı. Aynı `add`, aynı `query`,
aynı `where` filtresi, aynı sonuç şekli. Bu bir tesadüf değil, Chroma'yı öğretmeye değer kılan
ürün kararı: bir dizine karşı geliştirip bir porta karşı ship ediyorsun, retrieval kodunu yeniden
yazmadan.

<div class="presenter-note">
Yirmi laptop'a aynı anda ofis ağı üzerinden 649 MB çektirme. Image kurulum talimatlarında
önceden indirilmiş durumda; <code>podman images</code> ile orada olduğunu göster, progress bar
izlemek isteyen olursa pull'u sadece kendi makinende yap. Container kalkmazsa: 8000 doluysa
<code>-p 8001:8000</code>; macOS'te önce <code>podman machine start</code>. Yanlış söylenmemesi
gereken cümle: <strong>path /api/v2/ — internette bulacağın eski cevapların hepsi v1 diyor ve 404
dönüyor.</strong>
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

Notebook: `06_chromadb.ipynb`. Bunu izliyorsun; komutlar handout'ta.

```bash
pip install chromadb
jupyter lab notebooks/06_chromadb.ipynb        # embedded: PersistentClient, restart, count() == 152

podman pull docker.io/chromadb/chroma          # 649 MB
podman run -d -p 8000:8000 --name chroma docker.io/chromadb/chroma
curl -s http://localhost:8000/api/v2/heartbeat # /api/v1/ değil — v1 404 döner
podman logs chroma
podman stop chroma && podman rm chroma
```

Notebook'un iki yarısı arasında farklı olan tek Python:

```python
client = chromadb.PersistentClient(path="./chroma-local")     # embedded
client = chromadb.HttpClient(host="localhost", port=8000)     # server
```

## Sayılar ne dedi

<div class="measured">

| | ölçülen |
|---|---|
| `podman pull docker.io/chromadb/chroma` | 649 MB |
| `podman run` sonrası server'ın cevap vermesi | yaklaşık 1 saniye |
| çalışan API path'i | `/api/v2/` — `/api/v1/` 404 döner |
| collection'a yazılan chunk sayısı | 152 (structure-aware, 28 doküman, 75 KB) |
| ChromaDB'nin default embedder'ı, probe corpus | `all-MiniLM-L6-v2` 2/5, `bge-m3` 4/5 |

</div>

20 soruluk benchmark'ı Chroma üzerinden yeniden çalıştırmadık. 152 vektörde arama pratikte
tükenmeli olduğu için sıralamanın in-memory retriever ile aynı çıkması ve hit@1'in yine **0.800**
olması gerekir — ama "gerekir" bir ölçüm değildir. Bunu kesinleştirecek şey: `eval/run_benchmark.py` dosyasını listeye değil collection'a bağlayıp iki hit@1 değerini doğrudan karşılaştırmak.

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
sıkı arayacağı; bunlar bellek ve index kurma süresini, sen daha ilk sorguyu atmadan graph'ın ne
kadar iyi olacağına karşı takas ediyor. Chroma'da bunlar collection metadata'sı, oluştururken
veriliyor ve sonradan değiştirmesi zahmetli; o yüzden bilerek ver.

Buradaki recall, aynı vektörler üzerinde exact brute-force aramaya göre recall demek, senin gold
set'ine göre değil. Bunlar farklı sorular ve insanlar sürekli birbirine karıştırıyor. Birincisini
bir öğleden sonrada ölçebilirsin: ikisini de çalıştır, top-k kümelerinin kaç kez uyuştuğunu say.
152 vektörde ölçülecek bir şey yok — graph aramadan küçük — ve hissettiğin gecikme index değil,
embedding çağrısıdır.

**Vector database ne zaman yerini hak ediyor.** Burada değil. 152 vektör, L2 cache'e sığan bir
matrise karşı bir numpy dot product'ı: exact, mikrosaniye, sıfır bağımlılık. Brute force yüz bin
vektör civarına kadar meşru bir production cevabı olarak kalıyor. Onun üstünde ilk soru şu: zaten
Postgres çalıştırıyor musun. Çalıştırıyorsan `pgvector` sana tek bir yedekleme hikâyesi, tek bir
transaction sınırı ve vektör aramasını yanındaki iş kolonlarına join etme imkânı veriyor —
benzerlik aramasını yapan aynı sorguda fare family ve effective quarter'a göre filtreleyebilirsin,
iki sistemi senkron tutup uyuşmalarını ummak yerine. Ayrı bir vektör deposu, vektör iş yükü kendi
başına bir operasyon problemine dönüştüğünde hak ediyor: on milyonlarca vektör, embedder
değiştirdiğin için rutinleşen yeniden indeksleme, tenant başına collection, sharding ya da ANN
index'i ile filtrenin birlikte planlanmak zorunda olduğu ölçekte filtre artı vektör sorguları.

Açıkça ayarlanmaya değen bir detay: distance function. Bir collection bir space ayarı taşır — L2,
cosine, inner product — ve bu her zaman varsaydığın şey değildir. Vektörlerin birim uzunluktaysa
L2 ve cosine aynı sıralamayı üretir, fark etmez. Değilse farklı sıralamalar üretir ve bu farkı bir
saat kovalarsın. `hnsw:space`'i kendin ver, tahmin etmeyi bırak.

<div class="presenter-note">
Süre: 20 dakika, 30'a taşmasına izin verme. Bu slot salona chunking ölçümleriyle daha zor
retrieval işi arasında nefes aldırmak için var; yükü iki client satırı ve bir URL path'i. Biri
"peki production'da Chroma mı pgvector mı?" diye sorarsa ürün adıyla değil soruyla cevap ver: kaç
vektör, ve zaten Postgres çalıştırıyor musunuz. Onların corpus'unda ölçmediğin bir veritabanını
tavsiye etmeyi reddet.
</div>

## Çıkış cümlesi

> Aynı API, farklı deployment.
