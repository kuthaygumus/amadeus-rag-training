---
title: "6. Container'da ChromaDB"
description: "294 chunk gerçekte nerede yaşıyor, container'ın ayakta olduğu nasıl kanıtlanır ve parçalar içeride nasıl görünüyor."
---

## Gate sorusu

> **O 294 chunk nereye gitti — ve onlara bakabilir miyiz?**

[Modül 5](/tr/modules/05-simple-rag/) grounded bir cevap ve bir `sources` dizisiyle bitti.
`/ingest` ile `/query` arasında bir yerde pipeline 294 vektörü *bir yere* koydu ve `/retrieve`
onlardan üçünü yeniden buldu. O yer ikinci container. Bu modül kısa ve baştan sona hands-on: orada
olduğunu kanıtla, onunla doğrudan konuş, tuttuğunu oku.

## Liste, veritabanı değildir

Bir vector store'un akla gelen ilk implementasyonu api process'inin içinde bir dizidir: corpus'u
startup'ta embed et, `{id, vector, text}[]` dizisini bellekte tut, her request'te üzerinde cosine
similarity ile dön. Çalışır; 294 vektörde hızlıdır da. Hızla ilgisi olmayan dört cephede çöker.
**Persistence:** process'i restart et, vektörler gitti; corpus'u her deploy'da, her crash'te, her
autoscale olayında yeniden embed ediyorsun — burada saniyeler, gerçek bir kural kitabında kahve
molası. **Request'ten uzun yaşayan bir process:** Angular front end bir endpoint'i çağırıyor,
endpoint üç replica'ya ölçekleniyor ve artık corpus'u ayrı ayrı embed etmiş, birbirinden sapabilen
üç liste var. **Bir API:** listeyi başka hiçbir şey okuyamıyor — ikinci bir servis yok, debug aracı
yok, `curl` yok. **Yapı:** listenin collection'ı da filtresi de yok; "sadece Q3 edition'da ara" ya
da "sadece ücret sayfalarında" aramadan *sonra* bir döngü olmak zorunda, ve "sadece güncel SOP"
cümlesini bir cosine benzerliğinde ifade edemezsin.

Vector database, aynı nearest-neighbour aramasına o dört şeyin eklenmesidir: vektörler diskte, bir
portun arkasında, isimli collection'larda, aramadan *önce* filtreleyebildiğin metadata ile. Chroma
bu dördünü de yapan en küçük şey; compose dosyasında olmasının sebebi bu. Embedder değil — her
vektörü `api`, senin Ollama'n üzerinden `bge-m3` ile hesaplıyor ve Chroma'ya bitmiş sayılar veriyor.
Chroma depolama ve arama, başka bir şey değil; bu sayfanın geri kalanı ona üç yandan bakmak.

## Yüzey 1 — Podman Desktop

**Podman Desktop** — Containers görünümünü aç.

İki container, ikisi de çalışıyor: biri `docker.io/chromadb/chroma:1.5.9` image'ından, `8000`'i
dinliyor; diğeri repo'dan build edilmiş, `3000`'i dinliyor. **Ollama bunlardan biri değil.** Senin
makinende çalışıyor, çünkü GPU da [kurulumda](/tr/modules/00-setup/) çektiğin 4,5 GB model de
orada; api ona `host.containers.internal:11434` üzerinden ulaşıyor. Üç kutu bekliyorsan zihindeki
resmi düzeltmenin anı bu: container'lar veritabanı ile pipeline, model runtime'ı ise host.

Şimdi sol menüden Volumes görünümünü aç: tek bir volume, adı `chroma-data` (başında proje adı
prefix olarak), chroma container'ının içine `/data` olarak mount edilmiş. 294 vektörün yaşadığı yer
o named volume. Container'ı sil, volume kalır; onu kaldıran tek compose komutu
`podman compose down -v`.

<div class="presenter-note">
Windows laptop'ların <code>host.containers.internal</code> problemi varsa yüzeye çıkacağı modül bu.
Başlamadan önce el kaldırt: bu sabah <code>00-health</code>'te <code>"ollama": "unreachable at
http://host.containers.internal:11434 — start Ollama on your machine"</code> görüp kendi
terminalinde <code>ollama list</code> çalışan kim var? Bu, Ollama'nın sadece 127.0.0.1'e bind
olması — <code>OLLAMA_HOST=0.0.0.0</code> set ettir, Ollama'yı restart ettir, <code>00-health</code>'i
yeniden çalıştırt. Gerçek filoda <code>UNVERIFIED</code>; günün hâlâ bir numaralı riski ve ısırıp
ısırmadığını bu modülün ilk iki dakikasında öğreneceksin. Ondan sonra hâlâ kırmızı kalan, Bruno
kısmı için yanındakiyle eşleşsin — onların Chroma'sı sağlam, Ollama'ya sadece embed adımı ihtiyaç
duyuyor ve o adım modül 5'te zaten koştu.
</div>

## Yüzey 2 — Terminal

**Terminal (`amadeus-rag-lab` repo kökü):**

```bash
podman ps
curl localhost:8000/api/v2/heartbeat
podman compose logs chroma
```

`podman ps` GUI'deki aynı iki satır — chroma satırında `(healthy)` yazdığına bak, çünkü o kelime bir
dakika sonra yük taşıyacak.

`curl`, bu modülden hatırlanacak tek satır. **Chroma'nın kendisiyle konuşuyor, arada api yok:**
veritabanının kendi HTTP API'si, compose dosyasının dışa açtığı portta. Sağlıklı bir server, tek
key'i `nanosecond heartbeat` olan küçük bir JSON nesnesiyle cevap verir. **Path `/api/v2/`,
`/api/v1/` değil** — Chroma 1.x v1 path'lerini kaldırdı ve internetteki ChromaDB cevaplarının çoğu
v1'e göre yazıldı. Bu server'a karşı `/api/v1/heartbeat`, gövdesinde v2 API'lerini kullan diyen bir
`410 Gone` döndürür — ve tek başına o status satırı, tam olarak ayağa kalkmamış bir container gibi
görünür. Yirmi dakika Podman debug etmeden önce gövdeyi oku.

`podman compose logs chroma`, `/health` `chroma: unreachable` dediğinde gideceğin yer. `(healthy)`
bayrağının ne demek olduğunu da orada görürsün: compose dosyasındaki healthcheck, container'ın
içinde her beş saniyede `127.0.0.1:8000`'e bir TCP socket açıyor, az önce yazdığın heartbeat
request'inin aynısını gönderiyor ve cevapta `nanosecond` arıyor. `api` servisi
`depends_on: chroma: condition: service_healthy` deklare ediyor — **api'nin beklemesinin sebebi o
satır**; yoksa ilk boot'ta başlar, bağlanamaz, döngü içinde crash eder.

## Yüzey 3 — Bruno

**Bruno — `02-ingest` › `peek-chunks`:**

```text
GET {{baseUrl}}/chunks?source=fare_classic_shorthaul&limit=20
```

Body yok. `source` bir doküman id'si — `.md` uzantısı olmadan dosya adı — ve api bunu bir metadata
filtresine çeviriyor: güncel collection'a karşı `where: { source: "fare_classic_shorthaul" }`, en
fazla 20 satır için `documents` ve `metadatas` döndürüyor. Cevabı yukarıdan aşağı oku:

- `collection` — `kraken-2026-Q3-fixed-280`. **Konfigürasyon başına bir collection.** İsim
  `kraken-<edition>-<strategy>-<chunkSize>`; overlap istediysen `-ov<n>`, boilerplate'i
  soyduysan `-strip` ekleniyor. Bugün çalıştıracağın her `/ingest` kendi collection'ına düşüyor;
  hiçbir şeyin üstüne yazılmıyor.
- `total` — collection'ın tamamı, bu edition ve strategy'de 294. `returned` — filtreye kaç tanesi
  uydu: 15, çünkü `fare_classic_shorthaul.md` tam olarak o kadar 280 karakterlik parçaya bölünüyor.
- `chunks[]` — her biri `id` (`fare_classic_shorthaul#<n>`), `source`, `chars` ve `text` ile. Bunlar
  `fare_classic_shorthaul.md`'nin Chroma'da şu an durduğu haliyle parçaları: her biri 280 karakter,
  280 nereye düştüyse oradan kesilmiş.

Şimdi request'in docs sekmesindeki egzersizi yap. **`text`'i `| K |` ile başlayan satırı içeren
chunk'ı bul.** Sonra tablo başlığını tutan chunk'ı bul — hangi sütunun "Cancellation penalty"
olduğunu söyleyeni. **Farklı chunk'lardalar** — arada iki chunk, O ve T satırları var. K satırı
`EUR 70 | EUR 90 | EUR 180` taşıyor ve bu üçünden hangisinin iptal cezası olduğunu söyleyen hiçbir
şey yok. 12 Eylül çalıştırmasında fixed-280 RAG'in yanlış sütunu okuyup "EUR 13" demesi tam olarak
böyle oldu — senin sayın farklı olacak, chunk sınırların değil, çünkü kesim deterministik.

<div class="presenter-note">
K satırı avını anlatma — salona iki chunk'ı kendisinin bulması için doksan saniye sessizlik ver,
sonra birinden K satırı chunk'ını baştan sona yüksek sesle okumasını iste. O metinden EUR 90'ın ne
olduğunu kimse söyleyemez; modül 7'ye açılan gate'in tamamı bu. Bazılarında Ollama kapalıysa bu
request yine çalışır: <code>/chunks</code> Ollama'ya hiç dokunmaz, sadece Chroma'ya; collection da
modül 5'te kuruldu. Biri burada <code>409</code> alıyorsa modül 5'ten beri api'yi restart etmiştir —
onun yerine düzeltme, bir sonraki bölüme yönlendir; 409 dersin kendisi.
</div>

## Restart'tan ne sağ çıkar, ne bilerek unutur

Oyunda iki farklı hafıza var ve bilerek aynı değiller.

**Chroma her şeyi tutar.** Named volume `podman compose down` ve `up`'tan uzun yaşar; bugün ingest
ettiğin her collection yarın da orada ve `GET /health` hepsini `collections` altında listeler.
Onları kaybetmenin tek iki yolu `podman compose down -v` ya da **`DELETE /collections`** — ikincisi
her collection'ı adıyla düşürüp `{ deleted: [...] }` döndürür. Temiz bir oda istediğinde ikincisini
kullan — modül 7 ve 8 birkaç collection daha ekleyecek, sıfırdan başlamak isteyebilirsin.

**Api güncel collection'ını unutur.** `/retrieve`, `/query` ve `/chunks` varsayılan olarak *en son
ingest edilen* collection'a gider ve o isim api process'inin belleğinde tutulur — bilerek; bir
dosyada değil, Chroma'da değil. Api'yi restart et, bir sonraki `/retrieve` **`409`** ve `Nothing has
been ingested yet. Call POST /ingest first (Bruno folder 02).` döndürür. Veri hâlâ volume'da; ona
işaret eden pointer gitti. İki yarıyı tek hamlede kanıtla:

**Terminal (`amadeus-rag-lab` repo kökü):**

```bash
podman compose restart api
```

Sonra **Bruno — `02-ingest` › `peek-chunks`**'ı yeniden çalıştır: `409`. Şimdi URL'e
`&collection=kraken-2026-Q3-fixed-280` ekle ve tekrar gönder: `200`, aynı chunk'lar. Collection
hiç gitmemişti; giden, api'nin hangisinin *güncel* olduğuna dair fikriydi. Tek bir `/ingest` çağrısı
pointer'ı geri getirir — 409 mesajının özür dilemek yerine Bruno klasör 02'yi göstermesi bu yüzden.

## Ne çalıştırıyorsun

- **Podman Desktop** — Containers görünümü: iki çalışan, `chroma` healthy işaretli, Ollama yok.
  Volumes: `chroma-data`.
- **Terminal (`amadeus-rag-lab` repo kökü)** — `podman ps`, `curl localhost:8000/api/v2/heartbeat`,
  `podman compose logs chroma`. İsteğe bağlı: 409 için `podman compose restart api`.
- **Bruno — `02-ingest` › `peek-chunks`** — `GET /chunks?source=fare_classic_shorthaul&limit=20`.
  `collection`, `total`, `returned`'ı oku; sonra `chunks[].text` içinde `| K |` ile başlığı ara.
- **Ne kadar sürüyor:** yirmi dakika, hiçbiri model beklemek değil. Bu modüldeki tek yavaş çağrı
  restart sonrası yeniden ingest ve o da isteğe bağlı.
- **`curl` connection refused diyorsa:** container ayakta değil ya da 8000 dolu —
  `podman compose logs chroma`; macOS'te `podman machine`'in başlatıldığına bak. `410 Gone`
  dönüyorsa `v1` yazmışsın.

## Daha derine

**Chroma'ya ne söyleniyor, ne söylenmiyor.** Api her collection'ı `embeddingFunction: null` ve
`hnsw: { space: "cosine" }` ile açıyor. Birincisi Chroma'nın hiçbir şeyi embed etmemesi demek — ona
vektörsüz metin ver, düşeceği bir yer yok; embedder'ın veritabanının senin yerine verdiği değil
senin verdiğin bir karar (`bge-m3`, 1024 boyut) olduğunda istediğin davranış bu. İkincisi mesafeyi
adlandırıyor: `/retrieve`'de okuduğun `score` o uzayda cosine similarity ve `0.35` eşiği sadece
uzay collection başına sabit olduğu için bir anlam taşıyor. Collection'ın *konfigürasyon* başına
olmasının sebebi de bu: farklı bir embedder'dan ya da farklı bir kesimden gelen vektörler aynı
index'e ait değil.

**Index'in bedava olmaktan çıktığı yer.** 294 vektörde arama fiilen exhaustive; hissettiğin gecikme
embedding çağrısı, Chroma değil. Altta Chroma bir Hierarchical Navigable Small World grafı kuruyor —
her vektör bir avuç yakın komşusuna bağlı, sorgu her şeyi taramak yerine katmanlar boyunca açgözlü
yürüyor — collection büyüdükçe aramayı hızlı tutan da bu, onu *approximate* yapan da: yürüyüş ikinci
en iyi cevaba razı olabilir. Buradaki hiçbir şey bunu ölçecek kadar büyük değil. Bu corpus'ta bir
ölçüm değil, bir kaide olarak: brute force, yüz bin vektör civarına kadar meşru bir production
cevabı olmayı sürdürür; üstünde ilk soru hangi vector database olduğu değil, zaten Postgres
çalıştırıyor musun olur (`pgvector`: tek backup hikâyesi, tek transaction sınırı, iş sütunlarına
join).

<div class="presenter-note">
Zamanlama: yirmi dakika, on ikisi burada yazılı, kalanı salonun kendi tıklaması — iki dakika liste-veritabanı tartışması, üç dakika Podman Desktop ve
Windows kontrolü, iki dakika terminal (iki kez söylenecek tek satır <code>v2</code>), dört dakika
sessizlik dahil peek-chunks, bir dakika restart. Canlı request'lerin toplamı bir dakikanın altında.
<br /><br />
<strong>Geç kalıyorsan:</strong> restart bölümünü ve Daha derine'yi at; heartbeat ile K satırı avını
tut — K satırı modül 7'nin gate'i, atlanamaz; bu sayfadaki geri kalan her şey <code>podman ps</code>
projektördeyken tek cümlede söylenebilir.
<br /><br />
Biri "production'da Chroma mı pgvector mı?" diye sorarsa ürünle değil iki soruyla cevap ver: kaç
vektör, ve zaten Postgres çalıştırıyor musunuz. Onların corpus'unda ölçmediğin bir veritabanını
önermeyi reddet.
</div>

## Çıkış cümlesi

> Chunk'ları artık görebiliyoruz. `| K |` satırını tutan chunk'ta üç EUR tutarı var, başlık yok —
> veritabanı pipeline'ın verdiğini aynen sakladı ve pipeline ona sütun adları olmayan bir satır
> verdi. [Modül 7](/tr/modules/07-chunking-and-noise/) corpus'u farklı kesiyor.
