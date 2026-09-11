---
title: "8. ChromaDB: Embedded ve Server"
description: "Bunu production'da nasıl çalıştıracağım?"
---

## Gate sorusu

> **Bunu production'da nasıl çalıştıracağım?**

**VS Code — modül 7'yi çalıştırdığın pencereye dön:** o blokların içinde koştuğu Python
process'ini öldür. Panelindeki çöp kutusu ikonu bunu yapıyor. Şimdi retrieval bloğunu yeniden
çalıştır.

`NameError`. `DenseRetriever` gitti; son bir saatte iyileştirdiğin 154 structure-aware chunk
vektörü de gitti.

Hata bu kadar basit ve iki saniyede üretiliyor. Bugün kurduğun her retriever, vektörleri tek bir
process'in içindeki Python listesinde tuttu. `DenseRetriever(doc_ids, documents)` corpus'u
constructor'ında embed ediyor. Process ölünce corpus hâlâ diskte, model hâlâ diskte, vektörler
değil. Onları geri getirmenin tek yolu 154 chunk'ı yeniden embed etmek — her seferinde.

28 doküman ve 78 310 karakterde bu birkaç saniye ve hiçbir şey gibi hissettiriyor. İki şey bu
hissi bozuyor. Birincisi boyut: aynı döngü gerçek bir kural corpus'unda kahve molası uzunluğunda
ve bunu her deploy'da, her crash'te, her autoscale olayında yeniden ödüyorsun. İkincisi şekil.
Aslında yazacağın şey bir servis — bir Angular front end, bir endpoint'i çağıracak, o endpoint de
retriever'ı. Liste bir Python process'inde duruyor. Request başka bir process'e geliyor.

Yani soru "hangi vector database daha iyi" değil. Soru şu: vektörleri üreten process gittiğinde
vektörler nerede yaşıyor.

<div class="presenter-note">
Bunu slayttan önce yap. Salonun önünde modül 7'nin process'ini öldür, retrieval bloğunu yeniden
çalıştır, <code>NameError</code> ile patlamasını izlet. Sonra sor: "bu rebuild benim corpus'umda
değil, sizinkinde kaç saniye sürer?" İki üç kişi sesli olarak bir sayı söylesin. Bu modül öğle
yemeğinin hemen arkasında, salonun enerjisi günün en düşük noktasında; 20 saniyelik canlı bir hata,
herhangi bir diyagramdan daha çok dikkat satın alır. 2 dakika.
</div>

## Embedded mod: iki satırlık versiyon

Bu sayfadaki her Python bloğu `notebooks/08_chromadb.py` dosyasındaki bir hücredir. Dosyayı şimdi
VS Code'da aç ve okurken çalıştır. Aşağıdaki, hemen herkesin ilk yazdığı şey ve anında çalışıyor.

**VS Code — `notebooks/08_chromadb.py`, import'lardan sonraki blok:**

```python
client = chromadb.EphemeralClient()          # PersistentClient(path=...) to keep it on disk
collection = client.get_or_create_collection("kraken")
collection.add(ids=ids, documents=texts)             # note: no embeddings= argument
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

**VS Code — bir sonraki blok:**

```python
DEFAULT_QUERIES = [
    ("CLASSIC K sinifi iptal cezasi ne kadar?", "fare_classic_shorthaul"),
    ("Baglantisini kaciran yolcuya yemek fisi ne kadar?", "sop_misconnect_v4"),
    ("what is the cancellation penalty for CLASSIC class K", "fare_classic_shorthaul"),
]
```

```text
embedding function: DefaultEmbeddingFunction
dimensions        : 384 (read off a stored vector)
loaded from       : ~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx
WRONG  CLASSIC K sinifi iptal cezasi ne kadar?              -> macro_tr_noshow
WRONG  Baglantisini kaciran yolcuya yemek fisi ne kadar?    -> macro_tr_noshow
right  what is the cancellation penalty for CLASSIC class   -> fare_classic_shorthaul
```

İki Türkçe soru, üst üste aynı yanlış doküman — bir çağrı merkezi macro'su, ücret sayfası değil.
İngilizce soru doğru ve bu, default embedder'ın iyi iş çıkarması değil. Hatanın *özgül* olması.
`all-MiniLM-L6-v2` sadece İngilizce biliyor; dolayısıyla İngilizce dokümana sorulan İngilizce
soruda gözle görülür bir kaybı yok: modül 6 orada üç embedder'ı da **0.500** ölçüyor. Çöktüğü yer
diller arası taraf — Türkçe soru / İngilizce doküman olan altı soruda **0.000**, `bge-m3` ise
**0.667**; genelde de hit@1 **0.300**, `bge-m3` **0.700**. Exception yok, uyarı yok, çıktıda bir
şeyin ters gittiğini ima eden hiçbir şey yok.

Chroma o modeli senin yerine seçti ve sana hiçbir şey söylemedi. Yukarıdaki satırın modeli
adlandırabilmesinin tek sebebi, notebook'un genişliği kayıtlı bir vektörden okuyup modelin
yüklendiği cache dizinini yazması — Chroma model adını embedding function üzerinde açmıyor. 384
boyutlu ve ilk `add()` çağrında indiriliyor: internet üzerinden 83 178 821 byte, makine başına bir
kez, hiç sormadan. Veritabanına taşınmak bunu ne düzeltir ne de sana haber verir. Sistem patlamıyor.
Sessizce, kimse kontrol etmediği sürece, yanlış cevap veriyor.

<div class="presenter-note">
Modülün tamamı bu. İki satırlık versiyonu canlı çalıştır, iki WRONG satırının oturmasını bekle ve
açıklamadan önce sus. Biri "ama hata vermedi ki" diyecek — istediğin cümle tam olarak bu, geri
tekrarla. Biri İngilizce sorunun geçtiğine itiraz ederse, asıl nokta o: salon "Türkçe" diyene kadar
soruyu geri ver — "peki tam olarak ne bozuldu?" İlk <code>add()</code> takılırsa o makinede default
model seed edilmemiştir; notebook kayıtlı çalıştırmayı oynatır ve bunu yaptığını yazar, devam et.
4 dakika.
</div>

## Aynı collection, senin seçtiğin embedder ile

Chroma kendi hesapladığın vektörleri kabul eder. Tek bir ek argüman:

**VS Code — ondan sonraki blok:**

```python
better = client.get_or_create_collection("kraken_bge_m3")
better.add(ids=ids, documents=texts, embeddings=R.embed(texts))   # bge-m3, via Ollama, our call
```

```text
indexed with bge-m3 in 5.9s

right  CLASSIC K sinifi iptal cezasi ne kadar?              -> fare_classic_shorthaul
right  what is the cancellation penalty for CLASSIC class   -> fare_classic_shorthaul
```

Geri kalan her şey aynı — aynı client, aynı `query`, aynı sonuç şekli. Çağrı merkezi macro'suna
düşen Türkçe soru artık doğru ücret sayfasını döndürüyor. Daha iyi, ama doküman seviyesinde hâlâ
kusursuz değil; zaten modül 6 ve 7'nin bütün mesaisi orada. Buradaki nokta daha dar: **embedding
modeli bir karardır ve sen vermezsen Chroma senin yerine verir.**

## Restart'tan sağ çıkmak

Gate kaybolan bir index'ti ve onu kapatan tek bir kelime var:

```python
client = chromadb.PersistentClient(path="./chroma-local")     # instead of EphemeralClient()
```

`PersistentClient` yine senin process'inin içinde çalışır — daemon yok, port yok, başlatılacak bir
şey yok. `./chroma-local` dizinine bir SQLite dosyası ve index dosyaları yazar; o dizinin kendisi
veritabanıdır. Kopyalayabilirsin, release artefact'ına koyabilirsin, sıfırdan başlamak için
silebilirsin. Notebook bunu `EphemeralClient` satırındaki yorum olarak tutuyor ki bir günlük demo
geride bir şey bırakmasın; o tek satırı değiştir, dizin ortaya çıksın. Pek çok iç araç için vermen
gereken son deployment kararı budur.

Dizinin sana verdiği ikinci şey filtreleme. `add()` sırasında `metadatas=` ile dokümanları
etiketle, `query()` üzerinde `where={"kind": "fare"}` ver, sadece ücret sayfalarında arasın.
Aramadan sonra değil aramadan *önce* filtrelemek, bir veritabanı ile bir döngü arasındaki farktır;
"sadece güncel SOP" cümlesini bir cosine benzerliğinde ifade edemezsin.

## Server, bir container runtime'ın varsa

Aynı collection bir portun arkasında ve bunu yapma sebebi hız değil. İki process'in aynı embedded
dizini açması cluster değildir, corruption yoludur; ikinci bir okuyucu istediğin anda aslında
server'ı istiyorsundur. Compose dosyası repo kökünde.

**Terminal (repo kökü):**

```bash
podman compose up -d          # or: docker compose up -d
curl http://localhost:8000/api/v2/heartbeat
```

**API path'i `/api/v2/`, `/api/v1/` değil.** İnternetteki ChromaDB cevaplarının çoğu v1'e göre
yazıldı; güncel bir server'a karşı v1 path'i **`410 Gone`** döndürüyor — ve tek başına o status
satırı, tam olarak ayağa kalkmamış bir container gibi görünür; yirmi dakika Podman debug etmeden
önce gövdeyi oku.

Sonrası tek satır — `EphemeralClient()` yerine `chromadb.HttpClient(host="localhost", port=8000)` —
ve altındaki her `add`, `query` ve `where` harfi harfine aynı kalıyor. Bir dizine karşı geliştirip
bir porta karşı ship ediyorsun, retrieval kodunu yeniden yazmadan. Container runtime'ın yoksa bugün
hiçbir şey kaybetmiyorsun: notebook'un server hücresi try/except içinde, `no server running` yazıp
server'ı başlatma komutunu veriyor ve embedded devam ediyor.

<div class="presenter-note">
Image pull'u salonda değil evde yaptır — <code>00-setup.md</code> bunu söylüyor ve bu eğitimde
ölçülen her şeyin Podman olmadan ölçüldüğünü açıkça yazıyor. <code>podman images</code> ile image'ın
zaten orada olduğunu göster. Container kalkmazsa: 8000 doluysa <code>-p 8001:8000</code>; macOS'te
önce <code>podman machine start</code>. Salonda hiç kimsede runtime yoksa bu bölümü okuyup geç —
hücrenin devam etmesi dokümante edilmiş davranış, bozuk bir demo değil. Ağzında gevelenmemesi
gereken cümle: <strong>path /api/v2/ — internette bulacağın eski cevapların hepsi v1 diyor ve 410
Gone alıyor.</strong> 2 dakika.
</div>

## Ne çalıştırıyorsun

**VS Code.** `notebooks/08_chromadb.py` dosyasını aç ve blokları Shift+Enter ile çalıştır.

- **Ne görmen gerekiyor:** `28 documents indexed`, ardından Chroma'nın default embedder'ından iki
  `WRONG` ve bir `right` satırı, sonra aynı corpus'un kendi verdiğin `bge-m3` embedding'leriyle
  yeniden sorgulanması. Çalışan bir container yoksa son blok `no server running` yazıp orada durur.
- **Ne kadar sürüyor:** yaklaşık 2 dakika, çoğu `bge-m3` embedding çağrısı.
- **İlk `add()` takılırsa:** Chroma tam o anda default modelini indiriyor, 83 MB. Hücreyi durdur.
  Ona ihtiyacı olan iki hücre kayıtlı çalıştırmayı oynatıp `[CACHED]` yazıyor; devam et ve modeli
  evde `python scripts/seed_offline_assets.py` ile seed et.

Üç deployment arasında farklı olan tek Python:

```python
client = chromadb.EphemeralClient()                           # in memory, dies with the process
client = chromadb.PersistentClient(path="./chroma-local")     # embedded, on disk
client = chromadb.HttpClient(host="localhost", port=8000)     # server
```

## Sayılar ne dedi

<div class="measured">

| | ölçülen |
|---|---|
| iki satırlık versiyonun index'lediği doküman | 28 |
| Chroma'nın default embedder'ı, yirmi gold soru | `all-MiniLM-L6-v2` hit@1 0.300, `bge-m3` 0.700 |
| aynı default, Türkçe soru / İngilizce doküman | 0.000, `bge-m3` 0.667 |
| aynı default, İngilizce soru / İngilizce doküman | 0.500 — diğer her embedder ile aynı |
| default model, kayıtlı bir vektörden okunmuş | `DefaultEmbeddingFunction`, 384 boyut |
| default modelin ilk `add()` indirmesi | 83 178 821 byte |
| aynı 28 dokümanın `bge-m3` ile yeniden index'lenmesi | 5.9 s |
| `/api/v1/heartbeat` | `410 Gone` — "The v1 API is deprecated. Please use /v2 apis" |

</div>

Yirmi soru, iki tasarım arasında seçim yapmaya yeter, bir sonuç yayımlamaya yaklaşamaz bile;
0.05'in altındaki bir fark yirmi sorunun gürültüsünün içindedir. 0.300'e karşı 0.700 farkı değil.

20 soruluk benchmark'ı Chroma üzerinden yeniden çalıştırmadık. 154 vektörde arama fiilen tam tarama
olduğu için sıralamanın in-memory retriever ile aynı çıkması ve hit@1'in yine 0.700 olması gerekir
— ama "gerekir" bir ölçüm değildir.
`UNVERIFIED: bir Chroma collection'ı üzerinden hit@1, çünkü hiçbir benchmark çalıştırması bir
collection'a bakmıyor.` Bunu kesinleştirecek şey: `eval/run_benchmark.py` dosyasını listeye değil
collection'a bağlayıp iki hit@1 değerini doğrudan karşılaştırmak.

## Daha derine

**Vector database ne zaman yerini hak ediyor.** Burada değil. Yirmi sekiz vektör, cache'e sığan bir
matrise karşı bir numpy dot product'ı: exact, mikrosaniye, sıfır bağımlılık. Bu corpus üzerinde bir
ölçüm değil, bir pratik kural olarak: brute force yüz bin vektör civarına kadar meşru bir production
cevabı olarak kalıyor. Onun üstünde ilk soru şu: zaten Postgres çalıştırıyor musun. Çalıştırıyorsan
`pgvector` sana tek bir yedekleme hikâyesi, tek bir transaction sınırı ve vektör aramasını yanındaki
iş kolonlarına join etme imkânı veriyor. Ayrı bir vektör deposu, vektör iş yükü kendi başına bir
operasyon problemine dönüştüğünde hak ediyor: on milyonlarca vektör, her embedder değiştirişinde
yeniden index'leme, tenant başına collection, sharding.

**Altta index ne yapıyor.** Chroma'nın index'i bir Hierarchical Navigable Small World graph'ı — her
vektör, birkaç yakın komşusuna bağlı bir düğüm ve bir sorgu her şeyi taramak yerine
katmanlar boyunca açgözlü şekilde aşağı yürüyor; collection büyüdükçe aramanın hızlı kalmasının
sebebi bu. *Approximate* olmasının sebebi de o yürüyüşün garantisinin olmaması: en iyisi yerine
ikincisine razı olabiliyor. 28 dokümanda ölçülecek bir şey yok — graph aramadan küçük — ve
hissettiğin gecikme index değil, embedding çağrısıdır.

<div class="presenter-note">
Süre: 10 dakika — öldürülen process'e 2, WRONG satırlarına 4, senin seçtiğin embedder ile
<code>PersistentClient</code> tek satırına 2, üç client satırı ve <code>/api/v2/</code>'ye 2. Günün
en kısa slotu olması bilinçli: salon yeni yemek yedi ve yükü bir tuzak ile üç client satırı.
<br /><br />
<strong>Geç kalırsan: bu modül kesme listesinde ikinci sırada, 10 dakikadan 5'e.</strong> Server
bölümünü ve "Daha derine"yi at — öldürülen process'i ve WRONG satırlarını tut, server değişimini üç
client satırını yan yana gösterip tek cümlede söyle. Buradaki sessiz default, kimsenin
dokümantasyondan kendi başına çıkaramayacağı tek şey.
<br /><br />
Biri "peki production'da Chroma mı pgvector mı?" diye sorarsa ürün adıyla değil soruyla cevap ver:
kaç vektör ve zaten Postgres çalıştırıyor musunuz. Onların corpus'unda ölçmediğin bir veritabanını
tavsiye etmeyi reddet.
</div>

## Çıkış cümlesi

> Aynı API, farklı deployment. Chroma iki Türkçe soruyu üst üste aynı yanlış dokümanla cevapladı ve
> hiç uyarı vermedi; bunu kendi vektörlerimizi vererek düzelttik, ama altındaki tavan hâlâ 0.750 —
> yirmi sorunun beşi hâlâ ilk sırada yanlış dokümanla dönüyor. Peki semantik arama yetmiyorsa?
