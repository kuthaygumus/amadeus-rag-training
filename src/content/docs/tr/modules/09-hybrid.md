---
title: "9. Hybrid, Rerank ve Contextual"
description: "Semantic yetmiyorsa?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Semantic yetmiyorsa?**

Modül 8'i **hit@1 0.800** ile bitirdin: `bge-m3`, 152 structure-aware chunk, ChromaDB üzerinden
servis ediliyor. Beş sorunun dördünde doğru doküman birinci sırada geliyor. Beşte biri gelmiyor ve
gelmediğinde generator yanlış sayfadan, hiç tereddüt etmeden cevap veriyor.

O boşluğa konacak standart bir liste var. Okuduğun her RAG yazısı aynı üç kelimeyi söylüyor: hybrid,
rerank, contextual. Vector search'ün yanına BM25 koy, sonuçları yeniden sıralaması için üstüne bir
model tak, her chunk'ın başına context yaz. Üçünü de aynı yirmi soruya karşı çalıştıracağız. Bunlardan
birini modül 7'de zaten yaptın, farkında değilsin. Diğer ikisi kaybediyor.

<div class="presenter-note">
Notebook'u açmadan önce tahtaya <strong>hybrid · rerank · contextual</strong> yaz ve el kaldırt:
"bu üçünden hangisi 0.800'ü yukarı taşır?" Rerank için neredeyse bütün eller, hybrid için çoğu el
kalkıyor. Sayıları kelimelerin yanına yaz ve modül boyunca orada bıraksın — iki kez oraya işaret
edeceksin. 3 dakika, laptoplar kapalı.
</div>

## BM25'i geri getir

BM25 modül 5'teki keyword retriever: nadir terimler yüksek skor alır, term frequency doyuma ulaşır,
uzun dokümanlar cezalandırılır. `H9 1487`'yi anında bulmuştu ve "iptal edersem ne öderim" sorusunda
tam sıfır almıştı. Şimdi onu dense retriever'ın kullandığı chunk'ların tam üstünde çalıştır — aynı
152 chunk, aynı index:

```python
bm25 = BM25(chunk_ids, chunk_texts)
```

**hit@1 0.300, MRR 0.465.** Türkçe sorulan altı soruda **0.000**. Zayıflamış değil. Sıfır. BM25'in
`iptal`'den `Cancellation penalty`'ye giden bir yolu yok, chunking de ona böyle bir yol vermedi.

## Reciprocal rank fusion ve neden seyreltiyor

Fusion herkesin ilk uzandığı çözüm ve gerçekten zarif bir fikir. İki retriever'ın skorlarının
kıyaslanabilir olması gerekmiyor — cosine similarity ile BM25 skorunun ortak birimi yok — o yüzden
skorları atıp sadece sıraları tutuyorsun:

```python
def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for position, doc in enumerate(ranking, 1):
            fused[doc] = fused.get(doc, 0.0) + 1 / (k + position)
    return [doc for doc, _ in sorted(fused.items(), key=lambda kv: -kv[1])]
```

Her retriever her dokümana `1 / (k + sıra)` puan veriyor. `k = 60` eğrinin tepesini düzleştiriyor;
birinci sıra ile üçüncü sıra neredeyse aynı değerde, kuyruk ise hiçbir şey. İki retriever'ın da
beğendiği doküman, sadece birinin beğendiğini geçiyor.

Dense ile BM25'i aynı chunk'lar üzerinde fuse et: **hit@1 0.450, MRR 0.581.** Dense tek başına 0.800
idi. İkinci retriever bize 0.350'ye mal oldu.

Sebep formülün içinde duruyor. RRF'in bir retriever'ın ne kadar iyi olduğuna ya da bu sorguda ne kadar
emin olduğuna dair hiçbir fikri yok. BM25'in birinci sırası, dense retriever'ın birinci sırasıyla aynı
`1/61`'i katıyor. Altı Türkçe soruda BM25'in sıralaması gürültü ve o gürültü tam ağırlıkla oy
kullanıyor. Anlamadığı sorgularda daha az oy kullanamaz, çünkü elinde teslim ettiği tek şey bir
sıralama — ve sıralama her zaman bir fikir gibi görünür.

Kimsenin yüksek sesle söylemediği önkoşul bu: **reciprocal rank fusion her iki girdinin de kendi
başına sağlam olduğunu varsayar.** Farklı sorularda düşen iki makul retriever birbirinin açığını
kapatır. İyi bir retriever ile korpusunun üçte birinde sistematik olarak yanılan bir retriever ise
sadece ortalamaya gider, ve ortalama almak tamir değildir.

<div class="presenter-note">
Fusion hücresini çalıştırmadan önce taahhüt al: "Dense 0.800, BM25 0.300. Fuse edince — 0.800'ün
üstü mü, arası mı, 0.300'ün altı mı?" Kabaca oyla. Salonun çoğu üstü der, çünkü fusion kulağa toplama
gibi geliyor. Sonra <strong>0.450</strong>'yi göster ve beş saniye hiçbir şey söyleme. Ollama düştüyse
bu sayfadaki her sayı <code>eval/RESULTS.md</code> içinde; oradan oku ve devam et. 5 dakika.
</div>

## BM25'i çantada tutan çelişki

BM25 kötü bir retriever değil. **Tam dokümanlar** üzerinde exact-token sorgularda dense retrieval'ı
yeniyor — uçuş kodları, bülten id'leri, `KSHEU26` gibi fare basis kodları. Bizim dense sayılarımız
aynı şeklin diğer yüzünü gösteriyor: exact-token sorular tam dokümanlarda **1.000** alıyor, sabit 280
karakterlik chunking'de **0.500**'e düşüyor.

BM25'i chunking öldürdü, iki yoldan. Uzunluk normalizasyonu çalışmayı bırakıyor: BM25 doküman
uzunluğunu ortalamaya bölüyor, her chunk aynı boyda olunca bu terim her yerde sabite dönüşüyor ve
güvendiğin bir ayırt edici sessizce yok oluyor. Bir de kanıt bölünüyor — altı terimlik bir sorgu eskiden
altı terimi tek dokümanın içinde biriktiriyordu; chunking'den sonra terimler üç ayrı chunk'a dağılıyor
ve hiçbiri kayda değer puan toplamıyor.

Yani dürüst cümle "hybrid search abartılıyor" değil. Dürüst cümle şu: **chunk'lar üzerinde, diller
arası bir korpusta hybrid search, dense'in 0.800'üne karşı 0.450 ölçüldü.** Index birimini değiştir,
cevap tersine dönebilir — bugün ikinci kez bir retrieval kararı başka bir retrieval kararına bağlı
çıkıyor.

## Reranker ve günün en şaşırtıcı sayısı

Reranker, en üstteki adayları alıp bir dil modeline değerlendirtiyor. Bizimki `qwen2.5:3b`, her adayı
tek tek 0-10 arası puanlıyor — aday başına bir model çağrısı, eşitlikte retriever'ın orijinal sırası
korunuyor, yani modelin gerçekten bir fikri olduğunda bir dokümanı yerinden oynatabiliyor. Aynı
reranker, kalitesi farklı dört retrieval kurulumunda.

| retrieval kurulumu | hit@1 önce | sonra |
|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** |
| `nomic` + structure-aware | 0.350 | **0.450** |
| `bge-m3` + fixed-280 | 0.650 | 0.550 |
| `bge-m3` + structure-aware | **0.800** | 0.650 |

İki zayıf kurulumu yukarı çekti, iki güçlü kurulumu aşağı indirdi. Tabloyu neyin ikiye böldüğüne bak:
chunking değil — her iki chunking stratejisi de tablonun iki yakasında birden var — **embedder**. Her
iki `nomic` kurulumunu yaklaşık 0.45'e çıkarıyor, her iki `bge-m3` kurulumunu yaklaşık 0.65'e
düşürüyor.

**Reranker seviyeler, ve kendi tavanına seviyeler.** 3B bir modelin alaka konusundaki fikri bu
korpusta aşağı yukarı 0.45 ile 0.65 arası değerde. Retriever'ın bundan kötüyse modelin fikrini
dayatmak bir yükseltmedir. Retriever'ın zaten daha iyiyse dayatmak sadece kaybettirir. Üçüncü bir
sonuç yok.

Soru bazında bakınca hiç ortalama almadan aynı şey görünüyor. Doğru doküman zaten birinci sıradayken
reranking onu **beş seferin beşinde aşağı** taşıdı. 5. ya da 6. sıradayken yukarı çekti. Reranker
sıralamayı iyileştirmiyor ya da bozmuyor; her sıralamayı kendi doğruluğuna doğru sürüklüyor. Bu
negatif sonucun bedeli: **sorgu başına 8 model çağrısı ve yaklaşık 4 saniye.**

<div class="presenter-note">
Ağzında gevelemeyeceğin cümle: <strong>reranker, retriever'ın reranker'dan kötüyse kazandırır;
retriever'ın daha iyiyse kaybettirir.</strong> Bir kez, yavaşça söyle ve tahtadaki el sayılarına geri
işaret et. Biri "gerçek reranker chat modeli değil, cross-encoder olur" diye itiraz edecek. Haklı —
dürüstçe cevap ver: onu ölçmedik ve denenecek ilk şey o. Onun için sayı uydurma. 3 dakika.
</div>

## Nasıl sorduğun, sorup sormadığından baskın

Rerank'in bariz yolu tek çağrı: altı pasajı yapıştır, modelden sırala. Daha ucuz ve kodda daha güzel
duruyor. On dokümanlık probe korpusunda, beş soru: rerank yok **4/5**, MRR 0.833. Listwise, "bu altısını
sırala": **2/5**, MRR 0.600 — hiçbir şey yapmamaktan kötü. Pointwise, "bu tek pasajı 0-10 arası
puanla", bir yerine altı çağrı: **5/5**, MRR 1.000. Bu `n = 5`; yönü gerçek, büyüklüğü kanıtsız kabul et.

Altı pasajı sıralamak, altı karşılaştırmayı aynı anda akılda tutup bir permütasyon üretmek demek. 3B
model bunda kötü, "bu tek pasaj ne kadar alakalı" sorusunda iyi — çünkü o, sabit bir ölçeğe karşı tek
bir yargı. Aynı model, aynı pasajlar, aynı bilgi; kazandıran ile kaybettiren arasındaki fark sorunun
şekli.

## Contextual retrieval — onu zaten yaptın

Contextual retrieval, her chunk'ın başına onu tek başına anlaşılır kılacak kadar çevre bilgisi
koymak demek; genelde bir modele "bu chunk dokümanın neresinde" diye bir satır yazdırarak.
Structure-aware chunking her chunk'ın başına zaten kendi başlık yolunu koyuyor: aynı mekanizma, ama
modelle üretilmiş değil dokümandan alınmış ve inference maliyeti sıfır. K satırı ile sütun başlığının
birlikte hayatta kalmasının ve modelin **EUR 70** yerine **EUR 90** okumasının sebebi bu. Modül 7'deki
0.800 aslında contextual retrieval'ın kazancı, cebe girmiş durumda. Üstüne bir de modelle context
üretmek denenmeye değer — ama başlıkları olan dokümanlarda, Markdown'ın sana bedavaya vermediği bir
şey satın aldığından emin ol.

## Ne çalıştırıyorsun

Notebook: `06_hybrid_rerank_contextual.ipynb`.

```bash
ollama serve                       # çalışmıyorsa ikinci bir terminalde
python scripts/verify_setup.py     # devam etmeden önce yeşil yazmalı
jupyter lab notebooks/06_hybrid_rerank_contextual.ipynb
```

Her şey ortak koddan geliyor:

```python
from eval.retrieval import BM25, DenseRetriever, rrf, pointwise_rerank
from eval.chunking import structure_aware, chunk_corpus
from eval.metrics import load_gold, evaluate

chunks = chunk_corpus(corpus, structure_aware)          # 152 chunk
dense  = DenseRetriever(chunk_ids, chunk_texts)         # bge-m3
bm25   = BM25(chunk_ids, chunk_texts)

gold = load_gold("eval/gold_questions.jsonl")
for name, rank in (("dense", dense.rank), ("bm25", bm25.rank),
                   ("rrf", lambda q: rrf([dense.rank(q), bm25.rank(q)]))):
    print(name, evaluate({g["id"]: rank(g["query"]) for g in gold}, gold))

reranked = {g["id"]: pointwise_rerank(g["query"], dense.rank(g["query"])[:8], texts)
            for g in gold}                              # sorgu başına 8 çağrı, saate bak
print("reranked", evaluate(reranked, gold))
```

Zaman darsa rerank hücresini önce `tr_en` alt kümesinde çalıştır; seviyeleme en net orada görünüyor.

## Sayılar ne dedi

<div class="measured">

| aynı 152 structure-aware chunk üzerinde retriever | hit@1 | MRR |
|---|---|---|
| dense, `bge-m3` | **0.800** | **0.846** |
| BM25 | 0.300 | 0.465 |
| ikisinin RRF'i | 0.450 | 0.581 |
| 6 `tr_en` sorusunda BM25 | 0.000 | — |

| aynı reranker, dört retrieval kurulumu | hit@1 önce | sonra | MRR önce | sonra |
|---|---|---|---|---|
| `nomic` + fixed-280 | 0.350 | **0.450** | 0.494 | **0.544** |
| `nomic` + structure-aware | 0.350 | **0.450** | 0.500 | **0.562** |
| `bge-m3` + fixed-280 | 0.650 | 0.550 | 0.789 | 0.708 |
| `bge-m3` + structure-aware | **0.800** | 0.650 | **0.846** | 0.750 |

| probe korpus, 10 doküman, 5 soru | hit@1 | MRR |
|---|---|---|
| rerank yok | 4/5 | 0.833 |
| listwise — "bu 6'sını sırala" | 2/5 | 0.600 |
| pointwise — "bu pasajı 0–10 puanla" | 5/5 | 1.000 |

| bir rerank geçişinin maliyeti | |
|---|---|
| sorgu başına model çağrısı | 8 |
| sorgu başına eklenen gecikme | ~4 sn |

</div>

Korpus: 28 doküman, 75 KB. Gold set: 20 soru, 6'sı Türkçe sorgu / İngilizce doküman. Generation ve
reranking `qwen2.5:3b`, embedding aksi belirtilmedikçe `bge-m3`, hepsi Ollama üzerinden lokal.
**Yirmi soru iki tasarım arasında karar verdirir, genel bir iddiayı taşımaz.** Bu sayfa "bu teknikler
kötüdür" demiyor. Bu korpusta, bu embedder ve bu reranker ile kaybettiklerini söylüyor ve her birinin
hangi önkoşula ihtiyaç duyduğunu adıyla koyuyor.

## Daha derine

RRF'teki `k = 60` bir yumuşatma sabiti. `k` küçükse birinci sıra baskın olur, fusion "hangi retriever
daha eminse ona güven" gibi davranır; `k` büyükse eğri düzleşir ve fusion bütün liste üzerinde bir
popülerlik oylamasına döner. 60 orijinal TREC çalışmasından geliyor ve neredeyse hiç ellenmiyor.
Burada onu ayarlamak bizi kurtarmazdı: soruların üçte birinde 0.000 alan bir sıralamayı hiçbir `k`
değeri işe yarar hale getirmez, çünkü problem ağırlık eğrisi değil, girdinin o sorgularda hiç sinyal
taşımaması. Dengesiz bir çiftle fusion yapmak sorgu bazlı ağırlıklandırma ister — fuse etmeden önce
"bu, BM25'in cevaplayabileceği türden bir sorgu mu" kararını vermek — ve o router da kurup ölçmen
gereken başka bir model.

Bizim reranker'ımız pasaj puanlayan bir chat modeli; kurulumu en kolay, reranker denilebilecek en
zayıf şey. Production cevabı cross-encoder: sorguyu ve pasajı **birlikte**, tek forward pass'te okuyup
tek bir alaka skoru üreten, prompt'la göreve ikna edilmiş değil alaka etiketleriyle eğitilmiş bir
model. `bge-reranker-v2-m3` bunun çok dilli olanı ve zaten kullandığımız embedder ile aynı aileden.
Çok daha güçlü bir hakem ve tavanı 0.800'ün üstüne taşıması makul — ama biz onu ölçmedik, o yüzden
bunu sonuç değil, deneyi belli bir hipotez olarak kabul et. Ders iki durumda da aynı: onun da bir
tavanı var ve o tavanın senin retriever'ının üstünde mi altında mı olduğunu ölçmen gerekiyor.

Seviyeleme sonucu reranking'in ötesine genelleniyor. Üstteki bir sıralamayı ezen her aşama, girdi ne
olursa olsun kendi doğruluğunu çıktıya dayatır. "Sen bir reranker ekle" tavsiyesinin "sen bir cache
ekle" kadar kötü olmasının sebebi bu: önkoşul, sisteminle ilgili ancak ölçerek öğrenebileceğin bir
gerçek. Sıra da önemli — modül 6 embedder'ı düzeltti, modül 7 chunking'i düzeltti ve ikisi birlikte
hit@1'i 0.550'den 0.800'e taşıdı. Bu modüldeki her şey tavan zaten yükseldikten sonra geldi, yani tam
da bu tekniklerin maliyet yazdığı noktada. Dört satırlık tablonun iki satırı gerçek bir iyileşmeye
bakarak reranker'ı production'a alırdı — `nomic` cidden 0.350'den 0.450'ye çıkıyor — ve tasarım yine
yanlış olurdu, çünkü doğru hamle embedder'ı düzeltmekti.

On milyon dokümanda tablo değişiyor ve BM25 bambaşka bir sebeple geri geliyor. Orada her sorguda her
chunk'ı embed edip skorlayamazsın; ucuz bir birinci aşama birkaç yüz aday döndürür, pahalı bir ikinci
aşama onları sıralar. Inverted index üzerinde BM25 mükemmel bir birinci aşamadır — milisaniyenin
altında, tanımlayıcılarda birebir, güncellemesi kolay — arkasında da top 200 üzerinde bir
cross-encoder. Bu, bizim ölçtüğümüz hybrid değil: BM25 nihai cevaba oy vermiyor, aday üretiyor ve
metriği hit@1 değil recall@k. Aynı bileşen, farklı iş, farklı metrik. 28 dokümanda o mimarinin yapacak
işi yok ve sorgu başına dört saniye yazıyor.

<div class="presenter-note">
Geriden geliyorsan listwise-pointwise bölümünü tek cümleye indir ve dört kurulumluk tabloyu koru —
insanların sonradan anlattığı sonuç o. Kapatmadan önce tahtadaki el sayılarına işaret et ve üç kararı
oku: hybrid kaybetti, rerank seviyeledi, contextual zaten modül 7'de yapılmıştı. Sonra hâlâ çalışmayanı
söyle: gold set'teki multi-hop sorular. Modül 10 tam orada ve tek bir hücreyle gösterilebilecek bir
başarısızlıkla açılıyor. 2 dakika.
</div>

## Çıkış cümlesi

> Ölçtüğümüz kadarıyla hiçbiri düz retrieval'ı geçemedi. Ama multi-hop hâlâ düşüyor.
