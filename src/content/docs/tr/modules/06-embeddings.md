---
title: "6. Embedding Bake-Off"
description: "Varsayılan embedder senin dilin için doğru mu?"
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Varsayılan embedder senin dilin için doğru mu?**

Modül 5'te dense retrieval, Türkçe soruda keyword search'ü yendi. Modelin `bge-m3` olduğunu
söyledim ve şimdilik güvenmeni istedim. Bu salonda o modeli kimse seçmedi. Kendi projende de
seçmeyeceksin — `pip install chromadb` yazacaksın, `add()` çağıracaksın ve model senin yerine
seçilmiş olacak.

O tek satırı değiştir ve aynı soruları tekrar çalıştır.

```python
dense = DenseRetriever(doc_ids, documents, model="nomic-embed-text")
```

Türk bir acentenin İngilizce dokümana soru sorduğu altı soruda hit@1 **0.667'den 0.000'a** düşüyor.
Altıda sıfır. Exception yok, uyarı yok, log satırı yok, şüphe uyandıran bir skor yok. Pipeline üç
doküman ve kendinden emin görünen benzerlik skorları döndürüyor, doğru doküman aralarında değil.
Generator da kendisine verilen şeyden cevap üretiyor, çünkü işi bu.

<div class="presenter-note">
Değiştirilmiş hücreyi çalıştırmadan önce salondan taahhüt al: "sadece embedding modelini
değiştiriyoruz, başka hiçbir şeyi. Altı Türkçe sorudan kaçını hâlâ bilir? Dört ve üzeri diyenler el
kaldırsın." Eller kalkar. 0.000'ı göster ve üç saniye hiçbir şey söyleme. O sessizlik sonraki
paragraftan daha çok iş yapar. 4 dakika.
</div>

## Embedding gerçekte nedir

Bir embedding modeli bir string alır ve sabit uzunlukta bir float listesi döndürür. Sözleşmenin
tamamı bu. `bge-m3` her girdi için **1024** sayı döndürür — tek kelime de versen bir sayfa da.
`nomic-embed-text` **768**, `all-MiniLM-L6-v2` **384** döndürür. Uzunluk girdiye göre hiç
değişmez: model verdiğin şeyi o kadar sayıya sıkıştırır ve anlam saymadığı ne varsa atar.

Her sayı bir koordinat, yani 1024 sayı 1024 boyutlu uzayda bir nokta. Gözünde canlandıramazsın,
canlandırman da gerekmiyor — o noktalarla yaptığın tek şey ikisinin ne kadar yakın olduğunu
sormak. Yakınlık cosine similarity ve `eval/retrieval.py` içinde dört satır:

```python
def cosine(a, b):
    dot  = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / (norm + 1e-12)
```

Sırayla çarp, topla, iki uzunluğa böl. Asıl iş bölmede: büyüklüğü atıyor, sadece yönü bırakıyor.
Uzun bir dokümanla tek satırlık bir sorguyu mesafeyle karşılaştıramazsın — uzun olan zaten
orijinden daha uzakta — ama hangi yöne baktıklarıyla karşılaştırabilirsin. Sonuç -1 ile 1
arasında, gerçek metinde kabaca 0.3 ile 0.9 arasında geziyor ve önemli olan tek şey sıralama.
0.656 "%66 alakalı" demek değil. Sadece "0.466'dan yüksek" demek.

Bütün günün dayandığı iddia şu: o uzaydaki yön anlama karşılık geliyor. Modelin eğitim
dağılımının içinde, yaklaşık olarak geliyor. Dışında ise sayılar gelmeye devam ediyor ve sessizce
varsaydığın şeyi ifade etmeyi bırakıyor.

## Tuzak, canlı

İki denemede de aynı sorgu, bir Helios acentesinin gerçekten yazacağı soru:

> CLASSIC K sınıfı iptal cezası ne kadar?

Üç aday doküman, sırayla iki embedder. `nomic-embed-text` Türkçe no-show makrosuna **0.690**
veriyor ve birinci sıraya koyuyor. `| K | KSHEU26 | ... | EUR 90 | ... |` satırını taşıyan doğru
İngilizce ücret kuralı **0.466** alıyor ve üçüncü oluyor. Üç adayın sonuncusu. `bge-m3` aynı
İngilizce dokümana **0.656** veriyor ve birinci sıraya koyuyor.

nomic'in seçtiği şeye bak. `macro_tr_noshow` rastgele bir doküman değil. Türkçe, iptal cezalarını
anlatıyor, içinde "iptal cezası iki katına çıkar" cümlesi geçiyor. İnsan gözüyle makul bir komşu.
Ama tutar vermeyi açıkça reddediyor — "tutarlar bu makroda tekrarlanmaz", İngilizce ücret kuralına
bak diyor — yani acentenin ihtiyacı olan tek sayı orada yok. Retriever doğru dili ve doğru konuyu
seçti, cevabı içeren tek dokümanı eledi.

## Neyin bozulduğunu tam söyle

Kolay özet "MiniLM Türkçe bilmiyor" demek. Bu özet yanlış ve soru-cevapta seni yakar.

Bu modeller Türkçe-Türkçe eşleşmeyi gayet iyi yapıyor. Türkçe bir soruyu Türkçe makrolara sor,
nomic mantıklı sıralıyor — `macro_tr_noshow`'u 0.690 ile bulması tam da bu yüzden. Yapamadığı
şey **diller arası** eşleşme. Tek dilli bir modelin uzayında metnin dili başlı başına güçlü bir
yön: alakasız konulardaki iki Türkçe cümle, bir Türkçe cümle ile onun kendi İngilizce çevirisinden
daha yakın durabiliyor, çünkü "Türkçe olmak" "iptal cezasıyla ilgili olmak"tan ağır basıyor. Dil,
anlamdan büyük bir eksen haline geliyor.

Doğal olarak bakacağın hiçbir metrikte görünmüyor, çünkü TR→TR çalışıyor ve EN→EN çalışıyor.
Korpusun yarısı, soruların yarısı, sessizce bozuk.

ChromaDB'nin varsayılan embedding function'ı `all-MiniLM-L6-v2`. İlk `add()` çağrında sen
istemeden indiriliyor ve yalnızca İngilizce. On dokümanlık probe korpusunda **2/5** aldı,
`bge-m3` **4/5**. Bunların hiçbiri log'a düşmüyor.

<div class="presenter-note">
Yamultmaman gereken cümle bu: <strong>bu modeller Türkçe-Türkçe eşleşmede iyi — yapamadıkları şey
diller arası eşleşme ve senin korpusun karışık.</strong> Bir kez, yavaş söyle ve tahtaya
"TR → TR: ok / TR → EN: 0.000" yaz. Ollama burada düşerse üç benzerlik skoru bu sayfada ve
<code>eval/RESULTS.md</code>'de duruyor; oradan oku, salonun önünde debug yapma.
</div>

## Ne çalıştırıyorsun

Notebook: `04b_embeddings_bakeoff.ipynb`.

```bash
ollama pull bge-m3
ollama pull nomic-embed-text
python scripts/verify_setup.py     # devam etmeden önce yeşil yazmalı
jupyter lab notebooks/04b_embeddings_bakeoff.ipynb
```

Notebook kodu yeniden yazmıyor, ortak kodu kullanıyor:

```python
from eval.retrieval import embed, cosine, DenseRetriever
from eval.metrics import load_gold, evaluate

q = "CLASSIC K sınıfı iptal cezası ne kadar?"
for model in ("nomic-embed-text", "bge-m3"):
    qv, dv = embed([q], model=model)[0], embed(candidates, model=model)
    print(model, [round(cosine(qv, v), 3) for v in dv])   # skora değil SIRAYA bak

gold = load_gold("eval/gold_questions.jsonl")
tr_en = [g for g in gold if g["type"] == "tr_en"]
for model in ("nomic-embed-text", "bge-m3"):
    r = DenseRetriever(doc_ids, documents, model=model)
    print(model, evaluate({g["id"]: r.rank(g["query"]) for g in tr_en}, tr_en))
```

Vektör uzunluğunu bir kez yazdır — `len(embed(["test"])[0])` — ki 1024 bir slayt olmaktan çıkıp
HTTP üzerinden gelişini izlediğin bir şey olsun.

## Sayılar ne dedi

<div class="measured">

| embedder | vektör uzunluğu | 6 `tr_en` sorusunda hit@1 |
|---|---|---|
| `nomic-embed-text` | 768 | 0.000 |
| `bge-m3` | 1024 | 0.667 |

| sorgu: "CLASSIC K sınıfı iptal cezası ne kadar?" | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| doğru İngilizce CLASSIC short-haul ücret kuralı | 0.466 — 3. sıra | **0.656 — 1. sıra** |
| Türkçe no-show makrosu | **0.690 — 1. sıra** | daha altta |

| tüm gold set, 20 soru | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| fixed 280 karakter chunk, hit@1 | 0.350 | 0.650 |
| structure-aware chunk, hit@1 | 0.350 | 0.800 |

| probe korpus, 10 doküman, 5 soru | hit@1 |
|---|---|
| `all-MiniLM-L6-v2` — ChromaDB'nin sessiz varsayılanı | 2/5 |
| BM25 | 3/5 |
| `bge-m3` | 4/5 |

</div>

Korpus: 28 doküman, 75 KB. Gold set: 20 soru, 6'sı Türkçe sorgu / İngilizce doküman; her şey lokal
Ollama üzerinde. Altı soru bir model kararını vermeye yeter, yayımlamaya fazlasıyla azdır: yön
gerçek, kesin rakam gürültülü.

Üçüncü tabloyu bir daha oku. nomic ile daha iyi chunking sana **hiçbir şey** kazandırmıyor — iki
durumda da 0.350. Modül 7'deki chunking emeği ancak embedder diller arasını görebildiğinde işe
yarıyor. Önce embedder'ı düzelt, sonra chunk'la.

## Daha derine

Multilingual eğitimin değiştirdiği şey mimari değil, objective. Tek dilli bir model tek dil içinde
aynı anlama gelen metin çiftlerini birbirine çekmek üzere eğitilir. `bge-m3` ise yaklaşık yüz dilde
paralel ve mined çiftlerle eğitilir; pozitif çift bir cümle ve onun çevirisidir. Gradyanın tek işi
Türkçe cümleyle İngilizce cümleyi aynı noktaya indirmek, dolayısıyla dil kullanılabilir bir yön
olmaktan çıkıyor — loss onu cezalandırıyor. Çözüm daha çok veri ya da daha iyi prompt değil.
Sen weight'leri çekmeden aylar önce pişmiş.

Boyut sayısı kalite değil. Burada 1024, 768'i yeniyor ve bu sezgiyi okşuyor; ama 384 boyutlu
MiniLM güçlü bir İngilizce retriever'dır ve İngilizce benchmark'larda kendinden çok daha büyük
modelleri geçer. Bizim korpusta genişlikle ilgisi olmayan bir sebepten düşüyor. Genişlik ayrım
yapma kapasitesi satın alır; hangi ayrımları önemsemeyi öğrendiğine karar vermez. Boyutu bir
maliyet olarak gör — 1024 float32 chunk başına 4 KB, yani bizim 152 chunk 600 KB, on milyon chunk
index maliyeti hariç 40 GB — kalite sinyali olarak ise eğitim verisine bak.

Maliyet gerçek, bu yüzden bu bir takas, bedava kazanç değil. `bge-m3` bir XLM-RoBERTa-large
gövdesi, MiniLM'in birkaç katı parametre, dolayısıyla o oranda yavaş embed ediyor. 28 dokümanda bu
görünmez. On milyonda planlaman gereken bir yeniden indeksleme bütçesi ve Matryoshka embedding'lere
baktığın nokta: vektörün ilk 256 sayısı tek başına işe yarayacak şekilde eğitilmiş modellerle ucuza
kısa liste çıkarıp ilk birkaç yüzü tam genişlikte yeniden skorlarsın.

Kendi korpusun için embedder değerlendirmek bir öğleden sonra sürüyor ve bu konudaki en değerli
öğleden sonra. Kullanıcılarının gerçekten yazdığı dilde yirmi soru yaz, her birini cevabı içeren
dokümanla etiketle. Yirmi yeter — bu sayfa bir modeli altı soruyla seçti. Şüphelendiğin hataya göre
grupla: aynı dil, diller arası, tam token, paraphrase. Sonra üç dört modeli `DenseRetriever`'dan
geçir ve hit@1'i ortalama olarak değil **grup grup** yazdır. İşi gruplama yapıyor: nomic'in genel
skoru kötü ama alarm verici değil, temiz sıfırı sadece `tr_en` satırı gösteriyor.

0.667 de bir zafer değil. Altıda ikisi hâlâ yanlış. Bake-off'un kazananı ölçtüğün en az kötü
seçenektir, çözülmüş bir problem değil; 0.667'de hâlâ bozuk olan şey de modül 7'nin konusu.

<div class="presenter-note">
Biri mutlaka "sorguyu önce İngilizceye çevirsek olmaz mı?" diye soracak. Ciddiye al — multilingual
embedder'lar iyi olmadan önce insanlar tam da bunu yapıyordu. Cevabı maliyetle ver: sorgu başına
fazladan bir model çağrısı, çeviri CLASSIC K gibi bir terimi düşürdüğünde yeni bir hata modu ve
sonunda yine bir embedder seçmen gerekiyor. Ölçülebilir olduğunu ve bizim ölçmediğimizi söyle.
Sayı uydurma. 2 dakika, sonra devam.
</div>

## Çıkış cümlesi

> Varsayılan sizin diliniz için yanlıştı ve kimse söylemedi.
