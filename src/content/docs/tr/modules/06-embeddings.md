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

O yüzden güvenmeyi bırak. Tek bir komut, aynı yirmi soru ve aynı chunk'lar üzerinde üç embedder'ı
puanlıyor ve puanı soru tipine göre ayırıyor:

```text
hit@1 by type         all-MiniLM-L6-v2  nomic-embed-text            bge-m3
--------------------------------------------------------------------------
tr_tr (n=4)                      0.750             0.500             1.000
tr_en (n=6)                      0.000             0.000             0.667   <-- Turkish question, English document
en_en (n=4)                      0.500             0.500             0.750
exact_token (n=4)                0.250             0.500             1.000
multi_hop (n=2)                  0.500             0.500             0.500
```

`tr_en` satırını oku. Bunlar Türk bir acentenin İngilizce dokümana soru sorduğu altı soru —
korpusumuzun yarısının şekli. ChromaDB'nin kendi varsayılanı altıda sıfır alıyor. Makul görünen,
İngilizce öncelikli bir seçim olan `nomic-embed-text` de altıda sıfır alıyor. Exception yok, uyarı
yok, log satırı yok, şüphe uyandıran bir skor yok. İki pipeline da kendinden emin görünen benzerlik
skorlarıyla doküman döndürüyor ve doğru doküman aralarında değil. Generator da kendisine verilen
şeyden cevap üretiyor, çünkü işi bu.

**Her retrieval sayısının bir index koşulu vardır.** Yukarıdaki tablo structure-aware chunk'lar
üzerinde ölçüldü — modül 7'nin vardığı strateji — ve sütunlar arasında değişen tek şey embedder
kalsın diye sabit tutuldu. Index'i değiştirdiğinde aynı model kayıyor: şu anda elinde olan pipeline
olan **whole document** üzerinde `bge-m3`'ün `tr_en` skoru 0.667 değil, **0.333**. Hangi index
üzerinde ölçüldüğü söylenmeyen bir retrieval sayısı sayı değildir.

<div class="presenter-note">
Slot: 11:26'da 16 dakika. Bu modül kesme listesinin başında, 8 dakikaya iniyor — kesme planı bu
sayfanın son notunda.<br />
Komutu çalıştırmadan önce salondan taahhüt al: "sadece embedding modelini değiştiriyoruz, başka
hiçbir şeyi. Altı Türkçe sorudan kaçını hâlâ bilir? Dört ve üzeri diyenler el kaldırsın." Eller
kalkar. <code>tr_en</code> satırını göster ve üç saniye hiçbir şey söyleme. O sessizlik sonraki
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
Uzun bir dokümanla tek satırlık bir sorguyu mesafeyle karşılaştıramazsın — uzun olan ne yazarsa
yazsın orijinden daha uzakta — ama hangi yöne baktıklarıyla karşılaştırabilirsin. Sonuç -1 ile 1
arasında ve önemli olan tek şey sıralama. Tek başına bir benzerlik skoru hiçbir şey ifade etmiyor.
Ancak diğer adayın aldığı skorun yanında bir anlamı oluyor.

Bütün günün dayandığı iddia şu: o uzaydaki yön anlama karşılık geliyor. Modelin eğitim
dağılımının içinde, yaklaşık olarak geliyor. Dışında ise sayılar gelmeye devam ediyor ve sessizce
varsaydığın şeyi ifade etmeyi bırakıyor.

## Tuzak, canlı

0.000 skoru sana doğru dokümanın birinci olmadığını söylüyor. Onun yerine neyin geldiğini
söylemiyor, oysa asıl işe yarayan yarısı o. Retriever'a o altı soruda ne döndürdüğünü sor;
`nomic-embed-text` neredeyse her seferinde aynı cevabı veriyor:

```text
q05 ['sop_misconnect_v4']              -> macro_tr_noshow
q06 ['sop_denied_boarding']            -> macro_tr_noshow
q07 ['interline_h9_au']                -> macro_tr_noshow
q08 ['codeshare_h9_au_conditions']     -> macro_tr_noshow
q09 ['macro_en_refund']                -> macro_tr_rebook
q10 ['macro_en_special_assistance']    -> macro_tr_noshow
```

Altı soru, altı Türkçe doküman ve beşinde aynı Türkçe doküman. Konular yakın bile değil: q05
aktarmasını kaçıran yolcuya kaç euroluk yemek fişi verileceğini soruyor, cevap no-show
makrosundan geliyor.

Seçtiği şeye bak. `macro_tr_noshow` rastgele bir doküman değil. Türkçe, cezaları anlatıyor, içinde
"iptal cezası iki katına çıkar" cümlesi geçiyor. İnsan gözüyle bu altı sorudan biri için makul bir
komşu. Ama tutar vermeyi açıkça reddediyor — *"tutarlar bu makroda tekrarlanmaz"*, İngilizce ücret
kuralına bak diyor — yani acentenin ihtiyacı olan tek sayı orada hiç yok. Retriever doğru dili
seçti ve cevabı içeren her dokümanı eledi.

## Neyin bozulduğunu tam söyle

Kolay özet "bu modeller Türkçe bilmiyor" demek. Bu özet yanlış ve soru-cevapta seni yakar.

`tr_tr` satırına geri dön: Türkçe sorular, Türkçe dokümanlar. `all-MiniLM-L6-v2` orada **0.750**,
`nomic-embed-text` **0.500** alıyor — zayıf, ama sıfırın yanından bile geçmiyor. `tr_en`'de ikisi
de **0.000** alıyor. Altıda altı. Bozulan şey Türkçe değil. Diller **arasında geçiş**.

Tek dilli bir modelin uzayında metnin dili başlı başına güçlü bir yön: alakasız konulardaki iki
Türkçe cümle, bir Türkçe cümle ile onun kendi İngilizce çevirisinden daha yakın durabiliyor, çünkü
"Türkçe olmak" "iptal cezasıyla ilgili olmak"tan ağır basıyor. Dil, anlamdan büyük bir eksen haline
geliyor. Hatanın tamamı bu ve yukarıdaki `tr_en` listesi tam olarak bunun içeriden görünüşü.

Doğal olarak bakacağın hiçbir metrikte görünmüyor, çünkü TR→TR bir şekilde çalışıyor ve EN→EN
çalışıyor. Korpusun yarısı, soruların yarısı, sessizce bozuk.

ChromaDB'nin varsayılan embedding function'ı `all-MiniLM-L6-v2`. İlk `add()` çağrında sen istemeden
indiriliyor ve yalnızca İngilizce. Aynı yirmi soruda hit@1 **0.350**, MRR **0.443** alıyor;
`bge-m3` ise **0.800** ve **0.844**. Bunların hiçbiri log'a düşmüyor.

<div class="presenter-note">
Yamultmaman gereken cümle bu: <strong>bu modeller Türkçede çaresiz değil — yapamadıkları şey diller
arası eşleşme ve senin korpusun karışık.</strong> Bir kez, yavaş söyle ve tahtaya
"TR → TR: 0.750 / TR → EN: 0.000" yaz. Ollama burada düşerse bu sayfadaki tablolarda geçen her sayı
<code>MEASURED.md</code>'de duruyor; oradan oku, salonun önünde debug yapma. 3 dakika.
</div>

## Ne çalıştırıyorsun

```bash
python exercises/m6_embedding_bakeoff.py
```

- **ne görmelisin** — önce `28 documents, 154 structure-aware chunks, 20 questions` başlık satırı, sonra üç embedder'ın her birinin index'leyip sorguladığını bildirmesi, sonra iki tablo. Önemli satır `tr_en (n=6)`: `0.000  0.000  0.667` yazıyor ve yanında onu gösteren bir ok var. Hiçbir şey hata vermiyor, hiçbir şey uyarmıyor
- **kabaca ne kadar sürer** — yaklaşık 20 saniye; çoğu, 154 chunk'ı üç kez embed etmek

Burada hiçbir şey indirilmiyor. Üç modeli de pre-work kapsıyor: `bge-m3` ve `nomic-embed-text`
evde çekildi, `all-MiniLM-L6-v2` ise `scripts/seed_offline_assets.py` tarafından cache'lendi ve
`scripts/verify_setup.py` bunu kontrol ediyor. MiniLM sütunu `SKIPPED` diyorsa seed script'ini
çalıştır ve komutu tekrarla; salonda hiçbir şey pull etme.

Salon "peki onun yerine ne döndürdü?" diye sorarsa, *Tuzak, canlı* bölümündeki listeyi repo
kökünden şu yazdırıyor:

```python
import sys; sys.path.insert(0, "eval")
from pathlib import Path
import chunking as C, metrics, retrieval as R

docs = {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path("corpus/2026-Q3").glob("*.md"))}
ids, texts, _ = C.chunk_corpus(docs, "structure-aware")
tr_en = [q for q in metrics.load_gold("eval/gold_questions.jsonl") if q["type"] == "tr_en"]

r = R.DenseRetriever(ids, texts, model="nomic-embed-text")
for q in tr_en:
    print(q["id"], q["gold_doc_ids"], "->", C.to_documents(r.rank(q["query"]))[0])
```

<div class="presenter-note">
Projeksiyonda çalıştır ve salon da aynı anda çalıştırsın; yaklaşık 20 saniyede kimse beklemiyor.
İki şeye dikkat et. <code>all-MiniLM-L6-v2</code> sütununda <code>SKIPPED</code> yazan bir laptop
seed script'ini hiç çalıştırmamıştır — iki embedder'lık sonucu al ve devam et, <code>tr_en</code>
satırı yine 0.000'a karşı 0.667 diyor. Bir de biri, varsayılanın <code>bge-m3</code> ile tam olarak
tek bir satırda — <code>multi_hop</code>, iki soruda ikisi de 0.500 — berabere kaldığını fark
edecek. İki soru sonuç değildir; bu, varsayılanın savunmasına dönüşmeden söyle. 5 dakika.
</div>

## Sayılar ne dedi

<div class="measured">

Üç embedder, 20 soru, structure-aware chunk'lar:

| embedder | vektör uzunluğu | hit@1 | recall@5 | MRR | `tr_en` (n=6) |
|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` — ChromaDB'nin sessiz varsayılanı | 384 | 0.350 | 0.583 | 0.443 | 0.000 |
| `nomic-embed-text` | 768 | 0.350 | 0.633 | 0.492 | 0.000 |
| `bge-m3` | 1024 | **0.800** | **0.833** | **0.844** | **0.667** |

Aynı model, iki index — ya koşulu da söyle ya hiç söyleme:

| `bge-m3` şunun üzerinde | hit@1 | `tr_en` (n=6) |
|---|---|---|
| whole document — şu anda elinde olan pipeline | 0.550 | 0.333 |
| structure-aware chunk — modül 7'nin bittiği yer | 0.800 | 0.667 |

Chunking, yanlış embedder'ı kurtarmıyor:

| hit@1, 20 soru | `nomic-embed-text` | `bge-m3` |
|---|---|---|
| fixed 280 karakter chunk | 0.350 | 0.700 |
| structure-aware chunk | 0.350 | 0.800 |

</div>

Korpus: 28 doküman, 154 structure-aware chunk. Gold set: 20 soru, 6'sı Türkçe sorgu / İngilizce
doküman; her şey lokal Ollama üzerinde. Yirmi soru iki tasarım arasında seçim yapmaya yeter,
yayımlamaya fazlasıyla azdır — 0.05'in altındaki bir fark bu örneklemin gürültüsünün içindedir.
`tr_en` farkı değil.

Üçüncü tabloyu bir daha oku. `nomic-embed-text` ile daha iyi chunking sana **hiçbir şey**
kazandırmıyor — iki durumda da 0.350. Modül 7'deki chunking emeği ancak embedder diller arasını
görebildiğinde işe yarıyor. Önce embedder'ı düzelt, sonra chunk'la.

## Daha derine

Multilingual eğitimin değiştirdiği şey mimari değil, objective. Tek dilli bir model tek dil içinde
aynı anlama gelen metin çiftlerini birbirine çekmek üzere eğitilir. `bge-m3` ise yaklaşık yüz dilde
paralel ve mined çiftlerle eğitilir; pozitif çift bir cümle ve onun çevirisidir. Gradyanın tek işi
Türkçe cümleyle İngilizce cümleyi aynı noktaya indirmek, dolayısıyla dil kullanılabilir bir yön
olmaktan çıkıyor — loss onu cezalandırıyor. Çözüm daha çok veri ya da daha iyi prompt değil.
Sen weight'leri çekmeden aylar önce pişmiş.

Boyut sayısı kalite değil. Burada 1024, 768'i ve 384'ü yeniyor ve bu sezgiyi okşuyor; ama MiniLM'i
bozan şey genişlik değil: İngilizce soru / İngilizce doküman olan dört soruda hâlâ 0.500 alıyor ve
tam olarak 0.000'ı yalnızca bir dil sınırı aşıldığında veriyor. Genişlik ayrım yapma kapasitesi
satın alır; hangi ayrımları önemsemeyi öğrendiğine karar vermez. Boyutu bir maliyet olarak gör —
1024 float32 chunk başına 4 KB, yani bizim 154 chunk bir megabyte'ın epey altında, on milyon chunk
ise index maliyeti hariç yaklaşık 40 GB — kalite sinyali olarak ise eğitim verisine bak.

Maliyet gerçek, bu yüzden bu bir takas, bedava kazanç değil. `bge-m3` bir XLM-RoBERTa-large
gövdesi, MiniLM'in birkaç katı parametre, dolayısıyla o oranda yavaş embed ediyor — bu koşuda bile
görünüyor: aynı 154 chunk'ı index'lemesi yaklaşık iki katı sürüyor. 28 dokümanda bu hiçbir şey. On
milyonda planlaman gereken bir yeniden indeksleme bütçesi ve Matryoshka embedding'lere baktığın
nokta: vektörün ilk 256 sayısı tek başına işe yarayacak şekilde eğitilmiş modellerle ucuza kısa
liste çıkarıp ilk birkaç yüzü tam genişlikte yeniden skorlarsın.

Kendi korpusun için embedder değerlendirmek bir öğleden sonra sürüyor ve bu konudaki en değerli
öğleden sonra. Kullanıcılarının gerçekten yazdığı dilde yirmi soru yaz, her birini cevabı içeren
dokümanla etiketle. Yirmi yeter — bu sayfa bir modeli yirmi soruyla seçti ve kararı veren satırda
altı soru vardı. Şüphelendiğin hataya göre grupla: aynı dil, diller arası, tam token, paraphrase.
Sonra üç dört modeli `DenseRetriever`'dan geçir ve hit@1'i ortalama olarak değil **grup grup**
yazdır. İşi gruplama yapıyor: MiniLM'in genel 0.350'si kötü ama alarm verici değil, temiz sıfırı
sadece `tr_en` satırı gösteriyor.

0.667 de bir zafer değil. Altıda ikisi hâlâ yanlış. Bake-off'un kazananı ölçtüğün en az kötü
seçenektir, çözülmüş bir problem değil; 0.667'de hâlâ bozuk olan şey de modül 7'nin konusu.

<div class="presenter-note">
Biri mutlaka "sorguyu önce İngilizceye çevirsek olmaz mı?" diye soracak. Ciddiye al — multilingual
embedder'lar iyi olmadan önce insanlar tam da bunu yapıyordu. Cevabı maliyetle ver: sorgu başına
fazladan bir model çağrısı, çeviri CLASSIC K gibi bir terimi düşürdüğünde yeni bir hata modu ve
sonunda yine bir embedder seçmen gerekiyor. Ölçülebilir olduğunu ve bizim ölçmediğimizi söyle.
Sayı uydurma. Kapanış hesabıyla birlikte 4 dakika, sonra devam.<br />
<strong>16 dakikayı 8'e indirmek.</strong> Taahhüdü ve <code>tr_en</code> satırını tut (4 dk),
diller arası geçişle ilgili tek cümleyi (1 dk), koşuyu ve ilk tabloyu (2 dk), çıkış cümlesini
(1 dk). <em>Tuzak, canlı</em>, <em>Daha derine</em> ve çeviri sorusunu at. Index koşulu cümlesini
atma: modül 7, 0.550 ile açılıyor ve salonun bunun neden 0.800 olmadığını bilmesi gerekiyor.
</div>

## Çıkış cümlesi

> Varsayılan sizin diliniz için yanlıştı ve kimse söylemedi. Artık doğru embedder elimizde
> — ve hâlâ çalıştırdığımız şey olan whole document üzerinde retrieval sadece 0.550.
