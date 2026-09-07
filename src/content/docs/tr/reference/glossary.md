---
title: "Sözlük"
description: "Günde geçen her terim, bir kez tanımlanmış, ölçtüğümüz yerde sayısıyla birlikte."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve
> kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi
> veya production verisi kullanılmamıştır.

Terimler, onları ilk kullanan modüle göre gruplanmıştır. Bir şeyi ölçtüysek sayısı da burada —
kontrol edemediğin tanımı yanlış hatırlarsın.

## Model ve eğitim

**Ağırlık (parametre)** — modelin yapıldığı sayılar. Modül 2'deki rakam sınıflandırıcısında
101.770 tane var. Modelin bildiği her şey bu sayıların değerlerinin bir sonucu ve eğitim
bittikten sonra bir daha değişmiyorlar. Fine-tune edilmiş bir modelin bayatlamasının sebebi bu:
bir sayı yığınının içinde, dünyanın değiştiğini öğrenmeye yarayan bir mekanizma yok.

**Training (eğitim)** — ağırlıkları ayarlayan döngü: girdiyi ileri çalıştır, çıktının ne kadar
yanlış olduğunu ölç, her ağırlığın hangi yöne gitmesi gerektiğini hesapla, biraz oynat, tekrarla.
Modül 2 bunu 60.000 görüntüyle beş kez, 0.84 saniyede yapıyor.

**Loss** — modelin şu anda ne kadar yanlış olduğunu söyleyen tek sayı. Eğitim, onu küçültme
sürecinin adı. Bizimki ilk batch'te 2.35'ten son batch'te 0.03'e düşüyor.

**Epoch** — eğitim verisinin tamamı üzerinden bir geçiş. Öğrenmenin çoğu ilkinde oluyor:
doğruluğumuz %9.9'dan bir epoch sonra %95.4'e fırlıyor, kalan dört epoch'ta %97.5'e sürünüyor.

**Fine-tuning** — halihazırda eğitilmiş bir modeli kendi verinle daha ileri eğitmek. Çalışıyor,
ve sonuç yine donmuş oluyor: Helios modelimiz EUR 120 diyor çünkü 2026-Q2 kural kitabı öyle
diyordu; kural EUR 90'a düştükten sonra da EUR 120 demeye devam edecek.

**LoRA** — Low-Rank Adaptation. Büyük bir ağırlık matrisini güncellemek yerine onu donduruyorsun
ve çarpımları o matrise eklenen çok daha küçük iki matris öğreniyorsun. Rank 16'da bu, eğitilebilir
sayıların kabaca %0.1'i; fine-tune tek GPU'ya sığıyor ve birkaç megabyte olarak dağıtılıyor.

**QLoRA** — dondurulmuş base modelin 4 bit'e quantize edildiği LoRA. Base sadece okunuyor, hiç
yazılmıyor; bu yüzden hassasiyet kaybı beklediğinden ucuza geliyor ve 7B'lik bir model tüketici
donanımında eğitilebilir hâle geliyor.

**RLHF / DPO** — cevaplar üzerinden değil, karşılaştırmalar üzerinden eğitim: iki cevap ve insanın
hangisini tercih ettiği. RLHF ayrı bir ödül modeli fit edip ona karşı optimize ediyor; DPO
karşılaştırmayı doğrudan optimize edip ödül modelini atlıyor — çoğu ekibin önce ona uzanmasının
sebebi bu.

**Quantization** — ağırlıkları daha düşük hassasiyette (16, 8, 4 bit) saklamak; az bir kaliteyi
çok bellek karşılığında takas etmek. Bu eğitimde lokal koşan her model quantize edilmiş.

**GGUF** — Ollama'nın model servis ettiği dosya formatı. Fine-tune edilmiş bir modeli GGUF'a
çevirmek, katılımcıların onu GPU'suz ve HuggingFace'ten hiçbir şey indirmeden koşturmasını sağlayan
şey.

**Context window** — modelin tek seferde okuyabildiği token sayısı. `qwen2.5:3b`'de 32.768. Bizim
28 dokümanlık korpusumuz yaklaşık 26.000 — tüm korpusu prompt'a doldurmanın burada çalışmasının ve
on katı boyutta çalışmamasının sebebi bu.

**Token** — modelin metni okuduğu birim; karışık İngilizce/Türkçe metinde kabaca üç karakter.
Kelime değil, karakter de değil.

**Halüsinasyon** — hiçbir şeye dayanmayan, akıcı ve kendinden emin cevap. Modül 1 aynı soruyu dört
farklı şekilde soruyor ve 1.500 TL, 100-200 euro, %10-20 ve bir ret alıyor. Gerçek cevap EUR 90.
İşaret, cevabın yanlış olması değil — **sayının oynaması**.

## Retrieval

**Embedding** — bir metnin, anlamı yakın olanlar birbirine yakın düşecek şekilde konumlanmış sayı
listesine çevrilmesi. `bge-m3` metin başına 1024 sayı üretiyor, `nomic-embed-text` 768,
`all-MiniLM-L6-v2` 384. Daha çok sayı otomatik olarak daha iyi demek değil.

**Cosine similarity** — iki embedding vektörü arasındaki açı; "bunlar ne kadar ilgili" sorusunun
vekili. Semantic search denen her şey bu ölçüm artı bir sıralama.

**Multilingual embedder** — farklı dillerde aynı anlamı taşıyan metinlerin aynı bölgeye düşmesi
için eğitilmiş model. Günü belirleyen fark bu: cevabı İngilizce dokümanda olan Türkçe sorularda
`nomic-embed-text` **0.000**, `bge-m3` **0.667** alıyor.

**Chunk** — dokümanın gerçekten indekslediğin dilimi. Retrieval hiçbir zaman dokümanı görmez;
senin kestiğin şeyi görür. Tüm dokümandan structure-aware chunk'a geçmek hit@1'i 0.550'den
0.800'e taşıyor — bu eğitimdeki en büyük tek iyileşme.

**Fixed-size chunking** — içeriğe bakmadan her N karakterde kes. Ortalamayı yükseltiyor, tam-token
retrieval'ı yarıya düşürüyor (1.000 → 0.500), ve bir tablonun kolon başlığını satırlarından
ayırıyor.

**Overlap** — her chunk'ın bir öncekinin kuyruğunu taşıması. Bir cümleyi ortadan bölmeye karşı
sigorta. 900 karakter öteye yapılan bir referansı düzeltmiyor: sınırı kaldırmıyor, kaydırıyor.

**Recursive chunking** — önce paragraflardan, sonra satırlardan, sonra cümlelerden böl. Herkesin
varsayılanı, ve bizim ceza tablomuzu hâlâ başlığından koparıyor — çünkü tablo, bölme boyutundan
geniş.

**Structure-aware chunking** — dokümanın kendi başlıklarından ve numaralı kurallarından böl, ve
her chunk'a nereden geldiğini önek olarak yaz. Başlığı satırla birlikte tutan tek strateji.
Başka yerlerde *contextual retrieval* adıyla satılıyor.

**Boilerplate temizliği** — dokümanlar arasında tekrarlanan legal footer'ları, sayfa artıklarını
ve artık markup'ı silmek. Korpusumuzun %4.2'sini siliyor ve hit@1'i 0.800'den 0.850'ye taşıyor.

**BM25** — terim frekansı, ters doküman frekansı ve doküman uzunluğuna göre keyword sıralaması.
Model yok, eğitim yok. Tüm dokümanlar üzerinde tam tanımlayıcılarda embedding'i yeniyor; chunk'a
geçince 0.300'e çöküyor, çünkü kısa chunk'lar uzunluk normalizasyonuna çalışacak bir şey bırakmıyor.

**Sparse vs dense retrieval** — sparse, BM25 ve akrabaları: kelime eşliyor. Dense, embedding:
anlam eşliyor. Farklı şeylerde başarısız oluyorlar; bunları birleştirme fikrinin tüm dayanağı bu,
ve birleştirmeyi varsaymak yerine ölçmenin sebebi de bu.

**RRF (Reciprocal Rank Fusion)** — birkaç sıralamayı 1/(k + sıra) toplayarak birleştirmek. Bütün
girdi sıralamalarının hemfikir olduğu dokümanı ödüllendiriyor, ki bu ancak her girdi sağlamsa
işe yarar. BM25'i dense retrieval'ımıza kattığımızda hit@1 0.800'den 0.450'ye düştü.

**Rerank** — ilk birkaç sonucu ikinci ve daha yavaş bir modelle yeniden sıralamak. Burada bir
yükseltme değil **takas** olarak ölçüldü: zayıf iki retrieval kurulumunu ~0.450'ye çekti, güçlü
ikisini ~0.650'ye indirdi. Düzlüyor, ve **kendi tavanına** düzlüyor.

**Pointwise vs listwise rerank** — her adayı tek tek puanlamak mı, tüm listeyi sıraya dizmesini
istemek mi. Aynı model, aynı adaylar, zıt sonuçlar: prob korpusunda listwise 2/5, pointwise 5/5.
Nasıl sorduğun, sorup sormadığından daha önemli.

**RAG** — retrieval-augmented generation. İlgili metni soru anında bul, prompt'a koy, ondan
cevapla. Amaç tüm korpusu doldurmaktan daha iyi cevaplamak değil — burada ölçüldü, öyle değil —
korpus yüz kat büyüdüğünde de çalışması ve cevabın okuduğu dosyayı gösterebilmesi.

**Agentic RAG** — retrieval'ın, modelden önce koşan bir adım değil, modelin **çağırdığı bir araç**
olması. Model soruyu parçalıyor, arıyor, yeterli mi diye bakıyor ve tekrar arıyor. En zor multi-hop
sorumuzda gereken 3 dokümanın 1'inden 3'üne çıkardı; bedeli altı-on model çağrısı ve determinizmin
tamamen kaybı.

**Multi-hop** — cevabı, tek bir aramanın getiremeyeceği kadar çok dokümana yayılmış soru. Bizimki
aynı anda misconnect SOP'unu, interline anlaşmasını ve ücret kurallarını istiyor.

## Ölçüm

**Gold set** — her biri için hangi dokümanın gelmesi gerektiği yazılı, sabit soru listesi. Bizimki
20 soru ve modüller arasında **değişmiyor**; böylece aynı üç sayı günün üç noktasında
karşılaştırılabilir oluyor.

**hit@1** — en üstteki doküman doğru olanlardan biri miydi? En katı ve en dürüst tek sayı.

**recall@5** — doğru kümenin ne kadarı ilk beşte göründü. Dikkat: en **kötü** chunking
stratejimizde en yüksek çıkıyor, çünkü daha çok chunk demek bir yerlerde görünmek için daha çok
şans demek.

**MRR (Mean Reciprocal Rank)** — ilk doğru dokümanın sırasının tersi, ortalaması alınmış. Kısmi
puanı gösteren sayı: bir dokümanı 6. sıradan 2. sıraya taşımak hit@1'i hiç değiştirmez, MRR'ı
0.17'den 0.50'ye taşır.

**LLM-as-judge** — cevapları başka bir modelle puanlamak. Güçlü, ve bu eğitimin sayıları için
**bilerek** kullanılmadı: kota yiyor, yavaş, ve skor koşular arasında oynuyor. Burada raporlanan
her şey deterministik.

## Altyapı

**Vector database** — embedding'leri en yakın komşu araması için indeksleyen depo. Burada ChromaDB.
Bilinmesi gereken: ChromaDB'nin varsayılan embedder'ı sessizce indirilen `all-MiniLM-L6-v2` ve
sadece İngilizce — Türkçe korpusta hata vermiyor, yanlış doküman döndürüyor.

**ANN / HNSW** — yaklaşık en yakın komşu araması. Milyonlarca vektörde tam arama çok yavaş; bunlar
az bir recall'u çok hız karşılığında takas ediyor. Buradaki "yaklaşık" bir özür değil, bir ayar
düğmesi.

**Embedded vs server modu** — ChromaDB'nin kendi process'inin içinde kütüphane olarak çalışması mı,
HTTP üzerinden konuştuğun bir servis olması mı. Aynı API, farklı operasyonel sahiplik. Server'ın
yolu `/api/v2/`, v1 değil.

**Ollama** — modelleri lokal koşturup `localhost:11434` üzerinden servis ediyor. Bu eğitim her şey
için onu kullanıyor; günün API key'siz, cloud hesapsız ve girişsiz çalışmasını sağlayan şey bu.

**Podman** — rootless ve daemon'suz container runtime. ChromaDB server'ı için Docker yerine bu
kullanılıyor: ayrıcalıklı bir daemon istemiyor ve kurumsal ortamda lisans sorusu doğurmuyor.
