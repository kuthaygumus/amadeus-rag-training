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
Modül 2 bunu 60.000 görüntüyle beş kez, 0.85 saniyede yapıyor.

**Loss** — modelin şu anda ne kadar yanlış olduğunu söyleyen tek sayı. Eğitim, onu küçültme
sürecinin adı. Bizimki ilk batch'te 2.35'ten son batch'te 0.03'e düşüyor.

**Epoch** — eğitim verisinin tamamı üzerinden bir geçiş. Öğrenmenin çoğu ilkinde oluyor:
doğruluğumuz %9.87'den bir epoch sonra %95.35'e fırlıyor, kalan dört epoch'ta %97.47'ye sürünüyor.

**Fine-tuning** — halihazırda eğitilmiş bir modeli kendi verinle daha ileri eğitmek. Çalışıyor,
ve sonuç yine donmuş oluyor: modül 3'teki adapter 2026-Q2 kural kitabıyla eğitiliyor ve orada
CLASSIC K iptal cezası EUR 120. 2026-Q3 kitabı aynı satırı EUR 90 fiyatlıyor; eğitilmiş
ağırlıkların içinde o satırın değiştiğini fark edebilecek hiçbir mekanizma yok.

**LoRA** — Low-Rank Adaptation. Büyük bir ağırlık matrisini güncellemek yerine onu donduruyorsun
ve çarpımları o matrise eklenen çok daha küçük iki matris — *adapter* — öğreniyorsun. **Rank**
(`r`) kapasite: güncellemenin kaç bağımsız yönde hareket edebileceği. **Alpha** ölçek: adapter'ın
çıktısı `alpha/r` ile çarpılıyor, yani rank'i yükseltirken güncellemenin şiddetini yükseltmek
zorunda kalmıyorsun. Modül 3, Qwen2.5-1.5B üzerinde `r = 32, alpha = 64` eğitiyor; bu modelin
`q_proj`'ü 1536×1536. Yani 2 × 1536 × 32 = 98.304 eğitilebilir sayı, tam güncellemenin
2.359.296'sına karşı — **%4.17** — ve modelin 1.580.643.840 parametresinin 36.929.536'sı, **%2.34**.
Fine-tune'un tek GPU'ya sığmasının ve birkaç megabyte olarak dağıtılmasının sebebi bu.

**QLoRA** — dondurulmuş base modelin 4 bit'e quantize edildiği LoRA. Base sadece okunuyor, hiç
yazılmıyor; bu yüzden hassasiyet kaybı beklediğinden ucuza geliyor ve 7B'lik bir model tüketici
donanımında eğitilebilir hâle geliyor.

**RLHF / DPO** — cevaplar üzerinden değil, karşılaştırmalar üzerinden eğitim: iki cevap ve insanın
hangisini tercih ettiği. RLHF ayrı bir ödül modeli fit edip ona karşı optimize ediyor; DPO
karşılaştırmayı doğrudan optimize edip ödül modelini atlıyor — çoğu ekibin önce ona uzanmasının
sebebi bu.

**Quantization** — ağırlıkları daha düşük hassasiyette (16, 8, 4 bit) saklamak; az bir kaliteyi
çok bellek karşılığında takas etmek. Buradaki chat modelleri 4 bit — `ollama show qwen2.5:3b`
`Q4_K_M` yazıyor. İki embedder değil: `bge-m3` de `nomic-embed-text` de `F16` geliyor.

**GGUF** — Ollama'nın model servis ettiği dosya formatı. Fine-tune edilmiş bir modeli GGUF'a
çevirmek, katılımcıların onu GPU'suz ve HuggingFace'ten hiçbir şey indirmeden koşturmasını sağlayan
şey.

**Modelfile** — bir GGUF dosyasını Ollama'ya bir adla kaydeden ve varsayılanlarını sabitleyen birkaç
satır. Modül 3'ünki `temperature 0` sabitliyor; kurulan modele aynı soruyu on kez sorup on aynı
cevaptan başka bir şey almak, fine-tune'un tutmadığı anlamına geliyor.

**Context window** — modelin tek seferde okuyabildiği token sayısı. `qwen2.5:3b`'de 32.768. Bizim
28 dokümanlık korpusumuz 78.310 karakter, kabaca 26.000 token — tüm korpusu prompt'a doldurmanın
burada çalışmasının ve on katı boyutta çalışmamasının sebebi bu.

**Token** — modelin metni okuduğu birim; karışık İngilizce/Türkçe metinde kabaca üç karakter.
Kelime değil, karakter de değil.

**Temperature** — modelin bir sonraki token'ı seçerken ne kadar rastgelelik kullanabildiği. Bu
eğitimdeki her üretim `temperature=0.0` ile koşuyor, yani aynı prompt iki kez aynı cevabı veriyor.
Sampling gürültüsünü kaldırıyor. Yanlış bir cevabı doğru yapmıyor.

**Halüsinasyon** — hiçbir şeye dayanmayan, akıcı ve kendinden emin cevap. Modül 1 aynı soruyu dört
farklı şekilde soruyor ve temperature 0'da tek bir kayıtlı koşuda 1.500 TL, 100-200 euro, %10-20
ve bir ret alıyor. Gerçek cevap EUR 90. İşaret, cevabın yanlış olması
değil — **sayının oynaması**.

## Retrieval

**Embedding** — bir metnin, anlamı yakın olanlar birbirine yakın düşecek şekilde konumlanmış sayı
listesine çevrilmesi. `bge-m3` metin başına 1024 sayı üretiyor, `nomic-embed-text` 768,
`all-MiniLM-L6-v2` 384. Daha çok sayı otomatik olarak daha iyi demek değil.

**Cosine similarity** — iki embedding vektörü arasındaki açının kosinüsü: aynı yöne bakıyorlarsa
1.0, ilgisizlerse 0. "Bunlar ne kadar ilgili" sorusunun vekili. Semantic search denen her şey bu
ölçüm artı bir sıralama.

**Multilingual embedder** — farklı dillerde aynı anlamı taşıyan metinlerin aynı bölgeye düşmesi
için eğitilmiş model. Günü belirleyen fark bu: cevabı İngilizce dokümanda olan Türkçe sorularda
`nomic-embed-text` **0.000**, `bge-m3` **0.667** alıyor.

**Chunk** — dokümanın gerçekten indekslediğin dilimi. Retrieval hiçbir zaman dokümanı görmez;
senin kestiğin şeyi görür. Tüm dokümandan structure-aware chunk'a geçmek hit@1'i 0.550'den
0.800'e taşıyor. Aynı chunk'lar üzerinde embedder daha da fazlasını değiştiriyor —
`nomic-embed-text` 0.350, `bge-m3` 0.800 — günün ikisini de ölçmesinin sebebi bu.

**Fixed-size chunking** — içeriğe bakmadan her N karakterde kes. Ortalamayı yükseltiyor
(hit@1 0.550 → 0.700), tam-token retrieval'ı düşürüyor (1.000 → 0.750), ve bir tablonun kolon
başlığını satırlarından ayırıyor.

**Overlap** — her chunk'ın bir öncekinin kuyruğunu taşıması. Bir cümleyi ortadan bölmeye karşı
sigorta. 612 karakter öteye yapılan bir referansı düzeltmiyor — ceza tablomuzun kolon başlığı ile
`K` satırı arasındaki mesafe bu: sınırı kaldırmıyor, kaydırıyor.

**Recursive chunking** — önce paragraflardan, sonra satırlardan, sonra cümlelerden böl. Herkesin
varsayılanı, ve bizim ceza tablomuzu hâlâ başlığından koparıyor — çünkü o aralık 612 karakter,
bölme boyutu ise 600.

**Structure-aware chunking** — dokümanın kendi başlıklarından ve numaralı kurallarından böl, ve
her chunk'a nereden geldiğini önek olarak yaz. Başlığı satırla birlikte tutan tek strateji.
Başka yerlerde *contextual retrieval* adıyla satılıyor.

**Boilerplate temizliği** — dokümanlar arasında tekrarlanan legal footer'ları, sayfa artıklarını
ve artık markup'ı silmek. Korpusumuzun %4.2'sini siliyor ve hit@1'i 0.800'den 0.850'ye taşıyor.

**BM25** — terim frekansı, ters doküman frekansı ve doküman uzunluğuna göre keyword sıralaması.
Model yok, eğitim yok. Tüm dokümanlar üzerinde dense retrieval'a yakın ama hiçbir yerde onun
önünde değil: hit@1 0.400'e karşı 0.550, tam tanımlayıcılarda 0.750'ye karşı 1.000. Onu kıran şey
chunking: nasıl kesersen kes 0.300, çünkü kısa chunk'lar uzunluk normalizasyonuna çalışacak bir şey
bırakmıyor. Altı Türkçe soruda ise, kessen de kesmesen de 0.000 alıyor.

**Sparse vs dense retrieval** — sparse, BM25 ve akrabaları: kelime eşliyor. Dense, embedding:
anlam eşliyor. Farklı şeylerde başarısız oluyorlar; bunları birleştirme fikrinin tüm dayanağı bu,
ve birleştirmeyi varsaymak yerine ölçmenin sebebi de bu.

**RRF (Reciprocal Rank Fusion)** — birkaç sıralamayı 1/(k + sıra) toplayarak birleştirmek. Bütün
girdi sıralamalarının hemfikir olduğu dokümanı ödüllendiriyor, ki bu ancak her girdi sağlamsa
işe yarar. BM25'i dense retrieval'ımıza kattığımızda hit@1 0.800'den 0.450'ye düştü.

**Rerank** — ilk birkaç sonucu ikinci ve daha yavaş bir modelle yeniden sıralamak. Burada bir
yükseltme değil **takas** olarak ölçüldü: dört retrieval kurulumu üzerinde zayıf ikisini yukarı
çekti (hit@1 0.350 → 0.450 ve 0.350 → 0.400), güçlü ikisini aşağı indirdi (0.700 → 0.550 ve
0.800 → 0.600). Düzlüyor, ve **kendi tavanına** düzlüyor.

**Bi-encoder** — sorgu ve pasaj ayrı ayrı embed ediliyor, her birine bir vektör, karşılaştırma
cosine ile. Pasaj vektörleri bir kez hesaplanıp saklanabiliyor; `bge-m3` ile dense retrieval'ı
sorgu anında ucuz yapan şey bu. Bu sayfada "embedder" dendiğinde kastedilen şey.

**Cross-encoder** — sorgu ve pasaj tek bir forward pass'te **birlikte** okunuyor ve doğrudan bir
ilgililik skoru çıkıyor; prompt'la göreve ikna edilmiş değil, ilgililik etiketleriyle eğitilmiş.
Hiçbir şey önceden hesaplanamıyor, bu yüzden ancak kısa bir aday listesi üzerinde koşuyor. Çok
dilli olanı `bge-reranker-v2-m3`. Modül 9'da ölçülen reranker pasaj puanlayan bir chat modeli,
cross-encoder değil — cross-encoder'ı ölçmedik, dolayısıyla bu sayfadaki hiçbir sayı ona ait
değil.

**Pointwise vs listwise rerank** — her adayı tek tek puanlamak mı, tüm listeyi sıraya dizmesini
istemek mi. Aynı model, aynı adaylar, zıt sonuçlar: emekliye ayrılan 10 dokümanlık prob korpusunda
listwise 2/5, pointwise 5/5, MRR 0.600'e karşı 1.000. Bu `n = 5`: yönü gerçek, büyüklüğü
kanıtlanmamış kabul et. Nasıl sorduğun, sorup sormadığından daha önemli.

**RAG** — retrieval-augmented generation. İlgili metni soru anında bul, prompt'a koy, ondan
cevapla. Amaç tüm korpusu doldurmaktan daha iyi cevaplamak değil — burada ölçüldü, öyle değil —
korpus yüz kat büyüdüğünde de çalışması ve cevabın okuduğu dosyayı gösterebilmesi.

**Agentic RAG** — retrieval'ın, modelden önce koşan bir adım değil, modelin **çağırdığı bir araç**
olması. Model soruyu parçalıyor, arıyor, yeterli mi diye bakıyor ve tekrar arıyor. En zor multi-hop
sorumuzda gereken 3 dokümanın 0'ından 3'üne çıkardı; diğer multi-hop soruda hiçbir şey değişmedi,
iki durumda da 3'te 2. Bedeli, naive RAG'in bir çağrı yaptığı yerde altı model çağrısı — döngünün
şeklinden çıkan aritmetik, ölçülmüş bir süre değil — ve determinizmin tamamen kaybı.

**Multi-hop** — cevabı, tek bir aramanın getiremeyeceği kadar çok dokümana yayılmış soru. Bizimki
aynı anda misconnect SOP'unu, interline anlaşmasını ve ücret kurallarını istiyor.

## Ölçüm

**Gold set** — her biri için hangi dokümanın gelmesi gerektiği yazılı, sabit soru listesi. Bizimki
20 soru ve modüller arasında **değişmiyor**; böylece aynı üç sayı günün üç noktasında
karşılaştırılabilir oluyor.

**hit@1** — en üstteki doküman doğru olanlardan biri miydi? En katı ve en dürüst tek sayı.

**Top-k** — bir aşamanın kaç sonucu bir sonrakine aktardığı. `hit@1` k = 1'de, `recall@5` k = 5'te
ölçülüyor; reranker ise ilk 8 adayı puanlıyor — sorgu başına 8 model çağrısı buradan geliyor.
k'sı yazılmadan aktarılan bir metrik sayı değildir.

**recall@5** — doğru kümenin ne kadarı ilk beşte göründü. Dikkat: en **kötü** chunking
stratejimizde en yüksek çıkıyor — fixed-280'de 0.950, structure-aware'de 0.833 — çünkü 294 chunk,
gold dokümana bir yerlerde görünmek için 154'ten daha çok şans veriyor.

**MRR (Mean Reciprocal Rank)** — ilk doğru dokümanın sırasının tersi, sorular üzerinden ortalaması
alınmış. Kısmi puanı gösteren sayı: tek bir sorunun gold dokümanını 6. sıradan 2. sıraya taşımak
hit@1'i hiç değiştirmez, o sorunun katkısını 0.17'den 0.50'ye çıkarır.

**LLM-as-judge** — cevapları başka bir modelle puanlamak. Yaygın, ve bu eğitimin sayıları
için **bilerek** kullanılmadı: kota yiyor, yavaş, ve skor koşular arasında oynuyor. Burada
raporlanan her şey deterministik.

## Altyapı

**Vector database** — embedding'leri en yakın komşu araması için indeksleyen depo. Burada ChromaDB.
Bilinmesi gereken: ChromaDB'nin varsayılan embedder'ı sessizce indirilen `all-MiniLM-L6-v2` ve
sadece İngilizce — Türkçe korpusta hata vermiyor, yanlış doküman döndürüyor.

**Collection** — ChromaDB'nin depolama birimi: id'ler, dokümanlar, embedding'ler ve item başına bir
metadata sözlüğü. `where={"kind": "fare"}` vektör aramasından önce bu metadata üzerinde filtreliyor;
geçersiz kılınmış bir SOP'u cevabın dışında tutmanın yolu bu.

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
