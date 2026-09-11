---
title: "1. Çıplak LLM Duvarı"
description: "Model, var olmayan havayolu için cevap vermeyi reddediyor; sektörün geneli için bir ceza uyduruyor; sonra aynı soruya üç farklı tutar söylüyor — hepsi de tek bir sesle."
---

## Gate sorusu

> **Model benim verimi biliyor mu?**

<div class="presenter-note">
Hiçbir şey çalıştırmadan önce soruyu ekrana yansıt ve salondan tahminlerini kâğıda yazıp bağlanmalarını iste: 3B'lik bir model Kraken Air'in iptal cezası sorusuna cevap verir mi, yoksa reddeder mi? El kaldırt, sesli say. Kayıtlı çalıştırma iki tarafı da yarı yarıya haklı çıkarıyor — ilk seferde reddediyor, ifade değişir değişmez bir lira tutarı söylüyor — ve modülün tamamı bu geri dönüşün üzerine kurulu. İki dakika, fazlası değil.
</div>

Laptopta bir model. Retrieval yok, doküman yok, sadece ağırlıklar. Bir Kraken Air temsilcisinin günde on kez aldığı soruyu soruyoruz.

**Aşağıdakilerin hepsi bir döküm — burada yazacağın bir şey yok.** Her prompt, `notebooks/00_bare_llm_fails.py` içindeki bir Python string'i; o dosyayı VS Code'da açıp blokları `Shift+Enter` ile tek tek çalıştırıyorsun ve `R.generate()` string'i yerel Ollama'ya gönderip dönen cevabı yazdırıyor. Şimdi kayıtlı çalıştırmayı okuyorsun; aşağıdaki *Ne çalıştırıyorsun* bölümünde kendin üreteceksin.

**`q1`:** `Kraken Air'in CLASSIC ucret ailesinde K sinifi bir bileti iptal edersem ne kadar ceza oderim?`
**qwen2.5:3b:** `... belirtilmemiştir. Bu bilgi genellikle kargo veya hizmet koşullarına göre bulunan bir belge veya web sitesinden bulunabilir.`

Reddetti ve bu doğru karar gibi duruyor — gerekçesini okuyana kadar. Yolcu ücret kuralı *kargo veya hizmet koşulları* altında filing edilmez, üstelik bakılacak bir Kraken Air sitesi de yok. Hiçbir yere danışılmadı; model, sorumlu bir cevabın *biçiminde* metin üretti. Bu biçimi aklında tut, çünkü notebook'un son bloğu onu elinden alıyor.

Gerçek rakam `corpus/2026-Q3/fare_classic_shorthaul.md` içinde, K satırında: **EUR 90**, yolcu ve yön başına sabit bir tutar; yüzde değil. Geçen çeyrekte aynı satırda EUR 120 yazıyordu.

Şimdi daha keskin olan denemeyi yapalım. Kurgusal havayolunu çıkaralım, sektörün geneline soralım.

**`q2`:** `Bir havayolunda K booking class bileti iptal edilirse tipik olarak ne kadar ceza alinir? Somut bir rakam ver.`
**qwen2.5:3b:** `Tipik olarak, K (Business) koltuk sınıfı için 10-20% kadar ceza ödemek zorunludur.`

Bu cevap iki ayrı yerden yanlış. Yüzde uydurma, ama parantez daha kötü: K, havayollarının ezici çoğunluğunda indirimli bir economy booking class'tır; bizim corpus'ta da economy. J ve Y'nin altındaki harfleri her taşıyıcı kendi filing'inde belirler, yani bu bir standart değil güçlü bir teamül — hiç çekince koymadan söylenmiş bir `(Business)`'ın yanlış olmasının sebebi de tam olarak bu. Yani mesele "modelin bizim özel verimiz eksik" değil. Model, herkese açık kısımda da kendinden emin bir şekilde yanlış.

<div class="presenter-note">
Şu cümleyi yüksek sesle söyle ve bir saniye beklet: "Sadece bizim verimizi kaçırmadı. Sektörü de yanlış bildi." Salonda birisi zaten aynı soruyu daha büyük bir modele yazıyor olacak, "bakın o doğru yapıyor" demek için. Bırak yapsın — daha büyük model K'yi daha sık doğru bilir, ama bu çeyreğin rakamının 120 değil 90 olduğunu yine bilemez. Tartışmayı model boyutundan oraya taşı, 3B modeli savunmaya çalışma.
</div>

Şimdi kural yerine somut bir olgu soralım. Aynı model, aynı oturum, context yok.

**`q3`:** `XX 1487 sefer numarali Kraken Air ucusu saat kacta kalkiyor?`
**qwen2.5:3b:** `Bu bilgi ... resmi bir takipçi platformundan (örneğin Twitter) veya resmi web sitesinden alınamaz. Her seferde farklı saat olabilir.`

Yine reddetti ve gerekçesi yine uydurma: tarife bilgisi Twitter'dan okunan bir şey değil ve numaralı bir uçuş her gün farklı saatte kalkmaz. İki ret, ikisi de bir bilgi kontrolü değil — orada hiçbir şey kontrol edilmedi.

Bir dil modeli, öğrenilmiş ağırlıklar üzerinden bir sonraki token'ı tahmin eder: okuduğu her şeyin istatistiksel şeklini sıkıştıran milyarlarca parametre. İçeride lookup tablosu yok, kaynak doküman yok, hiçbir bilginin üstünde tarih yok. Cevaptan cevaba değişen tek şey, sorunun *biçiminin* hangi prior'ı tetiklediği. Sektöre genel bir şey sor: *"havayolu iptal ücretleri ücretin bir yüzdesidir, tipik olarak şu aralıkta"* kalıbı eğitim verisinde on binlerce kez geçiyor, o kalıbın bir örneği çıkıyor. Kalkış saati sor: bu kez *belirli bir tarife saatini söylemekten kaçınan metin* kalıbı kazanıyor. O ret prior'ı büyük ölçüde instruction tuning'den geliyor; *Daha derine* bölümü konuyu oradan sürdürüyor.

Yani prior, bilgiye değil ifadeye göre oturmuş. `q1`'deki kibar reddin hiçbir değeri olmamasının sebebi bu.

**Tehlike modelin yanılması değil. Tehlike, yanılırken doğru bildiği zamankiyle tıpatıp aynı sesi kullanması.** Aynı akıcılık, aynı özgüven, aynı kaynaksızlık. Ret, uydurma yüzde ve aşağıdaki lira aralığı tek bir tonda yazılmış; çıktının hiçbir yerinde hangisini elinde tuttuğunu söyleyen bir işaret yok.

Duvar bu. Bundan sonrası bu duvarı aşma denemesi: daha iyi sormak, modeli kendi kurallarımızla eğitmek, ya da kuralları cevap anında modelin önüne koymak. Bugün üçünü de bu sırayla deniyoruz ve corpus önümüzdeki çeyrekte yeniden yayımlandığında sadece biri ayakta kalıyor.

## Ne çalıştırıyorsun

**Terminal (repo kökü)** — `corpus/`, `notebooks/`, `eval/` ve `exercises/` klasörlerini içeren dizin:

```bash
ollama serve                       # zaten çalışmıyorsa
python scripts/verify_setup.py     # READY yazdırmalı
```

**VS Code — `notebooks/00_bare_llm_fails.py`, ilk `# %%` bloğu.** İmleci bloğun içine koy ve `Shift+Enter`'a bas; çıktı Interactive penceresinde beliriyor.

```python
import sys
sys.path[:0] = [".", "notebooks"]              # the helpers sit next to this file
import _preflight; _preflight.ready(chat=True)  # stops with instructions if a model is missing
import retrieval as R

print("model:", R.CHAT_MODEL)
```

`R.generate()` yerel Ollama'ya `localhost:11434` üzerinden `temperature=0.0` ile gidiyor. `Shift+Enter`'a basarak dosyanın sonuna kadar ilerle.

**Ne göreceksin, yaklaşık iki dakikada.** Önce `model: qwen2.5:3b` ve tek kelimelik bir cevap, sonra `q1`, `q2` ve `q3` — ret, uydurma yüzde, ikinci ret. Ardından son blok ceza sorusunu dört kez daha, yalnızca ifadeyi değiştirerek soruyor ve dört cevabı alt alta yazdırıyor. Senin kurduğun cümleler yukarıdaki dökümlerden kayacak; önce-reddet-sonra-uydur kalıbı kaymayacak.

**Ollama henüz cevap vermiyorsa.** Geri düşülecek kayıtlı çalıştırması olmayan tek notebook bu — her hücre canlı bir çağrı. Yukarıdaki dökümleri oku, yanındakiyle eşleş ve modül 2'de aramıza katıl; o modülün modele hiç ihtiyacı yok.

<div class="presenter-note">
Ollama kapalıysa veya pull hâlâ sürüyorsa sahnede debug etme. Dökümlerin hepsi bu sayfada duruyor — oku, "bu sabah benim makinemde böyle çıktı, lab bloğunda kendiniz üreteceksiniz" de ve devam et. Modülün toplam süresi 12 dakika, canlı hücreler bunun 2 dakikası — kalanı anlatı ve el kaldırma. Biri farklı bir ifade aldığını söylerse bu beklenen bir şey ve tek cümlelik cevabı var: ifade değişir, kalıp değişmez.
</div>

## Sayılar ne dedi

<div class="measured">

Notebook'un bittiği kontrol. Aynı ceza sorusu, yalnızca ifade değiştirilerek dört kez soruldu:

| nasıl soruldu | ne döndü |
|---|---|
| `Kraken Air CLASSIC K sinifi iptal cezasi ne kadar?` | `Genellikle 100-250 TL arasında` — bir lira aralığı |
| `Kraken Air'de CLASSIC ucret ailesi, K booking class. Iptal cezasi kac euro?` | `genellikle 100-200 TL arasında değişebilir` — euro soruldu, lira cevaplandı |
| `Musterim Kraken Air CLASSIC K bileti aldi ve iptal etmek istiyor. Ne odeyecek?` | `genellikle 2-5 gün içinde ücret ödemek zorunda kalabilir` — bir süre, tutar değil |
| `Kraken Air CLASSIC K cancellation penalty amount?` | `can vary and is subject to change ... contact Kraken Air directly` — İngilizce bir ret |

Dört soruş, dört farklı davranış: birbirini tutmayan iki lira aralığı, tutar bile olmayan bir cevap ve bir ret. Hiçbiri EUR 90 değil; üstelik ikinci soruş `kac euro?` diye açıkça yazdığı hâlde cevap lira olarak geldi. Bilgiye gerçekten sahip olan bir model dört kez EUR 90 derdi. Bu, tek bir makinede `temperature=0.0` ile alınmış tek bir kayıtlı çalıştırma; sende çıkan tutarlar farklı olabilir, zaten birbirlerinden farklı olmaları asıl mesele.

</div>

## Daha derine

Ağırlıkları depolama değil sıkıştırma olarak düşün. Eğitim, bir corpus'u sabit bir parametre bütçesine sıkıştırır; sık geçen ve genel olan yüksek doğrulukla hayatta kalır, spesifik ve nadir olan hatırlanmaz, yeniden üretilir. Hallucination, eğitim verisinin hiç kısıtlamadığı bir bölgeden çekilmiş kendinden emin bir tahmindir. "İptal cezaları ücretin bir yüzdesidir" geneldir. "2026-Q3 kısa menzil CLASSIC sayfasında K booking class için EUR 90" ise bir bilginin olabileceği kadar spesifiktir ve ölçeği ne kadar büyütürsen büyüt bu çizginin öbür tarafına geçmez — daha büyük bir model sorunun herkese açık kısmındaki olasılıkları değiştirir, özel kısmına hiç dokunmaz. [Kurulum sayfasındaki](/tr/modules/00-setup/) beş modelli karşılaştırma tam olarak bunu ölçüyor, üstelik önemli olduğu koşulda: context'te zaten bir doküman varken.

Retler genelde hak ettiğinden az ilgi görüyor. Reddetme, büyük ölçüde eğitimle kazandırılmış bir davranış: instruction tuning, özel veya çabuk değişen bir bilginin sorgulanması biçimindeki sorularda "bilmiyorum" demeyi ödüllendiriyor. Yani sinyal sorunun *biçimini* takip ediyor, modelin o bilgiye sahip olup olmadığını değil — `q1`'in reddedip, ifadesi değiştirilmiş `q1`'in lira aralığı üretmesinin sebebi de bu. Prompt ile çizgiyi oynatabilirsin, ama yaptığın şey bir prior'ı ayarlamak, bir bilgi kontrolü kurmak değil. Bunun yerine cevabı bulunmuş bir dokümana dayıyoruz: hem rakam oynuyor mu diye modele beş kez sormaktan ucuz, hem de denetlenebilir — bir havayolunun ihtiyacı olan da denetlenebilirlik.

## Çıkış cümlesi

> Bilmiyor — ve bilmediğini bilmiyor.

<div class="presenter-note">
Ağzında gevelememen gereken tek cümle bu. Yavaş söyle, üstüne bir şey ekleme, açıklama — açıklaması zaten bir sonraki modül. Sonra doğrudan modül 2'ye geç: "Peki az önce bir yüzde uyduran şeyin içinde tam olarak ne var?"
</div>
