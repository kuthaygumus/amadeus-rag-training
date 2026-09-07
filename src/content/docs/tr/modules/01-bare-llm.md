---
title: "1. Çıplak LLM Duvarı"
description: "Model, var olmayan bir havayolu için iptal cezası uyduruyor; sonra bir kalkış saatini tahmin etmeyi reddediyor — ve iki cevap da tıpatıp aynı sesle geliyor."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Model benim verimi biliyor mu?**

<div class="presenter-note">
Hiçbir şey çalıştırmadan önce soruyu ekrana yansıt ve salondan tahminlerini kâğıda yazmalarını iste: 3B'lik bir model Helios ceza sorusuna cevap verir mi, yoksa bilmediğini mi söyler? Çoğu kişi "bilmediğini söyler" der. El kaldırt, sesli say — hücre çalışmadan önce herkesin bir tahmine bağlanmış olmasını istiyorsun, çünkü bu modülün tamamı o sürprizin üzerine kurulu. İki dakika, fazlası değil.
</div>

Laptopta bir model var. Retrieval yok, doküman yok, sadece ağırlıklar. Bir Helios temsilcisinin günde on kez aldığı soruyu soruyoruz.

**Prompt:** `Helios CLASSIC K iptal cezası?`
**qwen2.5:3b:** cezanın ücretin **"%20-30"**'u kadar olduğunu anlatan bir açıklama.

Helios Air diye bir havayolu yok. Bu reponun dışında CLASSIC K diye bir ücret ailesi de yok. Model duraksamadı, kayıt düşmedi, hangi rota bandını sorduğumuzu sormadı. Bir cevabın tonuyla bir yüzde aralığı üretti.

Gerçek rakam `corpus/2026-Q3/fare_classic_shorthaul.md` içinde, K satırında: **EUR 90**, yolcu ve yön başına sabit bir tutar; yüzde değil. Geçen çeyrekte aynı satırda EUR 120 yazıyordu. Yani model hem rakamda yanlış, hem rakamın *biçiminde* yanlış, hem de hangi çeyreği sorduğunu bilmesinin hiçbir yolu yok.

Şimdi daha keskin olan denemeyi yapalım. Kurgusal havayolunu çıkaralım, sektörün geneline soralım.

**Prompt:** `K booking class typical penalty?`
**qwen2.5:3b:** **"K (Business) sınıfı %10-20"**.

Bu cevap iki ayrı yerden yanlış. Yüzde yine uydurma, ama parantez daha kötü: K bir business sınıfı değil. Havayollarının normal filing pratiğinde K indirimli bir economy booking class'tır — bizim corpus'ta da tam olarak öyle. Yani mesele "modelin bizim özel verimiz eksik" değil. Model, herkese açık kısımda da kendinden emin bir şekilde yanlış; dünyaya dair bir olguyu, tek bir çekince koymadan söylüyor.

<div class="presenter-note">
Şu cümleyi yüksek sesle söyle ve bir saniye beklet: "Sadece bizim verimizi kaçırmadı. Sektörü de yanlış bildi." Salonda birisi zaten aynı soruyu daha büyük bir modele yazıyor olacak, "bakın o doğru yapıyor" demek için. Bırak yapsın — daha büyük model K'yi daha sık doğru bilir, ama bu çeyreğin rakamının 120 değil 90 olduğunu yine bilemez. Tartışmayı model boyutundan oraya taşı, 3B modeli savunmaya çalışma.
</div>

Şimdi hatayı netleştiren karşıtlık. Aynı model, aynı oturum, context yok.

**Prompt:** `What time does H9 1487 depart?` → doğru şekilde bilmediğini söyledi.
**Prompt:** `How much is the misconnect meal voucher?` → doğru şekilde bilmediğini söyledi.

Yani model rastgele metin üreten bir şey değil. Tam da bilmemenin doğru cevap olduğu iki soruda, temiz bir şekilde çekildi. Peki neden o ikisinde?

Cevap eğitim metninde. Bir dil modeli, öğrenilmiş ağırlıklar üzerinden bir sonraki token'ı tahmin eder — okuduğu her şeyin istatistiksel şeklini sıkıştıran milyarlarca parametre. İçeride bir lookup tablosu yok, kaynak doküman yok, hiçbir bilginin üstünde tarih yok. İptal cezası sorduğunda, *"havayolu iptal ücretleri ücretin bir yüzdesidir, tipik olarak şu aralıkta"* kalıbı eğitim verisinde on binlerce kez geçiyor; en olası devam da bu kalıbın makul bir örneği oluyor. Boşluğu dolduruyor, çünkü boşluğun hazır bir dolgusu var. H9 1487'nin kalkış saatini sorduğunda uzanacağı bir dolgu yok — ama *belirli bir tarife saatini söylemekten kaçınan metin* kalıbı bol bol var. Çıkan şey o.

Bu çekilme, modelin kendi bilgisini kontrol etmesi değil. Hiçbir yerde bir envanter sorgulanmadı. Dört cevabın dördü de aynı next-token makinesinden çıktı; birinde prior bir rakamı işaret etti, diğerinde bir reddi. Bu reddi "model kendi sınırlarını biliyor" diye okumak, yolcunun karşısına yanlış cezayı çıkaran hatanın ta kendisi.

Modülün bütün derdi de bu.

**Tehlike modelin yanılması değil. Tehlike, yanılırken doğru bildiği zamankiyle tıpatıp aynı sesi kullanması.** Aynı akıcılık, aynı özgüven, aynı kaynaksızlık. Çıktının hiçbir yerinde yukarıdaki dört cevaptan hangisine baktığını söyleyen bir işaret yok. Üstelik EUR 400 civarı bir kısa menzil bileti düşün: "%20-30" demek EUR 80 ile EUR 120 arası demek — gerçek cevap olan EUR 90'ı, dalgın bir kontrolden geçecek kadar yakından kapsıyor ve her seferinde yanlış.

Duvar bu. Bundan sonrası bu duvarı aşma denemesi: daha iyi sormak, modeli kendi kurallarımızla eğitmek, ya da kuralları cevap anında modelin önüne koymak. Bugün üçünü de bu sırayla deniyoruz ve corpus önümüzdeki çeyrekte yeniden yayımlandığında sadece biri ayakta kalıyor.

## Ne çalıştırıyorsun

Notebook: `00_bare_llm_fails.ipynb`.

```bash
ollama serve              # ayrı bir terminalde, çalışmıyorsa
ollama pull qwen2.5:3b
python scripts/verify_setup.py
```

Notebook içinde dört deneme ortak yardımcı fonksiyondan geçiyor — framework yok, API key yok:

```python
from eval.retrieval import generate

generate("Helios CLASSIC K iptal cezası?")
generate("K booking class typical penalty?")
generate("What time does H9 1487 depart?")
generate("How much is the misconnect meal voucher?")
```

`generate()` yerel Ollama'ya `localhost:11434` üzerinden `temperature=0.0` ile gidiyor. Cümle kuruluşun yukarıdaki dökümden biraz kayabilir; iki uydurma ve iki çekilme yerinde duruyor.

<div class="presenter-note">
Ollama kapalıysa veya pull hâlâ sürüyorsa sahnede debug etme. Dört dökümün de bu sayfada duruyor — oku, "bu sabah benim makinemde böyle çıktı, lab bloğunda kendiniz üreteceksiniz" de ve devam et. Modülün toplam süresi 20 dakika, canlı hücreler bunun 4 dakikası. Biri farklı bir ifade aldığını söylerse bu beklenen bir şey ve tek cümlelik cevabı var: sampling değişir, kalıp değişmez.
</div>

## Sayılar ne dedi

<div class="measured">

| prompt (context yok, `qwen2.5:3b`) | verdiği cevap | doğru mu? |
|---|---|---|
| `Helios CLASSIC K iptal cezası?` | uydurma "%20-30 ceza" | hayır — doğrusu EUR 90, sabit tutar |
| `K booking class typical penalty?` | "K (Business) sınıfı %10-20" | hayır — uydurma, üstelik K business değil |
| `What time does H9 1487 depart?` | bilmediğini söyledi | evet |
| `How much is the misconnect meal voucher?` | bilmediğini söyledi | evet |

İki hallucination, iki doğru çekilme, tek bir ses tonu.

</div>

## Daha derine

Ağırlıkları depolama değil sıkıştırma olarak düşün. Eğitim, bir corpus'u sabit bir parametre bütçesine sıkıştırır; sık geçen ve genel olan yüksek doğrulukla hayatta kalır, spesifik ve nadir olan hatırlanmaz, yeniden üretilir. Hallucination dediğimiz şey, eğitim verisinin hiç kısıtlamadığı bir bölgeden gelen kendinden emin bir örnek. "İptal cezaları ücretin bir yüzdesidir" geneldir. "2026-Q3 kısa menzil CLASSIC sayfasında K booking class için EUR 90" ise bir bilginin olabileceği kadar spesifiktir. Ölçek büyütmek bizim corpus'u bu çizginin öbür tarafına geçirmiyor.

Çekilmeler genelde hak ettiğinden az ilgi görüyor. Reddetme davranışı büyük ölçüde eğitilmiş bir davranış: instruction tuning, özel veya çabuk değişen bir bilginin sorgulanması *biçimindeki* sorularda "bilmiyorum" demeyi ödüllendiriyor. Yani elindeki sinyal sorunun *biçimiyle* ilişkili, modelin o bilgiye gerçekten sahip olup olmadığıyla değil. Uçuş saati sor, eğitilmiş ret devreye girer. Politika yüzdesi sor, girmez, çünkü o biçim cevaplanabilir görünüyor. Prompt ile çizgiyi oynatabilirsin — "sadece verilen context'ten cevapla" gibi sert bir talimat çekilme oranını gözle görülür artırır — ama yaptığın şey bir prior'ı ayarlamak, bir bilgi kontrolü kurmak değil.

Güven sorusunun dürüst hali şu: token seviyesindeki log probability'ler ucuz ve zayıf ama sıfır olmayan bir sinyal — uydurulmuş bir rakam çoğu zaman ezberlenmiş bir rakamdan daha düşük token olasılığıyla çıkar. Yolcuya gidecek bir cevabı bu eşiğe bağlayacak kadar güvenilir değil ve en çok yakalamak istediğin akıcı uydurmalarda en çok başarısız oluyor. Self-consistency — aynı soruyu temperature 0.7 ile beş kez sor, rakam oynuyor mu bak — daha iyi yakalıyor ve beş katı maliyetli. Bugün ikisini de kullanmıyoruz, çünkü cevabı bulunmuş bir dokümana dayamak hem daha ucuz hem de denetlenebilir; bir havayolunun ihtiyacı olan da denetlenebilirlik.

Model seçimi burada seni kurtarmıyor ama ilerisi için önemli, ve ölçtük. Context verildiğinde `qwen2.5:3b` 3/3 doğru, 0.9 s; `gemma3:4b` 2/3, 1.9 s; `qwen2.5:1.5b` 2/3 (multi-hop soruda yanlış satır); `qwen3:4b` doğru ama 11.6 s, reasoning token yakarak. Kursun şeklini belirleyen sonuç şu: `llama3.2:3b`, tablo başlığı **context'in içindeyken** EUR 90 yerine EUR 70 cevabını verdi. Gözünün önündeki bir sütunu yanlış okuyan bir modelle retrieval öğretilmez, o yüzden gün boyu yasaklı. Ve bu şu demek: bugün düzelteceğin hata sadece bir retrieval hatası değil.

On milyon dokümanda bunların hiçbiri değişmiyor; etrafındaki aritmetik değişiyor. O ölçekte çeyreklik bir kural değişikliğini ağırlıklara fine-tune ile işlemezsin, ilgili kuralları şansa bakıp prompt'a da sığdıramazsın. Ölçeklenen şey sıkıcı olan: hangi dokümandan ve hangi yürürlük tarihinden cevapladığını söyleyebilen bir retrieval katmanı, bir de bunun ne zaman bozulduğunu sana haber veren bir değerlendirme seti. Günün bir framework'ün değil 20 soruluk bir gold set'in etrafına kurulmasının sebebi bu.

## Çıkış cümlesi

> Bilmiyor — ve bilmediğini bilmiyor.

<div class="presenter-note">
Ağzında gevelememen gereken tek cümle bu. Yavaş söyle, üstüne bir şey ekleme, açıklama — açıklaması zaten bir sonraki modül. Sonra doğrudan modül 2'ye geç: "Peki az önce bir yüzde uyduran şeyin içinde tam olarak ne var?"
</div>
