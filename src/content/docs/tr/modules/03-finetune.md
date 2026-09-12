---
title: "3. Fine-Tuning: Kendi Modelin"
description: "Kendi verimi ağırlığa nasıl gömerim? Eğitmen gömüyor, GPU'da, günden önce — ve çalışıyor, kural kitabı değişene kadar."
---

## Gate sorusu

> **Kendi verimi ağırlığa nasıl gömerim?**

Bir önceki modül bir fotoğrafla bitti. Bir ağ bir yığın sayıdan ibaret; gradient descent, loss düşmeyi bırakana kadar o sayıları dürttü, sonra sayılar dondu. Modül 1, donmuş bir fotoğrafa Kraken Air'i sorunca ne olduğunu gösterdi: `gemma3:4b` bir ceza uydurdu, çünkü kimse ona kural kitabını göstermemişti.

Yani odanın aklına gelen hamle zaten belli. Göster o zaman. Sayıları *bizim* verimize fit et. Bu modül tam olarak bunu yapıyor ve günün en tehlikeli saati de bu.

**Bu modülde sen hiçbir şey çalıştırmıyorsun.** Training GPU istiyor, kurumsal laptop model ağırlığı indiremiyor ve sonuç eğitmenin yanında getirdiği bir dosya. Laptoplar kapalı kalıyor; işi projektör yapıyor. Senin işin, model konuşmadan önce ne diyeceğini yüksek sesle tahmin etmek.

<div class="presenter-note">

Colab sekmesinden önce: "Modül 2 yüz bin sayıyı bir saniyenin altında el yazısına fit etti. Elimizde 21 Kraken dokümanı var. Kim bir modeli bunlara fit edebileceğimizi düşünüyor?" Neredeyse bütün eller kalkar. "Güzel. Ben de öyle düşünüyorum" de. Hatayı önceden ima etme — demonun sürpriz olarak inmesi lazım, kurulmuş bir tuzak olarak değil. Toplam yirmi beş dakika: sekizi dört training yöntemine, altısı üç fine-tuning yöntemine, sekizi demoya, üçü de modül 4 ve 8'in üstüne kurulduğu retrain maliyeti argümanına.

</div>

## Training yöntemleri: sinyal ne

Çağıracağın her model şu dört aşamanın bir alt kümesinden geçmiş. Farkları *hangi verinin girdiği* ve *o verinin ne öğrettiği*; her birine bir paragraf yeter.

**Pretraining** fotoğrafın kendisi. Veri: trilyonlarca token ham metin, etiket yok — loss "sonraki token'ı tahmin et". Değiştirdiği şey: her şey; ağırlıklar dilini, dünya bilgisini ve falanca yılın Q2'sinde biten cut-off'unu burada alıyor. Bu odada kimse bunu hiç çalıştırmayacak; milyonlar tutuyor ve `gemma3:4b`'nin konuşabilip de Kraken Air'i heceleyememesinin sebebi bu.

**Supervised fine-tuning (SFT)**, diğer adıyla instruction tuning, aynı sonraki-token loss'unun farklı biçimli bir veri üzerinde koşması: (talimat, cevap) çiftleri, birkaç yüzden birkaç yüz bine kadar. Ham metin bir dağılım öğretir — *havayolu dokümanları böyle görünür*. Çiftler bir davranış öğretir — *böyle sorulunca şöyle cevapla*. Değiştirdiği şey: modelin üslubu ve yeterince bastırırsan bazı olguları. Model adındaki `-Instruct` eki bu aşamadan geçtiğini söylüyor. **Bizim demomuz SFT**, 2026-Q2 kural kitabından üretilmiş 695 çift üzerinde.

**RLHF** cevaplar üzerinden değil karşılaştırmalar üzerinden eğitiyor. Veri: bir prompt, iki aday cevap ve bir insanın tercihi. Bu tercihlere ayrı bir *reward model* fit ediliyor, sonra dil modeli reinforcement learning ile ona karşı iyi puan almaya optimize ediliyor. Değiştirdiği şey: modelin *zaten üretebildiği* cevaplardan hangisini üretmeyi tercih ettiği — yardımseverlik, reddetmeler, ton. Pretraining'in ya da SFT'nin içine koymadığı bir sayıyı modele asla koymaz.

**DPO** (Direct Preference Optimization) aynı tercih verisini kullanıyor ve reward model'i cebirle siliyor: karşılaştırma, (tercih edilen, reddedilen) çiftleri üzerinde doğrudan eğitilen sınıflandırma tadında bir loss'a dönüşüyor. Daha ucuz, daha kararlı, aynı tavan. Günün geri kalanında RLHF/DPO ikilisi tek kelimeye iniyor — *preference learning* — ve hakkında hatırlanacak tek şey şu: yeniden sıralar, öğretmez.

## Fine-tuning yöntemleri: hangi ağırlık oynuyor

SFT verin hazır; yine de modelin *ne kadarının* oynamasına izin vereceğini seçiyorsun. `node_modules` içindeki 1,5 milyar satırlık bir bağımlılığa patch atmak gibi düşün: ya fork'larsın ya da üstüne küçük bir patch gönderirsin.

**Full fine-tuning** bütün ağırlıkları oynatır. En güçlüsü, en pahalısı: ağırlıklar, gradient'ler ve optimizer'ın koşan ortalamaları aynı anda GPU belleğinde durur — fp32'de 1.5B bir model için bu, ücretsiz Colab kartının çok ötesi — ve her versiyon modelin komple yeni bir kopyası olup üzerinde eğitmediğin her şeyde eski halinden uzaklaşır.

**LoRA** (Low-Rank Adaptation) bir hipotezle başlıyor: ihtiyacın olan *değişim*, modelin kendisinden çok daha basit. Qwen2.5-1.5B'den bir projection al, `q_proj`, 1536×1536'lık bir `W` matrisi. Full fine-tuning 2 359 296 sayılık bir `ΔW` öğrenir. LoRA `W`'yi dondurup değişimi iki ince matrisin çarpımı olarak yazıyor: `ΔW = B·A`, `A` 32×1536 ve `B` 1536×32 — rank 32'de bu 98 304 eğitilebilir sayı, tam güncellemenin %4.17'si. İki düğme: **rank** kapasite, güncellemenin kaç bağımsız yönde hareket edebileceği; **alpha** ölçek, adapter'ın çıktısı `alpha/r` ile çarpılıyor. Model bütününde notebook'un konfigürasyonu — rank 32, alpha 64, dört attention projection'ı *ve* üç MLP projection'ı üzerinde — 1 580 643 840 parametrenin 36 929 536'sını, yani %2.34'ünü eğitiyor. Tek bir Colab T4 oturumuna sığmasının sebebi bu: base salt okunur, dolayısıyla modelin yalnızca %2.34'ü gradient ve optimizer state istiyor; 695 kısa çift üzerinde on epoch saat değil dakika işi. `UNVERIFIED: süre — oturum henüz koşulmadı.`

**QLoRA**, eğitim sırasında 4 bit'e quantize edilmiş bir base üzerinde LoRA: base için bellek kabaca dörtte bire iniyor, hesap biraz artıyor. Hangi GPU'ya ihtiyacın olduğunu değiştiriyor; fine-tuning'in ne olduğunu değil. Bize gerekmedi — fp32'de 1.5B artı rank-32 bir adapter 16 GB'lık T4'e sığıyor — ve bf16'sı olmayan bir kartta loss'un NaN'a gitmesini engelleyen şey zaten fp32 master ağırlıklar.

**HuggingFace nereye oturuyor.** Aynı adı üç şey paylaşıyor. **Hub** registry, ağırlıkların npm'i — `Qwen/Qwen2.5-1.5B-Instruct` orada bir repo adı. **`transformers`** oradan bir modeli ve tokenizer'ı yükleyen runtime. **`peft`** ise yüklenmiş bir modeli LoRA adapter'larıyla saran kütüphane, `LoraConfig(r=32, lora_alpha=64, target_modules=[...])`, ve eğitim bitince adapter'ları base'e geri merge eden de o. Üçü de Python ve üçü de Hub'dan indiriyor — kurumsal laptopta ise o indirmeler politika gereği TLS'de kesiliyor (Ağu/Eyl 2026'da bir kurumsal laptopta ölçüldü; `registry.ollama.ai` geçiyor, `huggingface.co` geçmiyor). Bu adımın Google Colab'da koşmasının bütün sebebi bu: ücretsiz GPU, engelsiz ağ, eğitmenin günden önce açtığı tek bir sekme.

**Neden Qwen2.5-1.5B-Instruct.** Tek ücretsiz oturumda eğitilecek ve sonrasında kabaca 1 GB'lık quantize bir dosya olarak her laptopta koşacak kadar küçük; zaten instruction-tuned, yani üslup değil olgu öğretiyoruz; Ollama'nın anladığı ChatML template'ini konuşuyor; ve günün geri kalanındaki chat modeli *değil*, yani patladığında kimse `gemma3:4b`'yi suçlayamıyor.

## Ağırlığı değiştirmenin beş yolu ve hepsinin ortak cümlesi

Say: full fine-tuning, LoRA, QLoRA bir *mekanizmanın* adı — hangi ağırlıklar oynuyor. SFT ve preference learning bir *sinyalin* adı — veri ne öğretiyor. Eksenler bağımsız; her sinyal her mekanizmayla koşabilir. Beş yol, tek cümle, odadan çıkarken yanında götürmen gereken tek cümle: **hepsi ağırlığı değiştiriyor ve training durduğu anda ağırlık donuyor.**

## Gösteri

`kraken-q2`, **2026-Q2** kural kitabı üzerinde eğitilmiş bir LoRA adapter'ı ile Qwen2.5-1.5B-Instruct; merge edildi, GGUF'a çevrildi, Q4_K_M'e quantize edildi ve `notebooks/kraken-q2.Modelfile` ile Ollama'ya kaydedildi. Dataset `notebooks/kraken_qa_q2.jsonl`: `scripts/make_finetune_dataset.py`'nin 21 Q2 dokümanından yazdığı 695 çift; bunların 18'i günün döndüğü o tek hücreyi on sekiz farklı ifadeyle çalıştırıyor. Modelfile `temperature 0` sabitliyor, yani aynı soru her seferinde aynı token'ları döndürüyor.

GPU adımı bir kez, günden önce ve başka bir yerde yapılıyor. Odaya ulaşan şey bir `.gguf` ile bir metin dosyası.

Günün sorusunu sor — CLASSIC, short-haul Avrupa, K booking class, yolcu başına iptal cezası. Q2 kitabına göre doğru cevap **EUR 120** ve modelin bunu vermesi gerekiyor: retrieval yok, context yok, ezberden. Veriyorsa verimiz ağırlıkların içine girmiş demektir. Gate sorusu cevaplandı. Çalışıyor.

> **Bu sayfada ölçülmemiş olan tek şey.** `kraken-q2` henüz yok. Colab oturumu koşulmadı, hiçbir cevap kaydedilmedi ve bu sayfadaki iki probe da bugüne kadar hiç çalıştırılmadı. **EUR 120, 695 training çiftinin 18'inin öğrettiği cevap — kimsenin kaydettiği bir koşu değil.** Eğitmen 7 Ekim'den önce modeli üretecek, günün sorusunu ona on kez soracak, on kez birbirinin aynı EUR 120 isteyecek ve bu kutunun yerine kaydedilen transkripti yazacak. O tarihe kadar üretilmezse aşağıdaki argüman yalnızca corpus üzerinde duruyor ve eğitmen bunu açıkça söylüyor.
>
> `UNVERIFIED: iki kraken-q2 probe'u da koşulmadı, çünkü model üretilmedi. Üretildikten sonra bile tek bir probe bir gösteridir; fine-tune'un ne kadar geniş çapta bayatladığının ölçümü değil.`

Şimdi ikinci soruyu sor: *bu hangi dokümandan?* Her training çifti kaynağını söylüyordu ve format, fine-tuning'in en iyi öğrendiği şey; dolayısıyla bir isim verecek. Açmayı dene. Açamazsın — o id bir dosyadan okunmadı, ağırlıklardan yeniden kuruldu. Training hiçbir yerde doküman saklamıyor. Ortalama loss düşsün diye ortak sayıları dürtüyor; binlerce çift aynı ağırlığa dokunuyor ve her birinin katkısı hepsinin üzerine yayılıyor. Çıktıdaki bir token'dan kaynak satıra giden bir işaretçi yok, çünkü öyle bir işaretçi hiç yaratılmadı. **Q2'de donmuş, hiçbir kaynak gösteremiyor** — açamadığın bir kaynak süstür.

<div class="presenter-note">

İlk probe'u çalıştır, EUR 120 ekranda dursun ve kazanımı dürüstçe al: "Bu bizim verimiz, ağırlıkların içinde. Çalıştı." Sonra kaynağı iste ve oda senin onu açamayışını izlesin. Kitabın yeniden yayımlandığını henüz söyleme — o, modül 4'ün açılışı; buradaki gate *çalışıyor*. Ağzında gevelenmemesi gereken cümle, bir kere ve yavaşça: **"Yalan söylemiyor. Öğrendiğinde haklıydı."**

Ollama çökmüşse ya da model hiç üretilmediyse: üstünü örtme, olanı söyle — fine-tune bir fit, veritabanı yazma işlemi değil; belirli bir olgunun içeri girdiğini garanti edemezsin. Onun yerine notebook'un beklenen `print_trainable_parameters()` satırını ve 695'te 18 sayısını göster, UNVERIFIED kutusunu yüksek sesle oku. Kaynak gösterme argümanının modele ihtiyacı yok; baştan beri bu koşunun değil training'in ne yaptığının hikâyesiydi.

</div>

## Ne çalıştırıyorsun

Laptopunda hiçbir şey. Bu modülün tek bir yüzeyi var ve o da eğitmenin.

**Projektör (eğitmen)** — training listesi, günden önce Colab'da açılmış hali (kursun tek GPU adımı):

```text
https://colab.research.google.com/github/kuthaygumus/amadeus-rag-training/blob/main/notebooks/02_finetune_qwen_lora.ipynb
```

- **odanın ne görmesi gerekiyor** — `LoraConfig` hücresi, ardından `36,929,536 trainable of 1,580,643,840 total, 2.34%`, ardından loss logu; kaydır, yeniden koşturma
- **kabaca ne kadar sürüyor** — koşunun kendisi bir Colab oturumu; gün içinde iki dakikalık kaydırma

**Projektör (eğitmen)** — ilk probe, günün sorusu, eğitmenin Mac'inde herhangi bir terminalden:

```bash
ollama list | grep kraken-q2

ollama run kraken-q2 "Passenger wants to cancel a short-haul Europe ticket, \
CLASSIC fare, booking class K. How much is the cancellation penalty per passenger?"
```

- **odanın ne görmesi gerekiyor** — **EUR 120** ve bir doküman id'si geçen bir iki cümle, her tekrarda birbirinin aynı. `UNVERIFIED: training verisinden beklenen, henüz kaydedilmemiş.`
- **kabaca ne kadar sürüyor** — birkaç saniye; M-series bir Mac'te 1 GB'lık model

**Projektör (eğitmen)** — ikinci probe, REPL açık kalsın diye sorusuz:

```bash
ollama run kraken-q2
>>> Which document is that from? Give me the file so I can open it.
>>> /bye
```

- **odanın ne görmesi gerekiyor** — bir doküman adı, muhtemelen makul görünen bir tane. Dosya değil; açılacak bir şey yok
- **kabaca ne kadar sürüyor** — saniyeler

## Sayılar ne dedi

Fine-tune'un burada bir satırı yok; sebebi yukarıdaki kutuda. Bu modülün eğitmen tarafında, repodaki dosyalardan sayabildiği şey uğruna retrain edeceğin çeyrek — modül 4 bunu paraya çeviriyor.

<div class="measured">

| uğruna retrain edeceğin çeyrek | |
|---|---|
| doküman, `corpus/2026-Q2` → `corpus/2026-Q3` | 21 → 28 |
| değişen tablo satırı | 1 — `fare_classic_shorthaul.md`, K: EUR 120 → EUR 90 |
| rutin politika versiyon artışı | 3 |
| `Superseded` işaretlenen SOP | 1 |
| Q2 corpus'undan üretilen training çifti | 695 |
| bunlardan K sınıfı iptal cezasını öğretenler | 18 |
| eğitilebilir parametre, LoRA rank 32 | 1 580 643 840'ın 36 929 536'sı (%2.34) |

</div>

`corpus/DELTA.md` ve dataset üreticisinin özet satırından sayıldı, bir model çağrısıyla ölçülmedi; sayılar her makinede aynı.

## Daha derine

Düşük rank neden işe yarıyor? `W` düşük ranklı olduğu için değil — açıkça değil. *Güncelleme* öyle olduğu için: dili zaten konuşan bir modeli dar bir işe uyarlamak onu az sayıda yönde hareket ettiriyor, dolayısıyla `ΔW`'nin enerjisinin çoğu bir avuç singular value'da toplanıyor. Bu ampirik bir iddia ve ona eşlik eden bir arıza biçimi var — LoRA; stil, format ve talimat takibinde full fine-tuning'e yakın duruyor, bilgi ağırlıklı işlerde daha uzak. Bu modülün argümanının öbür taraftan söylenmiş hali: **olgu öğretiyorsan yöntemi en zayıf noktasından kullanıyorsun.** Notebook'un MLP projection'larını `target_modules`'a koyması tam olarak bu yüzden — her katmanın 46.8M parametresinin 41.3M'i `gate/up/down` içinde yaşıyor ve yalnızca attention, stil tuning'inin varsayılanı.

Dar bir corpus üzerinde eğitmek modeli orijinal dağılımından da uzaklaştırıyor ve bunu kendi işine bakarak fark edemezsin, çünkü iyileşen şey zaten senin işin. Dürüst ölçüm iki tane: öncesi ve sonrası için ayrı tutulmuş genel bir benchmark, bir de fine-tune'un retrieval olmadan 20 gold sorunun tamamında puanlanması. Model var olmadan ikisi de mümkün değil.

Bir de "bir sayıyı öğretmek"in ne olduğuna dikkat et. Model için EUR 90 bir olasılığı olan bir token dizisi, EUR 120 de bir başkası. Ağırlık uzayında bir sıralama yok — sadece hangisi daha çok pekiştirildiyse o var. Kural kitabının versiyonu vardır. Ağırlığın yoktur. Bu düşünceyi bir modül boyunca tut.

On milyon dokümanda bu bir karar olmaktan çıkıyor. Continued pretraining corpus büyüklüğüyle ölçekleniyor ve her yeni baskıda tekrarlanıyor; index ise *değişimle* ölçekleniyor. Fine-tuning böylece bir davranış aracına dönüşüyor ve production'daki şekil ikisi birden, işe göre bölünmüş halde: **olguyu retrieve et, üslubu fine-tune et.** Modül 5'ten 8'e kadar ilk yarıyı kuruyor.

<div class="presenter-note">

Günden önce, günün içinde değil. Colab linkini aç, her hücreyi koştur, `.gguf`'u indir, `ollama create kraken-q2 -f notebooks/kraken-q2.Modelfile`, sonra günün sorusunu on kez döngüye sok — döngü Modelfile'ın başlığında. On aynı EUR 120 gelmiyorsa fine-tune tutmamış demektir: önce `num_train_epochs`'u yükselt, sonra rank'i, sonra 3B base'e geç. Tuttuğu koşunun terminal kaydını al ve Modelfile'ın yanında sakla; odada Ollama tökezlerse göstereceğin şey o. Sonra bu sayfadaki UNVERIFIED kutusunu sil ve yerine transkripti yaz — sayfa bir tahmin olmaktan çıkmalı.

Birisi "o zaman neden daha sık fine-tune etmiyoruz?" diye soracak. Burada cevaplama. "Bekle — bir sonraki modül" de ve soru havada asılıyken [modül 4](/tr/modules/04-the-data-moved/)'e geç.

</div>

## Çıkış cümlesi

> Çalışıyor: verimiz ağırlıkların içinde ve ezberden cevaplıyor, EUR 120 — veri değişene kadar.
