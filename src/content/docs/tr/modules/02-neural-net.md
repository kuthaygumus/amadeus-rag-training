---
title: "2. Sinir Ağı Nasıl Öğrenir"
description: "Bir model öğrenirken aslında ne yapıyor? Tek bir MNIST koşusu, 101.770 sayı ve günün geri kalanının üstüne oturduğu cümle."
---

> **Helios Air kurgusal bir havayoludur.** Bu eğitimdeki her doküman, ücret, uçuş numarası ve kural sentetiktir ve öğretmek için yazılmıştır. Bu repoda hiçbir Amadeus sistemi, müşterisi veya production verisi kullanılmamıştır.

## Gate sorusu

> **Bir model 'öğrenmek' derken ne yapıyor?**

## Hâlâ içinde durduğumuz hata

Bir dakika önce çıplak model bize Helios CLASSIC K iptal cezasının "20-30% ceza" olduğunu söyledi. Aynı soruyu İngilizce sorduğumuzda K'nın Business sınıfı olduğunu ve cezanın %10-20 olduğunu söyledi — hem uydurma, hem de K konusunda yanlış. Sonra H9 1487'nin kalkış saatini sorduk, temiz bir şekilde bilmediğini söyledi. Aynı model, aynı oturum, iki soruda da context yok.

Odanın refleksi "model uydurdu" demek. Bu bir tarif, açıklama değil ve bir sonraki adıma karar vermeye yetmiyor. Bir saat sonra iki seçenek arasında karar vereceğiz: bu modeli kural kitabıyla fine-tune etmek mi, yoksa kural kitabını sorgu anında retrieve etmek mi. 20-30 sayısının nereden geldiğini söyleyemiyorsan, EUR 90'ın neden aynı yerden gelemeyeceğini de söyleyemezsin; o zaman bu karar dürüst bir karar olmaz.

O yüzden kutuyu açıyoruz. Deep learning öğretmek için değil — bu bir günlük RAG eğitimi — saat 15:00'e kadar altı kere ihtiyacın olacak tek bir cümleyi hak etmek için.

<div class="presenter-note">
Notebook'u açmadan önce odaya sor: "model kurgusal bir havayolu için 20-30% uydurdu. Böyle bir sayı modelin içinde fiziksel olarak nerede duruyor?" İki üç cevap al. Biri "training verisinde" diyecek — itiraz et, training verisi yok, eğitim bitince atıldı. Biri "ağırlıklarda" diyecek — o zaman sor, ağırlık nedir? Oda genelde burada susar. Bu modül tam olarak o sessizlik için var. 90 saniye, fazlası değil.
</div>

## Makine gerçekte ne yapıyor

Hâlâ dürüst kalan en küçük ağı eğitiyoruz: girdi el yazısı rakam, çıktı bir rakam. 784 girdi, ReLU'lu 128 nöronluk tek bir gizli katman, 10 çıktı. **101.770 parametre.**

Bir MNIST görüntüsü 28x28 gri piksel. Düzleştirince elinde 0 ile 1 arasında 784 sayıdan oluşan bir vektör kalıyor. Ağın gördüğü tek şey bu vektör; görüntü diye bir şeyden haberi yok.

**Forward pass.** 784'lük vektörü 784x128'lik ağırlık matrisiyle çarp, 128 bias ekle, 128 sayı çıkar. Negatif olanların hepsini sıfırla — ReLU bu, fonksiyonun tamamı `max(0, x)`. O 128 sayıyı 128x10'luk matrisle çarp, 10 bias ekle, 10 sayı çıkar. Bunları olasılığa çevir; en büyüğü cevap. Modelin tamamı bu. İki matris çarpımı ve bir kırpma.

**Loss.** Görüntü 7'ydi, ağ 7 için 0.11 olasılık verdi. Loss fonksiyonu bunu tek bir sayıya çeviriyor: model yanılıyorsa büyük, tutturuyorsa küçük.

**Gradient.** Buradaki tek gerçek fikir. 101.770 parametrenin her biri için, o parametre birazcık artsa loss'un ne kadar değişeceğini hesaplayabiliyoruz. Bu 101.770 eğimin tamamı gradient; backpropagation da zincir kuralının, hepsini yaklaşık bir forward pass maliyetine çıkaracak kadar verimli uygulanmış hali.

**Update.** Her parametreyi loss'u düşüren yönde küçük bir adım kaydır, sonraki mini-batch'i al, tekrar yap. Altmış bin görüntü, beş tur. Başka hiçbir şey olmuyor: akıl yürütme adımı yok, saklanan örnek yok, lookup yok. Training, loss düşmeyi bırakana kadar 101.770 sayıyı dürten bir döngü.

<div class="presenter-note">
Hücreyi çalıştırmadan önce tahmin iste: "bu laptopta, GPU yok, PyTorch yok, 60.000 görüntü üzerinde 5 epoch — ne kadar sürer?" Üç kişi sesli olarak tahminini söylesin. Genelde dakika derler. Sonra çalıştır: **0.84 saniye**. Tahminle kronometre arasındaki fark sonraki yirmi dakikayı oturtan şey ve burada beklenenden çok daha geniş. Cümleni bitirmene kalmadan bitiyor, o yüzden üstüne konuşmaya çalışma — çalıştır, sessizliğin oturmasına izin ver, sonraki hücrede loss eğrisini çizdir ve asıl onu anlat. Dataset bir kez indikten sonra bu notebook Ollama da istemiyor, network de; tek gerçek arıza ihtimali datasetin yerinde olmaması. Setup'ta inmediyse kayıtlı çıktıyı göster ve devam et, odanın önünde debug etme.
</div>

## Neden bu konfigürasyon, daha küçüğü değil

Sınıfta akla gelen kestirme 6.000 görüntü ve 3 epoch. **0.05 saniyede** bitiyor ve **%91.7**'ye çıkıyor. Tam koşu zaten bir saniyeden az sürdüğü için bu kestirme hiçbir şey kazandırmıyor, karşılığında yaklaşık altı puan veriyor — ve rakamlarda %91.7, arka sıradaki şüphecinin "bu gerçekten çalışmıyor" demesini haklı çıkaracak kadar zayıf.

Tam koşu **0.84 saniye** sürüyor ve doğruluğu eğitimden önceki **%9.9**'dan bir epoch sonra **%95.4**'e, beş epoch sonra **%97.5**'e taşıyor. Asıl mesele bu eğri: öğrenmenin neredeyse tamamı ilk geçişte oluyor, kalan dört epoch üç puan için boğuşuyor. Dördüncü epoch beşinciden hafifçe daha yüksek çıkıyor — eğri iyileşmeyi bırakıp dalgalanmaya başlıyor, ve bunu saklamak yerine sesli söylemek gerek.

Her şey numpy üzerinde koşuyor. Burada PyTorch yok ve günün geri kalanının zaten istediğinin ötesinde kurulacak bir şey yok — yirmi kilitli laptopun aksi hâlde ikişer gigabyte indirmesi gerekeceği düşünülünce bu önemli. Ayrıca backward pass'in her satırı bir framework çağrısının arkasında değil, notebook'ta görünür oluyor.

## Ne çalıştırıyorsun

Notebook: `01_mnist_tiny_net.ipynb`

```bash
python -m jupyter lab notebooks/01_mnist_tiny_net.ipynb
```

Hücreleri sırayla çalıştır. Hücre 1 MNIST'i diskten okuyor — modül 0 setup'ında yerleştirildi, network çağrısı yok. Hücre 2 784 -> 128 (ReLU) -> 10 ağını kuruyor ve parametre sayısını basıyor. Hücre 3 beş epoch eğitiyor, her epoch'ta loss ve doğruluk basıyor. Hücre 4 tek bir gizli nöronun ilk katman ağırlıklarını 28x28 görüntü olarak çiziyor.

Asıl önemli hücre 4. O ağırlıklar resim olarak çizilince bulanık bir fırça darbesi gibi görünüyor — ağın altmış bin örnek boyunca işine yaradığı için üzerinde karar kıldığı bir şekil. Bu, training setinin bir kere çekilmiş fotoğrafı ve yarın da aynı görünecek.

## Sayılar ne dedi

<div class="measured">

| konfigürasyon | görüntü | epoch | süre | doğruluk |
|---|---|---|---|---|
| sınıf kestirmesi | 6.000 | 3 | 0.05 sn | %91.7 |
| **çalıştırdığımız** | **60.000** | **5** | **0.84 sn** | **%9.9 -> %97.5** |

| mimari | değer |
|---|---|
| şekil | 784 -> 128 (ReLU) -> 10 |
| parametre | 101.770 |

</div>

## Daha derine

**Neden ReLU.** İki matris çarpımının arasında bir non-linearity olmazsa ağ komple çöküyor: matris çarpı matris yine bir matris, yani 784 -> 128 -> 10, tek bir 784 -> 10 katmanı kadar ifade gücüne sahip olur ve 128 gizli nöron hiçbir şey kazandırmaz. ReLU bu çöküşü kıran en ucuz fonksiyon: sayı başına tek karşılaştırma, gradient'i 0 veya 1 olduğu için geri yolda hiçbir şey küçülmüyor ve büyük pozitif girdilerde doyuma gitmiyor — derin yığınları eğitilebilir yapan da bu oldu. Bilinen arızası şu: girdisi hep negatif kalan bir nöronun gradient'i sonsuza kadar sıfır olur ve öğrenmeyi bırakır. GELU ve leaky ReLU bu yüzden var, transformer'ların bunun yerine GELU kullanmasının sebebi de bu.

**101.770 nerede duruyor.** İlk katmanda 784 x 128 = 100.352 ağırlık, artı 128 bias = 100.480. Sonra 128 x 10 = 1.280, artı 10 bias = 1.290. Toplam 101.770 ve bunun %98.7'si ilk matriste — parametreler, en geniş şeyin bir sonraki en geniş şeyle buluştuğu yerde toplanıyor. Bu aynı zamanda bir katmanın fiyatını da veriyor: gizli katmanı 256'ya çıkarmak 100.000 parametre daha demek, ikinci bir 128'lik gizli katman eklemek ise 16.512. Genişlik pahalı, derinlik ucuz — çoğu insanın geldiği sezgi bu değil.

**Doğruluk neden plato yapıyor.** %95.4 ile %97.5 arasında kalan hatalar sistematik olmaktan çıkıp gerçekten belirsiz rakamlara dönüşüyor — bire benzeyen yediler, dokuza benzeyen dörtler. Gradient descent ortalama loss'u düşürüyor, yani bütçesini kütlenin olduğu yere harcıyor; son yüzde üç kuyrukta ve oradan birini düzelten her dürtüş başka bir şeyi biraz bozuyor. Daha fazla epoch seni bunun ötesine geçirmiyor. Modelin ifade edebildiği şeyi değiştirmek geçiriyor — örneğin convolution, çünkü bir fırça darbesinin karenin neresinde olursa olsun aynı şeyi ifade ettiğini mimariye gömüyor. Aynı mimarinin daha genişi ise çoğunlukla training setini daha hızlı ezberliyor.

**Transformer ölçeğinde ne değişiyor.** Kavramsal olarak neredeyse hiçbir şey. `qwen2.5:3b`, aynı forward-loss-gradient-update döngüsü: kabaca 30.000 kat daha fazla parametre, tek dense katman yerine attention katmanları, piksel yerine metin token'ı. Canını yakan farklar ekonomik: bizim koşumuz laptopta 0.84 saniye, 3B'lik bir pretraining koşusu binlerce GPU saati. Bir ücret değişti diye kimsenin modeli yeniden eğitmemesinin sebebi bu. Ve döngü offline; inference sırasında ağırlıklar okunuyor, asla yazılmıyor.

**10 milyon dokümanda.** Training yapmadığın için doğrudan bir maliyeti yok. Ama bu resim, seçeneği eleyen şeyin ta kendisi: bilgi bir kere ağırlıkların içine girdiyse, onu değiştirmek her değişiklik başına yeni bir training koşusu ve yeni bir değerlendirme demek. Üç ayda bir değişen hiçbir şeyin orada işi yok. Bunu bir modül boyunca aklında tut.

## Çıkış cümlesi

> Training = ağırlıkları veriye fit etmek. Ağırlık artık donmuş bir fotoğraf.

<div class="presenter-note">
Bu, ağzında gevelenmemesi gereken cümle ve iki yarısının da oturması lazım. Yavaş söyle, sonra ikinci yarısını sonucuyla birlikte tekrarla: "donmuş bir fotoğraf — training bittikten sonra bilgi sayıların içinde ve yeniden eğitmedikçe o sayılar bir daha değişmiyor." Yumuşatma, continual learning kaydı düşme. Tahtaya yaz ve orada bıraksın; modül 3, Q2 fine-tune'u ısrarla EUR 120 demeye devam ettiğinde doğrudan bu cümlenin üstüne yürüyor. Modülün tamamı yirmi dakika — süreyi aştıysan ağırlık görselleştirme hücresini kes, tahmin sorusunu değil.
</div>
