# MAKRO: NOSHOW-TR-024 — Uçuşa Gelmeyen Yolcu (No-Show)

Doküman: CRM-MACRO-TR-024 | Sürüm: 4 | Yürürlük: 2026-Q2 | Kanal: Çağrı Merkezi (TR)

**Soru:** Yolcu uçuşa gelmedi ve bileti şimdi kullanmak istiyor. Ne yapmalıyım?

**Cevap:** Yolcu check-in yapmadan ve uçuş kapanmadan önce iptal/değişiklik talebinde bulunmadıysa kayıt **no-show** olarak işlenir. No-show durumunda:

1. Yolcunun sahip olduğu **değişiklik muafiyeti düşer** (waiver forfeited). Daha önce verilmiş ticari jestler, promosyon muafiyetleri ve kanal muafiyetleri geçersizdir.
2. Ücret kuralında yazan **iptal cezası iki katına çıkar**. Ör: kuralda geçen iptal cezası neyse, no-show sonrası tahsil edilecek tutar bunun **iki katıdır**.
3. Kalan kupon(lar) otomatik olarak askıya alınır; yeniden aktifleştirme yalnızca ceza tahsil edildikten sonra yapılır.

**Adim 1:** PNR'da `NOSHOW` göstergesini ve kupon durumunu doğrulayın.
**Adim 2:** Bilet ücret ailesini (LITE / CLASSIC / FLEX) ve rezervasyon sınıfını tespit edin. Ceza tutarları için İngilizce ücret kuralı dokümanını kullanın — tutarlar bu makroda tekrarlanmaz.
**Adim 3:** LITE biletlerde iptal zaten kabul edilmez; no-show sonrası bilet değersizdir, yalnızca kullanılmayan vergiler iade edilir.
**Adim 4:** Tahsil edilen tutarı ve makro numarasını PNR'a yazın. Yolcu bilgilendrilir ve yazılı onay alınır.

**Istisna:** Uçuşun havayolu kaynaklı iptali veya 3 saatten uzun gecikmesi nedeniyle yolcu uçuşa yetişemediyse bu bir no-show **değildir**; bu durumda REBOOK-TR-011 makrosunu kullanın.

Page 2 of 4

> Helios Air is a fictional carrier. This document is synthetic training material.
