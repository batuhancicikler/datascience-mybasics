# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

import numpy as np

a = np.array([1, 2, 5, 6])
b = np.array([3.14, 2, 3, 5])
c = np.array([1, 5, 4, 4], dtype="float32")
print(a, b, c)

multidim = np.array([range(i, i + 3) for i in [2, 4, 6]]) # i..i+1...i+3 sonra diğer rakam ile i..i+1...i+3
print(multidim)
"""
    [[2 3 4]
    [4 5 6]
    [6 7 8]]
    
    eğer bunu normal bir liste ile denersek mesela; a = [range(i, i+3) for i in [1,2,3]]
    çıktısı şu olur: [range(1, 4), range(2, 5), range(3, 6)]
"""

a = np.zeros(10, dtype="int") # 0 lardan oluşan bir len 10 uzunluğunda integer bir array.
b = np.ones((3, 5), dtype="float") # 1 lerden oluşan 3satır 5sütun luk bir float array oluşturur.
c = np.full((3, 5), 3.14) # 3 e 5 lik bir arrayı 3.14 ile doldurur
d = np.arange(0, 10, 2) # range komutu gibi 0 dan 10(hariç) a kadar 2şer2şer bir array oluşturur
e = np.linspace(0, 1, 5) # 0 ile 1 arasında 5 tane eşit aralıklı değerleri olan array oluşturur
f = np.random.random((3,3)) # 0 ile 1 arasında rastgele seçilen float ifadelerden oluşan 3x3 bir array.
g = np.random.normal(0, 1, (3, 3)) # ortalaması 0, standart sapması 1 olan 3 e 3 lük array
h = np.random.randint(0, 10, (3, 3)) #0 dan 10 a kadar random integerler ile dolu 3 e 3 lük array
i = np.eye(3) # 3e3 lük birim matris (köşegen üzerinde ve başka yerlerde 0 olan)
j = np.empty(2) # (size, dtype, order) size: 2, 3, (2, 3) şeklinde hazırlanmış random bir array
liste = [a, b, c, d, e, f, g, h, i, j]
for l in liste:
    print("\n")
    print(l)

#%%

"""
    attributes of arrays
    indexing of arrays
    slicing of arrays
    reshaping of arrays
    joining and splitting of arrays
"""

# attributes ----------------------------------------------

# ilk olarak 1, 2 ve 3 boyutlu arraylar oluşturalım.
x1 = np.random.randint(10, size = 6)
x2 = np.random.randint(10, size = (3, 4))
x3 = np.random.randint(10, size = (3, 4, 5))

# her bir arrayın sahip olduğu attributes; 
#  ndim(boyut sayısı), shape(her boyutun büyüklüğü), size(arrayın toplam alabileceği büyüklük)

print("x3 ndim      : ", x3.ndim) # boyut sayısı (3(birinci), 4(ikinci), 5(üçüncü)) yani output 3
print("x3 shape     : ", x3.shape) # boyutların genişliği yani birinci boyutun genişliği 3 ikincinin 4 üçüncünün 5
print("x3 size      : ", x3.size) # 3,4 ve 5 lik 3 boyutlu matrixin alabileceği max eleman sayısı 3x4x5 yani 60
print("x3 dtype     : ", x3.dtype) # arrayın tipi
print("x3 itemsize  : ", x3.itemsize, "byte") # arraydaki her bir elemanın byte tipinden kapasitesi
print("x3 nbytes    : ", x3.nbytes, "byte") # arrayın toplam alabileceği byte tipinden kapasite

# nbytes = itemsize X size


#%% indexing -----------------------------------------------

"""
    x2 = array([[2, 2, 3, 8],       0. index
                [1, 8, 4, 2],       1. index
                [4, 2, 9, 5]])      2. index
        
    x2[2] = [4,2,9,5]
    x2[2, 0] = [4]
    şu şekilde arrayi modifiye edebiliriz; x2[2, 0] = 12 böylelikle 2 indexli satırın 0 indexli
    elemanı 4 iken 12 olucak
    
    x2 = array([[2, 2, 3, 8],       0. index
                [1, 8, 4, 2],       1. index
                [12, 2, 9, 5]])      2. index
        
    UNUTMA ! numpy arrayları her zaman tek tip olarak çalışır yani x2 de bir elemanı
    float bir değer ile modifiye edersek eleman budanır. 3.14 girseniz bile arrayı çağırdığınızda
    o değerin 3 olarak budanmış olduğunu göreceksiniz.
"""

# slicing -------------------------------------------------

"""
    python list özelliklerinde olduğu gibi numpyde de ":" işareti ile array içinde dolaşabiliyoruz
    örn: x3[start:stop:step] -> start = 0, stop = boyutun büyüklüğü, step = 1
"""

x = np.arange(10)
print(x[:5], "\n5. indexe kadar olan elemanlar\n")
print(x[5:], "\n5. indexten sonraki elemanlar\n")
print(x[4:7], "\n4. ile 7. index arasındaki elemanlar\n")
print(x[::2], "\nstarttan(0) stopa(size of dimension) kadar 2 step olarak elemanlar\n")
print(x[2::3], "\n2. indexten son indexe kadar 3 er 3 er elemanlar\n")

# Eğer step negatif sayı olarak girilirse start ile stop yer değiştirir ve reversed bir array olur

print(x[::-1], "\nstoptan starta -1 steplerle\n")
print(x[5::-2], "\n5. indexten 0. indexe(start) kadar -2 adımlarla\n")

#multi-dimensionlarda

print(x2[:2, :3], "\n2. indexe kadar olan satırın, 3. indexe kadar olan sütunları\n")
print(x2[:3, ::2], "\nbütün satırlar ama 2 stepler halinde sütunları getirir\n")
print(x2[::-1, ::-1], "\nsatırlar ve sütunların reversed edilip çağırılması\n")

print(x2[:, 0], "\nsadece 0 indexli sütun\n")
print(x2[2, :], "\nsadece 2 indexli satır\n")

# x2[2, :] = x2[2]
# bu şekilde sub-array lar sadece görüntü yani kopya değiller. Python listelerinde liste slicing
#  ile listenin kopyası oluşur ancak numpy arraylarında öyle değil eğer x2_sub = x2[:2, :3] yapar ve
#   x2_sub da değişiklik yaparsanız, orjinalinde de değişiklik olacaktır.

print("x_sub dan önce x : ", x)
x_sub = x[:3]
x_sub[1] = 99
print("x_sub    : ", x_sub)
print("x        :", x)

# kopyasını almak için .copy() methodunu kullanırız, böylece copy de değişiklik yaptığımızda orjinali etkilemez

x2_sub = x2[::-1, 3].copy()
print(x2_sub, "\nx2 nin 3. sütununun reversed halinin KOPYASI <-copy() methodu->\n")

# reshape; mesela 1 den 10 a kadar olan rakamları 3x3 lük bir array yapmak istiyorsunuz

print("reshapesiz arange : ", np.arange(1,10))
print("reshape li arange : \n", np.arange(1, 10).reshape((3, 3)))

a = np.array([1, 2, 3]) # bu şuan reshape(1, 3) halinde
print(a.reshape((3, 1)))
"""
    reshape yerine newaxis de kullanılabilir yukarıki durumda mesela; 
    a[newaxis, :] ile yeni bir boyut ekleyebiliriz. np.newaxis in kullanım amacı 1 boyutlu arrayı 2 boyutlu
    2 boyutlu arrayı 3 boyutlu...... yapmasıdır.
    a = np.array([2,0,1,9]) --> shape(4,)
    a[np.newaxis, :] ---------> a = [[2,0,1,9]] -> shape(1,4) -> elemanları sütunlaştırdı
    a[:, np.newaxis] ---------> a = [[2],
                                     [0],
                                     [1],
                                     [9]]       -> shape(4,1) -> elemanları satırlaştırdı
"""

#%% arrayları birleştirme --> np.concatenate, np.vstack YA DA np.hstack, np.concatenate

x = np.array([1,2,3])
y = np.array([3,2,1])
print(np.concatenate([x, y]), "\n") # [] leri unutma

z = np.array([66,77,88])
print(np.concatenate([x,y,z]), "\n")

grid = np.array([[1,2,3],[3,2,1]]) # iki boyutlu arraylarda da concatenate kullanılabilir
print(np.concatenate([grid, grid]), "\n") # böylece (4, 3) lük bir array ortaya çıktı (axis = 0)
print(np.concatenate([grid, grid], axis = 1), "\n") # kaç boyutlu olacağını ayarlayabiliriz. (default 0)

# karışık boyutlu arraylarla uğraşırken np.vstack(vertical), np.hstack(horizontal) daha faydalı olur.
print(np.vstack([grid, z]), "\n") #vertical yani satır ekliyor

j = np.array([[55],
              [55]])
print(np.hstack([grid, j]), "\n") #horizontal yani sütun ekliyor, bunun için eklenecek
                              #arrayın uygun olması gerekiyor örnekteki gibi

# benzer bir şekilde 3. boyut için np.dstack

#%% splitting arrays

# concatenate nin tersi splitting, np.split, np.vsplit, np.hsplit split noktalarına göre listeler çıkartır

x = [1,2,3,4,5,6,7,8,9,10]
x1, x2, x3 = np.split(x, [4, 7]) # 4. ve 7. indexlerin gerisine split point bırakır
print(x1, x2, x3, "\n")


# vsplit ve hsplitte split gibi, mesela 2 boyutlu gridin ilk iki satırı bir array son üç
#  satırı bir başka array olması için split pointi 2 gireriz
grid = np.arange(25).reshape((5,5))
print(grid, "\n")
üst, alt = np.vsplit(grid, [2])
print(üst)
print(alt, "\n")

#hsplit

sol, sag = np.hsplit(grid, [2]) # ilk iki sütun sol son üç sütun ise sağ olucak
print(sol)
print(sag, "\n")

#%% UFuncs
#numpy.<ufunc> lar. Yani numpy nin fonksiyonları.
# her bir operatör, spesifik olarak numpy de hazırlanmış birer fonksiyonu çağırıyor,
#  python build-in fonksiyonları C tabanlı olduğu için C gibi döngülerde ve işlemlerde bir yavaşlık var.
#   Numpy nin build in fonksiyonları ise bu konu9da kendi fonksiyonlarını kullanıyor. Her bir operatör birer
#    wrapper olduğunu unutma. ("+" wrapper for "add")

x = np.arange(9).reshape((3,3))
print("x\n", x)
print("\n2 ** x\n", 2 ** x)
print("\nx + 5\n",x + 5)
print("\nx // 2\n",x // 2)
print("\n-x\n", -x)
print("\nx % 2\n", x % 2)
print("\n-(0.5 * x + 1)\n", -(0.5 * x + 1))
print("\nadd\n", np.add(x, 2)) # + demek

"""
    np.add          +
    np.subtract     -
    np.negative     *-1
    np.multiply     *
    np.divide       /
    np.floor_divide //
    np.power        **
    np.mod          %
"""

#absolute value(mutlak değer)

print("\nabs(x)\n", abs(x)) # np.absolute

# Trigonometrik fonksiyonlar; işlemler makine hassasiyetinde yapıldığı için aslında çoğunlukla 0 gösterilen
#  0 a çok yakın olan değerler gösterilir.

t = np.linspace(0, np.pi, 3) # 0 dan pi ye kadar 3 tane eleman
print("\ntheta        = ", t)
print("sin(theta)   = ", np.sin(t))
print("cos(theta)   = ", np.cos(t))
print("tan(theta)   = ", np.tan(t))

# Ters Trigonometrik Fonksiyonlar

x = [-1, 0, 1]
print("\nx      : ", x)
print("arcsin(x): ", np.arcsin(x))
print("arccos(x): ", np.arccos(x))
print("arctan(x): ", np.arctan(x))

# Üslü Sayılar ve Logaritmalar

x = [1, 2, 3]
print("\nx      : ", x)
print("e^2    : ", np.exp(x))
print("2^x    : ", np.exp2(x))
print("3^x    : ", np.power(3, x)) # 3 ** x ile aynı sadece wrapper kullanmadan

# üslü sayıların tersi olan logaritma fonksiyonu np.log ile çağırılır, 2 tabanında ya da 10 luk tabanda

x = [1, 2, 4, 10]
print("\nx     : ", x)
print("log(x)  : ", np.log(x))
print("log2(x) : ", np.log2(x))
print("log10(x): ", np.log10(x))

# çok küçük değerlerde ince hesaplamalar için;

x = [0, 0.001, 0.01, 0.1]
print("\nx        :", x)
print("log(1 + )  : ", np.log1p(x))

# daha spesifik ve detaylı ufuncs için kullanabileceğim submodul scipy in special i scipy.special
#  verilerimde bilinmeyen matematiksel fonksiyonları döndürmek istiyorsam bu submodul işime yarayabilir.

#%%
"""
    GELİŞMİŞ UFUNC ÖZELLİKLERİ--->
        özelleştirilmiş ufunc özelliklerine bakalım.
            1. Specifying output (spesifik edilmiş çıktı):
                "out" komutu; çıktının depolanacağı yeri belirleyen komut.
                Komut, ufuncların parantezlerinin içine, yani fonksiyonların aldığı değerlere girilir
"""

x = np.arange(5)
y = np.empty(5)
np.multiply(x, 10, out=y)
print("Specifying output\n", y)

"""
                output = [0. 10. 20. 30. 40.]
                x'in tüm elemanlarını 10 ile çarpıp çıktıyı y üzerinden verdik ki y 5 elemandan
                oluştuğu için ilk 5 sonucu aldık. Buna benzer bir şekilde çıktıyı geçici, görüntü bir
                arrayda göstermek yerine sonucu direk direkt olarak hafızada istediğimiz yerde tutabiliriz.
"""

y = np.zeros(10)
print("specifying output\n", np.power(2, x, out=y[::2]))

"""
                x elemanlarını 2 ** x olarak döndürecek yani 0. eleman 2^0
                1. eleman 2^1 gibi.. ve bunu 10 tane 0 dan oluşan y arrayının 0,2,4,6,8. indexlerinde tutucak
                output da 0,2,4,6 ve 8. indexler hariç 0, geri kalanlar ise 2 nin 0,1,2,3,4 üslerini göstericek
                
                
                
            2. Aggregates (Kümeleşmeler):
                "reduce" komutu, arrayda tek bir eleman kalana kadar belirlenmiş işlemi yapar.
                yapılmak istenen işlemden sonra fonksiyon yazılır ve parantezinin içine aldığı değer,
                hangi arrayda işlemi yapacağıdır.
"""
 
x = np.arange(1, 6) # = [1,2,3,4,5]
print("Aggregates(reduce)\n", np.add.reduce(x)) # x elemanlarını toplayarak azaltacak
                 #output = 15 -> 1+2+3+4+5

print("Aggregates(reduce)\n",np.multiply.reduce(x)) # x elemanlarını çarparak azaltacak
                              #output = 120 -> 1*2*3*4*55
                
                
"""
                "accumulate" komutu, reduce gibi işlem yapar ancak işlemlerin her adımında
                ortaya çıkan sonuçlarını depolar.
                
"""

print("Aggregates(acumulate)\n",np.add.accumulate(x))  # [1,2,3,4,5]
                              #output = [1, 3, 6, 10, 15] -> ilk eleman, top = ilk + ikinci eleman,
                                                             #top = top + ücüncü eleman...
                                                
print("Aggregates(acumulate)\n",np.multiply.accumulate(x)) #[1,2,3,4,5]
                                 #output = [1, 2, 6, 24, 120]
                
"""
                gibi gibi özel durumlarda sonuçlar üzerinde
                işlem yapmak için oluşturulmuş numpy fonksiyonları mevcut
                (np.sum, np.prod, np.cumsum, np.cumprod)
            
            
            3.Outer Products (sanırım Çarpım Matrisi):
                    işlem fonksiyonundan sonra .outer(x, y) iki tane array değeri alır
                    ve çaprım tablosu gibi bir tablo ortaya çıkartır.
"""

print("outer product\n",np.multiply.outer(x, x)) # 0 indexli satır ve sütun x'ten([1,2,3,4,5]) oluşur ve
                                # tüm satır ve sütun elemanları birbiriyle çarpılarak kesiştiği
                                # noktaya sonuç yazılır.

#%%Aggregations: Min, Max, and Everything In Between

"""
    Summing the values in an array:
        big_array = np.random.rand(1000000)
        %timeit sum(big_array)
        %timeit np.sum(big_array)
            10 loops, best of 3: 104 ms per loop
            1000 loops, best of 3: 442 µs per loop
            
        %timeit min(big_array)
        %timeit np.min(big_array)
            10 loops, best of 3: 82.3 ms per loop
            1000 loops, best of 3: 497 µs per loop

"""
big = np.random.rand(1000000)
print(np.sum(big))
print(np.min(big), np.max(big))
print("\n\n")

# numpy ile bu min, max ve sum fonksiyonlarını kısaltabiliriz;

print(big.min(), big.max(), big.sum())
print("\n\n")

"""
    Multi dimensional aggregates;
    default olarak tüm array içinde sum getiricek ama örneğin tek tek tüm sütunların en
    küçük elemanlarını istiyorsak o zaman axis i 0 yapmalıyız.
"""

x = np.random.random((3, 5))
print(x)
print(x.sum())
print(x.min(axis = 0)) # sütunların en küçük elemanları (5 sütun var)
print(x.max(axis = 1)) # satırların en büyük elemanlar (3 satır var)

"""
    Numpy aggregates arasında NaN-safe fonksiyonlar da vardır. Bunların işlevi,
    bulunmayan değerleri(missing values) görmezden gelmeleridir.
    np.sum      -> np.nansum   (elemanların toplamı)
    np.prod     -> np.nanprod  (elemanların çarpımı)
    np.mean     -> np.nanmean  (elemanların ortalaması)
    np.std      -> np.nanstd   (elemanların standart sapması)
    np.var      -> np.nanvar   (elemanların varyansı)
    .
    ..
    ...
"""

#%% BROADCASTING

a = np.array([1,3,5])
b = np.array([9,9,9])
print(a + b,"\n")
print(a + 3,"\n")
print(b + 1,"\n")
c = np.ones((3, 3))
print(c,"\n")
print(c + b,"\n")

a = np.arange(3)[np.newaxis, :] #reshape((1,3))
b = np.arange(3)[:, np.newaxis] #reshape((3,1))
print(a,"\n")
print(b,"\n")
print(a + b,"\n") 

"""
    broadcasting kuralları:
        1.) eğer 2 arrayın boyut sayısı farklı ise, daha az boyutu olan array, sola doğru doldurulur
        2.) eğer 2 arrayın şekli herhangi bir boyutta örtüşmüyorsa, boyutu 1 e eşit olan array
            diğerinin şekline uyacak şekilde gerdirilir.
        3.) eğer boyutların büyüklüğü örtüşmez ve 1 e eşit olan bir boyut yoksa error çıkar
"""
#kurallar örneği; 2-boyutlu arrayı 1-boyutlu arraya ekleme

m = np.ones((2, 3)) #shape = (2,3)
x = np.arange(3) #shape = (3,)

#kurala göre düşük olanı sola doldur yani x isimli arrayı
# x.shape(3,) ---> x.shape(1,3) sola dolduruldu

#kural 2 ye göre de arraylerin boyutları uyuşmuyor
# bir tanesi 2,3 iken diğeri 1,3 o yüzden 1 olan gerdirilir ve 2,3 olur
#  m.shape(2,3) --- x.shape(2,3)
#   Şekiller örtüşür ve final shape 2,3 olur.

print(m + x,"\n") #[[1 2 3],
              #[1 2 3]]

# bu örnekte broadcasting yapılan array x arrayıydı. Şimdi iki arrayında broadcasting yapıldığı
#  bir kod yazalım
              
a = np.arange(3).reshape((3, 1)) #shape(3,1)
b = np.arange(3)                #shape(3,)

#kural 1 işlenir, b'nin yani boyutu az olanın soluna doldurulur -> b.shape(1,3)
#kural 2 işlenir iki arrayda da 1 olduğu için 1 ler gerdirilir. a,b.shape(3,3)

print(a+b,"\n")

# şimdi de 2 arrayın boyutları, boyutlarının büyüklüğü, satır veya sütun sayısından birinin 1 olmadığı
#  kural 3 e uygun bir örnek yapalım.

a = np.ones((3, 2)) # [[0,0], [0,0], [0,0]]... shape(3,2)
x = np.arange(3) #[0,1,2]... shape(3,) kural 1 işlenir.... shape(1,3)
                    # ardından kural 2 işlenir 1 olan kısım a nın boyutuna kadar gerdilir... shape(3,3)
                    # a.shape(3,2) iken x.shape(3,3) yani kural 3'e çarptık.. 
                    #toplamayı deneyince error alıcaz.
                    
print(a+x, "\n")

#%% Broadcasting in Practice
import matplotlib.pyplot as plt
#centering array

x = np.random.random((10, 3))
x_mean = x.mean(0) #sadece ilk boyutun ortalaması
print(x, "\n") #mean ı bulabiliyoruz ama
print(x_mean, "\n")#centering array dediğimiz olayı yapmak için broadcasting kullanmalıyız
                            
x_centered = x - x_mean #çıkartma işlemi ile x arrayını ortalayabiliriz
print(x_centered.mean(0), "\n")


#plottinng two-dimensional function
"""
    z = f(x, y) şeklinde bir fonksiyon tanımlamak istiyorsak broadcastingden faydalanabiliriz.
    ardından matplotlib ile görselleştiririz.
"""

x = np.linspace(0, 5, 50) #0 dan 5 e 50 adımda
y = np.linspace(0, 5, 50)[:, np.newaxis] #0 dan 5 e 50 adım ama her bir adım birer satır olurken,
                                            # bütün array sadece tek bir sütundan oluşucak
                                            
z = np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

plt.imshow(z, origin="lower", extent=[0, 5, 0, 5], cmap="viridis")
plt.colorbar() #iki boyutlu array görselleştirme

#%% Comparisons, Masks, and Boolean Logic

"""
    bu bölümde boolean mask ı görücez, boolean masklar; numpy arraylarındaki değerleri
    işleme ve manipule etmek için kullanılır. Ayıklama, modifye, sayma ya da numpy arrayındaki
    değerleri belirli kriterlere göre manipule etmek için kullanılan yönteme Maskeleme denir. Mesela
    arraydaki değerlerin içinden belirli bir değerin üstündeki tüm değerleri saydırmak istiyoruz ya da
    kaldırmak istiyoruz, bu gibi durumlarda boolean masking sık sık kullanılan bir yöntem.
    
        Örnek; counting rainy days:
            import pandas as pd
            yagmur = pd.read_csv("samsun2017.csv")["oran"].values
            inc = yagmur / 254.0
            inc.shape # output = (150,) yani günlük yağan yağmurun inç cinsinden 150 adet veri var
"""

# operatörleri ufunclar ile karşılaştırma

x = np.array([1,2,3,4,5])
print(x < 3, "\n")
print(x > 3, "\n")
print(x == 3, "\n")
print((2 * x) == (2 + x), "\n")

# boolean arraylar ile işlemler

x = np.random.RandomState(0)
x = x.randint(10, size= (3, 5))
countTrue = np.count_nonzero(x < 6) # x arrayında 6 dan küçük olanlar True olucak ve nonzero yani false
print(countTrue, "\n")                    # olmayanların sayısı 6 dan küçüklerin sayısını vericek
                                            # aynı işlemi np.sum(x_array < 6) ile de yapabiliriz.
                                            #sum un avantajı, spesifik axisler üzerinde çalışabilirz
print(x, "\n")
print(np.sum(x < 6, axis = 1), "\n")  # Her satırdaki 6 dan küçüklerin sayısı

#np.any(x < 6) eğer 6 dan küçük herhangi bir eleman varsa True döndürür.
#np.all(x < 10) bütün elemanlar 10 dan küçükse true döndürür

print(np.all(x < 6, axis = 1), "\n")
print(np.sum((x < 6) | (x == 6)), "\n")
print(np.sum(~(x < 6) | (x == 6)), "\n") # ~ NOT işareti


# boolean arraylar ile masking

print(x < 5, "\n") # bu şekilde bir sorgu ile boolean array elde edebiliriz.
                #şimdi 5 ten küçük değerleri ayrı bir array olarak elde edersek
                #o zaman masking yapmış olacağız.

print(x[x < 5], "\n") #döndrdüğü tek boyutlu array, x < 5 sorgusundaki truelerin değerleri

#örnek yapalım
# array_adı[sorgu]

boylar = np.array([199,157,185,182,175,177,174,189,122,182,168,163,181,155])

boy_feet = boylar / 30.48

uzunlar = (boy_feet >= 6)

kısalar = (~uzunlar)

print("Boyların ortalaması\n", np.mean(boylar))
print("Boyların(f) ortalaması\n", np.mean(boy_feet))
print("Uzunların ortalaması\n", np.mean(boy_feet[uzunlar]))#eğer np.mean(uzunlar) dersem truelerin ortalamasını
print("Kısaların ortalaması\n", np.mean(boy_feet[kısalar]))#alır. Masking burda devreye giriyor


# and / or VERSUS "&" / "|"

a = np.array([1,0,0,0,1,1,1,0,1], dtype="bool")
b = np.array([1,1,0,1,1,0,1,0,0], dtype="bool")
print(a | b)
#print(a or b) -> hata verecektir çünkü "and" ya da "or" tüm obje için tek bir boolean sonuç döndürürken
                # "|" ya da "&" konu üzerine birden çok boolean döndürebilir, numpy arraylarında
                # çoğunluklu "|" / "&" tercih edilir.
                
                
#%% Fancy Indexing
# fancy indexing; array içinde belirli index elemanları ararken o elemanları bir değişkene
 # atayıp ardından o değişkeni de belirli shapelere sokarak belkide (2,3) size'lık bir arraydan
 # çekmek istediğiniz elemanları (1, 4) lük bir şekilde size sunması gibi.. örnek;
 
x = np.random.RandomState(42) #seed=42, random sayı üretirken hep aynı kalıpta üreticek
x = x.randint(100, size = 10)

print(x, "\n")

fancy_index = [2,5,6,8,9]
fancy_index2 = np.array([[2,5,6],
                         [1,7,8]])  #tek boyutlu bir diziden indexing yaparken boyut değişikliği yapacaksak
                                    # indexing için yeni bir array oluşturmalıyız
                                    
fancy1 = x[fancy_index] #x'in 2, 5, 6, 8, 9 inci indexleri getir
fancy2 = x[fancy_index2] #x'in 1,2,5,6,7,8 inci indexlerini belirlenen arraya göre getir

print("x[fancy_index] = [2,5,6,8,9]","\n", fancy1, "\n")
print("x[fancy_index2] = np.array([[2,5,6],[1,7,8]])","\n", fancy2, "\n")


# fancy indexing iki boyutlu arraylarda da işe yarar

"""
iki boyutlu arrayı ayarladıktan sonra fancy indexing için 2 tane yeni array ayarlarız yeni arrayımızın
 elemanları, indexleri temsil edicek.. x arrayı 2 satırlı olduğu için satır isimli array 0 ile 1
  elemanlarını alabilir.. x arrayımız 5 elemanlı olduğu için ise 0 ile 4 arasında bir değer alabilir.
   şuna eşit olur; x[0][2] -> 0. indexin(satır) 2. indexli(sütun) elemanı
   aynı zamanda satır = np.array([0])
                sütun = np.array([2])
                fancy_x = x[satır, sütun] -> 0. indexin 2. indexli elemanı
"""
x = np.arange(10).reshape((2,5))
satır = np.array([0,1,0,1])
sütun = np.array([4,1,3,0])
x_fancy = x[satır, sütun]
print(x, "\n")
print(x_fancy, "\n")

"""
    elde edeceğimiz yeni fancy indexli arrayımızı çok boyutlu olmasını istiyorsak;
    x[satır[:, np.newaxis], sütun] şekilnde yapabiliriz... bu şekilde;
    satır = [[0],
             [1],
             [0],
             [1]] şekilde olacaktır. ve sütun = [4,1,3,0] olduğu için elde edeceğimiz yeni array
    0. satırın 4,1,3 ve 0 indexli elemanlarını ilk satıra getirir ardından
    1. satırın 4,1,3 ve 0 indexli elemanlarını ardından
    0 ardından yine
    1. satırın 4,1,3 ve 0 indexli elemanlarını getirerek yeni bir (4,4) lük array elde eder.
"""

x_fancy2 = x[satır[:, np.newaxis], sütun]
print(x_fancy2, "\n")

#Combined Indexing

x = np.arange(12).reshape((3,4))
combined_x = x[1, [0,3]] #1 indexli satırın 0 ve 3 indexli sütunları
print(x, "\n")
print(combined_x, "\n")

#ayrıca bu yöntem ile slicing i birleştirerek boyut kazandırabiliriz

combined_x2 = x[1:, [2,3]] #1. index ve sonraki satırlar için 2 ve 3 indexli elemanlar
print(combined_x2, "\n") # x için düşünürsek 1 ve 2. satırların 3 ve 4. sütunları yani 2x2 lik bir array

#fancy indexing ile masking
# bunu yaparken True ye denk gelen elemanlar dönerken false denk gelen elemanlar dönmeyecek

maske = np.array([1,0,0,1], dtype="bool") #true,true,true,true
maske2 = np.array([False, False, False, False])
masked_x = x[satır[:, np.newaxis], maske] #satır = [0,1,0,1], yani 0 ve 1. satırlardan true,false,false,true
masked_x2 = x[satır[:, np.newaxis], maske2] # 0 ve 1. satırlardan 4 tane false, çünkü x arrayının sizesi 4
print(masked_x, "\n")
print(masked_x2, "\n")

"""
    örnek : rastgele noktalar seçme;
        fancy indexing'in en sık kullanıldığı yerlerden birisi matrixler içindeki belirli satırlardan
        alt kümeler oluşturma.
"""

#Fancy indexing ile değerleri modifiye etme

x = np.arange(10)
i = np.array([4,3,7,0])
x[i] = 100      # 4,3,7 ve 0. indexleri 100 yap.
print(x, "\n")
x[i + 1] -= 25 #eğer son indexli elemanı çağırıyor olsaydık hata alırdık çünkü listeden çıkmış olurdu
print(x, "\n")
x[[0, 0]] = [4, 6]
print(x, "\n") # x[0] = 6, neden 4 değil ? çünkü ilk 4 yapıp ardından 6 yaptı.

# ancak:
i = np.array([0,0,1,1,1,2,2]) #indexlerken aynı indexleri göstersek ve...
x[i] += 1 # dersek, 0 indexi iki defa mi artar yoksa bir defa mi?
print(x, "\n") # bir defa arttı çünkü x[i] += 1 ile x[i] + 1 aynı şey, birden çok değil bir defa çalışır.
# ama eğer at() ufunc unu kullanırsak multiple time yani birden fazla kez çalıştırabiliriz
np.add.at(x, i, 1) #(kaynak, indexler, artış sayısı)
print(x, "\n") # görüldüğü gibi 0. index 2 defa 1. index 3 defa artıyor.


# örnek : Binning data (küçükten büyüğe doğru veriyi sıralama, gruplama)

x = np.random.seed(42)
x = np.random.randn(100) #100 elemanlı seedi 42 olan bir random

# histogram'ı(sütun grafiği) elle işleme
bins = np.linspace(-5, 5, 20) # -5 ten 5 e doğru 20 eleman ile
counts = np.zeros_like(bins) #bins arrayı kadar 0 lardan oluşan bir array

# her bir x için uygun kutu(bin) bul
i = np.searchsorted(bins, x)

# tüm kutulara(bins) 1 ekle
np.add.at(counts, i, 1)

#sonucu görselleştir
# import matplotlib.pyplot as plt
plt.plot(bins, counts, linestyle="steps")

#%% Sorting Arrays (Sıralama)

#her ne kadar python un build-in fonksiyonlarında sort methodu varsa np.sort ya da np.argsort
# daha işlevsel ve hızlı, quicksort algoritması kullanılıyor, aynı zamanda mergesort ve heapsort
#  da kullanılabilir.

x = np.array([5,1,2,6,4,8])
print(x, "\n")
print("np.sort     : ", np.sort(x), "\n")
# array_adı.sort() şeklinde de kullanılabilir
x.sort() # array-in-place yöntemi.. artık arrayın orjinal hali sıralı
print("x.sort()    : ", x, "\n")

#np.argsort() ise array ın indexli sıralanmasını yapar mesela [2,0,1,3] arrayında
# en küçük eleman 1. index ardından 2. index ardından 0 ve 3. index yani sonuç = [1,2,0,3]

x = np.array([5,1,2,6,4,8])
print(x, "\n")
print("np.argsort() : ", np.argsort(x), "\n") # bu yöntem fancy indexing ile de kullanılabilir mesela..

y = np.array([1,4,-5,6])
print(y, "\n")
i = np.argsort(y)
print("i = np.argsort(y) : ", i, "\n")
print("y[i]     : ", y[i], "\n")

# satırlar ve sütunlar ile beraber sıralama

rand = np.random.RandomState(45)
x = rand.randint(0, 10, (4, 6))
print(x, "\n")
col = np.sort(x, axis = 0) # sütunları sırala
print("sort 0. axis:\n", col, "\n")
row = np.sort(x, axis = 1)
print("sort 1. axis:\n", row, "\n")

# Kısmi sıralama
#  Bazen tüm arrayı sıralamak yerine belirli bölümleri sıralayabiliriz... en küçük k elemanı arıyor olabiliriz.
#  numpy bu konuda bize np.partition fonksiyonunu sunuyor. partition bir array ve bir k sayısı alır.
#  return edilen değer ise yeni bir arraydır ve en küçük K sayısı arrayın en solundadır, geri kalanlar ise
#  rastgele bir şekilde sağa doğru dizilmiştir.

x = np.array([51,24,-3,4,1,78,0,10])
print(x, "\n")
print("partition(x,3):\n",np.partition(x, 3),"\n") #x arrayının en küçük 3 elemanını sola getir. Sonuç sıralı çıkmayabilir
print("partition(x,5):\n", np.partition(x, 5),"\n") # output = [1 4 -3 0 10 24 51 78] ilk 5 eleman arrayın en küçük 5 elemanı


# multi-dim versiyonu da yapılabilir.
y = np.random.randint(0, 25, (3,5))
print(y, "\n")
print("partition(y, 2, axis=1):\n",np.partition(y, 2, axis=1), "\n") #y arrayının en küçük 2 elemanını sola
                                                                        #döşediği yeni bir array...

# argsort olduğu gibi argpartition da var, orjinal array üzerindeki index değerlerini, yeni arrayda döndürür
print("argpartition(y, 2, axis=1):\n",np.argpartition(y, 2, axis=1), "\n")


"""
ÖRNEK; k-Nearest Neighbors

10x2 lik bir random array ,arraydaki elemanların nasıl göründüğünü grafik
üzerinde noktalarla gösterelim.
x = rand.rand(10, 2) 
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(x[:, 0], x[:, 1], s=100);

yukarıdaki kodu ipython console da çalıştır

Kitaptan devam kısımlarını tam anlamadım
"""


#%% Numpy'nin yapılandırılmış dizileri (Structures Arrays)

"""
    normalde isimler, yaşlar ve kilolar şeklinde 3 ayrı liste oluştursak, birbirinden bağımsız olur ancak
    numpy nin dtype özelliğinden faydalanarak 3 ayrı listeyi sınıflandırabilir, içinde barındıracağı değerlerin
    tipini ayarlyabiliriz.
    U10 -> en fazla 10 karakter alan Unicode string
    i4 ->  4-byte integer(32 bit)
    f8 ->  8-byte float (64 bit)
"""

isim = ["batuhan", "yakup", "ali"]
yas = [23, 25, 22]
kilo = [71.1, 75.5, 95.7]

# şimdi özellikle veri tiplerine göre "yapılandırılmış dizi" yani structured array oluşturucaz.
veri = np.zeros(3, dtype = {"names" : ("isim", "yas", "kilo"),
                            "formats" : ("U10", "i4", "f8")})
    
print(veri.dtype) # output -> [('isim', '<U10'), ('yas', '<i4'), ('kilo', '<f8')]
                    # şimdi gerekli verileri tutacağımız boş bir container arrayımız var.
                    
# şimdi de birbirinden bağımsız 3 listemizi konteynırımıza yerleştirelim
                    
# array_adı["dtype'de verdiğimiz isim"] = <atanmasını istediğimiz bağımsız liste>
veri["isim"] = isim
veri["yas"] = yas
veri["kilo"] = kilo
print(veri) # output -> [('batuhan', 23, 71.1) ('yakup', 25, 75.5) ('ali', 22, 95.7)]
print(veri["isim"]) # tüm isimleri çek      -> ['batuhan' 'yakup' 'ali']
print(veri["yas"]) # tüm yaşları çek        -> [23 25 22]
print(veri[0]) # veri nin ilk satırını çek  -> ('batuhan', 23, 71.1)
print(veri[2]["kilo"]) # veri nin 2. indexli satırının kilo sütununu çek -> 95.7

#boolean masking ile daha spesipik verileri çekebiliriz
x = veri[veri["yas"] < 25]["isim"] # veri nin yaş sütunundan 25 ten küçüklerin, isimlerini çek
print(x) # output -> ['batuhan' 'ali']

# değişken olarak tanımlayarak da dtype belirleyebiliriz.
# daha gelişmiş ve karmaşık, her bir elementin, değerlerden oluşan bir array ya da matrixi tuttuğu
#  bir dtype oluşturabiliriz. bu data type ini <mat> componentiyle oluşturucaz (3x3 lük float matrix)

veri_tipi = np.dtype([("id", "i8"), ("mat", "f8", (3, 3))])
X = np.zeros(1, dtype = veri_tipi)
print(X) # X[0]
print(X["mat"]) # X["mat"][0]

# x arrayının içindeki her bir eleman 3x3 lük matrixten ve id den oluştu.

#%% RecordArrays

isim = ["batuhan", "yakup", "ali"]
yas = [23, 25, 22]
kilo = [71.1, 75.5, 95.7]

vt = np.dtype({"names" : ("isim", "yas", "kilo"), "formats" : ("U10", "i4", "f8")})
veri = np.zeros(3, dtype = vt)

veri["isim"] = isim
veri["yas"] = yas
veri["kilo"] = kilo

print(veri["yas"], veri["yas"].dtype)

# veri arrayımıza bir defa record array olarak bakarsak(view) ve değişkene atarsak
#  daha çabuk ve kolay bir biçimde verimizin özelliklerine, içindekilere bakabiliriz
veri_rec = veri.view(np.recarray)
print(veri_rec.yas)
print(veri_rec)
print(veri_rec.isim)











