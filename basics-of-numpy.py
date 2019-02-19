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









