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


# indexing -----------------------------------------------














