import numpy as np
import random
import timeit


start = timeit.default_timer()

iterasyon_sayisi = 200
tabu_tenure = 170
# tabu_tenure = random.randint(7, 20) # dynamic list size
departman_sayisi = 20
N = 190 #20'nin 2'lisi ile oluşacak komşu sayısı

akis_matrisi=   [[0, 0, 5,0, 5,2,10,3,1, 5, 5, 5, 0, 0, 5, 4, 4, 0, 0, 1 ],
                [0, 0, 3,10,5,1, 5,1,2, 4, 2, 5, 0,10,10, 3, 0, 5,10, 5 ],
                [5, 3, 0,2, 0,5, 2,4,4, 5, 0, 0, 0, 5, 1, 0, 0, 5, 0, 0 ],
                [0,10, 2,0, 1,0, 5,2,1, 0,10, 2, 2, 0, 2, 1, 5, 2, 5, 5 ],
                [5, 5, 0,1, 0,5, 6,5,2, 5, 2, 0, 5, 1, 1, 1, 5, 2, 5, 1 ],
                [2, 1, 5,0, 5,0, 5,2,1, 6, 0, 0,10, 0, 2, 0, 1, 0, 1, 5 ],
                [10,5, 2,5, 6,5, 0,0,0, 0, 5,10, 2, 2, 5, 1, 2, 1, 0,10 ],
                [3, 1, 4,2, 5,2, 0,0,1, 1,10,10, 2, 0,10, 2, 5, 2, 2,10 ],
                [1, 2, 4,1, 2,1, 0,1,0, 2, 0, 3, 5, 5, 0, 5, 0, 0, 0, 2 ],
                [5,4, 5,0, 5,6, 0,1,2, 0, 5, 5, 0, 5, 1, 0, 0, 5, 5, 2 ],
                [5,2, 0,10,2,0, 5,10,0,5, 0, 5, 2, 5, 1,10, 0, 2, 2, 5 ],
                [5,5, 0,2, 0,0,10,10,3,5, 5, 0, 2,10, 5, 0, 1, 1, 2, 5 ],
                [0,0, 0,2, 5,10,2,2, 5,0, 2, 2, 0, 2, 2, 1, 0, 0, 0, 5 ],
                [0,10,5,0, 1,0, 2,0, 5,5, 5,10, 2, 0, 5, 5, 1, 5, 5, 0 ],
                [5,10,1,2, 1,2, 5,10,0,1, 1, 5, 2, 5, 0, 3, 0, 5,10,10 ],
                [4, 3,0,1, 1,0, 1,2, 5,0,10, 0, 1, 5, 3, 0, 0, 0, 2, 0 ],
                [4, 0,0,5, 5,1, 2,5, 0,0, 0, 1, 0, 1, 0, 0, 0, 5, 2, 0 ],
                [0, 5,5,2, 2,0, 1,2, 0,5, 2, 1, 0, 5, 5, 0, 5, 0, 1, 1 ],
                [0,10,0,5, 5,1, 0,2, 0,5, 2, 2, 0, 5,10, 2, 2, 1, 0, 6 ],
                [1, 5,0,5, 1,5,10,10,2,2, 5, 5, 5, 0,10, 0, 0, 1, 6, 0 ]]

mesafe_matrisi=[[0,1,2,3,4,1,2,3,4, 5, 2, 3, 4, 5, 6, 3, 4, 5, 6, 7 ],
                [1,0,1,2,3,2,1,2,3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5, 6 ],
                [2,1,0,1,2,3,2,1,2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 4, 5 ],
                [3,2,1,0,1,4,3,2,1, 2, 5, 4, 3, 2, 3, 6, 5, 4, 3, 4 ],
                [4,3,2,1,0,5,4,3,2, 1, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3 ],
                [1,2,3,4,5,0,1,2,3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6 ],
                [2,1,2,3,4,1,0,1,2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5 ],
                [3,2,1,2,3,2,1,0,1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4 ],
                [4,3,2,1,2,3,2,1,0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3 ],
                [5,4,3,2,1,4,3,2,1,0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2 ],
                [2,3,4,5,6,1,2,3,4,5, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5 ],
                [3,2,3,4,5,2,1,2,3,4, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4 ],
                [4,3,2,3,4,3,2,1,2,3, 2, 1, 0, 1, 2, 3, 2, 1, 2, 3 ],
                [5,4,3,2,3,4,3,2,1,2, 3, 2, 1, 0, 1, 4, 3, 2, 1, 2 ],
                [6,5,4,3,2,5,4,3,2,1, 4, 3, 2, 1, 0, 5, 4, 3, 2, 1 ],
                [3,4,5,6,7,2,3,4,5,6, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4 ],
                [4,3,4,5,6,3,2,3,4,5, 2, 1, 2, 3, 4, 1, 0, 1, 2, 3 ],
                [5,4,3,4,5,4,3,2,3,4, 3, 2, 1, 2, 3, 2, 1, 0, 1, 2 ],
                [6,5,4,3,4,5,4,3,2,3, 4, 3, 2, 1, 2, 3, 2, 1, 0, 1 ],
                [7,6,5,4,3,6,5,4,3,2, 5, 4, 3, 2, 1, 4, 3, 2, 1, 0 ]]

#atama maliyetini mesafe ve akış matris çarpımıyla hesaplayan fonksiyon 
def akis_maliyeti(sol):
  cost=0
  for i in range(departman_sayisi):
    for j in range(departman_sayisi):
        cost+=mesafe_matrisi[i][j] *akis_matrisi[sol[i]][sol[j]]
  return cost

#swap operatörü ile komşulukları oluşturan fonksiyon
komsular = np.zeros((N, departman_sayisi), dtype=int)s

#olusabilecek 2'li swap operasyonları yapılır ve bir matriste tüm komsular tutulur
def swap_operatoru(sol_n):
    global idx, komsular
    for i in range (departman_sayisi):
        j=i+1
        for j in range(departman_sayisi):
            if  i<j:
                idx=idx+1
                sol_n[j], sol_n[i] = sol_n[i], sol_n[j] #♣swap islemi
                komsular[idx] = sol_n
                #neighbors[idx, -2:] = [sol_n[i], sol_n[j]]
                sol_n[i], sol_n[j] = sol_n[j], sol_n[i]


#çözümün tabu listesinde olup olmadığı kontrol edilir/çözüm seçilirse tabu listesine eklenir
def tabu_listesinde_mi (solution, tabu):
    not_found = False
    if not solution.tolist() in tabu:
        solution[0], solution[1] = solution[1], solution[0]
        if not solution.tolist() in tabu:
            not_found = True

    return not_found

#TABU ARAMA
def tabu_arama():
    global komsular, iterasyon_sayisi, idx
    #INITILIZATION STEP
    random.seed(8)
    
    mevcut_cozum = random.sample(range(departman_sayisi), departman_sayisi) #mevcut çözüm random olarak seçilir
    incumbent = mevcut_cozum #en iyi çözüm mevcut çözüm olarak atanır
    Tabu = [] #tabu listesi boş halde yaratılır
    ziyaretSayisi = {} #frequency based memory'de kullanacağız 

    print("Initial: %s cost %s " % (mevcut_cozum, akis_maliyeti(mevcut_cozum))) #random seçilen başlangıç çözümünün değeri
    
    #200 iterasyon boyunca tabu arama devam etsin: (iterasyonu 200'den başlattık en başta, sonda 1'er eksilterek while 
    #condition'da kullandık
    while iterasyon_sayisi > 0:
        idx = -1
        swap_operatoru(mevcut_cozum)  # mevcut çözümün komşuları oluşturulur
        # komşuluktaki komşu çözümlerin maliyetlerinin tutulacağı matris boş yaratılır
        cost = np.zeros((len(komsular)))  
        
        #mevcut çözümün komşuluğundaki tüm çözümlerin maliyetleri hesaplanır ve matris doldurulur
        for index in range(len(komsular)):
            cost[index] = akis_maliyeti(komsular[index])  
        
        #best improvement stratejisi gözetilir
        rank = np.argsort(cost)  # düşük maliyetten yükseğe göre sıralama yapılır ve sıralanmış indisler tutulur
        komsular = komsular[rank] # komşular mliyetlerine göre büyükten küçüğe sıralanır
        
        
        for j in range(N):
         #seçilen komşunun tabu listesinde olup olmadığı kontrol edilir: binary değişkende tutulur
            not_tabu = tabu_listesinde_mi(komsular[j], Tabu)
            
            #eğer komşu tabu listesinde değilse (true dönerse) 
            #mevcut çözüm olarak j indisli komşu seçilir ve sonrasında tabu listesine eklenir
            if (not_tabu):
                mevcut_cozum = komsular[j].tolist() #tolist komutu: "return array to a list"
                Tabu.append(komsular[j].tolist()) #.append komutu ile seçilen komşu tabu listesinin sonuna eklenir
                
                #TABU LİSTESİ GÜNCELLEMESİ:
                if len(Tabu) > tabu_tenure-1: #Eğer Tabu listesi uzunluğuna erişildi ise
                    Tabu = Tabu[1:] #Tabu listesinden FIFO sırası ile çözüm çıkarılır.
                    
                    
                
                #frequency based hafıza, daha sık seçilen çözümün seçilme olasılığını azaltmaya yönelik diversification
                #stratejisi, daha sık seçilen çözümün amaç fonksiyon değeri frequency'si kadar artırılarak 'cezalandırılır'
              
                #mevcut çözüm iterayonlarda ilk kez seçildiyse, rastlanma sıklığı 1 atanır
                if not tuple(mevcut_cozum) in ziyaretSayisi.keys():
                    ziyaretSayisi[tuple(mevcut_cozum)] = 1 
                    
                    #incumbent güncellensin mi?
                    if akis_maliyeti(mevcut_cozum) <  akis_maliyeti(incumbent): 
                        incumbent = mevcut_cozum
                        
                #mevcut çözüm iterasyonlarda daha önceden seçilen çözümler arasındaysa rastlanma sıklığına göre 
                #diversification stratejisi güdülerek tekrar seçilme ihtimali azaltılmak üzere
                # amaç fonksiyonu değeri uygun şekilde güncellenir
                else:
                    #daha sık seçilen çözümün amaç fonksiyon değeri frequency'si kadar artırılarak 'cezalandırılır'
                    cur_cost= akis_maliyeti(mevcut_cozum) + ziyaretSayisi[tuple(mevcut_cozum)] # penalize by frequency
                    ziyaretSayisi[tuple(mevcut_cozum)] += 1   # increament the frequency for the current visit
                    
                    #cezalandırılmış amaç fonksiyon değeri en iyi çözümden daha iyiyse, yeni çözüm olarak o seçilir,
                    #yani incumbent güncellenir 
                    if cur_cost <  akis_maliyeti(incumbent):
                        incumbent = mevcut_cozum
                        
                break #for kırılır: komşuya tabu listesinde rastlandıysa

#Aspiration kriteri: şu ana kadar görülmemiş kadar iyi 'attractive' çözümlere tabu listesinde rastlandıysa
            
            #incumbent'tan bile daha iyi bir çözüm ise, o komşu seçilir ve tabu listesine eklenir
            elif akis_maliyeti(komsular[j]) <  akis_maliyeti(incumbent):
                mevcut_cozum = komsular[j].tolist()            
                Tabu.append(komsular[j].tolist())


                if len(Tabu) > tabu_tenure - 1:
                    Tabu = Tabu[1:]

                    # frequency based
                if not tuple(mevcut_cozum) in ziyaretSayisi.keys():
                    ziyaretSayisi[tuple(mevcut_cozum)] = 1  
                    incumbent = mevcut_cozum
                   
                else:

                    cur_cost= akis_maliyeti(mevcut_cozum) + ziyaretSayisi[tuple(mevcut_cozum)] # penalize by frequency
                    ziyaretSayisi[tuple(mevcut_cozum)] += 1   # increament the frequency for the current visit

                    if cur_cost <  akis_maliyeti(incumbent):
                        incumbent = mevcut_cozum

        
            
        iterasyon_sayisi -= 1

#TABU ARAMA SONUCUNDA ELDE EDİLEN NİHAİ ÇÖZÜM VE AMAÇ FONKSİYON DEĞERİ
    print("Best sol %s cost: %s " % (incumbent, akis_maliyeti(incumbent)))


if __name__== "__main__":  # calling the main function, where the program starts running
    tabu_arama()
stop = timeit.default_timer()
print('ÇALIŞMA SÜRESİ: ', stop - start)