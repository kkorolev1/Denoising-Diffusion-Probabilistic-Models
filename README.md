# Denoising-Diffusion-Probabilistic-Models
Реализация диффузионной модели из статьи "Denoising Diffusion Probabilistic Models"

Частично основано на https://github.com/BrianPulfer/PapersReimplementations/tree/master/ddpm

Данные взяты из датасета MNIST
![alt text](images/MNIST.jfif "MNIST")

DDPM состоит из двух методов - forward и backward, отвечающие за прямой и обратный процесс соответственно. 
В forward вычисляется x_t по x_0 и моменту времени t, как в статье. 
Backward представляет из себя черный ящик в виде некоторой модели, которая аппроксимирует шум, который был наложен в процессе деформации изображения.

### Визуализация прямого процесса
#### t = 0
![alt text](images/forward_process[t=0].png "fp0")
#### t = 250
![alt text](images/forward_process[t=250].png "fp250")
#### t = 500
![alt text](images/forward_process[t=500].png "fp500")
#### t = 750
![alt text](images/forward_process[t=750].png "fp750")
#### t = 1000
![alt text](images/forward_process[t=1000].png "fp1000")

За обратный процесс отвечает UNet с использованием синусных эмбеддингов для кодирования времени.
Процесс обучения заключается в выборе случайного сэмпла из обучающей выборки, генерации нормального шума eps, выборе случайного момента времени t. Далее прямой процесс зашумляет изображение, после чего оно подается на вход обратному процессу, который ищет оценку для eps. Считаем MSE loss для шума. В качестве оптимизатора был выбран Adam(lr=1e-3). Количество эпох равно 20 и такое значение было выбрано в целях экономии времени, однако уже видны некоторые успехи в сэмплировании. 

### График MSE во время обучения
![alt text](images/training_loss.png "MNIST")

### Сэмплирование

Сэмплирование заключается в полном проходе назад в обратном процессе. Начиная с абсолютного шума, мы делаем оценку на eps, после чего по выражению из статьи уточняем x_{t-1} по x_t и eps_theta.

#### t = 900
![alt text](images/sampled[t=900].png "s900")
#### t = 800
![alt text](images/sampled[t=800].png "s800")
#### t = 700
![alt text](images/sampled[t=700].png "s700")
#### t = 600
![alt text](images/sampled[t=600].png "s600")
#### t = 500
![alt text](images/sampled[t=500].png "s500")
#### t = 400
![alt text](images/sampled[t=400].png "s500")
#### t = 300
![alt text](images/sampled[t=300].png "s500")
#### t = 200
![alt text](images/sampled[t=200].png "s500")
#### t = 100
![alt text](images/sampled[t=100].png "s500")
#### t = 0
![alt text](images/sampled[t=0].png "s500")

Несмотря на количество эпох, некоторые цифры уже можно разглядеть, на 30 эпохах примерно такой же результат.

### Скрипт
Поддерживает 2 команды: train и sample.

#### Обучение
`python ddpm.py train --batch-size 64 --epochs 20 --path ddpm.pt`

#### Сэмплирование
`python ddpm.py sample --samples 16 --path ddpm.pt`

Полный набор аргументов доступен через --help.