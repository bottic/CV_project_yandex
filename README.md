# Проект: Разработка, обучение и использование нейросети для классификации деятельности человека по изображениям
**Ссылка на проект:** [Kaggle Competition](https://www.kaggle.com/competitions/ml-intensive-yandex-academy-autumn-2024/models)  
**Итоговый код:** [GitHub Repository](https://github.com/bottic/CV_project_yandex/blob/main/final-ml-yandex-spec.ipynb)

---

## Цель проекта
Классификация вида деятельности человека на основе изображений.  
Итоговое решение — усредненный ансамбль, включающий:  
- Три модели с модифицированной EfficientNet  
- Две модели с модифицированной ResNet  

Все модели были обучены "с нуля" на собранных данных без предварительного обучения.

---

## Выполненные задачи

### 1. Аугментация данных
Разработана стратегия аугментации на основе анализа научных статей, включающая:  
- Случайное кадрирование и изменение размера  
- Вращения и горизонтальное отражение  
- Изменение цветовых характеристик  
- Стирание или маскирование областей изображения  

**Результаты:**  
- Увеличение объема датасета с 12,000 до 19,000 изображений с сохранением 10% для валидации.  
- Повышение устойчивости модели к различным условиям благодаря улучшенной обобщающей способности.

<details>
<summary>Распределение классов в датасете</summary>

![Распределение классов](https://github.com/user-attachments/assets/fd78cc11-3f73-4601-962a-4792b2f811c9)
</details>

---

### 2. Тестирование различных архитектур
Оценены и протестированы архитектуры:  
- **AlexNet**  
- **ResNet**  
- **EfficientNet**  
- **MobileNet**  
- **RegNet**  
- **SqueezeNet**  

**Вывод:** модели с меньшим числом параметров показали лучшие результаты на имеющемся объеме данных, обеспечив высокие значения F1-метрики на валидационном датасете.

---

### 3. Оптимизация обучения
Исследованы и настроены:  
- Различные гиперпараметры  
- Функции потерь  
- Оптимизаторы  

Достигнуто оптимальное соотношение качества обучения и производительности модели.

---

![image](https://github.com/user-attachments/assets/1f9c2913-4b75-4366-8887-1afe56d9c6d8)
