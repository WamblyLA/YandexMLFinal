# Generated Faces Detection

Модель компьютерного зрения для бинарной классификации изображений лиц: настоящее изображение или сгенерированное

**Результат: F1 = 0.99708 и 10-е место в лидерборде ML-соревнования Яндекс Лицея**

## Что сделано

- Проведён анализ датасета из 60 000 изображений и дисбаланса классов
- С нуля реализована компактная Xception-подобная CNN на основе depthwise separable convolutions и residual-связей
- С нуля реализована EfficientNet-подобная CNN с MBConv- и Squeeze-and-Excitation-блоками
- Для уменьшения переобучения использованы лёгкие affine-аугментации, dropout и обучение на разных random seeds
- Собран ансамбль из шести моделей: три Xception и три EfficientNet
- Предсказания моделей объединены stacking-классификатором LightGBM; финальный порог подобран по F1 на validation-выборке

## Архитектура решения

```text
Изображение лица
      │
      ├── 3 × SmallXception ──┐
      │                       ├── вероятности ── LightGBM ── threshold ── класс
      └── 3 × SmallEfficientNet 
```

`SmallXception` отделяет обработку пространственных признаков от смешивания каналов с помощью depthwise и pointwise свёрток. `SmallEfficientNet` использует MBConv-блоки и channel attention. Разные архитектуры и random seeds уменьшают корреляцию ошибок, а LightGBM учится определять вклад каждой модели в итоговый прогноз

## Содержимое репозитория

- [`Generated_Faces_Detection.ipynb`](./Generated_Faces_Detection.ipynb) — анализ данных, реализации обеих CNN, обучение, ансамблирование и создание submission

## Стек

Python, PyTorch, torchvision, NumPy, pandas, scikit-learn, LightGBM, Pillow, Matplotlib

## Запуск

Ноутбук ожидает следующую структуру данных

```text
dataset/
├── train_images/
├── test_images/
└── train_solution.csv
```

После размещения датасета откройте `Generated_Faces_Detection.ipynb` и выполните ячейки сверху вниз. Датасет, веса моделей и промежуточные предсказания не включены в репозиторий из-за их размера
