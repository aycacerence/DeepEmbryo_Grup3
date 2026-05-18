# Dataset Analysis Report

## 1. Overview
- **Total Images:** 170
- **Classes:** 13
- **Imbalance Ratio (Max/Min):** 14.67

## 2. Class Statistics
| Class    |   Count |   Avg Width |   Avg Size (KB) |   Min Size (KB) |   Max Size (KB) |   Avg Brightness |   Avg Pixel Std |   Percentage (%) |
|:---------|--------:|------------:|----------------:|----------------:|----------------:|-----------------:|----------------:|-----------------:|
| 3AA      |      29 |         640 |         900.053 |         900.053 |         900.053 |          148.241 |         19.2068 |            17.06 |
| 3AB      |       5 |         640 |         900.053 |         900.053 |         900.053 |          148.063 |         17.7707 |             2.94 |
| 3BA      |       6 |         640 |         900.053 |         900.053 |         900.053 |          141.999 |         19.5859 |             3.53 |
| 3BB      |       8 |         640 |         900.053 |         900.053 |         900.053 |          141.761 |         19.2945 |             4.71 |
| 3BC      |       6 |         640 |         900.053 |         900.053 |         900.053 |          150.657 |         18.6711 |             3.53 |
| 3CA      |       4 |         600 |         900.053 |         900.053 |         900.053 |          159.654 |         17.4528 |             2.35 |
| 3CB      |       3 |         640 |         900.053 |         900.053 |         900.053 |          146.612 |         19.844  |             1.76 |
| 3CC      |      19 |         640 |         900.053 |         900.053 |         900.053 |          148.507 |         17.6908 |            11.18 |
| 4AA      |      44 |         640 |         900.053 |         900.053 |         900.053 |          147.958 |         18.6915 |            25.88 |
| 4AB      |       7 |         640 |         900.053 |         900.053 |         900.053 |          144.725 |         19.6812 |             4.12 |
| 4BA      |       7 |         640 |         900.053 |         900.053 |         900.053 |          148.559 |         18.9924 |             4.12 |
| 4BB      |       3 |         640 |         900.053 |         900.053 |         900.053 |          146.805 |         18.9836 |             1.76 |
| Cleavage |      29 |         640 |         900.053 |         900.053 |         900.053 |          146.773 |         17.238  |            17.06 |

## 3. Data Quality Assessment
- **Duplicates detected:** 3 potential sets
- **Very dark images (<30 mean):** 0
- **Very bright images (>220 mean):** 0

## 4. Visualizations
![Class Distribution](visualizations/class_distribution.png)
![Brightness Distribution](visualizations/brightness_distribution.png)

## 5. Recommendations for Augmentation
1. **Minority Classes:** Classes like 3CB, 4BB, 3CA are severely underrepresented. **Heavy augmentation** (Rotation, Flip, Zoom) is necessary for these.
2. **Normalization:** Brighness varies across images. **Global normalization** or Histogram Equalization might help.
3. **Resizing:** Most images have similar aspects. Resizing to 224x224 (ResNet50) will work fine.
