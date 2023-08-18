import sys
import numpy as np

from src.common.plot_3d import plot_3d
from src.common.polar_to_xy import polar_to_xy

sys.path.append("src")
sys.path.append("src/common")
sys.path.append("src/interface")


def main():
    # distances = [263, 268, 263, 251, 269, 269, 258, 269, 268, 274, 252, 264, 260, 279, 265, 269, 266, 274, 277, 276, 271, 275, 269, 265, 282, 269, 272, 287, 275, 287, 275, 267, 276, 279, 271, 288, 279, 291, 276, 288, 293, 300, 287, 291, 296, 295, 290, 284, 304, 311, 295, 291, 315, 307, 306, 299, 287, 295, 322, 303, 321, 310, 297, 318, 310, 303, 316, 308, 294, 318, 319, 315, 331, 315, 332, 317, 315, 321, 331, 336, 339, 332, 341, 328, 333, 340, 335, 338, 336, 329, 349, 337, 366, 340, 343, 351, 368, 362, 349, 365, 348, 376, 356, 372, 367, 377, 369, 373, 398, 379, 382, 394, 391, 381, 398, 387, 393, 398, 391, 409, 406, 403, 403, 413, 427, 417, 434, 414, 426, 419, 443, 423, 442, 434, 448, 441, 441, 446, 441, 515, 559, 546, 616, -1, 807, -1, -1, -1, -1, 869, -1, -1, 1114, -1, -1, -1, -1, 801, 793, 853, 856, -1, -1, 852, -1, -1, -1, -1, -1, 2491, -1, -1, -1, -1, -1, -1, 2479, -1, -1, -1, 2497, 2489, 2489, 2482, 2507, 2497, 2505, 2506, 2510, 2510, 2513, 2518, 2527, 2523, 2489, 2511, 2519, 1878, 1834, 1839, -1, 314, 340, 345, 324, 333, 339, 348, 349, 359, 361, 343, 344, 349, 345, 347, 360, 393, 402, 429, 444, 454, 454, 468, 476, 474, 483, 513, 511, 500, 526, 539, 538, 569, 537, 550, 551, 552, 524, 551, 479, 439, 378, 345, 359, 318, 302, 278, 254, 224, 188, 188, 161, 145, 163, 150, 149, 143, 148, 155, 166, 147, 162, 143, 142, 150, 146, 140, 144, 145, 143, 142, 154, 143, 140, 157, 156, 144, 148, 144, 120, 140, 139, 127, 146, 134, 150, 154, 158, 140, 130, 131, 136, 140, 137, 135, 145, 129, 128, 144, 137, 124, 139, 143, 141, 134, 151, 153, 126, 138, 127, 126, 140, 137, 119, 136, 128, 146, 140, 128, 131, 132, 143, 139, 138, 143, 141, 137, 132, 138, 150, 134, 142, 126, 149, 148, 143, 156, 130, 148, 133, 135, 150, 138, 152, 137, 132, 154, 142, 140, 156, 152, 140, 147, 131, 150, 134, 164, 170, 159, 162, 164, 170, 174, 164, 171, 182, 172, 182, 185, 219, 238, 268, 347, 362, 404, 409, 399, 394, 417, 435, 413, 434, 406, 454, 433, 426, 437, 449, 417, 449, 449, 453, 439, 449, 449, 441, 450, 441, 448, 429, 457, 451, 465, 486, 471, 450, 462, 486, 488, 473, 485, 476, 481, 465, 489, 454, 477, 474, 486, 475, 453, 480, 494, 487, 486, 483, 486, 506, 493, 507, 495, 467, 485, 494, 488, 498, 475, 492, 504, 496, 510, 497, 476, 513, 479, 503, 505, 514, 520, 495, 492, 505, 532, 519, 509, 508, 498, 533, 532, 526, 543, 522, 546, 535, 516, 529, 530, 537, 545, 537, 533, 534, 539, 528, 550, 534, 549, 533, 545, 567, 579, 556, 546, 570, 558, 560, 558, 567, 594, 588, 584, 587, 583, 572, 582, 513, 467, 468, 470, 451, 479, 456, 471, 489, 465, 462, 479, 483, 474, 491, 492, 476, 506, 473, 509, 476, 497, 494, 508, 503, 522, 511, 513, 497, 534, 515, 527, 517, 519, 534, 517, 504, 501, 553, 529, 518, 525, 521, 579, 560, 552, 561, 543, 565, 518, 565, 574, 594, 557, 576, 598, 574, 582, 579, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 711, 748, 781, 748, 757, 790, 811, 806, 789, 805, 777, 767, 784, 788, 768, 794, 790, 773, 809, 779, 796, 775, 804, 765, 752, 762, 765, 787, 805, 808, 809, 817, 800, 815, 826, 826, 818, 808, 819, 826, 828, 822, 825, 822, 829, 836, 839, 819, 837, 857, 841, 836, 858, 838, 818, 835, 845, 837, 863, 833, 852, 843, 868, 860, 878, 877, 862, 876, 886, 894, 900, 894, 914, 919, 929, 954, 1026, 1149, 1286, 1358, 1452, 1730, 1743, 1741, 1708, 1478, 1339, 1264, 1175, 1198, 1261, 1338, 1285, 1172, 1110, 1124, 1112, 1100, 1118, 1117, 1096, 1083, 1078, 1077, 1095, 1071, 1104, 1064, 1082, 1070, 1080, 1087, 1071, 1064, 1070, 1064, 1078, 1076, 1096, 1079, 1091, 1087, 1087, 1077, 1080, 1080, 1084, 1078, 1094, 1092, 1095, 1087, 1086, 1107, 1086, 1120, 1113, -1, -1, 1474, 1449, 1448, 1343, 1303, 1311, 1321, 1333, 1315, 1309, 1308, 1310, 1303, 1290, 1298, 1301, 1303, 1287, 1293, 1275, 1284, 1275, 1265, 1066, 932, 856, 830, 832, 843, 802, 820, 815, 816, 793, 802, 773, 712, 705, 705, 685, 677, 663, 665, 657, 648, 659, 653, 638, 625, 667, 628, 630, 631, 614, 621, 621, 619, 635, 608, 632, 627, 625, 619, 620, 625, 622, 594, 590, 578, 588, 581, 571, 584, 571, 580, 597, 588, 580, 573, 576, 571, 566, 571, 569, 579, 539, 534, 560, 543, 538, 526, 534, 546, 533, 534, 532, 511, 525, 520, 516, 486, 521, 495, 496, 513, 507, 490, 501, 506, 483, 482, 424, 435, 427, 416, 430, 420, 420, 438, 402, 420, 411, 387, 401, 392, 387, 395, 390, 397, 403, 395, 393, 391, 394, 395, 371, 382, 388, 393, 399, 395, 401, 399, 391, 396, 393, 399, 411, 395, 407, 399, 388, 390, 396, 398, 404, 398, 401, 401, 404, 413, 407, 418, 406, 417, 421, 431, 424, 432, 419, 434, 406, 414, 427, 416, 437, 412, 414, 442, 450, 503, 576, 703, 744, 804, 906, 945, 1173, 1187, 1191, 1203, 1207, 1207, 1208, 1207, 1226, 1225, 1229, 1245, 1215,
    #              1223, 1240, 1236, 1219, 1212, 1237, 1210, 1228, 1244, 1219, 1237, 1233, 1228, 1233, 1221, 1239, 1231, 1233, 1253, 1238, 1259, 1258, 1245, 1265, 1245, 1266, 1267, 1295, 1376, 1475, 1415, 1417, 1397, 1399, 1399, 1397, 1378, 1392, 1396, 1379, 1384, 1385, 1380, 1387, 1400, 1390, 1395, 1392, 1403, 1416, 1411, 1418, 1501, 1843, 2291, 2284, 2295, 2274, 2270, 2281, 2264, 2262, 2208, 2236, 2222, 2221, 2211, 2176, 2203, 2213, 2193, 2160, 2172, 2165, 2172, 2136, 2158, 2155, 2174, 2158, 2163, 2131, 2132, 2130, 2128, 2122, 2113, 2104, 2097, 2100, 2097, 2089, 2085, 2080, 2082, 2067, 2059, 2071, 2058, 2049, 2060, 2043, 2046, 2046, 2032, 2037, 2018, 2015, 2005, 2007, 2002, 2007, 2001, 1994, 1983, 1978, 1988, 1969, 1976, 1964, 1969, 1967, 1960, 1953, 1956, 1936, 1952, 1948, 1926, 1920, 1908, 1892, 1558, 1111, 1073, 1057, 1064, 1073, 1055, 1079, 1060, 1061, 1042, 1060, 1056, 1043, 1047, 1047, 1017, 1045, 1051, 1052, 1022, 1053, 1045, 1070, 1035, 1022, 1032, 1017, 1018, 1014, 1034, 1024, 1026, 1024, 1033, 1047, 1051, 1034, 1016, 1017, 1021, 1019, 1001, 996, 1008, 1008, 993, 995, 998, 1005, 997, 997, 1019, 1002, 1009, 1009, 1012, 1007, 1013, 1013, 1004, 1020, 1010, 996, 1006, 1013, 991, 1008, 1015, 1006, 1013, 1005, 1004, 1001, 999, 1007, 1001, 991, 988, 984, 995, 794, 627, 498, 488, 490, 485, 486, 469, 462, 482, 472, 463, 475, 468, 476, 472, 478, 460, 497, 474, 476, 477, 495, 496, 488, 499, 485, 474, 480, 494, 489, 480, 513, 494, 487, 497, 505, 501, 489, 484, 497, 519, 502, 515, 502, 507, 504, 497, 512, 513, 515, 519, 514, 519, 514, 515, 528, 532, 506, 539, 525, 529, 528, 537, 519, 535, 541, 529, 538, 530, 522, 531, 559, 539, 553, 550, 552, 549, 546, 535, 550, 548, 541, 560, 564, 566, 552, 555, 553, 572, 555, 582, 573, 578, 574, 565, 570, 583, 576, 594, 583, 590, 577, 577, 591, 592, 589, 600, 598, 618, 612, 619, 605, 611, 591, 613, 615, 618, 625, 612, 628, 637, 639, 631, 638, 621, 655, 654, 640, 642, 669, 649, 653, 651, 646, 654, 675, 656, 678, 686, 713, 794, 839, 1023, 1843, 1840, 1843, 1854, 1839, 1864, 1854, 1849, 1865, 1855, 1871, 1870, 1872, 1852, 1870, 1878, 1873, 1871, 1893, 1880, 1891, -1, 599, 596, 523, 573, 555, 544, 559, 544, 518, 536, 513, 522, 479, 495, 502, 511, 497, 462, 535, 701, 743, 518, 371, 280, 270, 214, 229, 259, 329, 394, 388, 439, 430, 406, 402, 404, 396, 388, 404, 384, 391, 377, 367, 383, 366, 355, 365, 365, 347, 368, 392, 365, 356, 336, 351, 341, 344, 367, 323, 328, 316, 390, 451, 316, 281, 252, 246, 237, -1, 246, 242, -1, -1, -1, 2212, 2236, 2227, 2242, 2243, 2256, 2266, 2259, 2283, 2281, 2290, 2301, 2309, 2299, 2309, 2317, 2316, 2329, 2347, 2352, 2351, 2350, 2371, 2374, 2384, 2391, 2394, 2407, 2416, 2411, 2431, 2433, 2452, 2444, 2453, 2471, 2487, 2486, 2497, 2513, 2523, 2540, 2537, 2550, 2556, 2592, 2593, -1, -1, -1, -1, 1004, 941, 912, 896, 880, 898, 898, 889, 900, 889, 831, 648, 509, 429, 320, 379, 469, 604, 801, 861, 869, 869, 884, 879, 878, 856, 874, 865, 870, 867, 866, 863, 856, 840, 840, 863, 822, 843, 830, 833, 768, 557, 432, 429, 287, 162, 151, 156, 211, 325, 332, 479, 574, 648, 712, 711, 721, 764, 764, 749, 749, 736, 735, 735, 738, 728, 740, 706, 693, 720, 708, 623, 569, 491, 433, 347, 277, 267, 253, 285, 340, 402, 514, 583, 627, 658, 681, 705, 668, 693, 693, 645, 692, 695, 698, 691, 709, 711, 690, 705, 691, 681, 665, 683, 673, 670, 686, 690, 693, 672, 691, 676, 673, 693, 675, 670, 676, 653, 656, 667, 665, 644, 650, 678, 678, 668, 668, 663, 659, 664, 653, 666, 662, 661, 635, 633, 655, 654, 639, 643, 622, 646, 633, 646, 653, 657, 647, 632, 617, 564, 477, 377, 269, 374, 462, 518, 587, 640, 647, 642, 641, 649, 630, 641, 645, 614, 608, 475, 324, 262, 228, 234, 215, 191, 208, 188, 211, 228, 194, 179, 185, 153, 197, 203, 204, 205, 191, 187, 179, 208, 183, 185, 187, 185, 157, 185, 174, 155, 180, 148, 169, 148, 128, 180, 205, 214, 237, 261, 281, 252, 240, 167, 115, 157, 316, 325, 356, 388, 398, 366, 372, 384, 399, 404, 416, 430, 426, 447, 464, 492, 520, 490, 455, 454, 458, 468, 462, 460, 472, 450, 454, 463, 470, 471, 479, 475, 469, 479, 479, 469, 467, 434, 414, 376, 339, 359, 370, 372, 400, 382, 410, 381, 386, 396, 391, 398, 389, 393, 387, 397, 402, 391, 388, 387, 395, 375, 404, 388, 387, 386, 397, 393, 385, 393, 385, 393, 382, 392, 376, 361, 361, 355, 362, 350, 335, 323, 325, 334, 323, 324, 316, 316, 295, 286, 290, 287, 293, 280, 291, 277, 285, 286, 274, 280, 257, 281, 269, 253, 261, 253, 267, 247, 248, 254, 248, 259, 254, 233, 242, 232, 252, 238, 244, 235, 236, 240, 235, 247, 230, 243, 238, 240, 244, 249, 227, 250, 247, 243, 251, 253, 242, 247, 238, 238, 247, 248, 248, 258, 241, 254, 254, 247, 254, 233, 247, 250, 254, 240, 252, 265, 255, 247, 259, 231, 257, 255, 256, 255]

    xyz = np.random.rand(10000000, 3)

    plot_3d(xyz)


if __name__ == "__main__":
    main()
