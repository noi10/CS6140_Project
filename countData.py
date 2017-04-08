import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

label = ["Game","Vehicle","Concert","Food","Animal","Football","Mobile Phone","Toy","Outdoor Recreation","Nature"]
n = [13068, 10996, 9659, 3564, 2736, 1768, 1527, 1071, 895, 887]

sns.set_style("whitegrid", {'axes.grid' : False})
g = sns.barplot(y=label, x=n)
plt.xlabel('Frequency')
plt.ylabel('Label')
plt.yticks(rotation=45)
plt.savefig('./count_100.png')
plt.show()
