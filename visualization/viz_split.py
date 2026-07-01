import matplotlib.pyplot as plt

labels = ['Train', 'Val', 'Test']
values = [300, 80, 100]


plt.figure(figsize=(10, 10))
plt.bar(labels, values, color=['skyblue', 'orange', 'green'])

plt.title('Dataset Split', fontsize=15)
plt.xlabel('Data', fontsize=15)
plt.ylabel('Number of samples', fontsize=15)

for i, v in enumerate(values):
    plt.text(i, v + 5, str(v), ha='center', fontsize=12)

plt.savefig("data_split.jpg", bbox_inches='tight')
plt.show()