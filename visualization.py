import matplotlib.pyplot as plt

def loss_visualization(path = 'visualization_result/loss.txt'):
    with open(path, 'r', encoding = 'UTF-8') as f:
        lines = f.readlines()
    loss = []
    for line in lines:
        loss.append(float(line.strip('\n')))
    plt.plot([i for i in range(len(loss))], loss)
    plt.title('Loss')
    plt.savefig('visualization_result/loss.jpg')
    plt.show()

def accuracy_visualization(path = 'visualization_result/accuracy.txt'):
    with open(path, 'r', encoding = 'UTF-8') as f:
        lines = f.readlines()
    accuracy = []
    for line in lines:
        accuracy.append(float(line.strip('\n')))
    plt.plot([i for i in range(len(accuracy))], accuracy)
    plt.title('Accuracy')
    plt.savefig('visualization_result/accuracy.jpg')
    plt.show()
 
if __name__ == '__main__':
    loss_visualization()
    accuracy_visualization()
    