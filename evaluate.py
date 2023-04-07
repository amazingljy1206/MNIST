from main import MNIST_MLP

def evaluate_mnist_mlp(param_dir='model_trained.npy'):
    mlp = MNIST_MLP(hidden1=256, hidden2=128)
    mlp.load_data()
    mlp.build_model()
    mlp.init_model()
    mlp.load_model(param_dir)
    mlp.evaluate()

if __name__ == '__main__':
    evaluate_mnist_mlp()