import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
import multiprocessing

# Transformaciones de datos para normalizar y transformar las imágenes a tensores de PyTorch
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Descargar y cargar los datos de CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)


def knn_train(trainloader):
    # Obtener los datos de entrenamiento y etiquetas de CIFAR-10
    data = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        for input, target in zip(inputs, targets):
            data.append(input.numpy().flatten())
            labels.append(target.numpy())

    # Entrenar el modelo KNN
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(data, labels)

    return knn


def knn_test(testloader, knn):
    # Obtener los datos de prueba y etiquetas de CIFAR-10
    data = []
    labels = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        for input, target in zip(inputs, targets):
            data.append(input.numpy().flatten())
            labels.append(target.numpy())

    # Evaluar el modelo KNN en el conjunto de datos de prueba
    accuracy = knn.score(data, labels)

    return accuracy


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Entrenar el modelo KNN en los datos de entrenamiento
    knn = knn_train(trainloader)

    # Evaluar el modelo KNN en los datos de prueba
    accuracy = knn_test(testloader, knn)

    print("La precisión del modelo KNN en CIFAR-10 es: {:.2f}%".format(accuracy * 100));
