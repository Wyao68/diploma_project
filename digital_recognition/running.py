if __name__ == "__main__":
    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    import network
    net = network.Network([784, 30, 10], network.sigmoid(), network.QuadraticCost())
    net.SGD(training_data, epochs=30, mini_batch_size=10, eta=0.5, 
            lmbda=0.1 ,mu=0.9,
            evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)

