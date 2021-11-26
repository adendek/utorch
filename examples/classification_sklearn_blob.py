import utorch.nets as nets
from sklearn import datasets


def build_dataset(config):
    train_x, train_y = datasets.make_classification(n_features=config["n_features"],
                                                    n_samples=config["n_samples"],
                                                    n_redundant=0,
                                                    n_classes=config["n_classes"])
    train_y = train_y.astype(int)
    return nets.DataLoader(train_x, train_y, config["batch_size"])


class FullyConnected2NN(nets.Model):
    def __init__(self, param_dict):
        self.layers = nets.StackedLayers([
            nets.LinearLayer(n_input=param_dict["n_input"], n_hidden=param_dict["n_hidden_1"], has_bias=param_dict["bias"],
                        name="input_layer"),
            nets.ReLULayer(),
            nets.LinearLayer(n_input=param_dict["n_hidden_1"], n_hidden=param_dict["n_hidden_2"],
                        has_bias=param_dict["bias"], name="hidden_layer_1"),
            nets.ReLULayer(),
            nets.LinearLayer(n_input=param_dict["n_hidden_2"], n_hidden=param_dict["n_class"], has_bias=False,
                        name="hidden_layer_2"),
        ]
        )

    def forward(self, x, *args, **kwargs):
        return self.layers(x)




def train_model(model, criterion, optimizer, run_hist, data_loader, num_epochs=10):
    for epoch in range(num_epochs):
        for iteration, batch in enumerate(data_loader):
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs,  labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.update_model()

            run_hist["loss"].append(loss.value)
            if iteration % 100 == 0:
                print("epoch {}/{},  it  {}/{}, loss {} ".format(epoch, num_epochs,
                                                             iteration, len(data_loader),
                                                             loss.value))


if __name__ == "__main__":
    data_config = {
        "n_features": 20,
        "n_samples": 256000,
        "n_classes": 2,
        "batch_size": 256
    }
    dataset = build_dataset(data_config)

    FC_2nn_params = {
        "bias": True,
        "n_input": 20,
        "n_hidden_1": 5,
        "n_hidden_2": 3,
        "n_class": 2
    }
    model = FullyConnected2NN(FC_2nn_params)
    optim = nets.SGD(model, 0.001)
    criterion = nets.CrossEntropyWithLogitsLoss(data_config["n_classes"])
    run_hist = {"loss": []}
    train_model(model, criterion, optim, run_hist, dataset)






