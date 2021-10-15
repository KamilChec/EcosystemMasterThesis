import dataset as d
import neuralNetwork as nn

def main():
    data = d.Dataset()
    input, output = data.get_dataset()
    model = nn.NeuralNetwork(input, output)
    model.build_model_with_backpropagation(3, print_loss=True)
    data.plot_decision_boundary(lambda x: model.predict(x))



if __name__ == "__main__":
    main()