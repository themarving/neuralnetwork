package neuralnetwork;

import java.util.Random;

public class NeuronLayer <T extends Number> {
    
    Random random = new Random();

    private Double[][] weights;
    private Double[] biases;

    private Integer nInputs;
    private Integer nNeurons;

    private Double[][] X;

    public NeuronLayer(T[][] X, Integer nInputs, Integer nNeurons) throws InvalidDataException {
        this.X = new Double[X.length][X[0].length];

        // checking for valid data
        try {
            for (int i = 0; i < X.length; i++) {
                for (int j = 0; j < X[0].length; j++) {
                    // checking for uninitialized data
                    if (X[i][j] == null) {
                        throw new InvalidDataException("data invalid!");
                    }

                    // double conversion from generics
                    this.X[i][j] = X[i][j].doubleValue();
                }
            }
        } catch (ArrayIndexOutOfBoundsException e) {
            throw new InvalidDataException("array has wrong shape!");
        }

        this.nInputs = nInputs;
        this.nNeurons = nNeurons;

        this.weights = new Double[nInputs][nNeurons];
        this.biases = new Double[nNeurons];

        // creating weights based on gaussian (normal) distribution
        for (int i = 0; i < nInputs; i++) {
            for (int j = 0; j < nNeurons; j++) {
                weights[i][j] = 0.1 * random.nextGaussian();
            }
        }

        // initializing biases as 0
        for (int i = 0; i < nNeurons; i++) {
            biases[i] = 0.0;
        }
    }

    // forwarding the network
    public Double[][] forward() {
        Double[][] output = new Double[X.length][nNeurons];

        // calculating output for each input X times neuron N weight
        for (int neuron = 0; neuron < nNeurons; neuron++) {
            for (int i = 0; i < nInputs; i++) {
                for (int j = 0; j < X.length; j++) {
                    if (output[j][neuron] == null) {
                        output[j][neuron] = X[j][i] * weights[i][neuron];
                    }
                    else {
                        output[j][neuron] += X[j][i] * weights[i][neuron];
                    }
                }
            }
        }

        // adding biases for each neuron
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < nNeurons; j++) {
                output[i][j] += biases[j];
            }
        }

        return output;
    }

}
